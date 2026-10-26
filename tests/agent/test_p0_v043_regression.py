"""v0.4.3 regressions: DAG source proof, file-by-file QC and ReAct completion."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from eeg_agent.execution import parse_plan
from eeg_agent.plan_versions import activate_plan
from eeg_agent.schema import Plan
from eeg_agent.workflow_support import ready_batch
from eeg_agent.tools import EEGTools
from test_graph_offline import AIMessage, simulated_graph, tools_and_file


def _step(sid, tool, source='$input', deps=()):
    return {'id':sid,'tool':tool,'args':{'source':source},'depends_on':list(deps)}


def test_latest_reference_without_producer_is_rejected_before_any_execution(tools_and_file):
    tools,src=tools_and_file
    bad=[_step('E1','analyze_spectrum','$latest_npz'),
         _step('E2','preprocess_eeg','$input')]
    with pytest.raises(ValidationError,match='exactly one dependency-reachable'):
        Plan.model_validate({'steps':bad})
    with pytest.raises(ValidationError,match='exactly one dependency-reachable'):
        parse_plan({'steps':bad})
    with pytest.raises(ValidationError,match='exactly one dependency-reachable'):
        ready_batch({'steps':bad,'evidence':{}},tools)
    # Planning rejection is atomic: no new plan ID or stale mutable latest pointer.
    state={'plan_version':1,'latest_npz':'old.npz','evidence':{'E1':{'output_npz':'old.npz'}}}
    with pytest.raises(ValidationError,match='exactly one dependency-reachable'):
        activate_plan(state,bad)
    assert state['plan_version']==1 and state['latest_npz']=='old.npz'


def test_latest_reference_with_one_producer_is_bound_to_exact_evidence():
    steps=[_step('E3','analyze_spectrum','$latest_npz',['E2']),
           _step('E1','inspect_eeg'),
           _step('E2','preprocess_eeg','$input',['E1'])]
    plan=Plan.model_validate({'steps':steps})
    assert plan.steps[0].args['source']=='$E2.output_npz'
    assert parse_plan({'steps':steps})[0]['args']['source']=='$E2.output_npz'
    first=activate_plan({'plan_version':0,'evidence':{}},steps)
    assert first['steps'][0]['args']['source']=='$E2.output_npz'
    second=activate_plan({'plan_version':1,'evidence':{'E1':{'output_npz':'old'}}},steps)
    assert second['steps'][0]['args']['source']=='$P2:E2.output_npz'
    # Even if a parallel branch modifies the mutable latest pointer, this step
    # only ever references the correct producer evidence.
    assert all('$latest_npz' not in str(st['args']) for st in second['steps'])


def test_multiple_producers_ambiguous_even_if_both_finished():
    steps=[_step('E1','preprocess_eeg'),_step('E2','preprocess_eeg'),
           _step('E3','analyze_spectrum','$latest_npz',['E1','E2'])]
    with pytest.raises(ValidationError,match='found 2'):
        Plan.model_validate({'steps':steps})


def test_explicit_result_reference_must_be_dependency_reachable():
    with pytest.raises(ValidationError,match='requires depends_on'):
        Plan.model_validate({'steps':[
            _step('E1','analyze_spectrum','$E2.output_npz'),
            _step('E2','preprocess_eeg')]})
    # Qualified cross-plan evidence is permitted only by activate_plan's prior
    # plan evidence checks; a local Plan validator does not claim it exists.
    with pytest.raises(ValueError,match='Cross-plan evidence is not available'):
        activate_plan({'plan_version':1,'evidence':{}},[
            _step('E1','analyze_spectrum','$P1:E5.output_npz')])


def _make_raw(path: Path, *, flat: bool):
    sf=250.
    t=np.arange(1250)/sf
    x=np.vstack([
        23*np.sin(2*np.pi*10*t)+0.2*np.random.default_rng(1).normal(size=len(t)),
        17*np.sin(2*np.pi*8*t)+0.2*np.random.default_rng(2).normal(size=len(t)),
    ])
    if flat:x[0,:]=0
    np.savez(path,data=x.astype('float32'),sfreq=np.asarray(sf),
             channel_names=np.asarray(['F3','F4']),data_unit=np.asarray('uV'))


def test_batch_qc_is_per_file_and_prevents_unsafe_preprocessing(tmp_path,monkeypatch):
    data=tmp_path/'data';data.mkdir()
    good=data/'a_good.npz';bad=data/'b_flat.npz'
    _make_raw(good,flat=False);_make_raw(bad,flat=True)
    tools=EEGTools(data,tmp_path/'out')
    real=tools.invoke;preprocessed=[];quality=[]
    def spy(name,args):
        if name=='preprocess_eeg':preprocessed.append(args['source'])
        if name=='assess_eeg_quality':quality.append(args['source'])
        return real(name,args)
    monkeypatch.setattr(tools,'invoke',spy)
    result=tools.invoke('batch_preprocess',{
        'source_dir':str(data),'preprocessing':{'use_ica':False}})
    assert set(quality)=={str(good),str(bad)}
    assert preprocessed==[str(good)]
    assert result['status']=='partial_failure'
    assert (result['completed'],result['failed'],result['requires_review'])==(1,0,1)
    manifest=json.loads(Path(result['manifest_json']).read_text(encoding='utf-8'))
    assert manifest['files'][str(good)]['qc']['status']=='pass'
    assert manifest['files'][str(bad)]['status']=='requires_review'
    assert 'F3' in manifest['files'][str(bad)]['qc']['flagged_channels']
    assert 'output_npz' not in manifest['files'][str(bad)]
    # Batch-level approval does not grant permission to process the flagged file.
    repeated=tools.invoke('batch_preprocess',{'source_dir':str(data),
                                              'preprocessing':{'use_ica':False}})
    assert repeated['requires_review']==1
    assert preprocessed==[str(good),str(good)]  # second run is a content-cache hit
    assert repeated['first_files'][0]['cache_hit'] is True


def test_batch_qc_unavailable_is_review_not_preprocessing(tmp_path,monkeypatch):
    data=tmp_path/'data';data.mkdir()
    source=data/'bad.npz';source.write_bytes(b'not an eeg file')
    tools=EEGTools(data,tmp_path/'out')
    invoked=[];original=tools.invoke
    def spy(name,args):
        if name=='preprocess_eeg':invoked.append(name)
        return original(name,args)
    monkeypatch.setattr(tools,'invoke',spy)
    result=tools.invoke('batch_preprocess',{'source_dir':str(data),
                  'preprocessing':{'use_ica':False},'continue_on_error':False})
    assert result['requires_review']==1 and result['completed']==0
    assert result['status']=='partial_failure'
    assert not invoked
    manifest=json.loads(Path(result['manifest_json']).read_text())
    assert manifest['files'][str(source)]['qc']['status']=='unavailable'


def _react_state(source, **kwargs):
    return {'task':'analyze the EEG','input_path':str(source),'strategy':'react',
            'iterations':0,'replans':0,'tool_cursor':0,'steps':[],
            'evidence':{},'trace':[],'error':None,'status':'running',**kwargs}


def test_react_text_without_tool_cannot_claim_analysis_completed(simulated_graph,tools_and_file):
    tools,source=tools_and_file
    class TextOnly:
        def react(self,*args):return AIMessage(content='Analysis complete. Everything is normal.')
    graph=simulated_graph.AgentGraphs(TextOnly(),tools).build('react')
    state=graph.invoke(_react_state(source))
    assert state['status']=='failed'
    assert 'No successful tool execution evidence' in state['error']
    assert state['trace']==[]
    assert 'Stopped:' in state['final_answer']


def test_react_explicit_chat_mode_allows_text_without_tools(simulated_graph,tools_and_file):
    tools,source=tools_and_file
    class TextOnly:
        def react(self,*args):return AIMessage(content='A general explanation, not an EEG result.')
    graph=simulated_graph.AgentGraphs(TextOnly(),tools).build('react')
    state=graph.invoke(_react_state(source,task_mode='chat'))
    assert state['status']=='completed'
    assert 'general explanation' in state['final_answer']


def test_react_contract_requires_specific_tool_and_artifact(simulated_graph,tools_and_file):
    tools,source=tools_and_file
    class OneInspection:
        calls=0
        def react(self,*args):
            self.calls+=1
            if self.calls==1:
                return AIMessage(tool_calls=[{'id':'call1','name':'inspect_eeg','args':{'source':'$input'}}])
            return AIMessage(content='Completed preprocessing and spectrum analysis.')
    # An inspection is evidence, but not evidence for preprocessing.
    for contract in [
        {'required_tools':['preprocess_eeg']},
        {'required_artifacts':['output_npz']},
    ]:
        graph=simulated_graph.AgentGraphs(OneInspection(),tools).build('react')
        state=graph.invoke(_react_state(source,**contract))
        assert state['status']=='failed'
        assert len(state['trace'])==1 and state['trace'][0]['status']=='ok'
        assert state['error'].startswith('Required ')


def test_react_contract_passes_with_successful_tool(simulated_graph,tools_and_file):
    tools,source=tools_and_file
    class OneInspection:
        calls=0
        def react(self,*args):
            self.calls+=1
            if self.calls==1:
                return AIMessage(tool_calls=[{'id':'call1','name':'inspect_eeg','args':{'source':'$input'}}])
            return AIMessage(content='Verified EEG was inspected.')
    graph=simulated_graph.AgentGraphs(OneInspection(),tools).build('react')
    state=graph.invoke(_react_state(source,required_tools=['inspect_eeg'],
                                     required_artifacts=['n_channels']))
    assert state['status']=='completed'
    assert len(state['trace'])==1


def test_react_missing_artifact_file_does_not_count_as_output(simulated_graph,tools_and_file):
    tools,source=tools_and_file
    class TextOnly:
        def react(self,*args):return AIMessage(content='Analysis done.')
    graph=simulated_graph.AgentGraphs(TextOnly(),tools).build('react')
    state=graph.invoke(_react_state(source,required_artifacts=['output_npz'],
        trace=[{'tool':'preprocess_eeg','status':'ok',
                'result':{'output_npz':str(source.parent/'not_generated.npz')}}]))
    assert state['status']=='failed'
    assert 'output_npz' in state['error']
