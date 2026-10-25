"""Regression tests for the first five audited correctness/engineering defects."""
from __future__ import annotations
import numpy as np
import pytest
from pydantic import ValidationError
from pathlib import Path
from eeg_universal import load_raw
from eeg_agent.schema import Plan, PreprocessArgs
from eeg_agent.execution import execute_step
from eeg_agent.workflow_support import ready_batch, execute_ready
from eeg_agent.qc_policy import evaluate_tool
from eeg_agent.plan_versions import activate_plan
from test_graph_offline import AIMessage, simulated_graph, tools_and_file


def test_continuous_npz_auto_detects_v_and_mv_and_legacy(tmp_path):
    # A 10-uV sine stored in V, mV and uV respectively must have same amplitude.
    sfreq=250.
    for unit, peak in [('V',10e-6),('mV',1e-2),('uV',10.)]:
        path=tmp_path/f'wave_{unit}.npz'
        x=peak*np.sin(np.arange(1000)*2*np.pi*10/sfreq)
        np.savez(path,data=x[None,:],sfreq=sfreq,data_unit=unit,channel_names=['F3'])
        raw=load_raw(path)
        try:assert np.max(raw.get_data())==pytest.approx(10e-6*np.max(np.sin(np.arange(1000)*2*np.pi*10/sfreq)),rel=1e-6)
        finally:raw.close()
        with pytest.raises(ValueError,match='conflicts'):
            load_raw(path,mat_unit='uV' if unit!='uV' else 'V')
    legacy=tmp_path/'legacy.npz'
    np.savez(legacy,data=np.full((1,500),10.),sfreq=sfreq,channel_names=['F3'])
    raw=load_raw(legacy)
    try:assert raw.get_data()[0,0]==pytest.approx(10e-6)
    finally:raw.close()


def test_preprocess_args_exposes_input_unit_and_rejects_conflicts(tools_and_file):
    tools,source=tools_and_file
    with np.load(source) as z:
        content={k:z[k] for k in z.files}
    content['data_unit']=np.asarray('V')
    np.savez(source,**content)
    assert PreprocessArgs.model_validate({'source':str(source),'input_unit':'V'}).input_unit=='V'
    with pytest.raises(ValueError,match='conflicts'):
        tools.invoke('preprocess_eeg',{'source':str(source),'input_unit':'uV','use_ica':False})


def test_dag_unordered_dependencies_and_cycle_detection():
    # Forward and noncontiguous references are valid if the graph is acyclic.
    Plan.model_validate({'steps':[
        {'id':'E3','tool':'inspect_eeg','args':{'source':'$input'},'depends_on':['E2']},
        {'id':'E1','tool':'inspect_eeg','args':{'source':'$input'},'depends_on':[]},
        {'id':'E2','tool':'inspect_eeg','args':{'source':'$input'},'depends_on':['E1']} ]})
    with pytest.raises(ValidationError,match='Cyclic'):
        Plan.model_validate({'steps':[
            {'id':'E1','tool':'inspect_eeg','args':{'source':'$input'},'depends_on':['E2']},
            {'id':'E2','tool':'inspect_eeg','args':{'source':'$input'},'depends_on':['E1']} ]})


def test_dag_ready_across_gaps_and_replan_isolation(tools_and_file):
    tools,source=tools_and_file
    initial={'input_path':str(source),'evidence':{},'trace':[],
             'plan_version':0,'plan_history':[]}
    # E1 is blocked by E2, but E2 and E3 can run before E1.
    plan=[{'id':'E1','tool':'inspect_eeg','args':{'source':'$E2.source'},'depends_on':['E2']},
          {'id':'E2','tool':'inspect_eeg','args':{'source':'$input'},'depends_on':[]},
          {'id':'E3','tool':'validate_eeg','args':{'source':'$input'},'depends_on':[]}]
    updated=activate_plan(initial,plan)
    state={**initial,**updated}
    batch=ready_batch(state,tools,0,max_parallel=3)
    assert [x.id for x in batch]==['E2','E3']
    result=execute_ready(batch,state,tools)
    state.update(result)
    assert state['step_status']['E2']=='completed'
    assert [x.id for x in ready_batch(state,tools,99)]==['E1']
    state.update(execute_ready(ready_batch(state,tools),state,tools))
    assert set(state['evidence'])=={'E1','E2','E3'}
    revision=activate_plan(state,[{'id':'E1','tool':'inspect_eeg','args':{'source':'$input'},'depends_on':[]}])
    state.update(revision)
    assert state['step_status']=={}
    assert [x.id for x in ready_batch(state,tools)]==['P2:E1']
    assert state['evidence']['E1']['source']==str(source)


def test_failing_step_records_duration_and_failure(tools_and_file):
    tools,source=tools_and_file
    state={'input_path':str(source),'evidence':{},'trace':[]}
    step={'id':'E1','tool':'analyze_spectrum','args':{'source':'$E9.output_npz'},'depends_on':[]}
    from eeg_agent.schema import Step
    result=execute_step(Step.model_validate(step),state,tools)
    assert result['error'].startswith('ValueError:')
    assert result['trace'][0]['status']=='error'
    assert result['trace'][0]['attempt_id']=='E1:attempt:1'
    assert result['trace'][0]['duration_ms']>=0
    assert result['trace'][0]['started_at']
    assert result['trace'][0]['finished_at']


def test_qc_gate_stops_flagged_and_missing_preflight():
    for status in ('requires_review','unavailable'):
        decision=evaluate_tool('preprocess_eeg',{'target_sfreq':250}, {'status':status})
        assert decision.action=='review'
    assert evaluate_tool('inspect_eeg',{}, {'status':'requires_review'}).action=='allow'
    assert evaluate_tool('preprocess_eeg',{'target_sfreq':250,'bandpass_high':150},
                         {'status':'pass'}).action=='reject'


def test_react_failed_tool_never_claims_completed(simulated_graph,tools_and_file,monkeypatch):
    tools,source=tools_and_file
    class FailingModel:
        def react(self,*a):
            return AIMessage(tool_calls=[{'id':'c1','name':'inspect_eeg','args':{'source':'$input'}}])
    def broken(*a):raise RuntimeError('synthetic failure')
    monkeypatch.setattr(tools,'invoke',broken)
    state=simulated_graph.AgentGraphs(FailingModel(),tools,max_steps=4).build('react').invoke({
        'task':'inspect','input_path':str(source),'cursor':0,'iterations':0,'replans':0,
        'steps':[],'evidence':{},'trace':[],'error':None,'status':'running'})
    assert state['status']=='failed'
    assert 'synthetic failure' in state['error']
    assert state['tool_failures']==1
    assert state['trace'][0]['status']=='error'
    assert state['trace'][0]['attempt_id']=='c1:attempt:1'
    assert 'Stopped:' in state['final_answer']


def test_react_qc_policy_blocks_preprocessing_without_approval(simulated_graph,tools_and_file,monkeypatch):
    tools,source=tools_and_file
    sfreq=200.
    x=np.random.default_rng(2).normal(0,2,(8,4,200))
    x[:,0,:]=0.
    artifact=source.with_name('qc_flagged.npz')
    np.savez(artifact,data=x,sfreq=sfreq,channel_names=['F3','F4','C3','C4'],data_unit='uV')
    original=tools.invoke
    called=[]
    def invoked(name,args):
        if name=='preprocess_eeg':called.append(args)
        return original(name,args)
    monkeypatch.setattr(tools,'invoke',invoked)
    class WantsUnsafe:
        def react(self,*a):
            return AIMessage(tool_calls=[{'id':'c1','name':'preprocess_eeg',
                'args':{'source':'$input','use_ica':False}}])
    state=simulated_graph.AgentGraphs(WantsUnsafe(),tools,max_steps=3).build('react').invoke({
        'task':'preprocess','input_path':str(artifact),'cursor':0,'iterations':0,
        'steps':[],'evidence':{},'trace':[],'error':None,'status':'running'})
    assert state['qc_summary']['status']=='requires_review'
    assert state['status']=='failed'
    assert 'QC approval required' in state['error']
    assert not called


def test_progress_includes_failed_phase_and_context(simulated_graph,tools_and_file,monkeypatch):
    tools,source=tools_and_file
    events=[]
    monkeypatch.setattr(simulated_graph.AgentGraphs,'_progress',
                        staticmethod(lambda name,phase,**kw:events.append({'tool':name,'phase':phase,**kw})))
    class Break:
        def plan(self,s,tools,**kw):
            return [{'id':'E1','tool':'analyze_spectrum',
                     'args':{'source':'$input'},'depends_on':[]}]
        def summarize(self,s):return 'Trace includes failure.'
    state=simulated_graph.AgentGraphs(Break(),tools,max_replans=0).build('plan-execute').invoke({
        'task':'analyze','input_path':str(source),'cursor':0,'iterations':0,
        'replans':0,'steps':[],'evidence':{},'trace':[],'error':None,'status':'running'})
    assert state['status']=='failed'
    assert events[-1]['phase']=='failed'
    assert events[-1]['step_id']=='E1'
    assert events[-1]['error']
    assert state['trace'][-1]['finished_at']


def test_dag_dynamic_registered_tool_works_across_plan_activation(tools_and_file):
    from eeg_agent.schema import StrictModel
    tools,source=tools_and_file
    class Args(StrictModel):
        source:str
    tools.register_tool('custom_read', 'Read-only third party extension', Args,
                        lambda args:{'source':args.source},parallel_safe=True)
    state={'input_path':str(source),'evidence':{},'trace':[]}
    plan=activate_plan(state,[{'id':'E1','tool':'custom_read','args':{'source':'$input'},
                               'depends_on':[]}],allowed_tools=set(tools.registry))
    state.update(plan)
    batch=ready_batch(state,tools)
    assert [step.tool for step in batch]==['custom_read']
    outcome=execute_ready(batch,state,tools)
    assert outcome['evidence']['E1']['source']==str(source)
