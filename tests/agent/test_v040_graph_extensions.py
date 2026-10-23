"""No real LangGraph package here; verify nodes with existing offline simulator.
Actual interrupt/checkpointer integration requires installing LangGraph.
"""
import sys,types
from pathlib import Path
import numpy as np
import pytest
from eeg_agent.schema import PreprocessArgs
from test_graph_offline import simulated_graph,tools_and_file


def test_human_parameter_edit_requires_second_approval(monkeypatch, simulated_graph,tools_and_file):
    tools,src=tools_and_file
    requests=[]
    replies=iter([{'arguments':{'target_sfreq':200.,'use_ica':False}}, {'approved':True}])
    def interrupt(request):
        requests.append(request)
        return next(replies)
    mod=types.ModuleType('langgraph.types');mod.interrupt=interrupt
    monkeypatch.setitem(sys.modules,'langgraph.types',mod)
    engine=simulated_graph.AgentGraphs(None,tools,approval_tools={'preprocess_eeg'})
    orig={'source':str(src),'target_sfreq':250.,'use_ica':False}
    edited=engine._approval('preprocess_eeg',orig,'E2')
    assert edited['target_sfreq']==200.
    assert len(requests)==2
    assert requests[0]['arguments']['target_sfreq']==250.
    assert requests[1]['arguments']['target_sfreq']==200.
    assert orig['target_sfreq']==250.


def test_invalid_edit_rejected_before_second_approval(monkeypatch,simulated_graph,tools_and_file):
    tools,src=tools_and_file
    calls=[]
    def interrupt(request):
        calls.append(request)
        return {'arguments':{'target_sfreq':50.,'bandpass_high':70.}}
    mod=types.ModuleType('langgraph.types');mod.interrupt=interrupt
    monkeypatch.setitem(sys.modules,'langgraph.types',mod)
    engine=simulated_graph.AgentGraphs(None,tools,approval_tools={'preprocess_eeg'})
    with pytest.raises(Exception):
        engine._approval('preprocess_eeg',{'source':str(src),'use_ica':False})
    assert len(calls)==1


def test_qc_review_triggers_replanning(simulated_graph,tools_and_file):
    tools,source=tools_and_file
    # New 3D windows with obvious artifact in one channel.
    sf=200.;x=np.random.default_rng(99).normal(0,2,(8,4,200))
    x[:,0,:]=0
    sample=Path(source).with_name('artifact.npz')
    np.savez(sample,data=x,sfreq=np.asarray(sf),data_unit=np.asarray('uV'),
             channel_names=np.asarray(['F3','F4','C3','C4']))
    class Adaptive:
        def plan(self,state,tools,**kwargs):
            if state.get('replans',0)==0:
                return [{'id':'E1','tool':'assess_eeg_quality',
                    'args':{'source':'$input'},'depends_on':[]}]
            assert state['qc_summary']['status']=='requires_review'
            # A safe revised plan stops at introspection instead of immediately
            # running destructive preprocessing without a researcher decision.
            return [{'id':'E1','tool':'inspect_eeg','args':{'source':'$input'},'depends_on':[]}]
        def summarize(self,s):return 'QC-informed plan created'
    for strategy in ('plan-execute','rewoo'):
        graph=simulated_graph.AgentGraphs(Adaptive(),tools,max_steps=5,max_replans=1).build(strategy)
        state=graph.invoke({'task':'QC and replan','input_path':str(sample),
             'cursor':0,'iterations':0,'replans':0,'steps':[],
             'evidence':{},'trace':[],'error':None,'status':'running'})
        assert state['status']=='completed', (strategy,state.get('error'))
        assert state['replans']==1
        assert state['qc_summary']['status']=='requires_review'
        assert state['trace'][-1]['tool']=='inspect_eeg'

def test_invalid_edit_does_not_resume_checkpoint(monkeypatch, tmp_path):
    from eeg_agent.session import EEGSession
    data=tmp_path/'data';data.mkdir()
    sf=250;src=data/'x.npz'
    np.savez(src,data=np.random.default_rng(1).normal(size=(3,2,250)),
             sfreq=np.asarray(float(sf)),channel_names=np.asarray(['F3','F4']))
    s=EEGSession(data_root=data,output_root=tmp_path/'out')
    s.set_file(src)
    t='test_thread'
    s.store.create_run(session_id=s.session_id,thread_id=t,task='test',input_file=str(src),
                       strategy='react',model='fake')
    s.store.update_run(t,status='awaiting_approval',pending=[{'tool':'preprocess_eeg',
        'arguments':{'source':str(src),'target_sfreq':250,'use_ica':False}}])
    with pytest.raises(Exception):
        list(s.resume_run(t,parameters={'bandpass_high':140}))
    assert s.get_run(t)['status']=='awaiting_approval'
