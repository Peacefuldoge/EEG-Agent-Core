"""Check approval occurs BEFORE calling scientific tools, including ReAct batches."""
import pytest
from test_graph_offline import FakeModel,simulated_graph,tools_and_file


def test_gate_before_side_effects(simulated_graph,tools_and_file,monkeypatch):
    tools,source=tools_and_file
    calls=[]
    def gate(self,name,args,step_id=None):
        calls.append(name)
        return name!='preprocess_eeg'
    monkeypatch.setattr(simulated_graph.AgentGraphs,'_approval',gate)
    for strategy in ('plan-execute','rewoo','react'):
        calls.clear()
        model=FakeModel()
        graph=simulated_graph.AgentGraphs(model,tools,max_steps=8,approval_tools={'preprocess_eeg'}).build(strategy)
        state=graph.invoke({'task':'test','input_path':str(source),'cursor':0,'iterations':0,
                            'replans':0,'steps':[],'trace':[],'evidence':{},'error':None,'status':'running'})
        assert 'preprocess_eeg' in calls
        assert state['status']=='rejected',strategy
        assert not state.get('latest_npz'),strategy

@pytest.mark.parametrize('strategy',['plan-execute','rewoo','react'])
def test_langgraph_interrupt_is_not_swallowed(simulated_graph,tools_and_file,monkeypatch,strategy):
    tools,source=tools_and_file
    class SimulatedInterrupt(Exception):pass
    def pause(self,name,args,step_id=None):
        if name=='preprocess_eeg':raise SimulatedInterrupt('pause for human approval')
        return True
    monkeypatch.setattr(simulated_graph.AgentGraphs,'_approval',pause)
    graph=simulated_graph.AgentGraphs(FakeModel(),tools,max_steps=8,
                                      approval_tools={'preprocess_eeg'}).build(strategy)
    with pytest.raises(SimulatedInterrupt,match='human approval'):
        graph.invoke({'task':'test','input_path':str(source),'cursor':0,'iterations':0,
                      'replans':0,'steps':[],'trace':[],'evidence':{},'error':None,'status':'running'})
