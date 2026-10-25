"""Optional real LangGraph smoke tests; skipped when the extra is not installed."""
from __future__ import annotations
import numpy as np
import pytest
pytest.importorskip('langgraph')
pytest.importorskip('langchain_core')
from langchain_core.messages import AIMessage
from eeg_agent.strategies import AgentGraphs
from eeg_agent.tools import EEGTools


@pytest.mark.parametrize('strategy',['react','plan-execute','rewoo'])
def test_graph_executes_under_real_langgraph(tmp_path,strategy):
    source_root=tmp_path/'data';source_root.mkdir()
    source=source_root/'source.npz'
    sfreq=200.
    signal=10*np.sin(2*np.pi*10*np.arange(1200)/sfreq)
    np.savez(source,data=signal[None,:],sfreq=sfreq,channel_names=['F3'],data_unit='uV')
    tools=EEGTools(source_root,tmp_path/'out')
    class Model:
        calls=0
        def plan(self,*args,**kw):
            return [{'id':'E1','tool':'inspect_eeg','args':{'source':'$input'},'depends_on':[]}]
        def react(self,*args):
            self.calls+=1
            if self.calls==1:
                return AIMessage(content='',tool_calls=[{'id':'call_1','name':'inspect_eeg','args':{'source':'$input'}}])
            return AIMessage(content='Inspection completed using tool evidence.')
        def summarize(self,*args):return 'Inspected EEG with evidence.'
    graph=AgentGraphs(Model(),tools,max_steps=4,max_replans=0).build(strategy)
    result=graph.invoke({'task':'inspect','input_path':str(source),'strategy':strategy,
                         'cursor':0,'iterations':0,'replans':0,'tool_cursor':0,'steps':[],
                         'evidence':{},'trace':[],'error':None,'status':'running'})
    assert result['status']=='completed'
    assert len(result['trace'])==1
    assert result['trace'][0]['status']=='ok'
    assert result['final_answer']
