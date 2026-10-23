"""Graph NODE tests with a tiny local graph simulator when LangGraph is unavailable.

This does NOT claim external LangGraph integration was tested; the real package
and a local running Qwen/vLLM endpoint are still required for integration tests.
"""
import importlib
import sys
import types
from pathlib import Path
import numpy as np
import pytest
from eeg_agent.tools import EEGTools

class Msg:
    def __init__(self,content='',tool_calls=None,**kwargs):
        self.content=content
        self.tool_calls=tool_calls or []

class AIMessage(Msg):pass
class HumanMessage(Msg):pass
class SystemMessage(Msg):pass
class ToolMessage(Msg):pass

class MiniGraph:
    def __init__(self,*args,**kwargs):
        self.nodes={};self.edges={};self.routes={}
    def add_node(self,name,fn):self.nodes[name]=fn
    def add_edge(self,start,dest):self.edges[start]=dest
    def add_conditional_edges(self,src,fn,mapping):self.routes[src]=(fn,mapping)
    def compile(self,checkpointer=None):return self
    def invoke(self,initial,config=None):
        state=dict(initial);node=self.edges['START']
        for _ in range(60):
            if node=='END':return state
            patch=self.nodes[node](state)
            if 'messages' in patch:
                patch['messages']=state.get('messages',[])+patch['messages']
            state.update(patch)
            if node in self.routes:
                fn,mapping=self.routes[node]
                node=mapping[fn(state)]
            else:
                node=self.edges[node]
        raise RuntimeError('mini-graph infinite loop')

class FakeModel:
    def __init__(self):self.calls=0
    def plan(self,state,tools,**kwargs):
        return [{'id':'E1','tool':'inspect_eeg','args':{'source':'$input'},'depends_on':[]},
                {'id':'E2','tool':'preprocess_eeg','args':{'source':'$input','target_count':3,'window_seconds':1,'use_ica':False},'depends_on':['E1']},
                {'id':'E3','tool':'analyze_spectrum','args':{'source':'$E2.output_npz'},'depends_on':['E2']}]
    def react(self,messages,tool_schemas):
        self.calls+=1
        names=['inspect_eeg','preprocess_eeg','analyze_spectrum']
        if self.calls<=3:
            a={'source':'$input'} if self.calls==1 else (
              {'source':'$input','target_count':3,'window_seconds':1,'use_ica':False} if self.calls==2 else
              {'source':'$latest_npz'})
            return AIMessage(tool_calls=[{'id':f'call{self.calls}','name':names[self.calls-1],'args':a}])
        return AIMessage(content='The three tool results were observed. No medical diagnosis.')
    def summarize(self,s):return f"Evidence from {len(s['evidence'])} completed EEG operations."

@pytest.fixture
def simulated_graph(monkeypatch):
    graph=types.ModuleType('langgraph.graph')
    graph.StateGraph=MiniGraph;graph.START='START';graph.END='END'
    root=types.ModuleType('langgraph')
    message=types.ModuleType('langgraph.graph.message');message.add_messages=lambda a,b:a+b
    langchain_core=types.ModuleType('langchain_core')
    lc_msgs=types.ModuleType('langchain_core.messages')
    for k,v in [('AIMessage',AIMessage),('HumanMessage',HumanMessage),('SystemMessage',SystemMessage),('ToolMessage',ToolMessage)]:
        setattr(lc_msgs,k,v)
    with monkeypatch.context() as mp:
        for k,v in [('langgraph',root),('langgraph.graph',graph),('langgraph.graph.message',message),
                    ('langchain_core',langchain_core),('langchain_core.messages',lc_msgs)]:
            mp.setitem(sys.modules,k,v)
        sys.modules.pop('eeg_agent.strategies',None)
        yield importlib.import_module('eeg_agent.strategies')
    sys.modules.pop('eeg_agent.strategies',None)

@pytest.fixture
def tools_and_file(tmp_path):
    data=tmp_path/'data';data.mkdir()
    t=np.arange(1000)/200
    x=np.array([12*np.sin(2*np.pi*(i+8)*t) for i in range(4)])
    source=data/'small.npz'
    np.savez(source,data=x,sfreq=np.array(200.),channel_names=np.array(['F3','F4','C3','C4']))
    return EEGTools(data,tmp_path/'out'),source

@pytest.mark.parametrize('strategy',['plan-execute','rewoo','react'])
def test_end_to_end_graph_nodes_simulated(simulated_graph,tools_and_file,strategy):
    tools,source=tools_and_file
    fake=FakeModel()
    graph=simulated_graph.AgentGraphs(fake,tools,max_steps=8).build(strategy)
    state=graph.invoke({'task':'preprocess and analyze','input_path':str(source),
            'cursor':0,'iterations':0,'replans':0,'steps':[],
            'evidence':{},'trace':[],'error':None,'status':'running'})
    assert state['status']=='completed',state.get('error')
    assert len(state['evidence'])==3
    assert Path(state['latest_npz']).exists()
    assert len(state['trace'])==3
    assert state['final_answer']

def test_step_budget_does_not_claim_success(simulated_graph,tools_and_file):
    tools,source=tools_and_file
    fake=FakeModel()
    graph=simulated_graph.AgentGraphs(fake,tools,max_steps=2,max_replans=0).build('rewoo')
    state=graph.invoke({'task':'full pipeline','input_path':str(source),'cursor':0,
            'iterations':0,'replans':0,'steps':[],'evidence':{},'trace':[],'error':None})
    assert state['status']=='failed'
    assert 'budget' in state['error']
