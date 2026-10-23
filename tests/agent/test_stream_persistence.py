"""Contract tests run without LangGraph, vLLM or UI installed."""
from __future__ import annotations
import sqlite3
import sys
import types
from pathlib import Path
import numpy as np
import pytest
from eeg_agent.session import EEGSession
from eeg_agent.persistence import SessionStore


def dirs(tmp_path):
    d=tmp_path/'data';d.mkdir(exist_ok=True)
    x=d/'small.npz'
    np.savez(x,data=np.zeros((2,3,100)),sfreq=100)
    return d,x,tmp_path/'out'


def test_stable_sessions_chat_restored_with_history(tmp_path,monkeypatch):
    if 'langchain_core.messages' not in sys.modules:
        core=types.ModuleType('langchain_core')
        msgs=types.ModuleType('langchain_core.messages')
        class Message:
            def __init__(self,content):self.content=content
        for k in ('AIMessage','HumanMessage','SystemMessage'):
            setattr(msgs,k,Message)
        monkeypatch.setitem(sys.modules,'langchain_core',core)
        monkeypatch.setitem(sys.modules,'langchain_core.messages',msgs)
    d,x,o=dirs(tmp_path)
    class Chat:
        def stream(self,msgs):
            assert msgs[-1].content == 'hi' or msgs[-1].content == 'next'
            yield type('Chunk',(),{'content':'EEG '})()
            yield type('Chunk',(),{'content':'answer'})()
    model=Chat()
    s=EEGSession(data_root=d,output_root=o,llm=model)
    sid=s.session_id;s.set_file(x);s.set_strategy('rewoo')
    events=list(s.stream_chat('hi'))
    assert ''.join(e['text'] for e in events if e['type']=='token')=='EEG answer'
    assert s.history==[('hi','EEG answer')]
    restored=EEGSession(data_root=d,output_root=o,session_id=sid,llm=model)
    assert restored.current_file==str(x.resolve())
    assert restored.strategy=='rewoo'
    assert restored.history==[('hi','EEG answer')]
    assert any(z['session_id']==sid for z in restored.list_sessions())
    assert restored.chat('next')=='EEG answer'
    restored.clear()
    assert EEGSession(data_root=d,output_root=o,session_id=sid).history==[]


def test_session_run_store_and_cross_session_isolation(tmp_path):
    d,x,o=dirs(tmp_path)
    s=EEGSession(data_root=d,output_root=o)
    sid=s.session_id
    s.store.create_run(session_id=sid,thread_id='thread01',task='inspect',input_file=str(x),strategy='react',model='fake')
    s.store.update_run('thread01',status='awaiting_approval',pending=[{'tool':'preprocess_eeg'}])
    assert s.get_run('thread01')['pending'][0]['tool']=='preprocess_eeg'
    assert s.list_runs()[0]['status']=='awaiting_approval'
    s2=EEGSession(data_root=d,output_root=o)
    with pytest.raises(ValueError,match='does not belong'):
        s2.get_run('thread01')


def test_human_approval_cannot_be_enabled_without_checkpoint(tmp_path):
    d,_,o=dirs(tmp_path)
    with pytest.raises(ValueError,match='checkpoint'):
        EEGSession(data_root=d,output_root=o,checkpoint=False,approval_tools={'preprocess_eeg'})


def test_graph_stream_node_token_interrupt_and_resume(monkeypatch,tmp_path):
    d,x,o=dirs(tmp_path)
    class Snapshot:
        def __init__(self,pending=False):
            self.values={'task':'test','status':'running' if pending else 'completed',
                         'final_answer':'' if pending else 'done from checkpoint',
                         'trace':[],'evidence':{}}
            self.next=('tools',) if pending else ()
            self.tasks=[types.SimpleNamespace(interrupts=[types.SimpleNamespace(value={
                'type':'approval_required','tool':'preprocess_eeg','arguments':{'source':'$input'}})])] if pending else []
    class Command:
        def __init__(self, resume):self.resume=resume
    monkeypatch.setitem(sys.modules,'langgraph',types.ModuleType('langgraph'))
    types_mod=types.ModuleType('langgraph.types');types_mod.Command=Command
    monkeypatch.setitem(sys.modules,'langgraph.types',types_mod)
    class Graph:
        def __init__(self):self.pending=True
        def stream(self, initial, *, config,stream_mode,durability):
            assert 'messages' in stream_mode
            assert 'debug' in stream_mode
            self.pending=not isinstance(initial,Command)
            yield ('debug', {'type':'task','payload':{'name':'tools'}})
            yield ('messages', (types.SimpleNamespace(content='Hello'),{'langgraph_node':'reason'}))
            yield ('custom', {'tool':'inspect_eeg','phase':'started'})
            if self.pending:
                yield ('updates',{'__interrupt__':()})
            else:
                assert initial.resume=={'approved':True}
                yield ('updates',{'tools':{'status':'completed','final_answer':'done from checkpoint'}})
        def get_state(self,config):return Snapshot(self.pending)
    graph=Graph()
    class FakeAgentGraphs:
        def __init__(self,*args,**kwargs):pass
        def build(self,*args):return graph
    import eeg_agent.session as sess
    class DummySaver:pass
    class FakeConn:
        def close(self):pass
    monkeypatch.setattr(sess.EEGSession,'_checkpoint',lambda self:(DummySaver(),FakeConn()))
    mp=types.ModuleType('eeg_agent.strategies');mp.AgentGraphs=FakeAgentGraphs
    monkeypatch.setitem(sys.modules,'eeg_agent.strategies',mp)
    s=EEGSession(data_root=d,output_root=o)
    s.set_file(x)
    s._model=object()  # fake graph doesn't use actual LLM
    original=list(s.stream_agent('test'))
    t=original[0]['thread_id']
    assert any(z['type']=='node_started' for z in original)
    assert any(z['type']=='token' for z in original)
    assert any(z['type']=='tool_progress' for z in original)
    assert any(z['type']=='approval_required' for z in original)
    assert s.get_run(t)['status']=='awaiting_approval'
    assert s.get_run(t)['pending'][0]['tool']=='preprocess_eeg'
    restored=EEGSession(data_root=d,output_root=o,session_id=s.session_id)
    restored._model=object()
    with pytest.raises(ValueError,match='Approval decision required'):
        list(restored.resume_run(t))
    resumed=list(restored.resume_run(t,approved=True))
    assert resumed[-1]['record']['status']=='completed'
    assert restored.get_run(t)['status']=='completed'
    assert restored.history[-1][0]=='test'
