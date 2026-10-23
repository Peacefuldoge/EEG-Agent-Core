"""Offline interactive session tests; do not claim live vLLM/Chainlit testing."""
from __future__ import annotations
import json
import sys
import types
from pathlib import Path
import numpy as np
import pytest
from eeg_agent.session import EEGSession


def fixture_files(tmp_path):
    data = tmp_path/'data'; data.mkdir()
    src = data/'a.npz'
    np.savez(src, data=np.zeros((2, 3, 100)), sfreq=np.asarray(100))
    return data, src, tmp_path/'out'


def test_session_paths_and_strategy(tmp_path):
    data,src,out=fixture_files(tmp_path)
    s=EEGSession(data_root=data,output_root=out,checkpoint=False)
    assert s.set_file(src)==str(src.resolve())
    assert s.strategy=='react'
    s.set_strategy('rewoo')
    assert s.strategy=='rewoo'
    with pytest.raises(ValueError):s.set_strategy('invented')
    other=tmp_path/'outside.edf';other.write_bytes(b'raw')
    with pytest.raises(ValueError):s.set_file(other)


def test_no_file_produces_error(tmp_path):
    data,_,out=fixture_files(tmp_path)
    s=EEGSession(data_root=data,output_root=out,checkpoint=False)
    with pytest.raises(ValueError,match='choose an EEG file'):
        s.run_agent('inspect')


def test_repl_saves_independent_run_and_last_npz(monkeypatch,tmp_path):
    data,src,out=fixture_files(tmp_path)
    produced=out/'processed.npz'
    out.mkdir()
    np.savez(produced,data=np.zeros((1,3,100)),sfreq=np.array(100.))
    class FakeGraph:
        def stream(self, initial, *, config, stream_mode):
            assert stream_mode=='updates'
            assert config['configurable']['thread_id']
            assert initial['input_path']==str(src.resolve())
            yield {'plan': {'steps':[{'id':'E1'}]}}
            yield {'execute': {'latest_npz':str(produced), 'trace':[{'tool':'inspect_eeg','status':'ok'}]}}
            yield {'summary': {'final_answer':'Computed result', 'status':'completed'}}
    class FakeAgentGraphs:
        def __init__(self, *args, **kwargs):pass
        def build(self,*args):return FakeGraph()
    mp=types.ModuleType('eeg_agent.strategies')
    mp.AgentGraphs=FakeAgentGraphs
    monkeypatch.setitem(sys.modules,'eeg_agent.strategies',mp)
    s=EEGSession(data_root=data,output_root=out,checkpoint=False)
    s.set_file(src);s._model=object()
    nodes=[]
    r=s.run_agent('inspect EEG',on_node=lambda n,v:nodes.append(n))
    assert r['status']=='completed'
    assert r['answer']=='Computed result'
    assert nodes==['plan','execute','summary']
    assert Path(r['manifest']).is_file()
    assert s.current_file==str(produced.resolve())
    assert len(s.history)==1
    manifest=json.loads(Path(r['manifest']).read_text(encoding='utf-8'))
    assert manifest['input_file']==str(src.resolve())
    assert manifest['trace'][0]['tool']=='inspect_eeg'
