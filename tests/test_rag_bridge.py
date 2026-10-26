"""Offline contract tests: no Milvus, vLLM or GPU required."""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag_bridge import RAGClient, register_rag_tool


def _mock_client(callback):
    return httpx.Client(transport=httpx.MockTransport(callback))


def test_retrieve_bounded_cited_and_header():
    def handler(req):
        assert req.url.path == '/v1/retrieve'
        assert req.headers.get('x-api-key') == 'secret'
        assert req.read().find(b'"top_k":5') >= 0
        return httpx.Response(200, json={'query':'EEG', 'results':[
            {'document_id':'doc1','chunk_id':'chunk1','title':'Paper',
             'doi':'10.test/xyz','content':'data'*600,'page':7},
        ]})
    with _mock_client(handler) as http:
        client=RAGClient(api_key='secret', http_client=http)
        result=client.retrieve('EEG evidence')
        assert result['status']=='ok'
        assert result['evidence_count']==1
        assert result['sources'][0]['citation']=='S1'
        assert result['sources'][0]['doi']=='10.test/xyz'
        assert len(result['sources'][0]['excerpt'])==1600


def test_retrieve_no_results_ok():
    with _mock_client(lambda req:httpx.Response(200,json={'results':[]})) as http:
        result=RAGClient(http_client=http).retrieve('Unknown document')
    assert result['evidence_count']==0
    assert result['sources']==[]


def test_retrieve_invalid_payload_rejected():
    with _mock_client(lambda req:httpx.Response(200,json={'results':None})) as http:
        with pytest.raises(ValueError, match='results list'):
            RAGClient(http_client=http).retrieve('bad response')


def test_retrieve_http_error_raises():
    with _mock_client(lambda req:httpx.Response(503,json={'detail':'down'})) as http:
        with pytest.raises(httpx.HTTPStatusError):
            RAGClient(http_client=http).retrieve('Some question')


def test_answer_calls_knowledge_endpoint():
    def handler(req):
        assert req.url.path=='/v1/answer'
        return httpx.Response(200,json={'answer':'Supported [S1]', 'sources':[]})
    with _mock_client(handler) as http:
        assert RAGClient(http_client=http).answer('What is ICA?')['answer']=='Supported [S1]'


def test_tool_registration_is_strict(monkeypatch):
    fake_pkg = types.ModuleType('eeg_agent')
    fake_schema = types.ModuleType('eeg_agent.schema')
    class StrictModel(BaseModel):
        model_config=ConfigDict(extra='forbid')
    fake_schema.StrictModel=StrictModel
    monkeypatch.setitem(sys.modules, 'eeg_agent', fake_pkg)
    monkeypatch.setitem(sys.modules, 'eeg_agent.schema', fake_schema)
    client = Mock(spec=RAGClient)
    client.retrieve.return_value={'status':'ok', 'sources':[], 'evidence_count':0}
    tools = Mock()
    session = Mock(tools=tools)
    register_rag_tool(session, client)
    name, description, model, callback = tools.register_tool.call_args.args
    assert name == 'search_eeg_literature'
    assert tools.register_tool.call_args.kwargs == {'parallel_safe':True}
    with pytest.raises(Exception):
        model.model_validate({'query':'EEG', 'top_k':11})
    args=model.model_validate({'query':'How to check bad EEG channels?'})
    assert callback(args)['status']=='ok'
    client.retrieve.assert_called_once_with(**args.model_dump())


def test_reject_unbounded_top_k():
    with _mock_client(lambda req: (_ for _ in ()).throw(AssertionError('unexpected HTTP'))) as http:
        with pytest.raises(ValueError):
            RAGClient(http_client=http).retrieve('question',top_k=999)
