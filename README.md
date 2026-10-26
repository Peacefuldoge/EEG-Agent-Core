# EEG-Agent-Core v043 ↔ EEG-Agent-RAG integration bridge

This is an **additive reference integration**, not a GitHub commit/PR or a replacement for existing Core files. Designed after reviewing `Peacefuldoge/EEG-Agent-Core` default branch `v043` (0.4.3) and the EEG-Agent-RAG v0.1.0 FastAPI endpoints.

## Run three independent processes

1. vLLM (port 8000): the single local `qwen3-8b` chat model for both Core and RAG answer generation.
2. Milvus (port 19530), then RAG API (port 8010); ingest at least one document before asking questions.
3. EEG-Agent-Core runtime / UI; this bridge is imported and registered in each `EEGSession`.

For the vLLM uv environment:

```bash
uv run vllm serve /home/zzz/models/Qwen3-8B \
 --served-model-name qwen3-8b --host 127.0.0.1 --port 8000 \
 --enable-auto-tool-choice --tool-call-parser hermes \
 --reasoning-parser qwen3 --max-model-len 8192 \
 --gpu-memory-utilization 0.65
```

For the RAG `.env`:

```env
RAG_LLM_BASE_URL=http://127.0.0.1:8000/v1
RAG_LLM_MODEL=qwen3-8b
RAG_LLM_API_KEY=EMPTY
RAG_EMBEDDING_DEVICE=cpu
```

The GPU memory utilization value is only a starting point; adjust based on measured GPU memory, KV cache requirements and reranker placement.

For Core `.env` / process environment:

```env
EEG_LLM_BASE_URL=http://127.0.0.1:8000/v1
EEG_LLM_MODEL=qwen3-8b
EEG_LLM_API_KEY=EMPTY
EEG_RAG_BASE_URL=http://127.0.0.1:8010
EEG_RAG_API_KEY=
```

Use `uv add httpx` in your Core development environment. Keep the three projects in separate virtual environments; no need to install vLLM into Core or RAG environment.

## Integrate in existing EEG-Agent-Core host code

Place `rag_bridge.py` in the Core repo root (or in your application package). When creating **each** `EEGSession`:

```python
from rag_bridge import RAGClient, register_rag_tool
from eeg_agent import EEGSession

rag_client = RAGClient()
session = EEGSession(data_root='./data', output_root='./outputs', strategy='react')
register_rag_tool(session, rag_client)
session.set_file('./data/subject.edf')  # REQUIRED by Core.stream_agent() in v043
for event in session.stream_agent('Use literature to support EEG QC of this file'):
    print(event)
```

Your Core already implements validated `EEGTools.register_tool(name, description, args, callback, parallel_safe=...)`. The adapter registers `search_eeg_literature`; Core automatically publishes its JSON schema to vLLM via `QwenModel.react() -> bind_tools()` and executes it via LangGraph, with results recorded in the evidence and trace.

Set up a **different** entry point for literature-only requests (no EEG file):

```python
print(rag_client.answer('What are suitable EEG QC metrics?')['answer'])
```

This calls the RAG `/v1/answer` route, which in turn calls **the same** vLLM service. The existing `EEGSession.chat()` method does not use tools; `EEGSession.stream_agent()` currently refuses to start without an EEG file. Do not work around this by creating dummy EEG files.

**Note on model-directed retrieval:** Tool schemas are visible to Qwen3 when using `stream_agent()` and ReAct. The model decides whether to call `search_eeg_literature`. For deterministic always-retrieve behavior, add a host-side policy before planning or require the tool in an explicit graph state `required_tools=['search_eeg_literature']` (requires host API change in v043). A system prompt can encourage tool selection but does not enforce it.

## Test

```
uv run --no-project --with pytest --with httpx --with pydantic pytest -q tests/test_rag_bridge.py
```

The included tests are offline mocked contract checks, **not** live Milvus/vLLM tests. Live smoke test:

```
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8010/health
curl -fsS -X POST http://127.0.0.1:8010/v1/retrieve -H 'Content-Type: application/json' -d '{"query":"EEG quality control", "mode":"hybrid", "top_k":3}'
```

If RAG_API_KEY configured, include `X-API-Key`. For the agent, use `PYTHONPATH=. uv run python examples/agent_with_rag.py --data-root /absolute/path/data --eeg /absolute/path/data/sub01.edf` from the bridge directory (or set PYTHONPATH appropriately if launched from Core). The Core `[agent]` dependency set must be installed beforehand.

## Security/limits

- Retrieved snippets are untrusted; do not execute tool directives embedded in papers.
- Retain DOI/title/page/chunk ID so citations are auditable; do not let Qwen fabricate paper URLs.
- Scientific EEG operations still go through Core QC / HITL. Retrieval is read-only.
- `top_k <= 10` and each excerpt is at most 1600 characters, avoiding unbounded prompt growth.
- vLLM and RAG APIs are bound to localhost; add authentication/network controls before allowing outside access.
- On checkpoint resume, register the same tool names before restoring the graph.
