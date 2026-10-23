# EEG-Agent Core v0.3.2 — Streaming, approvals and durable sessions

This package contains no Web UI, terminal frontend or vLLM server. Existing host applications can install `eeg-agent-core[agent]` and use its Python API directly.

## What changed

- **Token streaming**: `EEGSession.stream_chat()` yields incremental text fragments. `EEGSession.stream_agent()` yields LangGraph `messages` fragments for answer/summarization nodes, while excluding raw planner JSON and tool-call arguments.
- **Node status**: events include `node_started` (`debug` stream), `node_completed` (`updates` stream), and `tool_progress` (`custom` stream). A long scientific tool currently reports start/finish, **not per-file percentages**.
- **Human approval**: by default, `preprocess_eeg` is gated by LangGraph `interrupt()` in all three strategies. Approval is required *before* calling MNE. `resume_run(thread_id, approved=True|False)` uses the **same** `thread_id` and `Command(resume=...)`.
- **Durable sessions**: `agent_sessions.sqlite3` stores session IDs, chat turns, run metadata, strategy and current file; `agent_checkpoints.sqlite3` stores LangGraph checkpoints. Individual outputs and manifests continue to live in `output_root`.
- **Recovery**: after restarting Python, create `EEGSession(..., session_id=previous_id)` and use `list_runs()` / `resume_run()`. Completed history is stored in full. Only the 10 most recent turns are automatically passed to Qwen as chat context.

## Installation (in the Agent environment, NOT the vLLM environment)

```bash
conda activate eeg_agent
cd EEG-Agent-Core
uv pip install -e '.[agent]' --python "$CONDA_PREFIX/bin/python"
```

The vLLM inference server remains an independent process at `http://127.0.0.1:8000/v1`. EEG-Agent Core does not install vLLM, CUDA, Chainlit or Qwen model weights.

## Local Python usage

```python
from eeg_agent import EEGSession

s = EEGSession(
    data_root='/mnt/d/EEG',
    output_root='./agent_runs',
    model='qwen3-8b',
    base_url='http://127.0.0.1:8000/v1',
    strategy='react',
    checkpoint=True,
)
print('SAVE THIS SESSION ID:', s.session_id)
s.set_file('/mnt/d/EEG/subject01.edf')

for event in s.stream_agent('Inspect and preprocess the current EEG'):
    if event['type'] == 'token':
        print(event['text'], end='', flush=True)
    elif event['type'] in ('node_started', 'node_completed', 'tool_progress'):
        print('\n', event)
    elif event['type'] == 'approval_required':
        thread_id = event['thread_id']
        print('Approval required:', event['requests'])
        # Stop here and ask the human. Can be resumed in another process.
        break

# Later, even after process restart:
s2 = EEGSession(data_root='/mnt/d/EEG', output_root='./agent_runs', session_id=s.session_id)
for event in s2.resume_run(thread_id, approved=True):
    print(event)
```

For ordinary local chat:

```python
for event in s.stream_chat('What is differential entropy in EEG?'):
    if event['type'] == 'token':
        print(event['text'],end='',flush=True)
```

To inspect history:

```python
s.get_history()     # all saved user/assistant messages
s.list_sessions()   # session IDs in this local output directory
s.list_runs()       # tasks and statuses for this session
```

## Host UI integration contract

Events are plain Python dictionaries with `type` and optional fields:

| Event | Fields | UI behavior |
|---|---|---|
| `token` | `text`, `node`, `scope` | append visible text |
| `node_started` | `node`, `thread_id` | show running step |
| `node_completed` | `node`, `status` | complete step |
| `tool_progress` | `data.tool`, `data.phase` | show tool state |
| `approval_required` | `thread_id`, `requests` | display Approve/Reject UI |
| `run_finished` | `record` | show final output / link to artifacts |
| `completed` | `answer` | normal chat completed |

The UI is free to use Chainlit, React, Rich or an existing LangGraph interface. For an existing LLM client, use `EEGSession(llm=my_chat_model, ...)`; for a custom graph, `build_agent(strategy=..., model=..., tools=..., checkpointer=..., approval_tools=...)`.

## Scientific and reliability limits

1. **Resume is checkpoint-level, not arbitrary instruction-level resume.** If a process stops inside a long ICA/preprocessing node, its checkpoint may precede that node and restarting can repeat the operation. For large batch pipelines, add per-file idempotency/caching before relying on crash recovery.
2. A saved `thread_id` must use the same strategy graph and compatible package version. Changing topology mid-run may invalidate checkpoints.
3. The default policy gates `preprocess_eeg`; you can pass `approval_tools={'preprocess_eeg', 'extract_features'}`. Approval cannot be enabled with `checkpoint=False`.
4. The first resume after a valid HITL interruption must explicitly supply `approved=True` or `approved=False`; the core will not interpret `None` as consent.
5. SQLite is intended for **trusted local single-user usage**. Do not expose it as a multi-tenant service without authentication, authorization, access controls, retention policy and encryption/PHI precautions. Session/run data can include paths and other potentially identifiable scientific metadata.
6. `debug` streaming is used internally to detect node starts but its full payload is never forwarded to the UI. Tool results are persisted in manifests; outputs must remain in the configured data/output roots.
7. Automated numerical tests and mocked LangGraph execution are provided. The source archive was **not** end-to-end tested against your live WSL2 vLLM/Qwen3-8B service, and Chainlit browser behavior requires local validation.
