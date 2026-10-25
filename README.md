[中文项目说明](README.zh-CN.md) · [中文可视化指南](docs/visualization.zh-CN.md)

> **v0.4.2**: Fix ReAct failure propagation, numeric NPZ units, shared QC gate, dependency-aware DAG scheduling, and tool failure tracing. See [FIXES_V042.md](FIXES_V042.md).
> **v0.4.1**: Automatic paired pre/post QC reports, standardized output metadata sidecars, and new Agent read-only tools. See [README_QC_METADATA.md](README_QC_METADATA.md).

# EEG-Agent Core v0.4.0

**Current release:** Independent QC Agent Tools, resumable batch preprocessing, six connectivity methods, scientific parameter reviews, QC-aware planning, read-only parallel steps, editable HITL requiring reapproval, dynamic strict tool registration, content-addressed cache, integrity reports and reproducibility manifests. See **[README_V040.md](README_V040.md)**.

The scientific package remains independent of Chainlit, CLI, vLLM and downloaded model weights.

---

## v0.3.2 additions

See **[README_RUNTIME.md](README_RUNTIME.md)** for streamed events, approvals, checkpoint recovery and Python API.

# EEG-Agent Core · v0.3.1

UI-agnostic EEG scientific tools with **optional** LangGraph execution strategies: Plan-and-Execute, ReAct, ReWOO. This project does **not** install Chainlit, a chat CLI, vLLM, or model weights. It works with the user’s own Python and model setup.

## Installation (Conda + uv, WSL2/Linux)

```bash
conda activate eeg_agent
uv pip install -e . --python "$CONDA_PREFIX/bin/python"             # EEG tools only
uv pip install -e ".[agent]" --python "$CONDA_PREFIX/bin/python"   # also LangGraph + plotting
```

Use the vLLM server in your **other** `vllm` Conda environment; the Agent consumes `http://127.0.0.1:8000/v1` and doesn't install vLLM.

## New: handcrafted features — differential entropy and relative EEG bandpower

The `extract_features` tool exports a `(windows, channels, 34)` matrix with
raw/band Gaussian differential entropy, delta/theta/alpha/beta/gamma absolute,
relative and log bandpower, Hjorth, spectral entropy, peak frequency and
standard temporal features. Relative powers use **1–45 Hz** as denominator.
It preserves source window/subject/label metadata when available.

```python
from eeg_agent import EEGTools, extract_handcrafted_features

tools = EEGTools('/mnt/d/EEG', './agent_runs')
result = tools.invoke('extract_features', {
    'source': '/mnt/d/EEG/processed/sub01_preprocessed.npz',
    'max_windows': 128,
})
print(result['output_csv'], result['features_npz'])
```

For raw EDF/BDF, call `preprocess_eeg` first. For a NumPy tensor, call
`extract_handcrafted_features(data, sfreq=250, data_unit='uV')` directly.
See **[README_FEATURES.md](README_FEATURES.md)** for definitions, custom bands,
DE units/methods, window-length caveats and full calling examples.

## Use EEG tools without a UI or LLM

```python
from eeg_agent import EEGTools

tools = EEGTools(data_root='/mnt/d/EEG', output_root='./agent_runs')
print(tools.invoke('inspect_eeg', {'source': '/mnt/d/EEG/sub01.edf'}))
```

## Plug into your existing LangGraph code (optional)

```python
from langchain_openai import ChatOpenAI
from eeg_agent import EEGTools, build_agent

model = ChatOpenAI(model='qwen3-8b', base_url='http://127.0.0.1:8000/v1',
                   api_key='EMPTY', temperature=0)
tools = EEGTools('/mnt/d/EEG', './agent_runs')
graph = build_agent(strategy='rewoo', model=model, tools=tools)
result = graph.invoke({
    'task': 'Inspect EEG, calculate PSD, then describe results',
    'input_path': '/mnt/d/EEG/sub01.edf',
    'strategy': 'rewoo', 'cursor': 0, 'iterations': 0, 'replans': 0,
    'steps': [], 'trace': [], 'evidence': {},
    'latest_npz': None, 'error': None, 'status': 'running',
}, config={'recursion_limit': 100})
print(result.get('final_answer'))
```

For managed execution in an existing app, the stable adapter is also `EEGSession`, which accepts `llm=your_langchain_chat_model`, `strategy='react'`, and `checkpoint=True`. This remains the public integration contract for EEG-Agent-UI 0.3.x.

## Compatibility and interfaces

- Public: `EEGTools`, `EEGSession`, `STRATEGIES`, `SUPPORTED_RAW`, `build_agent`, `AgentGraphs`.
- Tools: `inspect_eeg`, `preprocess_eeg`, `analyze_spectrum`, `analyze_connectivity`, `visualize_eeg`, `extract_features`.
- If plotting isn't installed (`.[viz]` or `.[agent]`), use other tools; plotting needs Matplotlib.
- Per-run logs and SQLite checkpoints are supported, but complete interactive pause/resume, token streaming and durable chat history are **not yet implemented**.
- No uploaded EEG is transferred to an external API by this library; users choose their LLM client.
- The source tree retains original four-dataset preprocessing scripts and older notebooks for research traceability; they are not installed as import dependencies.

## GUI and CLI (separate project)

See `EEG-Agent-UI`: an optional package including Chainlit Web UI and terminal REPL. The core has **no import or install dependency** on this UI.

## Tests

```bash
python -m pytest -q tests
```

Synthetic example inputs live under `examples/demo_npz/`. They are generated data, not clinical samples.

## Benchmark all three strategies (no UI required)

```bash
python scripts/compare_strategies.py \
  --data-root /mnt/d/EEG --input /mnt/d/EEG/sub01.edf \
  --task 'Inspect, compute spectral power and explain results' \
  --output-root ./benchmarks/local_runs
```

An already-running compatible LLM endpoint is needed. The optional `examples/start_vllm_example.sh` is a user-editable recipe, **not** an install dependency.

Built-in 26/64 and HydroCel electrode layout JSONs are included in the installed Python package. Access with `from eeg_agent import channel_config_path; print(channel_config_path("standard_26"))`.
