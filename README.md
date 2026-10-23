# EEG-Agent: Local LLM Agent for Reproducible EEG Analysis

An open-source-ready EEG automation project combining **LangGraph + locally served Qwen through vLLM** with three selectable execution strategies: **Plan-and-Execute, ReAct and ReWOO**.

**Start here:** [README_AGENT.md](README_AGENT.md) — model startup, install, commands, execution strategies, validation, and current limitations.

**EEG library guide:** [README_EEG_TOOLKIT.md](README_EEG_TOOLKIT.md) — channel mapping, 1-second or arbitrary windows, old PRED+CT/HBN/MODMA/TDBRAIN adapters, 17 types of visualization, notebooks and data formats.

## Architecture

```text
User task → LangGraph strategy (plan-execute | react | rewoo)
                               ↓
                   shared EEGTools registry
                               ↓
            eeg_universal + eeg_visualization
                               ↓
             NPZ / HTML figures / QC / traces
```

DeepSeek Harness migration is explicitly postponed to a second stage. It will use the same validated tool registry rather than duplicating algorithms.

## Quick start

1. Run `bash scripts/start_vllm.sh` in a CUDA/WSL2 environment with vLLM installed.
2. In a separate Python environment: `pip install -e '.[agent,dev]'`.
3. Verify the server: `python scripts/healthcheck.py`.
4. Run the local agent:

```bash
python -m eeg_agent.cli \
  --strategy plan-execute --data-root /data/eeg \
  --input /data/eeg/sub-01.edf --output-root ./agent_runs \
  --task "Preprocess to 26 channels and 1 second windows, analyze PSD/FC and generate plots."
```

Run `python -m eeg_agent.cli --help` for all options. For locally sourced demonstration data, see `examples/` and tests.

> **Validation scope**: Core EEG tools and three graph-node control flows have been tested offline with synthetic data. Real LangGraph execution and GPU-hosted Qwen tool calling are not verified in the artifact-generation environment; test locally before using for research production.

See [README_AGENT.md](README_AGENT.md) for security boundaries and limitations. Not a medical diagnostic system.


## Interactive CLI & Web UI

See [README_UI.md](README_UI.md). First start your existing vLLM Qwen service, then use `eeg-agent-chat` or `chainlit run eeg_agent/web_app.py --host 127.0.0.1 --port 8002` from a separate `eeg_agent` Conda environment.
