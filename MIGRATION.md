# Migration from EEG-Agent-LangGraph v0.2 to split v0.3

**Core project:** `EEG-Agent-Core` installs the importable `eeg_agent` Python module, scientific EEG pipelines, EEGTools, EEGSession and the three optional LangGraph strategies. It does not install `eeg-agent-chat` or `eeg-agent-web` anymore. Python scripts referencing `eeg_agent.cli` move to `eeg_agent_ui.batch_cli` (command: `eeg-agent-run`).

**UI project:** `EEG-Agent-UI` owns the web frontend and terminal CLI together, imports `EEGSession` and `SUPPORTED_RAW` from Core and declares `eeg-agent-core[agent]>=0.3,<0.4`.

To migrate editable installs, in your **eeg_agent** Conda env, not the vLLM env:

```bash
uv pip uninstall eeg-agent-local --python "$CONDA_PREFIX/bin/python" || true
uv pip install -e './EEG-Agent-Core[agent]' -e './EEG-Agent-UI[all]' --python "$CONDA_PREFIX/bin/python"
```

Your existing vLLM server and cached Qwen weights are unchanged. Generated recordings/reports are kept outside the packages. SQLite checkpoints from older graph code are not guaranteed to deserialize under a changed graph; retain a backup and do not assume full cross-version resume.
