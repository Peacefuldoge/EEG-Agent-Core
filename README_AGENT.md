# EEG-Agent — LangGraph + vLLM + local Qwen

A local EEG research automation agent built **on top of the existing EEG_Preprocess_Universal science engine**. Three independent selectable strategies are implemented in `eeg_agent/strategies.py`:

- `plan-execute` (Planner → validated Executor → optional Replanner → Result)
- `react` (Model tool call → ToolMessage → model loop)
- `rewoo` (single upfront dependency plan → reference-aware Worker → Solver)

All three use the same deterministic MNE/NumPy/SciPy tools and vLLM OpenAI-compatible model endpoint. **DeepSeek Harness is intentionally NOT included in this first-phase release.** A future adapter should wrap the stable tool interface (`EEGTools.invoke`), not duplicate EEG algorithms.

## Requirements / architecture

- Python 3.11+ on WSL2 Ubuntu or Linux; Windows native can run the orchestration and EEG code, but **vLLM CUDA service should run on WSL2/Linux**.
- CUDA GPU with enough VRAM for your chosen checkpoint. `Qwen/Qwen3-8B` is the starting option.
- Local EEG input: `.edf`, `.bdf`, `.set`, `.vhdr`, `.fif`, `.cnt`, `.npy`, `.mat`, 2D continuous `.npz`; existing processed 3D `.npz` is supported by analysis tools.
- Input and output live under explicit allowlisted roots. Raw files are read-only; processed EEG/figures are saved under `--output-root`.

```
User task + allowed local EEG file
          ↓
 LangGraph runtime (strategy selected)
 ├── PlanExecute: Planner → Executor ↔ Replanner → Summary
 ├── ReAct: LLM ↔ Tool execution → answer
 └── ReWOO: Planner → DAG Worker → Solver/Summary
          ↓ (shared)
      EEGTools registry
          ↓
 eeg_universal.py + eeg_visualization.py
          ↓
 NPZ / figures / report.json / index.html / execution trace
```

## Install and start (WSL2 / Linux)

**Model server environment** (separate from scientific env recommended):

```bash
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
pip install vllm
bash scripts/start_vllm.sh
```

The startup flags use the Qwen3 Hermes tool-call parser and `deepseek_r1` reasoning parser described in the Qwen function-calling guide. Consult the exact model/vLLM-version documentation if using another Qwen checkpoint. Keep the endpoint loopback-only unless authenticated.

**Agent/science environment** (second terminal):

```bash
python3 -m venv .venv-agent
source .venv-agent/bin/activate
pip install -r requirements.txt -r requirements-agent.txt
python scripts/healthcheck.py
python scripts/smoke_tool_call.py   # verifies real Qwen tool calling
python -m pytest -q tests/
```

For Windows PowerShell with vLLM already running in WSL2:

```powershell
py -m pip install -r requirements.txt -r requirements-agent.txt
python scripts/healthcheck.py
```

## Run a workflow

Set `--data-root` to the folder containing your EEG files. Paths outside this folder (or a previous agent output root) are rejected.

```bash
python -m eeg_agent.cli \
  --strategy plan-execute \
  --task "Inspect the recording, preprocess to 26 channels and 1 s windows; compute bandpower and Pearson connectivity, then visualize results. Do not infer any medical diagnosis." \
  --data-root /data/eeg \
  --input /data/eeg/sub-01_rest.edf \
  --output-root ./agent_runs
```

Switch with `--strategy react` or `--strategy rewoo` using the **same dataset and tool registry**. For preprocessed NPZ choose a task that analyzes the NPZ directly, with no preprocessing. To continue or inspect a checkpoint, use the `thread_id` in output JSON and the SQLite checkpoint store; this release exposes persistence but does not yet implement a dedicated human approval/resume UI.

### PowerShell example

```powershell
python -m eeg_agent.cli `
  --strategy rewoo `
  --data-root "D:\EEG" `
  --input "D:\EEG\sub01.edf" `
  --output-root "D:\EEG-Agent-Results" `
  --task "Inspect and preprocess to 26 channels with 250 Hz and 1-second non-overlapping windows, then visualize FC and PSD."
```

### Tool names and reference binding

- `inspect_eeg`: sample rate, channels, duration, annotations or existing NPZ shape.
- `preprocess_eeg`: runs the unchanged MNE preprocessing engine with parameter validation; returns `output_npz`, QC, ICA metadata.
- `analyze_spectrum`: Welch PSD + channel/time-aggregated bandpower.
- `analyze_connectivity`: Pearson FC summary and top connections.
- `visualize_eeg`: original plotting core → PNG, JSON, self-contained HTML gallery.

Plans use *safe references*, such as `$input`, `$latest_npz`, `$E2.output_npz`. Only these references are resolved; **no `eval`, Python code generation, arbitrary shell, or URL downloading**.

Example ReWOO plan:

```json
{"steps": [
  {"id":"E1","tool":"inspect_eeg","args":{"source":"$input"},"depends_on":[]},
  {"id":"E2","tool":"preprocess_eeg","args":{"source":"$input","target_count":26,"window_seconds":1,"target_sfreq":250},"depends_on":["E1"]},
  {"id":"E3","tool":"analyze_spectrum","args":{"source":"$E2.output_npz"},"depends_on":["E2"]},
  {"id":"E4","tool":"visualize_eeg","args":{"source":"$E2.output_npz","plots":"default"},"depends_on":["E2"]}
]}
```

**Channel-count caveat**: selecting 26 arbitrary existing electrodes is not equivalent to harmonizing named 26-channel layouts. Prefer `target_channels` with an explicit shared channel list for cross-dataset comparisons. Missing target positions/interpolation require special verification. If source metadata has annotations with zero durations, prepare an events TSV or explicitly choose `ignore_annotations` only if methodologically appropriate.

## Safety and reproducibility

1. EEG tool paths are restricted to `--data-root` and `--output-root`; input files aren't overwritten.
2. Pydantic forbids unknown parameters; tool names are whitelisted, ReWOO DAG dependencies validated.
3. Token-budget-equivalent limits: `--max-steps` caps strategy iterations, `--max-replans` caps replanning.
4. Artifacts are separated into UUID folders; model outputs are not trusted as numeric scientific findings.
5. Trace JSON and optional SQLite checkpoints enable auditing. Set `LANGGRAPH_STRICT_MSGPACK=true`.
6. No cloud API required. Do not expose raw patient data or public unauthenticated vLLM access.
7. ICA/component removal and target-channel interpolation can materially affect results. For real patient datasets, human review is recommended; **automatic approval interrupts are not yet implemented**.

## Test status and limitations

The included offline tests cover EEG preprocessing, spectrum/connectivity tools, artifact creation, schema validation, path guards, plan DAG/reference binding, and the existing visualization tests. LangGraph/Qwen runtime integration needs a real local deployment with installed dependencies: the artifact-building environment has no network access and did not have LangGraph installed. Do not report end-to-end GPU inference as tested until you run it.

## Phase 2 (not implemented): DeepSeek Harness adapter

Implement separate `adapters/deepseek_harness/` wrapping *the same* `EEGTools` schema/invoke functions, with a TypeScript↔Python transport (HTTP/MCP). Do not import DeepSeek Harness into the LangGraph runtime; maintain separate execution entrypoints and compare both harnesses against identical EEG tool tests.

For the legacy dataset-specific adapters and visualization notebook documentation see [`README_EEG_TOOLKIT.md`](README_EEG_TOOLKIT.md).

## Compare the three strategies on your local Qwen server

Use the included **synthetic preprocessed NPZ** as a safe first real-model demonstration. Ask for analysis only (no preprocessing of pre-windowed data):

```bash
python scripts/compare_strategies.py \
  --data-root ./examples/demo_npz \
  --input ./examples/demo_npz/synthetic_TDBRAIN.npz \
  --output-root ./benchmarks/local_runs \
  --task "Inspect this already preprocessed NPZ, calculate spectrum and connectivity, and generate visualizations. Do not preprocess it again."
```

`comparison.csv` contains **measured** elapsed seconds, tool calls and failure counts for each strategy. These are not estimates and must be generated using your real local model. Each strategy runs in a separate subprocess and writes independently namespaced artifacts.
