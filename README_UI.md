# EEG-Agent: LangGraph + Qwen chat UI (WSL2)

**Quick start once dependencies are installed:** `bash scripts/start_chat.sh` (terminal) or `bash scripts/start_web.sh` (browser at `http://127.0.0.1:8002`).

**This adds two local frontends to the original project** (three strategies unchanged): a Codex-like interactive CLI and a Chainlit browser chat. Both reuse `eeg_agent.session.EEGSession` and the `EEGTools` allowlist. A **chat mode** talks to Qwen; an **agent mode** really runs LangGraph and EEG tools. Plain vLLM/Chat UI alone does not execute MNE code.

## 1. Run your existing Qwen3-8B vLLM server

Use your already working `conda activate vllm` and `vllm serve` startup setup. **Do not reinstall it.** The model has served name `qwen3-8b`, the API URL `http://127.0.0.1:8000/v1`.

Check from a second WSL2 shell:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

For the RTX 5090 / WSL2 environment, retain whichever UVA, FlashInfer and nvcc settings previously worked. Do not randomly mix the new `eeg_agent` environment into the vLLM Conda environment.

## 2. Install LangGraph/UI into a separate Conda environment

```bash
conda create -n eeg_agent python=3.12 -y
conda activate eeg_agent
python -m pip install -U uv
cd /mnt/c/Users/ZZZ/Work/LLM/EEG_Agent_LangGraph   # change this to the actual unpacked directory
uv pip install -e '.[agent,ui,dev]' --python "$CONDA_PREFIX/bin/python"
python scripts/healthcheck.py
python scripts/smoke_tool_call.py --model qwen3-8b
```

`uv` is used as the package installer targeting Conda's current Python; no `uv venv` is required. Model weights stay in the vLLM server environment/cache.

## 3A. Interactive CLI (Codex-inspired)

```bash
python -m eeg_agent.chat_cli \
  --data-root ./examples/demo_npz \
  --output-root ./agent_runs \
  --model qwen3-8b \
  --mode chat
```

Try:

```text
[chat:react] > 你好，解释一下 alpha 节律
[chat:react] > /file examples/demo_npz/synthetic_TDBRAIN.npz
[chat:react] > /agent
[agent:react] > 检查这个预处理后的 NPZ，计算 PSD 和 FC，生成可视化报告
[agent:react] > 比较一下连接最强的电极对
[agent:react] > /strategy rewoo
[agent:rewoo] > 重新分析当前文件，重点生成可视化
[agent:rewoo] > /status
[agent:rewoo] > /exit
```

Available commands: `/chat`, `/agent`, `/file PATH`, `/strategy plan-execute|react|rewoo`, `/clear`, `/status`, `/help`, `/exit`.

**Safety note**: only files under `--data-root` or `--output-root` can be selected. If data is in `/mnt/d/EEG` specify `--data-root /mnt/d/EEG`; on WSL do not pass `D:\EEG\...`.

## 3B. Chainlit Web UI

From the project root in `eeg_agent` environment:

```bash
export EEG_DATA_ROOT="$(pwd)/examples/demo_npz"
export EEG_OUTPUT_ROOT="$(pwd)/agent_runs"
export EEG_LLM_MODEL=qwen3-8b
export EEG_LLM_BASE_URL=http://127.0.0.1:8000/v1
chainlit run eeg_agent/web_app.py --host 127.0.0.1 --port 8002
```

Open on Windows: **http://127.0.0.1:8002**.

Switch strategy/mode in Chat Settings (or use `/agent`, `/chat`, `/strategy react`). Select an existing allowlisted EEG file via chat `/file /absolute/path/file.npz` or upload a single file with `/upload` (512 MB limit; multi-file formats should use `/file`). Text conversation mode works without selecting a file. EEG tool mode requires a file.

Examples when using `EEG_DATA_ROOT=/mnt/d/EEG`:

```text
/file /mnt/d/EEG/subject01.edf
/agent
检查通道与坏道，预处理到 250 Hz，切 1 秒窗，再生成 PSD、FC 和可视化报告。
```

**Do not open port 8002 to the public Internet without authentication.** The initial UI is local-only and has no authorization, cancellation, upload approval, or clinical oversight layer. Real patient-data runs need human review for interpolation/ICA choices.

## How conversations and files work

- Chat mode: a small in-memory rolling Qwen conversation, no EEG tools executed.
- Agent mode: every task runs a new LangGraph execution thread and writes `agent_run_<id>.json` plus optional reports; the UI retains brief conversation context.
- A successful `preprocess_eeg` changes the active file to the generated NPZ so follow-up analysis naturally uses the processed data. Use `/file` to select the raw input again.
- Original one-shot CLI `python -m eeg_agent.cli` and Benchmark scripts remain available.
- The UI shows completed LangGraph node names/steps and exposes generated HTML/JSON as download elements. It does **not** reveal hidden model reasoning, and the whole job (rather than each token) is awaited before the final response.

## Troubleshooting

- **404 model not found**: the served model name is `qwen3-8b` not `Qwen/Qwen3-8B` when `--served-model-name qwen3-8b` is used.
- **No connection**: confirm `curl /v1/models`; check WSL/Windows forwarding and endpoint.
- **Could not access EEG file**: Web/CLI executes inside WSL, so use `/mnt/c/...`, `/mnt/d/...`; set the enclosing `EEG_DATA_ROOT` explicitly.
- **Chainlit import/module error**: make sure you run from the project root and installed `.[agent,ui]` in `eeg_agent` conda environment.
- **Tool call only**: plain chat is not Agent mode; select `/agent` and `/file` first.
- **Runs take time**: scientific processing remains CPU/GPU work in MNE, not accelerated simply by Qwen.

## Verification status

Offline unit tests and Python syntax checks are possible without your local vLLM. This artifact has **not** been executed against your own RTX 5090 / Qwen service or an installed Chainlit browser session. Run both provided smoke scripts before real EEG datasets.
