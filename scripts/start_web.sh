#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export EEG_DATA_ROOT="${EEG_DATA_ROOT:-$(pwd)/examples/demo_npz}"
export EEG_OUTPUT_ROOT="${EEG_OUTPUT_ROOT:-$(pwd)/agent_runs}"
export EEG_LLM_MODEL="${EEG_LLM_MODEL:-qwen3-8b}"
export EEG_LLM_BASE_URL="${EEG_LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
exec chainlit run eeg_agent/web_app.py --host 127.0.0.1 --port 8002 "$@"
