#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m eeg_agent.chat_cli \
    --data-root "${EEG_DATA_ROOT:-./examples/demo_npz}" \
    --output-root "${EEG_OUTPUT_ROOT:-./agent_runs}" \
    --model "${EEG_LLM_MODEL:-qwen3-8b}" \
    --base-url "${EEG_LLM_BASE_URL:-http://127.0.0.1:8000/v1}" \
    "$@"
