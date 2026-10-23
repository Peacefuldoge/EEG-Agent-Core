#!/usr/bin/env bash
set -euo pipefail
# Run on a CUDA Linux / WSL2 machine after installing vLLM in a separate environment.
# Qwen3 base model uses the Hermes tool parser according to official Qwen documentation.
MODEL="${EEG_LLM_MODEL:-Qwen/Qwen3-8B}"
PORT="${EEG_LLM_PORT:-8000}"
exec vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser deepseek_r1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.80
