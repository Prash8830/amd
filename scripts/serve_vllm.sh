#!/usr/bin/env bash
# Serve the fine-tuned expert model via vLLM (OpenAI-compatible API) on ROCm.
# Run in a terminal; then launch the app with VLLM_URL=http://localhost:8200/v1
set -e
cd "$(dirname "$0")/.."

MODEL_DIR="${MERGED_MODEL_DIR:-./models/truthline-merged-qwen3-14b}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "Merged model not found at $MODEL_DIR — building it first..."
  python export_merged.py
fi

echo "Starting vLLM on :8200  (served-model-name: truthline-14b)"
exec vllm serve "$MODEL_DIR" \
  --port 8200 \
  --served-model-name truthline-14b \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.6
