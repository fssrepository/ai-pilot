#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
exec nice -n 10 .tools/llama-b11429/llama-server \
  --model "$PWD/models/Qwen3.5-0.8B-Q4_K_M.gguf" \
  --alias Qwen3.5-0.8B-Q4_K_M \
  --host 127.0.0.1 --port 8088 \
  --cors-origins http://127.0.0.1:8088,http://localhost:8088 \
  --threads 2 --threads-batch 2 --ctx-size 2048 --parallel 1 \
  --n-gpu-layers 0 --batch-size 128 --ubatch-size 128 \
  --jinja --reasoning off
