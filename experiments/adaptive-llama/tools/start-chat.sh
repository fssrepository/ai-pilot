#!/usr/bin/env bash
# Same pinned llama-server and built-in UI as the Qwen launcher, with the tiny snapshot.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"
MODEL=${1:-"$ROOT/artifacts/adaptive-llama/continual-01/seed-42/adaptive-replay/selected-f16.gguf"}
[[ -f "$MODEL" ]] || { echo "Missing trained GGUF: $MODEL" >&2; exit 1; }
mkdir -p .runtime
exec 9>.runtime/adaptive-llama.lock
flock -n 9 || { echo 'Adaptive training/chat is already running; stop that owner first.' >&2; exit 1; }
available=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
((available>=512*1024)) || { echo 'Less than 512 MiB available; defer chat.' >&2; exit 1; }
printf 'llama.cpp chat: http://127.0.0.1:8088/\nModel: %s\nUse w=0,x=1,y=2,a=R,b=0> (expected 2,2;).\nOnly the latest user message is sent to this toy model. Ctrl+C stops the server.\n' "$MODEL"
exec nice -n 10 .tools/llama-b11429/llama-server \
  --model "$MODEL" --alias Adaptive-Llama-Toy \
  --host 127.0.0.1 --port 8088 \
  --cors-origins http://127.0.0.1:8088,http://localhost:8088 \
  --threads 2 --threads-batch 2 --threads-http 2 --ctx-size 128 --parallel 1 \
  --n-gpu-layers 0 --batch-size 128 --ubatch-size 128 \
  --temp 0 --top-k 1 --repeat-penalty 1 --n-predict 4 \
  --jinja --chat-template-file "$ROOT/experiments/adaptive-llama/ui/last-user.jinja" \
  --reasoning off --ui-config-file "$ROOT/experiments/adaptive-llama/ui/llama-settings.json"
