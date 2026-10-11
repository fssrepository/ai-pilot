#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"
mkdir -p .runtime
exec 9>.runtime/adaptive-build.lock
flock -n 9 || { echo 'The adaptive Llama build already has an owner.' >&2; exit 1; }
available=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
((available>=2*1024*1024)) || { echo 'Less than 2 GiB available; defer compilation.' >&2; exit 1; }
if [[ -z "${LIBTORCH_ROOT:-}" ]]; then
  candidates=("$ROOT"/.venv/lib/python*/site-packages/torch)
  LIBTORCH_ROOT=${candidates[0]}
fi
LLAMA_ROOT="$ROOT/.tools/llama-b11429"
[[ -f "$LIBTORCH_ROOT/include/torch/csrc/api/include/torch/types.h" ]] || { echo 'Set LIBTORCH_ROOT to an existing LibTorch installation.' >&2; exit 1; }
[[ -f "$LLAMA_ROOT/include/llama.h" ]] || { echo 'Run experiments/adaptive-llama/tools/setup-headers.sh first.' >&2; exit 1; }
mkdir -p .tools/adaptive-llama
# This project has no watcher. One translation unit, one compiler, low priority.
# Official torch 2.8 Linux CPU wheel uses the C++11 ABI.
nice -n 15 "${CXX:-g++}" -std=c++17 -O1 -g0 -D_GLIBCXX_USE_CXX11_ABI=1 \
  -DLLAMA_BACKEND_DIR="\"$LLAMA_ROOT\"" \
  -isystem "$LIBTORCH_ROOT/include" -isystem "$LIBTORCH_ROOT/include/torch/csrc/api/include" \
  -I "$LLAMA_ROOT/include" experiments/adaptive-llama/src/main.cpp \
  -L "$LIBTORCH_ROOT/lib" -Wl,-rpath,"$LIBTORCH_ROOT/lib" \
  -L "$LLAMA_ROOT" -Wl,-rpath,"$LLAMA_ROOT" \
  -ltorch -ltorch_cpu -lc10 -lllama -lggml -lggml-base -pthread \
  -o .tools/adaptive-llama/adaptive-llama.tmp
mv .tools/adaptive-llama/adaptive-llama.tmp .tools/adaptive-llama/adaptive-llama
printf 'Built: %s\n' "$ROOT/.tools/adaptive-llama/adaptive-llama"
