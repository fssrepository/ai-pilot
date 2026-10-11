#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
if [[ ! -x .venv/bin/jupyter || ! -f models/Qwen3.5-0.8B/REVISION ]]; then
  printf 'Run ./tools/setup.sh first.\n' >&2
  exit 1
fi
mkdir -p .runtime/jupyter .cache/matplotlib
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export JUPYTER_RUNTIME_DIR="$PWD/.runtime/jupyter"
export JUPYTER_CONFIG_DIR="$PWD/.runtime/jupyter-config"
export JUPYTER_DATA_DIR="$PWD/.runtime/jupyter-data"
exec nice -n 10 .venv/bin/jupyter lab \
  --ip=127.0.0.1 --port=8890 --ServerApp.port_retries=0 --no-browser \
  --ServerApp.root_dir="$PWD" \
  --ServerApp.default_url=/lab/tree/experiments/qwen/notebooks/01-qwen-internals.ipynb
