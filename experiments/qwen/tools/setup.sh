#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
mkdir -p .cache .runtime
python3 -m venv --without-pip .venv
if [[ ! -f .cache/pip.pyz ]]; then
  python3 - <<'PY'
import urllib.request
urllib.request.urlretrieve('https://bootstrap.pypa.io/pip/pip.pyz', '.cache/pip.pyz')
PY
fi
export PIP_CACHE_DIR="$PWD/.cache/pip"
.venv/bin/python .cache/pip.pyz install 'pip==25.2'
.venv/bin/python -m pip install 'torch==2.8.0+cpu' --index-url https://download.pytorch.org/whl/cpu
if [[ -f requirements.lock.txt ]]; then
  .venv/bin/python -m pip install -r requirements.lock.txt
else
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python -m ipykernel install --prefix "$PWD/.venv" --name ai-pilot-qwen --display-name 'AI Pilot Qwen (CPU)'
.venv/bin/python tools/download_model.py
printf '\nReady. Start manually: ./tools/start-playground.sh\n'

