#!/usr/bin/env bash
# Compatibility entry point; implementation belongs to the Qwen experiment.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$ROOT/experiments/qwen/tools/start-playground.sh" "$@"
