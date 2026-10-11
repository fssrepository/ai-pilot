#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"
RUN=${1:-artifacts/alternative-architecture/pilot-03}
[[ -f "$RUN/status.json" ]] || { echo "No recorded measurement: $RUN" >&2; exit 1; }
exec node experiments/adaptive-llama/tools/live.cjs "$RUN" 8091 \
  experiments/alternative-architecture/ui/report.html
