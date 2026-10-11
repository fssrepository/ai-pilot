#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"
BINARY="$ROOT/.tools/adaptive-llama/adaptive-llama"
[[ -x "$BINARY" ]] || { echo 'Run experiments/adaptive-llama/tools/build.sh first.' >&2; exit 1; }
mkdir -p .runtime artifacts/adaptive-llama
exec 9>.runtime/adaptive-llama.lock
flock -n 9 || { echo 'An adaptive Llama experiment is already running.' >&2; exit 1; }
MODE=${1:-run}
[[ "$MODE" == run || "$MODE" == test || "$MODE" == resume || "$MODE" == campaign || "$MODE" == campaign-resume ]] || { echo 'Usage: run.sh run|test [directory] [seed] [steps] [extra] [pause], resume DIRECTORY [pause], or campaign|campaign-resume DIRECTORY [pause]' >&2; exit 1; }
if [[ "$MODE" == resume ]]; then
  [[ -n "${2:-}" && -f "$2/state.pt" ]] || { echo 'Resume requires an existing experiment directory with state.pt.' >&2; exit 1; }
fi
OUTPUT=${2:-"$ROOT/artifacts/adaptive-llama/$(date -u +%Y%m%dT%H%M%SZ)-$MODE"}
available=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
((available>=2*1024*1024)) || { echo 'Less than 2 GiB available; defer this experiment.' >&2; exit 1; }
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# Foreground execution only. SIGINT/SIGTERM requests a checkpoint at the next update boundary.
if [[ "$MODE" == campaign || "$MODE" == campaign-resume ]]; then
  args=("$MODE" "$OUTPUT" "$ROOT/experiments/adaptive-llama/ui/campaign.html" "${3:-0}")
elif [[ "$MODE" == resume ]]; then
  args=(resume "$OUTPUT" "$ROOT/experiments/adaptive-llama/ui/report.html" "${3:-0}")
else
  args=("$MODE" "$OUTPUT" "${3:-42}" "${4:-600}" "${5:-200}" "$ROOT/experiments/adaptive-llama/ui/report.html" "${6:-0}")
fi
timeout --signal=TERM --kill-after=30s 15m nice -n 15 "$BINARY" "${args[@]}"
printf '\nState and outputs: %s\n' "$OUTPUT"
