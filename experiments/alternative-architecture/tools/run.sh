#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"
mkdir -p .runtime artifacts/alternative-architecture
exec 9>.runtime/repair-graph.lock
flock -n 9 || { echo 'Repair Graph is already building/running.' >&2; exit 1; }
available=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
((available>=512*1024)) || { echo 'Less than 512 MiB available; defer run.' >&2; exit 1; }
MODE=${1:-campaign}
OUTPUT=${2:-"artifacts/alternative-architecture/$(date -u +%Y%m%dT%H%M%SZ)-$MODE"}
[[ "$MODE" == campaign || "$MODE" == test || "$MODE" == campaign-v2 || "$MODE" == test-v2 || "$MODE" == test-v3 || "$MODE" == campaign-v3 ]] || { echo 'Usage: run.sh campaign|campaign-v2|campaign-v3|test|test-v2|test-v3 [new-output-directory]' >&2; exit 1; }
args=("$MODE" "$OUTPUT")
[[ "$MODE" != campaign && "$MODE" != campaign-v2 && "$MODE" != campaign-v3 ]] || args+=(experiments/alternative-architecture/ui/report.html)
# One native thread, manual foreground lifetime, no network, no autostart.
ulimit -v 524288
timeout --signal=TERM --kill-after=5s 15m nice -n 15 .tools/repair-graph/repair-graph "${args[@]}"
printf '\nOutputs: %s\n' "$OUTPUT"
