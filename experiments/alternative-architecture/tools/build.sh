#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"
mkdir -p .runtime .tools/repair-graph
exec 9>.runtime/repair-graph.lock
flock -n 9 || { echo 'Repair Graph is already building/running.' >&2; exit 1; }
available=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
((available>=1024*1024)) || { echo 'Less than 1 GiB available; defer build.' >&2; exit 1; }
nice -n 15 g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  experiments/alternative-architecture/src/main.cpp -o .tools/repair-graph/repair-graph.next
mv .tools/repair-graph/repair-graph.next .tools/repair-graph/repair-graph
