#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
DEST="$ROOT/.tools/llama-b11429/include"
mkdir -p "$DEST"
# Public API headers from the same revision as the installed binary; about 180 KB.
while read -r digest source; do
  target="$DEST/${source##*/}"
  if [[ ! -f "$target" ]] || ! printf '%s  %s\n' "$digest" "$target" | sha256sum -c --status; then
    curl --fail --location --silent --show-error "https://raw.githubusercontent.com/ggml-org/llama.cpp/d81235049/$source" -o "$target.tmp"
    printf '%s  %s\n' "$digest" "$target.tmp" | sha256sum -c --status
    mv "$target.tmp" "$target"
  fi
done < "$ROOT/experiments/adaptive-llama/headers.sha256"
printf 'Verified pinned llama.cpp/ggml headers.\n'
