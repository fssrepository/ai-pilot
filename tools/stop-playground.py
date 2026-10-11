#!/usr/bin/env python3
"""Stop only this repository's llama.cpp and Jupyter processes on Linux."""
import os
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
stopped = []
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    try:
        if (entry / 'cwd').resolve() != ROOT or entry.stat().st_uid != os.getuid():
            continue
        args = (entry / 'cmdline').read_bytes().split(b'\0')
        # Match the exact executable/script used by the versioned launchers.
        is_chat = args[0] == b'.tools/llama-b11429/llama-server'
        is_lab = len(args) > 1 and args[1] == b'.venv/bin/jupyter-lab'
        if is_chat or is_lab:
            os.kill(int(entry.name), signal.SIGTERM)
            stopped.append(entry.name)
    except (OSError, ProcessLookupError):
        continue
print('Stop requested for owned process IDs: ' + ', '.join(stopped) if stopped else 'No playground servers running.')
