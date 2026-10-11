#!/usr/bin/env python3
"""Compatibility entry point for experiments/qwen/tools/export_token_graph.py."""
from pathlib import Path
import runpy
globals().update(runpy.run_path(str(Path(__file__).resolve().parents[1] / "experiments/qwen/tools/export_token_graph.py"), run_name=__name__))
