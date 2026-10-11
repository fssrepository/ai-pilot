# Qwen inspection playground

This experiment owns the existing **Qwen3.5-0.8B** analyzer, notebook, visual
assets and launch/setup scripts. It does not train or grow Qwen.

From the repository root:

```bash
./experiments/qwen/tools/start-chat.sh
./experiments/qwen/tools/start-playground.sh
```

Use one loaded Qwen runtime at a time. Open the authenticated URL printed by
Jupyter; the canonical notebook is `experiments/qwen/notebooks/01-qwen-internals.ipynb`.
The repository-level `tools/start-*.sh`, `playground.py`, `notebooks/` and old
asset paths remain forwarding entry points for existing bookmarks and imports.
There is only one implementation of the analyzer and each launcher.

Shared `.venv`, downloaded `models/`, `.tools`, `.cache`, `.runtime`, documentation
and generated outputs retain their existing locations; no large files are copied.
Stop the Qwen services with the existing `python3 tools/stop-playground.py`.
Nothing starts at boot or login.

[Guide](../../docs/QWEN_PLAYGROUND_GUIDE.md) ·
[Token graph](../../docs/TOKEN_REPRESENTATION.md)
