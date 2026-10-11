# myscoutee-ai

Local AI research playground.

## Qwen playground

The installed model is **Qwen3.5-0.8B**. Everything starts manually; setup creates
no boot/login service, cron entry or desktop autostart item.

| Purpose | Runtime and UI | Start |
| --- | --- | --- |
| Everyday local chat | C++ **llama.cpp**, built-in web UI, Q4_K_M GGUF | `./tools/start-chat.sh` → http://127.0.0.1:8088 |
| Inspect the model | **Transformers + CPU PyTorch**, JupyterLab, BertViz | `./tools/start-playground.sh` → open the token URL printed in the terminal |

Run these commands from this directory. In Jupyter, open
`experiments/qwen/notebooks/01-qwen-internals.ipynb`, choose **Run → Run All Cells**, then use
**Analyze prompt**. The model loads only when an experiment is requested.
Use one loaded model runtime at a time to conserve memory.

Stop a foreground server with **Ctrl+C**, or stop this project's two servers
and Jupyter kernels with `python3 tools/stop-playground.py`. Closing a browser
tab does **not** stop a server.

- [Guide: tools, experiments, limitations and verification](docs/QWEN_PLAYGROUND_GUIDE.md)
- [Printable PDF guide](docs/QWEN_PLAYGROUND_GUIDE.pdf)
- [Analysis notebook](experiments/qwen/notebooks/01-qwen-internals.ipynb)
- [Token representation graph and algorithms (Hungarian)](docs/TOKEN_REPRESENTATION.md)
- [Standalone interactive token graph](docs/examples/token-representation.html) — measured
  embedding/layer coordinates, step-through playback, matrix shapes and explanations;
  works offline without loading a model. New analyses expose the same view in **Token flow**.

To reproduce on Linux x86-64 / Python 3.10:

```bash
./tools/setup-llama.sh   # pinned official C++ binary + 533 MB GGUF
./tools/setup.sh         # isolated .venv + original 1.75 GB analysis weights
```

Downloads need internet. Installed chat, inference and visualization work
locally without an external AI API. Weights, virtual environments, runtime
tokens and generated experiment outputs are gitignored. The Qwen inspection
experiment does not train or permanently update Qwen. All starts are manual,
with no boot/login autostart.
