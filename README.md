# myscoutee-ai
playground of self-learning AI

[Research conversation summary](CONVERSATION_SUMMARY.md) — English summary of the project's research discussions and proposed direction.

Two independent development directions have English plans:

- [Adaptive Llama demonstrator](docs/ADAPTIVE_LLAMA_PLAN.md) — a tiny C++ learner
  with bounded structural adaptation, an inspectable graph and llama.cpp exports.
- [Alternative-architecture research](docs/MINIMAL_LEARNING_MODEL_PLAN.md) — a
  learner that develops its representations and learns which repairs to make;
  its architecture is not constrained by Llama or GGUF compatibility.

## Experiment layout

| Directory | Purpose | Status |
| --- | --- | --- |
| [experiments/qwen](experiments/qwen/README.md) | Existing Qwen chat and internal inspection | Working pretrained reference |
| [experiments/adaptive-llama](experiments/adaptive-llama/README.md) | Native C++ training, four-to-five-layer growth, GGUF and graph report | Demonstrator complete and closed; measured limitations documented |
| [experiments/alternative-architecture](experiments/alternative-architecture/README.md) | A different learning architecture | Research plan only |

Shared runtime dependencies stay in `.venv` and `.tools`; downloaded weights stay
in `models`. Generated adaptive runs go to `artifacts/adaptive-llama`. These
directories are gitignored. Existing Qwen entry points forward to its experiment
directory so earlier notebook links and launch commands continue to work.

## Adaptive Llama

The C++ learner reuses the installed LibTorch and llama.cpp libraries. It trains
on a tiny movement-prediction task. A validation rule decides whether to try a
fifth layer, then compares it with continued four-layer training. Experiments
can be paused and resumed with their optimizer and decision state intact. This is an initial structural-adaptation experiment,
not the alternative-architecture research system or a general chat model.

```bash
./experiments/adaptive-llama/tools/build.sh
./experiments/adaptive-llama/tools/run.sh test
./experiments/adaptive-llama/tools/run.sh run
```

Open the generated `index.html` in the printed output directory. This offline
graph/report needs no Jupyter login or running model. The
[experiment README](experiments/adaptive-llama/README.md) documents dependencies,
manual execution, acceptance rules and limitations.

[Current two-world comparison](experiments/adaptive-llama/examples/continual.html)
· [English results](docs/ADAPTIVE_LLAMA_RESULTS.md)
· [PDF report](docs/ADAPTIVE_LLAMA_RESULTS.pdf)
· [Earlier adaptation and resume results](experiments/adaptive-llama/examples/ADAPTIVE_RESULTS.md)
· [Initial forced-proposal pilots](experiments/adaptive-llama/examples/RESULTS.md)

To talk to the tiny model in the **official llama.cpp chat UI**:

```bash
./experiments/adaptive-llama/tools/start-chat.sh
```

Open **[http://127.0.0.1:8088/](http://127.0.0.1:8088/)**. Use the
[documented messages, expected answers and observed errors](experiments/adaptive-llama/README.md#official-llamacpp-chat-port-8088).
The [recorded chat configuration](experiments/adaptive-llama/README.md#recorded-chat-configuration--2026-10-11) identifies the exact model, settings and measured answers.
The model is a coordinate predictor, not a natural-language assistant. Stop it
before training or switching back to Qwen; both use the same chat port.

To inspect the recorded campaign through the separate measurement viewer:

```bash
node experiments/adaptive-llama/tools/live.cjs artifacts/adaptive-llama/continual-01
```

Open **[http://127.0.0.1:8090/](http://127.0.0.1:8090/)** while that manual viewer
runs. It shows live measurements during training and the final results afterward.
The offline HTML link above works without a server. To start a new twelve-condition
campaign, use `run.sh campaign DIRECTORY` as documented in the experiment README.

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
experiment does not train or permanently update Qwen; adaptive Llama trains its
own separate tiny model. All starts are manual, with no boot/login autostart.
