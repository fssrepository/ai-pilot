# Adaptive Llama: native C++ experiment

This experiment owns a tiny Llama-compatible decoder, its trainer, native
checkpoint format, bounded four-to-five-layer growth experiment and GGUF exporter.
It is separate from [Qwen](../qwen/README.md) and the
[alternative architecture research](../alternative-architecture/README.md).

Start with the [two-world comparison](examples/continual.html),
[English results](../../docs/ADAPTIVE_LLAMA_RESULTS.md) or
[PDF](../../docs/ADAPTIVE_LLAMA_RESULTS.pdf); these require no running model.
The [single-world adaptation and resume results](examples/ADAPTIVE_RESULTS.md)
are also retained. The [initial forced-proposal pilots](examples/RESULTS.md)
are retained separately as historical evidence.

## Closure — 2026-10-11

The requested demonstrator's development and verification are complete: changing
rules, forgetting measurements, twelve fixed/adaptive comparison runs, live and
offline visualization, GGUF deployment in the official llama.cpp chat, and the
English PDF with the AI Pilot cover. Native checkpoint/resume, export parity
and browser checks passed; see the [verification record](examples/continual-verification.json).

This closes the bounded experiment, with no remaining implementation item in
that scope. It does not establish superior adaptive learning: no growth proposal
was accepted in the two-world campaign, and the selected chat snapshot makes the
documented prediction error. Those measured limitations are retained without
post-hoc retuning. Alternative-architecture research remains a separate next task.
The chat and measurement viewer are left available for manual inspection and
have no boot/login startup.

The executable uses **LibTorch C++** for tensor operations, gradients and Adam,
and the already installed **llama.cpp** libraries for exported inference.
Training, checkpointing, growth and GGUF writing execute in C++; there is no
Python training process. On this machine the headers/libraries are reused from
the existing CPU PyTorch 2.8.0 installation. `LIBTORCH_ROOT` can instead point to
an ABI-compatible standalone LibTorch distribution (C++11 ABI).

## Build and run manually

From the repository root:

```bash
./experiments/adaptive-llama/tools/setup-headers.sh
./experiments/adaptive-llama/tools/build.sh
./experiments/adaptive-llama/tools/run.sh test
./experiments/adaptive-llama/tools/run.sh run
```

## Two-world campaign and live view

The [predeclared protocol](PROTOCOL.md) compares changing-world acquisition and
old-world forgetting for fixed four/five-layer models and the adaptive model.
It includes a no-replay forgetting control, three seeds, a matched matrix-work
budget and the cost of both adaptive search branches.

Start a new campaign manually from the repository root:

```bash
./experiments/adaptive-llama/tools/run.sh campaign artifacts/adaptive-llama/my-campaign
```

In another terminal, serve that campaign's live view:

```bash
node experiments/adaptive-llama/tools/live.cjs artifacts/adaptive-llama/my-campaign
```

**Open [http://127.0.0.1:8090/](http://127.0.0.1:8090/).** The viewer reads actual
native measurements; the trainer updates them after each 50-update chunk. The
browser checks for updates once a second and stops polling when complete.
This animates measurement/architecture changes, not individual neuron activity.
An optional final viewer argument changes its port. Only the report/status are served on loopback; this viewer does not expose
model checkpoints or implement a separate chat interface.

The recorded campaign for this implementation is `artifacts/adaptive-llama/continual-01`.
Its final `index.html` also opens offline, after the server is stopped. See the [versioned offline report](examples/continual.html) and
[English PDF](../../docs/ADAPTIVE_LLAMA_RESULTS.pdf).

Ctrl+C stops the trainer at an update boundary and saves the current segment.
Resume with `run.sh campaign-resume DIRECTORY`. Completed condition results and
pretraining checkpoints are reused. A partial segment restores model, Adam,
random state and its training-time counters atomically; the log may contain a
repeated observation if interrupted between saving and logging. Final report
work may finish before responding to a stop. Keep the entire run directory.
A completed campaign resume changes no files. An optional final integer pauses
after that many training chunks, for recovery verification.

Stop the Node measurement viewer separately with Ctrl+C. It is not installed as
a boot/login service and runs no model. The learner and model runtime remain C++.

### Official llama.cpp chat: port 8088

Start the existing llama.cpp built-in web UI with the trained snapshot:

```bash
./experiments/adaptive-llama/tools/start-chat.sh
```

Open **[http://127.0.0.1:8088/](http://127.0.0.1:8088/)** and start a new conversation. The model name is
`Adaptive-Llama-Toy`; the default snapshot is seed 42, adaptive with replay.
This is the same llama.cpp interface used for Qwen, now serving the tiny model.
The separate measurement viewer stays at port 8090.

Send the following strings exactly, without Markdown backticks:

| Message | Expected | Observed model answer |
| --- | --- | --- |
| `w=0,x=1,y=2,a=R,b=0>` | `2,2;` | `0,2;` (wrong) |
| `w=1,x=1,y=2,a=R,b=0>` | `0,2;` | `0,2;` (correct) |
| `w=1,x=1,y=2,a=R,b=1>` | `1,2;` | `1,2;` (correct) |

The first error illustrates old-world forgetting. The expected answers come
from the world rules; the chat displays only the model's generated answer,
without a simulator replacing it. Checks use greedy generation and four tokens.

The custom chat template sends only the latest user message, stripped of outer
whitespace. System messages and previous turns are absent from the predictor's
input. This provides no conversational memory, chat learning or natural-language
competence. Use a text-only coordinate prompt. The visible chat history is a UI
record. Do not attach files.

Defaults are temperature 0, top-k 1, repeat penalty 1 and max tokens 4. Previous
browser preferences may override UI defaults: reset those fields if necessary.
The model is held in RAM until Ctrl+C stops the server. This manual server holds
the adaptive experiment lock; stop it before training. Qwen uses the same port:
stop the existing server before switching. An optional first launcher argument
selects another generated GGUF snapshot. No boot/login service is installed.

### Recorded chat configuration — 2026-10-11

This is the current configuration of the manually started local chat. Any change
to the model, template, runtime, port or generation settings must update this
record and the reproducible launcher/config files in the same change. Expected
answers must remain separate from observed model outputs. Documentation files
are repository content; generated model/checkpoint directories are Git-ignored.

| Setting | Recorded value / source |
| --- | --- |
| Runtime and UI | Existing official llama.cpp b11429 / d81235049, built-in web UI |
| Launcher | [tools/start-chat.sh](tools/start-chat.sh) |
| Loaded model | `artifacts/adaptive-llama/continual-01/seed-42/adaptive-replay/selected-f16.gguf` |
| Model identity | Four layers, 787,584 parameters, F16 GGUF, 1,592,576 bytes |
| GGUF SHA-256 | `f0fc2c5998dc95bded256f7435b5e5933b43f1e3a834dbe4ac6491939f5c41ca` |
| Chat name | `Adaptive-Llama-Toy` |
| Address | `http://127.0.0.1:8088/` — loopback only |
| Measurement viewer | `http://127.0.0.1:8090/` — separate read-only Node viewer |
| Input format | [ui/last-user.jinja](ui/last-user.jinja): latest user message only, outer whitespace trimmed |
| UI defaults | [ui/llama-settings.json](ui/llama-settings.json): temperature 0, top-k 1, repeat penalty 1, max tokens 4 |
| Runtime bounds | Two inference/batch threads, two HTTP threads, one slot, 128-token context, CPU only |
| Lifetime | Manual foreground launch; Ctrl+C stops the model server; no boot/login startup |
| Training exclusion | `.runtime/adaptive-llama.lock` shared with the experiment runner |

The actual built-in UI test sent all three documented messages in one visible
conversation and verified the four-token greedy settings. The first prediction
was **`0,2;`, not the expected `2,2;`**. Both remaining predictions were correct.
This is recorded in [continual-verification.json](examples/continual-verification.json).

To switch snapshots, stop the current server first, then pass the desired GGUF
path to this same launcher. Do not start a second model server alongside it.
To return to Qwen, stop this server, then run the existing root `tools/start-chat.sh`.
The adaptive setup does not change the Qwen launcher's model or flags.

## Original single-world experiment

The following `run`/`resume` details describe the earlier single-world protocol;
use `campaign`/`campaign-resume` above for changing-world experiments.

Header setup downloads only pinned public headers and verifies SHA-256 hashes.
Build reuses g++, LibTorch and the existing `.tools/llama-b11429` libraries; it
never starts a service. A fresh installation needs those dependencies first;
the launcher does not download them automatically.

The run command prints a new output directory under `artifacts/adaptive-llama/`.
Open its `index.html` in a browser. It is an offline report of actual measurements,
not a live training UI, chat assistant or Jupyter notebook. Phase buttons show
the baseline, equal-step control, growth candidate (when proposed) and selected
model. A validation-history slider shows measurements recorded every 50 baseline
updates; it does not animate individual neurons or live computation.

Optional explicit run arguments:

```bash
./experiments/adaptive-llama/tools/run.sh run artifacts/adaptive-llama/my-run 42 600 200
```

Arguments are output directory, random seed, baseline steps and additional steps
per branch. Existing output directories are refused. Runs are foreground-only,
limited to two CPU threads and 15 minutes. Concurrent experiments are rejected
by a file lock. No boot/login services are installed.

### Stop and resume

During training, Ctrl+C or SIGTERM requests a checkpoint after the current
optimizer update. Wait for `PAUSED` before closing the process. Continue the same
experiment in a fresh process with:

```bash
./experiments/adaptive-llama/tools/run.sh resume artifacts/adaptive-llama/my-run
```

`state.pt` stores the experiment phase, decision, validation history, branch
timing, active model, Adam state and random-generator state. Keep the entire run
directory: later phases also need the saved baseline/control branch checkpoints.
Checkpoints are replaced atomically every 50 updates and at phase transitions.
A forced kill can lose updates since the last saved checkpoint; power-loss
durability is not guaranteed. The 15-minute launcher timeout requests a soft
stop, then allows 30 seconds before killing a stuck process.

`COMPLETE` appears only after final evaluation and GGUF checks finish. A paused
run is not a completed result. Resuming a completed run is a no-op. A stop during
final export may finish the bounded report work before exiting.

For reproducible pause/resume checks, an optional final argument pauses after
a number of training chunks (normally 50 updates each). This sequence pauses
in the baseline, control and candidate phases, then completes seed 44:

```bash
./experiments/adaptive-llama/tools/run.sh run artifacts/adaptive-llama/resume-demo 44 600 200 5
./experiments/adaptive-llama/tools/run.sh resume artifacts/adaptive-llama/resume-demo 8
./experiments/adaptive-llama/tools/run.sh resume artifacts/adaptive-llama/resume-demo 4
./experiments/adaptive-llama/tools/run.sh resume artifacts/adaptive-llama/resume-demo
```

Stop the Qwen runtimes before training to conserve memory:

```bash
python3 tools/stop-playground.py
```

This existing helper stops only this repository's Qwen chat/Jupyter processes.
It does not stop the adaptive learner or unrelated backend/frontend processes.

## What is actually learned

The toy task predicts a movement result in a 4×4 world:

```text
x=1,y=2,a=R,b=0>  ->  2,2;
```

`R/L/U/D/W` mean right/left/up/down/wait. U increases y, D decreases it.
`b=1` means the selected move is obstructed; boundaries also prevent movement.
The 160 observable combinations are deterministically partitioned into 112
training, 24 validation and 24 final-test examples. No identical input occurs in
multiple splits. This is held-out combination testing, not evidence of transfer
to a new world or actuator regime.

A fixed byte tokenizer maps UTF-8 bytes to IDs 3–258, with reserved IDs 0–2 and
unused slots up to a 512-entry vocabulary. GGUF uses the GPT-2 byte alphabet with
no merges and the supported `default` pre-tokenizer. This avoids a separate
learned tokenizer while preserving the proposed model size. It does not make the
neural architecture GPT-2: the predictor is a Llama-style decoder.

The four-layer model has 787,584 parameters; five layers have 951,680. The decoder
uses RMSNorm, interleaved RoPE, causal attention and SwiGLU with untied embeddings.
Only outcome-token cross-entropy trains the model. Generation is unconstrained;
incorrect/malformed predictions count as failures.

After the requested baseline budget, a rule checks recorded validation behavior.
It skips growth when validation accuracy is at least 98% and loss is at most
0.02. Otherwise it requires at least 200 baseline updates and two consecutive
50-update windows with less than 5% relative loss improvement in each. Ongoing
improvement or insufficient history also skips growth. This is one bounded
decision at the end of baseline training, not continuous architecture search.

The four-layer control always continues for the additional budget. When growth
is proposed, a second branch resumes the same baseline checkpoint and sees the
same sampled examples for the same number of updates, but adds a fifth layer. Its output projections start at zero, preserving the
existing function at insertion. Existing Adam state is retained; the new layer
gets a new optimizer parameter group.

**The current proposal trigger is measurement-driven and rule-based, not
learned.** The archived initial pilots deliberately forced a proposal. A
separate acceptance rule accepts growth only when validation loss improves
by at least 1% over the continued four-layer control, exact validation accuracy
does not regress, and replay accuracy drops by at most two percentage points
relative to the baseline. Otherwise the continued control is selected. Replay
accuracy here is not a separate old-task retention test. Extra layers consume
extra computation: equal updates are not equal CPU time; both times are reported.

## Artifacts and checks

- `state.pt`: resumable experiment lifecycle, history and active learner.
- `baseline.pt`, `fixed-control.pt`, `growth-candidate.pt` (only if proposed),
  `selected.pt`: native
  C++ model/Adam/random-generator checkpoints, with version and layer metadata.
- `*-f32.gguf`, `*-f16.gguf`: independently tested inference snapshots. The
  selected F16 snapshot is also evaluated through llama.cpp on final-test inputs.
- `episodes.jsonl`, `summary.json`, `index.html`: task records, recorded validation
  history/metrics and the browser report. Training losses are printed to stdout.
- `COMPLETE`: successful evaluation/export marker.

`run.sh test` checks exact continuation of the weights/Adam/batch RNG after
save/reload, initial output equivalence after growth, updates to the new block,
proposal/accept/reject guards, tokenizer round-trips including whitespace/UTF-8, and
next-token logit agreement between LibTorch and llama.cpp for both layer counts.
The predeclared maximum absolute logit tolerances are 0.0003 (F32) and 0.02 (F16).
The trainer's interleaved Q/K layout is written directly; applying the HF
split-half Q/K permutation here would be wrong.

The GGUF contains the frozen predictor, not the native learner's growth policy.
Resuming adaptation requires the C++ implementation and native checkpoint.
The `resume` CLI continues the complete experiment, including a partially
trained candidate. The `compare STATE_A STATE_B` native diagnostic checks
decisions, validation history, all branch weights and batch RNG, and exact
agreement after another optimizer update; it does not alter the files.
No internet exploration, learned
repair selector, open-ended continual learning or general chatbot is implemented.
The separate two-world campaign above now measures old-task retention on held-out
examples and compares fixed-size controls under an approximate compute budget.

For a fresh-process prediction from one of the generated native checkpoints:

```bash
.tools/adaptive-llama/adaptive-llama predict artifacts/adaptive-llama/pilot-42/selected.pt 'x=1,y=2,a=R,b=0>'
```

Replace the checkpoint path with your run directory; the expected answer for
this movement example is `2,2;`. The command exits after the prediction.

See the [plan](../../docs/ADAPTIVE_LLAMA_PLAN.md) for the wider experiment scope.
