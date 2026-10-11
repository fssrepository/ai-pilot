# Repair Graph — alternative architecture research

Status on 2026-10-11: **the bounded prototype implementation is complete and
verified**, including retention, observation-based context routing, learned
repair abstractions and active inquiry with noise controls. These capabilities
now have C++ implementations and measurements; unrestricted self-improvement
and general intelligence are not claimed.

Start with the [current interactive experiment](examples/pilot-v3.html),
[English results](../../guides/publication/reports/repair-graph/CONTINUAL_INQUIRY_RESULTS.md), or
[PDF](../../guides/publication/reports/repair-graph/CONTINUAL_INQUIRY_RESULTS.pdf).
The [v3 protocol](PROTOCOL_V3.md) fixes the experimental design before final
measurement. [Verification](examples/verification-v3.json) records the original
fixture failure, final checks, reproducibility and artifact hashes.

## Current measured outcome

V3 has 12 continual streams / 576 windows and 288 inquiry conditions. A bank
retains 16 learned graphs and routes from actual observations without task IDs.
On return, it reuses old behavior without new search. The single graph also
recovers with enough new training, so this is a relearning-cost benefit, not
an accuracy improvement over an already-perfect after-window control.

The bank learns parameterized templates from shared structures in accepted
graphs. Transfer to new coefficient values reduces search work by 48.61% versus
the same bank without templates. Counting the additional declared operation
proxies, the saving is 37.36%. Those units are not exact FLOPs or a speed claim.

At eight queries, active inquiry reaches 100% accuracy versus 89.78% for random
inquiry and 64.88% for raw error seeking, but costs roughly 56.7 times as many
hypothesis evaluations as random. It detects evidence inconsistent with its
model class and stops querying unsupported sources. That is not proof that an
unknown source is intrinsically random. Source names do not guide selection.

The current graph alone still forgets other contexts. The bounded graph bank
and routing preserve usable old knowledge; they do not predict a hidden change
before receiving observations. Learned abstractions combine existing primitives,
and curiosity uses a designed disagreement rule. These precise limits are
part of the result, not absent implementations of those capabilities.

## Earlier repair-selection follow-up

The [v2 protocol](PROTOCOL_V2.md), [viewer](examples/pilot-v2.html),
[results](../../guides/publication/reports/repair-graph/REPAIR_POLICY_RESULTS.md), [PDF](../../guides/publication/reports/repair-graph/REPAIR_POLICY_RESULTS.pdf)
and [verification](examples/verification-v2.json) preserve the 648-condition
repair-policy experiment. It reached 87.50% versus 73.76% for fixed ordering at
a 6,000-evaluation cap, and saved 10.53% online work at the larger cap. Its
policy-training cost did not amortize in that campaign. V3 does not rewrite
those outcomes or use final labels to tune them.

## Historical first pilot

The [v1 protocol](PROTOCOL.md), [recorded report](examples/pilot.html),
[English report](../../guides/publication/reports/repair-graph/ALTERNATIVE_ARCHITECTURE_RESULTS.md),
[PDF](../../guides/publication/reports/repair-graph/ALTERNATIVE_ARCHITECTURE_RESULTS.pdf) and
[original verification](examples/verification.json) remain historical evidence.
It had 264 conditions, no learned-selector advantage, and weak composed-task
performance. V2 fixes were motivated by that failure. All original scientific
rows and all 198 checkpoint bytes still reproduce with the extended executable.
Historical source hashes refer to the earlier implementation, not today's files.

## Build and run manually

```bash
./experiments/alternative-architecture/tools/build.sh
./experiments/alternative-architecture/tools/run.sh test
./experiments/alternative-architecture/tools/run.sh test-v2
./experiments/alternative-architecture/tools/run.sh test-v3
./experiments/alternative-architecture/tools/run.sh campaign-v3
```

Run from the repository root. An optional second argument chooses a new output
directory; existing campaign directories are rejected. The command prints its
output path. `campaign` retains the v1 experiment; `campaign-v2` runs the repair-policy follow-up, and `campaign-v3` runs the
continual and active-inquiry experiments.
One C++17/O2 compiler process is used, with warnings treated as errors. Native
execution is one thread, nice 15, a 512 MiB virtual-memory ceiling and 15-minute
timeout. The launcher checks available RAM. No new dependency/model download,
Python training, startup service or background learner is used.

Schemas `REPAIR_GRAPH 1/2` remain readable. `REPAIR_CONTINUAL 1` stores the
active learner, protected graph bank, witnesses, descriptors, replay buffer and
learned templates. `REPAIR_EXPLORER 1` stores hypothesis IDs, predictions, query
history summaries and RNG state. Exact sizes are in the current report.
These files are interpreted by this C++ runtime; they are not GGUF and cannot
run directly in stock llama.cpp. Learner restore and identical next decisions
are tested. Campaign-level restart after interruption is not provided; use a
fresh directory to reproduce a campaign.

## Recorded viewer configuration — 2026-10-11

Every change to the active run, template, address, model semantics or launch flags
must update this record and its versioned source in the same change.

| Setting | Current value |
| --- | --- |
| Measurement | `artifacts/alternative-architecture/pilot-03` |
| Address | http://127.0.0.1:8091/ — loopback only |
| Launcher | [tools/start-viewer.sh](tools/start-viewer.sh) |
| Server | Existing adaptive [read-only server](../adaptive-llama/tools/live.cjs), optional template argument |
| Template | [ui/report.html](ui/report.html), supports v1, v2 and v3 reports |
| Read routes | `/`, `/index.html`, `/status.json`; all other paths rejected |
| State | Completed recorded repairs; no training started by the page |
| Prediction form | Evaluates the displayed frozen graph locally; parity checked with native C++ |
| Lifetime | Manual foreground Node process; Ctrl+C stops it; no boot/login startup |

```bash
./experiments/alternative-architecture/tools/start-viewer.sh
```

An optional first argument selects another completed run, including pilot-01 or pilot-02.
Stop the current viewer before starting its replacement. New measurements do
not silently replace the configured run. The current report also opens offline
from `examples/pilot-v3.html`. The earlier Qwen/adaptive chat stays at 8088 and
the adaptive comparison at 8090; neither model's launch settings were changed.

In the viewer, select **Kísérlet**:

- **Memória és tanult javítási sablonok:** select `201 · bank-macros`, move the
  window slider into the return phase, and observe automatic graph retrieval,
  zero new search, library supports and free coefficients. **Lejátszás** replays
  the recorded decisions. **Javítás előtti állapot** shows the previous graph.
- **Aktív megfigyelés és zaj:** choose a run with `active` and an eight-query cap.
  The timeline shows the chosen anonymous source, action, predicted/observed
  output and shrinking hypothesis set. Source ground truth is shown for
  explanation only; the learner did not receive it.

The form computes the displayed frozen graph locally. It neither retrains nor
creates a natural-language chat. Actual training occurs through `campaign-v3`.
To inspect a persisted final state through native C++:

```bash
.tools/repair-graph/repair-graph predict \
  artifacts/alternative-architecture/pilot-03/201-bank-macros.model 2 1 0 0 -1
.tools/repair-graph/repair-graph predict-inquiry \
  artifacts/alternative-architecture/pilot-03/201-3--2-active-8.inquiry 1 2 1 0 0 -1
```

The inquiry example selects source 1, then x=2, action=1, gate=0, noise=0,
previous action=-1. Its learned gated-delay rule predicts **4**; changing gate
to **1** predicts **2**, and setting gate=0/previous action=1 predicts **0**.
`predict` arguments after the checkpoint are x, action, gate, noise, optional
previous action. `predict-inquiry` additionally takes source 0/1 first.
These calls do not modify the saved model or guess a new hidden context.

## Reports and checks

```bash
.venv/bin/python experiments/alternative-architecture/tools/report.py \
  artifacts/alternative-architecture/pilot-03
```

Python renders documents and plots only. The common report entry point recognizes
all three report schemas. Native tests cover protected retention, context return,
learned parameter holes and transfer, memory bounds, budget rollback, final-label
isolation, schema restore, next-window/query continuation and contradictory
sources. Repeated v3 measurements reproduce all scientific rows/events and
300 checkpoints exactly; v1/v2 results and checkpoint bytes remain unchanged.

The existing `tests/check-report.cjs` covers online/offline graph playback,
automatic return evidence, learned supports, inquiry history and native/browser
prediction parity for both new persisted states. It retains legacy UI coverage,
mobile checks and read-only route checks. Set `PLAYWRIGHT_MODULE` to an existing
installation when needed; no package download is required.

## Implementation map and completion boundary

| Owner | Implemented capability |
| --- | --- |
| `src/main.cpp` | Shared world, bounded graph, fitting, legacy persistence and native prediction |
| `src/repair_policy.h` | V2 learned repair ranking and scheduler |
| `src/continual.h` | Protected graph bank, evidence-based routing, anti-unified templates, active inquiry, v3 persistence/tests/measurement |
| `ui/report.html` | Shared v1/v2/v3 measured graph and decision playback |
| `tools/report.py` / `report_v3.py` | Validation, current English results and PDF generation |

The current pilot closes the implementation gates in the research plan. Its
finite grammar, deterministic worlds, fixed observation windows and small
controlled measurements do not establish open-ended learning or a novel frontier
architecture. They give concrete working mechanisms to inspect and challenge.
Hypothesis search and candidate fitting stay in bounded memory; checkpoints are
written per completed condition, not inside scoring loops. The viewer reads a
recorded measurement once, without background training or polling.

## User-provided research reference

On 2026-10-11 the user identified [TypeSafe AI's System One Models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
as a reference for the next research discussion. Preserve this exact reference;
it is Jev, not JEPA. The article describes typed probabilistic decisions,
parallel output sampling and Reinforcement Learning for Calibrated Decisions
(RLCD). These are the vendor's descriptions, not independently reproduced results.
Schema validity does not establish factual correctness. Evaluate its relevance
after the adaptive Llama campaign; do not silently substitute it for the existing
alternative-architecture plan or assume its implementation/weights are open.
