# Measurement-driven growth and resumable experiments

Measured on 2026-10-11, using the existing CPU LibTorch and llama.cpp libraries.
Training, growth selection and resume run in C++. No new model was downloaded.

## Open the recorded graphs

- [Seed 42: solved task, growth skipped](adaptive-seed-42.html)
- [Seed 44: stalled validation, growth tried and rejected](adaptive-seed-44.html)
- [Verification, provenance and artifact hashes](adaptive-verification.json)
- [Earlier forced-proposal results](RESULTS.md)

The HTML reports work offline without a model, server or Jupyter login. Phase
buttons change the displayed architecture; the history slider shows actual
recorded validation measurements. These are recorded experiments, not live
neuron animation. The growth button is disabled when no candidate was trained.

## What changed

Growth is no longer proposed unconditionally. After 600 baseline updates, these
runs apply the documented rule to the latest two 50-update validation windows.
A solved task skips growth; otherwise sustained improvement below 5% can trigger
a candidate. The thresholds are hand-written, not learned. Final-test examples
are consulted only after selecting the model.

Each run uses two CPU threads, batch size 16, 112 training combinations, 24
validation combinations and 24 held-out test combinations. The four-layer control
receives 200 further updates. A proposed five-layer candidate receives the same
200 updates and sampled examples, starting from the same baseline checkpoint.

| Seed | Proposal | Control validation loss | Candidate validation loss | Selected test accuracy | llama.cpp F16 accuracy |
| --- | --- | --- | --- | --- | --- |
| 42 | Skipped: target reached | 0.00111211 | No candidate | 100% | 100% |
| 44 | Tried: validation stalled; rejected | 0.17940557 | 0.18090321 | 91.67% | 91.67% |

Both select four layers: 787,584 parameters and a **1,592,576-byte F16 GGUF**.
The rejected five-layer candidate has 951,680 parameters and a 1,921,824-byte
F16 GGUF. Rejection preserves the continued four-layer control.

Seeds 42 and 44 deliberately exercise different proposal paths found in the
earlier pilots; they are not a random benchmark sample. There is still **no
demonstrated exact-accuracy advantage from structural growth**. Seed 44's loss
actually worsens while accuracy stalls: the rule treats insufficient improvement
as a reason to try growth, not proof that capacity caused the problem.

## Resume verification

The complete seed-44 experiment was repeated with fresh-process pauses after
baseline update 250, control update 650 and candidate update 650. Phase-local
training budgets resume from the saved global update count. Comparison against
the uninterrupted run passed for:

- The proposal, acceptance decision and full recorded validation history.
- Every branch's exact weights and batch random-generator state, including the
  rejected five-layer candidate.
- Exact agreement after one additional optimizer update from each checkpoint.

Separately, an actual SIGTERM stopped seed 42 after baseline update 52. It wrote
`state.pt`, printed `PAUSED` and left no `COMPLETE` marker. Fresh-process resume
matched the uninterrupted run under the same comparisons. Resuming an already
completed run changed no artifact bytes. Forced-kill and power-loss recovery
were not tested; only atomic replacement and periodic saving are implemented.

Use the [resume commands and checkpoint description](../README.md#stop-and-resume)
to reproduce. Native `compare STATE_A STATE_B` performs the comparisons without
changing either run. Keep all checkpoints in the run directory together.

## Resource observations

| Run | Total elapsed | Peak process RSS | Extra control training | Extra candidate training |
| --- | --- | --- | --- | --- |
| Seed 42, growth skipped | 16.45 s | 171.47 MiB | 3.380 s | Not run |
| Seed 44, growth rejected | 28.51 s | 185.74 MiB | 3.750 s | 9.046 s |

For context, the earlier seed-44 pilot with the same training/data budgets took
19.97 s and 211.12 MiB. The current observation is 42.8% longer overall and
12.0% lower in peak RSS. It adds periodic saving and validation; unrelated laptop
workloads were active, so these observations do not isolate the cause of either
change or establish performance parity. Branch training times exclude checkpoint
writing and validation. The five-layer branch was 141.2% slower than its control
in the current seed-44 run. Equal update budgets are not equal compute budgets.

The earlier seed-42 pilot took 21.43 s, but trained and exported an additional
candidate. Its workload differs; skipping that work is not a matched speed test.
The recorded peak RSS covers the native process, not the compiler, browser or
other programs. A resumed process's RSS does not include its earlier invocations.

## Checks and remaining boundaries

Native tests passed checkpoint/optimizer continuation for both architectures,
causal masking, function-preserving layer insertion, updates to new weights,
proposal/acceptance guards, UTF-8 tokenization and F32/F16 GGUF logit agreement.
The new reports passed browser checks for the skipped/rejected paths, history
selection, displayed metrics, narrow viewport layout and offline operation.

The final comparison build additionally checks rejected branch checkpoints and
makes completed resume a no-op; these verification/CLI changes do not alter the
training algorithm used for the recorded runs. Provenance hashes identify an
uncommitted working tree, not a claim that these changes are in the base commit.

The checkpoint holds the learning lifecycle; GGUF holds only the frozen inference
model. No learned proposal policy, independent old-task retention evaluation,
fixed-five-layer-from-start control, equal-compute comparison or new world rules
have been implemented. The alternative-architecture research remains separate.
Everything starts manually; no boot/login service or learning daemon is created.
