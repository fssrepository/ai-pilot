# Adaptive Llama — Independent Demonstrator Plan

Date: 2026-10-11. Status: bounded demonstrator complete and closed, including development,
verification, official llama.cpp chat and the branded PDF; the broader
research claims below remain unproven. See the
[implementation and commands](../../../experiments/adaptive-llama/README.md).

## Current implementation status

The requested four-part campaign is implemented and measured: changing-world
learning, held-out old-world forgetting, fixed/adaptive controls under a declared
matrix-work budget, and a live browser viewer with an English PDF report.

See the [results and limitations](../../publication/reports/adaptive-llama/ADAPTIVE_LLAMA_RESULTS.md),
[PDF](../../publication/reports/adaptive-llama/ADAPTIVE_LLAMA_RESULTS.pdf), and
[offline comparison](../../../experiments/adaptive-llama/examples/continual.html).
Twelve conditions cover three seeds and four methods. Old-example replay reduced
mean forgetting from 20.83 to 8.33 percentage points for the fixed four-layer
model. The adaptive policy accepted no growth and did not outperform its fixed
replay control. In one seed its strict guard rejected a candidate that improved
both validation tasks over its control, revealing a policy limitation.

The protocol charges both adaptive branches to the same approximate matrix-work
budget as the fixed controls. Exact equal-time comparison, learned repair
selection, unannounced rule changes and open-ended continual learning remain
unproven. The hand-written rule and supplied world ID are explicit limitations.

Native checkpoints retain model, optimizer, random state and segment timing.
The original single-world experiment also retains phase/history/decision state.
GGUF exports were checked through llama.cpp. The live viewer reports actual
measurements; it does not visualize neuron activity. The earlier
[single-world results](../../../experiments/adaptive-llama/examples/ADAPTIVE_RESULTS.md)
and [forced-proposal pilots](../../../experiments/adaptive-llama/examples/RESULTS.md)
remain historical evidence.

For the minimal tokenizer, the implementation uses a fixed UTF-8 byte alphabet
with no BPE merges instead of fitting SentencePiece; it retains 512 vocabulary
slots and verifies tokenization through llama.cpp. The learner and exporter run
in C++, reusing the installed CPU LibTorch libraries; no new large model was
downloaded.

## Purpose and boundary

Build a small, inspectable Llama-compatible model that learns a controlled task and can propose, evaluate and accept or reject one bounded structural change. This is an independent demonstrator alongside the [alternative-architecture research project](../repair-graph/MINIMAL_LEARNING_MODEL_PLAN.md). It is not that project's architecture, substitute or prerequisite.

The existing Qwen3.5-0.8B playground remains a separate pretrained-model reference. Here, “Llama” describes the proposed predictor architecture; “llama.cpp” describes the runtime used to execute exported snapshots. This project trains a tiny model from random initialization, not a downloaded Meta model.

## Initial deliverable

Use a four-layer decoder with hidden width 128, four attention/KV heads, feed-forward width 256 and a target vocabulary of 512 tokens. With untied embeddings and no attention/MLP biases, the expected starting parameter count is 787,584. Confirm the actual count in code. Its FP16 weights alone would occupy approximately 1.58 MB; GGUF metadata adds overhead, and the full training checkpoint is larger.

The adaptive component owns prediction, experience replay, weight updates, structural proposals and their evaluation. Start with one structural operation: add a fifth compatible decoder block. Initialize it to preserve the existing residual computation and verify that property before learning. Update optimizer parameter ownership explicitly. The five-layer configuration would have an expected 951,680 parameters under the same assumptions; verify both configurations independently.

An internal, explicitly implemented policy proposes a change when recorded validation behavior justifies trying it. Compare growth with spending the same additional training budget on the four-layer model. Acceptance requires recorded validation benefit within retention and resource constraints; rejection is a valid result. A test may force a proposal to verify the mechanism, but the UI must label it as forced rather than learned or autonomous.

The first selector is rule-based. This demonstrates bounded structural adaptation; it does not demonstrate a learned architecture-search policy or a novel learning paradigm.

## Implementation

Prefer C++ with LibTorch for the trainer and integrated adaptive component. Reuse existing tensor operations, automatic differentiation and optimizers. Implement the small model composition and adaptation logic, rather than a new numerical library. Inspect available compiler/library assets before installing dependencies. The existing Python analysis environment can be an optional numerical reference; the learner's execution and training should not depend on a Python controller.

Use the existing llama.cpp runtime setup, existing GGUF writer API and supported Llama tensor conventions for export. Test tokenization, tensor layouts and numerical agreement early. Native checkpoints retain optimizer, replay, random and architecture state; GGUF holds an inference snapshot. Stock llama.cpp does not execute our structural adaptation code.

Reuse the existing visual assets for an offline-capable browser viewer showing the actual four/five-layer graph, prediction errors, selected examples, proposed change, validation results and accepted/rejected checkpoint versions. Keep this experiment's model identifiers, checkpoints and runs separate from Qwen and the alternative-architecture experiment.

## Milestones and rough effort estimate

These estimates concern focused engineering effort, not model-training duration or guaranteed elapsed delivery. They assume the small task and single structural operator described here. Compiler/LibTorch compatibility and native GGUF export are the main uncertainties; revise the estimate after the initial feasibility work.

| Milestone | Evidence | Estimated effort |
| --- | --- | --- |
| Baseline and compatibility | Tiny C++ model learns a simple simulated-world transition task; checkpoint round-trip and an initial llama.cpp export comparison | 1–2 working days |
| Adaptive behavior | Four-to-five-layer proposal, continued training, acceptance/rejection, retention and fixed-size controls | 1–2 additional working days |
| Inspectable, reproducible demo | Graph/timeline, verified exports for both sizes, restart checks and a concise English report | 1–2 additional working days |

The original **3–6 working days** range was a pre-implementation estimate, not a current remaining-work estimate. The original single-world pilots took roughly 16–29 seconds; the twelve-condition two-world campaign took about five minutes on this shared CPU laptop, including evaluation/export. Neither estimates the duration of broader research. The status and measured results above take precedence over the original milestone estimates.

## Acceptance and resource rules

- The task has separate training, validation and final test episodes; candidates are not selected on the final test set.
- The viewer distinguishes a real graph change from changing numerical weights and from a forced diagnostic proposal.
- Record accuracy/loss, forgetting, parameter count, CPU time, peak memory, dataset/checkpoint hashes and random seeds. Include fixed four- and five-layer controls; additional capacity alone is not evidence of a better adaptation policy.
- Verify native resume and GGUF load independently for both structures. Report failures rather than silently changing model families.
- Run training and inference checks sequentially after checking machine capacity. Start with two CPU threads and a bounded pilot. Do not change unrelated workloads.
- Start manually; no boot/login services or background learning daemon. Keep generated artifacts in existing ignored model/artifact directories.

## Relationship to the research project

The demonstrator can teach us about numerical training, forgetting, graph inspection and export mechanics. The alternative-architecture project remains free to use a different representation and its own runtime. Share experiment interfaces or viewer components only where they actually fit; do not make the second project inherit Llama layers, tokenization or GGUF compatibility requirements.

Useful existing foundations: [LibTorch C++ frontend](https://docs.pytorch.org/tutorials/advanced/cpp_frontend.html), [pinned Llama conversion mapping](https://github.com/ggml-org/llama.cpp/blob/d81235049/conversion/llama.py), [GGUF C API](https://github.com/ggml-org/llama.cpp/blob/d81235049/ggml/include/gguf.h).
