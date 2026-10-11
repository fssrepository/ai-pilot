# Initial adaptive Llama results

Measured on 2026-10-11. These are results from a tiny, synthetic movement task,
not evidence of a general-purpose model or a novel self-learning architecture.

These historical pilots forced a growth proposal. The
[current measurement-driven version and resume results](ADAPTIVE_RESULTS.md)
are recorded separately; these files retain the original evidence.

## Inspect the runs

- [Seed 42: growth accepted](seed-42.html)
- [Seed 43: growth accepted](seed-43.html)
- [Seed 44: growth rejected](seed-44.html)
- [Machine-readable evidence and artifact hashes](verification.json)

The HTML files are self-contained and work offline. On repository websites,
download/open the HTML locally if it is displayed as source.

## Measurements

All runs used two CPU threads, batch size 16, 600 baseline updates, and another
200 updates per comparison branch. Each branch resumed the same baseline and
used identical sampled examples. There were 112 training, 24 validation and 24
final-test input combinations, with no identical input across splits.

| Seed | Four-layer validation accuracy | Five-layer validation accuracy | Decision | Selected final-test accuracy | llama.cpp F16 final-test accuracy |
| --- | --- | --- | --- | --- | --- |
| 42 | 100% | 100% | Accept growth: lower validation loss | 100% | 100% |
| 43 | 100% | 100% | Accept growth: lower validation loss | 100% | 100% |
| 44 | 91.67% | 91.67% | Reject growth: worse validation loss | 91.67% | 91.67% |

The four-layer column is the continued control, not the earlier baseline.
All selections were made before consulting final-test outcomes. Seeds were not
selected for favorable results: all three consecutive seeds are reported.

**No exact-accuracy advantage from adding the fifth layer was demonstrated.**
The first two runs accepted a small loss improvement under the predefined rule;
the third rejected growth. The task is too small/easy to establish useful
architecture-search behavior. A learned proposal policy, changing world rules,
separate old-task retention and equal-compute controls remain future work.

## Compute and storage

| Seed | Four-layer extra training | Five-layer extra training | Increase |
| --- | --- | --- | --- |
| 42 | 3.378 s | 5.941 s | 75.9% |
| 43 | 3.449 s | 4.482 s | 30.0% |
| 44 | 3.396 s | 4.183 s | 23.1% |

These are sequential wall-time observations on a shared laptop with unrelated
workloads active, not isolated performance benchmarks. Growth has more work per
update and was slower in every run. The comparison matches update/data budgets,
not CPU time. The seed-42 experiment including evaluation and exports took
21.43 seconds overall and peaked at 208.33 MiB process RSS. This excludes the
compiler, browser and other processes.

The four-layer F16 GGUF is **1,592,576 bytes**; the five-layer F16 GGUF is
**1,921,824 bytes**. They contain 787,584 and 951,680 model parameters respectively.
Native resumable checkpoints additionally store optimizer/random state.

## Verification boundary

Native self-tests passed exact continuation after checkpoint restore for both
architectures, unchanged output immediately after zero-residual growth, actual
updates to new-layer weights, and acceptance/rejection guards. Export tests
checked whitespace/UTF-8 tokenization and next-token logits across LibTorch and
llama.cpp. F32 tolerance was 0.0003 and F16 tolerance 0.02, fixed before training.
Actual trained-run errors are recorded in the evidence JSON and are far below
those thresholds.

The browser report was checked for actual phase changes, correct selected-layer
counts, recorded metrics, narrow viewport layout and offline operation without
JavaScript errors or network requests. This is replay of recorded measurements,
not live neural animation.

The Qwen notebook, token-flow template and vendored JavaScript/license files
were moved without changing their bytes. Import and path checks passed without
loading Qwen. Qwen inference was not repeated because its implementation was
only relocated and its existing entry points retained.

Efficiency review: examples are held in memory during training; weights and
optimizer state are saved at experiment boundaries. The tiny inference parity
and generation checks intentionally recompute bounded prefixes (at most four
output tokens), so their timings are not optimized serving benchmarks.
