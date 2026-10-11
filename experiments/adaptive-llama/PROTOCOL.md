# Continual-learning comparison protocol

Declared before the campaign measurements on 2026-10-11. This is a bounded
synthetic experiment, not a language-chat benchmark. Reuse the existing native
model, optimizer, checkpoint and GGUF implementation.

## Tasks and splits

World A uses the original 4x4 movement rules. World B reverses R/L and U/D;
wait, obstacles and boundary clamping stay unchanged. Prompts include `w=0,`
or `w=1,`, making both rules representable without contradictory labels. The
world ID is supplied, not inferred. This tests contextual task switching, not
autonomous discovery of a hidden change.

Use the existing deterministic 112/24/24 train/validation/test partition of
160 coordinate/action/obstacle combinations. A combination keeps its split in
both worlds: no test combination appears in training under either world ID.
Also report accuracy on test combinations whose answer actually changes, since
blocked/wait/boundary cases can mask failure to learn the new rule.

## Four conditions, seeds 42, 43, 44

1. Fixed four layers, B training without A replay (forgetting control).
2. Fixed four layers, 12 B + 4 A examples in every 16-example adaptation batch.
3. Fixed five layers from initialization, with the same replay ratio.
4. Adaptive four-to-five layers, with the same replay ratio and paid search cost.

All four-layer conditions share the exact A checkpoint for a seed. The five-layer
control starts randomly with the same seed; it has its own initial training.
Do not choose seeds or revise thresholds based on test outcomes.

## Compute accounting

Pretraining budget: 600 four-layer update equivalents. Adaptation budget: 400.
Use a declared dense-matrix multiply-add proxy for a forward/backward update:

`3 * batch * sequence * (width*vocabulary + layers * (4*width^2 + 3*width*FF + 2*sequence*width))`

Sequence is the actual teacher-forced input length. All examples have equal
length. This counts dense projections, attention products and an approximate
3x forward/backward factor. It excludes nonlinearities, normalization, optimizer,
validation, serialization and export. It is not an exact FLOP or wall-time
measurement. Integer updates may underspend by less than one update per branch.
Record update counts, proxy work, training process CPU/wall time, total campaign
CPU/wall time and peak RSS. All runs use two CPU threads, sequentially.

## Adaptive rule and selection

Train the four-layer model for 100 adaptation updates. If new-world validation
accuracy is at least 98%, loss at most 0.02 and old-world validation accuracy
has fallen by no more than two percentage points, continue without growth.
Otherwise propose an identity-initialized fifth block. This trigger is a
hand-written adaptation/retention target, not a learned policy or a claim that
capacity is the cause of failure.

Split the remaining proxy budget equally between continued four-layer training
and the five-layer candidate, both from the warm checkpoint. **Both branches
consume the adaptive method's budget**, including a rejected branch. Branches
receive the same sampling prefix; five layers get fewer updates under the proxy.
Accept only if new-world validation loss improves at least 1%, new-world
accuracy does not regress, and old-world validation accuracy is no worse than
the control and within two percentage points of its pre-switch reference.
Otherwise select the continued four-layer control. No test metric is used here.

## Final measures and interpretation

Evaluate selected models on A and B test sets only after selection. Forgetting
is pre-switch A test accuracy minus final A test accuracy in percentage points;
negative means improvement. Report B acquisition relative to pre-switch B
accuracy, changed-answer subset accuracy, all seeds and mean/range. Initial
test metrics are recorded for reporting only, never supplied to proposal or
selection logic. Verify the selected F16 GGUF through llama.cpp on both tasks.

This campaign can establish a result for this toy setup, including a negative
result for growth. Three seeds and 24 test examples per world do not establish
general superiority. Replay, extra capacity and structural adaptation are
different interventions. The second architecture project remains separate.
