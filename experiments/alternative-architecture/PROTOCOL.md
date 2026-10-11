# Repair Graph pilot v1 — frozen before measurement

Date: 2026-10-11. This is the first bounded implementation of the broader
[research plan](../../guides/development/repair-graph/MINIMAL_LEARNING_MODEL_PLAN.md), not completion of its
open-ended research agenda. No transformer, pretrained model or Jev service is used.

## Claim and falsification

Test separately whether changing a symbolic predictor's structure helps, and
whether a policy learned from earlier repair outcomes chooses useful repairs
sooner than a fixed order. Failure to outperform either control is a valid result.
This is a constrained program-search pilot, not a new frontier architecture.

## World and grammar

A one-dimensional cart has integer position x, action a in {-1,0,1}, an observed
binary gate b and an unrelated random binary sensor n. A two-step episode exposes
the previous action through an explicit Delay(a) node if the learner introduces
one. The learner receives observations and outcomes, never task type or gain.
The simulator uses parameter changes, gating, delayed actuation, or their
combination. This narrows the proposed movement/object world to one axis and a
binary gate; object manipulation, latent perception and exploration are later gates.

The supplied scaffold is x_next = x + delta. A graph predicts delta using an
integer affine expression c0+c1*a, optionally c2*Delay(a), and optionally one
conditional split on b or n. Coefficients: c0 in [-1,1], c1/c2 in [-2,2]. These
arithmetic/conditional/delay primitives and the position scaffold are designed
by us. The learner searches their coefficients and connections; it does not
invent primitives or arbitrary source code. Maximum two affine leaves, one
split and one delay dependency; all graph edges within a step are acyclic.

Operators: refit existing coefficients; try splitting on b and n; add a delay
input. Existing topology is preserved when fitting. Split and delay can compose.
Full finite coefficient enumeration minimizes training squared error with
simpler coefficients winning ties. Complexity is 3 base units + 2 for delay +
3 for a split + one per nonzero leaf coefficient; this is a designed penalty,
not the number of SVG boxes. An operator's candidate must reduce
validation MSE + 0.001 * structural complexity by more than 1e-9 to be accepted.
The current program is kept on rejection; no test result enters selection.

## Data, controls and costs

Seeds 42,43,44. Per task: 64 training, 32 validation and 128 final test episodes.
Positions are disjoint: train [-3,3], validation [-7,-4], test [4,7]. Other input
combinations can recur across partitions; this is compositional generalization
across positions, not an unseen-action-combination claim. Each episode's first
action populates history, then its second transition is predicted. The noisy
sensor is independent of outcomes. Every method sees identical examples.

Policy training tasks: parameter/gate/delay, gains -1 and +1 (six tasks per seed).
Every operator is evaluated on each training task from the same initial model;
its validation improvement and search cost are stored with an error descriptor.
The descriptor consists of empirical correlations between residuals and a,
previous a, a*b and previous a*b, plus normalized residual MSE, all from training
examples. A fixed 3-nearest-neighbor policy ranks operators by stored validation
improvement per candidate-example evaluation; it is frozen on development/test
worlds. The fixed selector is refit, split, delay. Disabling repair-policy memory
is exactly that fixed-order control, with identical available operators.

Development: parameter gain 0 and gated-delay gains +/-1. Final transfer:
parameter/gate/delay/gated-delay, gains +/-2 (eight tasks per seed). Gated-delay
is absent from policy training; report that family separately. The gain range
is deliberately within the grammar. No tuning after inspecting final results.

Controls: tabular delta memory keyed by a,b,n (the same x+delta scaffold); fixed affine without delay/split;
structural search with fixed selector; structural search with learned selector.
Run both 2,500 and 20,000 candidate-example-evaluation caps, at most six trials
(two passes over three operators). Count rejected and abandoned search work.
Only fully fitted/evaluated candidates may be accepted. Report final accuracy,
first-trial accuracy, recovery trials, total work, CPU/wall time and structural complexity.
The proxy counts each candidate/example prediction equally; graph size, ranking
and sorting overhead are additionally reported through wall/CPU measurements.
It is not exact FLOPs or equal elapsed time. Offline policy training cost is
reported separately and is not free; no amortization claim without accounting.

Retention: report old-world accuracy before/after replacing the active graph,
and accuracy using its preserved version. Version retrieval uses an explicit
caller-selected checkpoint, not inferred task identity or learned routing.
This distinguishes actual active-model forgetting from persistence.

## Persistence and verification

A versioned native text checkpoint retains current/archived graphs, previous
observed action, episode memory, learned repair-policy records and RNG state.
Save through a temporary file and rename. Verify exact restore and identical
next repair/RNG behavior; reject unknown schema and invalid graph fields.
Power-loss durability is not claimed. Native reports contain real accepted and
rejected graph states and their measured validation metrics.

C++17, standard library only, one CPU thread, manual launch, 512 MiB address-space
limit for native execution and 15-minute timeout. Build uses existing g++, one
process. No new model/dependency download, startup service or daemon. Generated
checkpoints live under ignored artifacts/alternative-architecture. Reuse the
existing read-only viewer server and English PDF builder and logo-free cover layout; the
new graph semantics require their own report template. Qwen/adaptive model
sources and archived results remain intact.

## Separate later gates

Active experiment selection versus random exploration, learning-progress noise
avoidance, probabilistic calibration, abstraction invention, unrestricted graph
growth, learned task routing and open-ended learning are not implemented in this
pilot. Passive rejection of a noisy feature does not establish curiosity.
