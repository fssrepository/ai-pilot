# Continual memory, reusable repairs and active observations — pilot v3

Declared 2026-10-11 before final measurements. This completes the bounded
prototype's remaining experiment gates, not an open-ended intelligence claim.
V1/v2 evidence remains historical. No final-outcome tuning is allowed.

## Reuse and extension boundary

Reuse `Graph`, `Example`, `world`, `fit`, `adapt_v2`, graph serialization,
read-only viewer, report entry point and PDF builder. The existing learner has
one active graph and one archive; it cannot route among learned contexts,
create reusable repair templates, or choose observations. Add these capabilities
in `src/continual.h`, included by the same native executable. Training remains
C++17/standard library. No download, Python training, or autostart.

## A. Retention and automatic routing

Use seeds 201/202/203 and fresh RNG domains above 5,000,000. A regular observation
window contains 64 training and 32 validation episodes; 128 independent test
episodes are evaluation only. The learner receives examples, never task/family,
gain, segment index or a boundary flag. A fixed-size batch boundary is observable.
Each of 24 segments supplies two windows: eight family/gain(-1,+1) tasks,
eight family/gain(-2,+2) tasks, then a return to the first eight. Shuffle task
order once per seed. This is window-level routing, not instant hidden-change
recognition. Report accuracy before observing the window as well as after it.

Controls: single graph, single graph with equal-size replay mixture, protected
graph bank with evidence-based routing, and the same bank with learned reusable
repair templates. Per-window search cap 20,000. Route by lowest validation
MSE + .001 complexity, with deterministic ties; reuse only a zero-error graph.
If none fits, search from a blank graph with the existing fixed-012 v2 scheduler.
If single edits fail, all four methods may try a delayed-and-split topology
through the existing proposer with the remaining cap. No intermediate graph
is committed without validation improvement. This lookahead was added after
a pre-measurement fixture exposed a small-effect composition barrier.
Store at most 16 distinct accepted graphs and each graph's 32 validation witness
examples. Old graphs are immutable. Exceeding the limit rejects insertion;
no hidden eviction or unlimited memory. Replay uses 32 new + 32 retained train
examples and 16 new + 16 retained validation examples, selected deterministically
from a bounded 256-example FIFO. It cannot magically distinguish contradictory
worlds without context. All methods see the same new observations.

Measure return accuracy, pre-observation errors, fresh-task accuracy, active
old-task forgetting, witness retention, graph/library size and work. Routing,
validation and retention checks are charged separately from native search.
Report both total work and search work, not a false equal-compute claim.
No task ID is accepted by the model API. Ambiguous evidence cannot establish a
unique context; first-window errors and finite-memory limits must stay visible.

## B. Learned reusable abstractions

Extract a parameterized graph template only after two distinct accepted graphs
share a topology. Anti-unify differing coefficients into parameter holes; keep
common coefficients constant. A template stores its actual support IDs and
training descriptors. Rank templates by distance to those observed descriptors,
then propose the closest one with the existing fitter, restricted to its holes.
Accept on current validation improvement using the same complexity penalty.
If rejected, use the existing search with the remaining cap. Charge template
matching, proposal fitting, validation and extraction. Persist the library.
This is bounded abstraction of existing arithmetic graphs, not invention of
new primitive instructions. Compare with bank routing without templates on
new coefficient values; a copied old graph is not a successful transfer.

Use identical simulator evidence, routing and acceptance for both bank methods.
Measure template support, holes, accepted uses, validation error, saved search
and full work, including the cost of extracting reusable structures. Do not
report guaranteed speed or unlimited invention. No primitive encodes a task ID.

## C. Active observation selection and unpredictable output

Run separately to isolate exploration from routing. The common hypothesis class
is every distinct graph permitted by the existing coefficient/topology bounds:
affine/delayed, unsplit or split on either sensor. No true world label/gain enters
the explorer. Two anonymous sources accept the same 36 action/sensor probes.
One supplies the existing world transition; the other supplies independent
uniform random deltas in [-2,2]. Randomize source order by seed. Candidate
version spaces start identical. A contradictory source exhausts its version
space and is marked unsupported, not proven to be noise.

Compare uniform random queries, raw recent prediction-error seeking, and active
version-space disagreement (Gini disagreement over predicted outputs). Active
selection chooses the source/probe with greatest disagreement and stops when
no discriminating supported query exists. Empty hypothesis sets have no supported
information gain. This is uncertainty-driven inquiry with a designed
contradiction guard, not calibrated probabilities or learned curiosity rewards.

Seeds 201/202/203; four families, gains -2/+2, query caps 4/8/16/32: 288 conditions.
All methods use the same hypothesis class and observation interface. Charge
actual simulator queries and hypothesis evaluations separately. Count queries
to the unpredictable source, contradiction events, exact held-out transition
accuracy and remaining hypotheses. Evaluate all methods at all caps; include
per-seed results and query-selection overhead. Reduced interaction count does
not imply lower CPU cost. Test source permutation, action necessity and a
controlled contradictory fixture. Compare checkpoints and next query after
restore, and prove evaluation labels never change selection.

## Evidence

Native tests: legacy persistence/regression, bank and library restore and next
window, protected witness retention, unknown context, memory bound, held-out
label isolation, template support/holes/equivalence and learned transfer,
query-state/RNG restore, contradictory and unsupported sources, and budgets.
Repeat final campaigns in fresh directories and compare all scientific rows,
events and checkpoint bytes excluding CPU/wall/RSS only. Record original
failures honestly and do not turn failed final outcomes into a tuned success.

Extend the existing graph viewer with recorded window routing, library events
and active-query decisions. No background learner or remote API. Update English
plan/results and logo-free PDF, exact launch/configuration record and verification
manifest. Existing Qwen and adaptive Llama remain separate and preserved.
One native CPU thread, 512 MiB address-space ceiling, 15-minute timeout per run.
