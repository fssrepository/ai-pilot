# Repair selection and composition — pilot v2

Frozen 2026-10-11 before evaluating the new final instances. Pilot v1 and its
results remain unchanged. This is a follow-up motivated by v1, not independent
confirmation on previously unknown task families.

## Diagnosis from policy-training evidence

Seed 42 v1 utility per unit search for a delay -1 training task was
0.0008516 (refit), 0.0004047 (split), 0.0002906 (delay). The inexpensive partial
repair outranked the complete solution. Three-neighbor averaging also mixed
operator preferences. The scheduler retried refit after fitting every coefficient
in an accepted candidate; on composed tasks that spent 4,832 evaluations without
adding a new topology. These are hypotheses about the failure, not proof that
our proposed replacement improves final performance.

## Fixed design

Reuse the same world, grammar, fitting, acceptance criterion and 64/32/128
train/validation/test sizes. Keep v1 commands and checkpoint reading. No new
primitive or gain beyond the declared [-2,2] coefficients is introduced.

The common v2 scheduler first tries refit (charged), then tries available
structural edits. An already delayed graph needs no add-delay operation; an
already split graph needs no split operation. All fitted proposals optimize
all coefficients on the same training data, so accepted fitted graphs do not
need another refit. A rejected structural operator may be retried only after
the current graph changes. At most six actual trials. Exhausted candidates
cannot replace the current graph. All control orders share this scheduler.

Compare six fixed priority permutations of refit/split/delay, the v1 empirical
selector, and a v2 empirical selector. The mandatory initial refit means some
fixed permutations are equivalent; retain and label them instead of treating
them as independent evidence. Also run the original v1 fixed-selector scheduler
as a paired legacy control to isolate the scheduler improvement. Learned v2
versus v1 selector isolates policy design under the same new scheduler.

V2 policy: one nearest training descriptor, with the v1 five residual features
plus two structural flags (2*has-delay, 2*has-split). Rank operators by
max(0, (old validation score - candidate score) / max(old score, 1e-12))
minus 0.01 * operator work / 20000. The score remains MSE + 0.001*complexity.
This rewards completion relative to the remaining error and weakly charges
cost; it is not claimed optimal. The flags prevent mixing incompatible stages.
No reward/neighbor/budget sweep or final-test tuning is allowed in this pilot.

Policy training: single-mechanism parameter/gate/delay, gains -1/+1, four
independent episode sets per task. For each, collect the initial graph, its
refitted graph, and accepted single-operator outcomes, deduplicated within that
task. Evaluate the same three operators from every state and store actual
outcomes; no task-family identifier enters the selector. V1 selector receives
its original six-record training procedure on new seed streams. Report both
training costs and memory sizes. Freeze both policies before final evaluation.

Seeds 101,102,103. World RNG domains: initial stable world 1,000,000+seed;
policy training 2,000,000+seed*1000+family*100+gain-index*10+replicate;
final worlds 3,000,000+seed*1000+family*100+gain-index*10. V1 policy's internal
seed is offset by 4,000,000. These domains are disjoint from pilot v1.
Final tasks: parameter/gate/delay/gated-delay at gains -2/+2. The combined family
is excluded from policy training but was already encountered in v1 research;
claim transfer to new instances/composition, not a never-seen research family.

Caps 2,500, 6,000, 20,000 candidate/example evaluations. The middle cap lies
between the analytically known costs of refit+delay (5,824) and
refit+split+delay (7,904). This cap is chosen before any final v2 evaluation.
Report all caps, all seeds and all fixed orders. Never compare only with the
worst hand-written order. Final outcomes do not enter ranking or acceptance.

## Evidence and boundaries

Record exact accuracy, composed-family accuracy, search work, recovery trials,
all accepted/rejected edits, active forgetting and explicit archived retention.
Work includes incomplete/rejected candidates and initial refit. Policy training
is separate overhead; compute any break-even using measured positive savings,
and report when there is no saving. Candidate/example work is not exact FLOPs;
CPU/wall observations and memory remain separate.

Schema 2 stores seven descriptor values and up to 512 records; schema 1 retains
its five values and exact legacy serialization. Test both and next-repair
continuation, policy freeze, no test-label leakage, scheduler eligibility,
budget rollback and identical repeated scientific results. Existing v1 results
must reproduce after the extension. UI uses the existing graph owner and viewer.
Extend the English report/PDF with this pilot; preserve v1 as historical evidence.

This addresses repair ranking and bounded composition only. Active curiosity,
noise-directed experiment selection, task routing and abstraction invention
remain distinct later gates. One native CPU thread, existing 512 MiB and
15-minute launcher limits, manual execution, no new dependencies or autostart.
