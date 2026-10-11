"""Render pilot-v2 results through the existing report entry point and PDF builder."""
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import matplotlib.pyplot as plt


def render(root, run, data):
    rows = data['results']
    assert len(rows) == 648 and len({r['id'] for r in rows}) == 648
    assert set(data['methods']) == {'legacy-fixed', 'fixed-012', 'fixed-021', 'fixed-102', 'fixed-120', 'fixed-201', 'fixed-210', 'learned-v1', 'learned-v2'}
    for r in rows:
        assert r['work'] <= r['cap'] and sum(e['work'] for e in r['events']) == r['work']
        assert r['archived_old'] == r['old_before']
        assert (run / (r['id'] + '.model')).stat().st_size == r['checkpoint_bytes']
        for e in r['events']:
            if e['exhausted']:
                assert not e['accepted'] and e['candidate'] is None and e['candidate_mse'] is None
            if e['accepted']:
                assert e['candidate_mse'] + .001 * e['candidate']['complexity'] < e['before_mse'] + .001 * e['before']['complexity'] - 1e-9
            else:
                assert e['selected'] == e['before']

    def select(method, cap=20000, seed=None, family=None):
        return [r for r in rows if r['method'] == method and r['cap'] == cap
                and (seed is None or r['seed'] == seed) and (family is None or r['family'] == family)]

    def mean(method, key, cap=20000, seed=None, family=None):
        return statistics.mean(r[key] for r in select(method, cap, seed, family))

    labels = {'legacy-fixed': 'Legacy scheduler', 'learned-v1': 'Old learned policy', 'learned-v2': 'New learned policy'}
    labels.update({m: 'Fixed ' + m[-3:] for m in data['methods'] if m.startswith('fixed-')})
    fixed = [m for m in data['methods'] if m.startswith('fixed-')]
    best_accuracy = max(mean(m, 'accuracy', 6000) for m in fixed)
    best_work = min(mean(m, 'work') for m in fixed)
    saving = best_work - mean('learned-v2', 'work')
    training = statistics.mean(r['v2_work'] for r in data['policy_runs'])
    v1_training = statistics.mean(r['v1_work'] for r in data['policy_runs'])
    offline = root / 'experiments/alternative-architecture/examples'
    shutil.copyfile(run / 'status.json', offline / 'pilot-v2.json')
    template = (root / 'experiments/alternative-architecture/ui/report.html').read_text()
    (offline / 'pilot-v2.html').write_text(template.replace('__REPORT_JSON__', json.dumps(data).replace('<', r'\u003c')))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    display = ['legacy-fixed', 'fixed-012', 'fixed-021', 'learned-v1', 'learned-v2']
    tick_labels = ['Legacy', 'Split first', 'Delay first', 'Learned v1', 'Learned v2']
    for ax, cap in zip(axes, [6000, 20000]):
        ax.bar(range(5), [100 * mean(m, 'accuracy', cap) for m in display], color=['#8892a3', '#6787b5', '#559bab', '#8679b1', '#29a694'])
        for i, method in enumerate(display):
            ax.scatter([i-.12, i, i+.12], [100*mean(method, 'accuracy', cap, seed=s) for s in [101, 102, 103]], color='black', s=14, zorder=3)
        ax.set_xticks(range(5), tick_labels, rotation=20, ha='right')
        ax.set_ylim(0, 105)
        ax.set_ylabel('Exact accuracy (%)')
        ax.set_title(f'{cap:,} candidate/example evaluations')
        ax.grid(axis='y', alpha=.2)
    fig.savefig(root / 'docs/examples/repair-policy-v2.png', dpi=160)
    plt.close(fig)
    text = f'''# Repair Graph v2: repair selection and composition

Measured 11 October 2026. The first follow-up addresses repair ranking and
composed changes; active curiosity, automatic task routing and abstraction
invention remain separate later gates. This is bounded symbolic program search,
not a frontier architecture or unrestricted self-improving system.

## Measured result

On 24 fresh final instances per condition, with a 6,000-evaluation search cap,
the new learned selector reached **{100*mean('learned-v2', 'accuracy', 6000):.2f}%**
versus **{100*best_accuracy:.2f}%** for the best fixed priority order. The improvement
is **{100*(mean('learned-v2', 'accuracy', 6000)-best_accuracy):.2f} percentage points**.
All fixed orders are reported below, including equivalent permutations.

At the 20,000 cap, the new scheduler's fixed and learned selectors all reached
100% on these test instances, including the composed gated-delay family. The
legacy scheduler reached {100*mean('legacy-fixed', 'accuracy'):.2f}% overall.
The new learned selector used **{mean('learned-v2','work'):.0f}** online evaluations
per task versus **{best_work:.0f}** for the least costly fixed order: a
**{100*saving/best_work:.2f}%** reduction in this search-work proxy.

This is a positive pilot result for repair ranking and composition within the
supplied grammar. It does not establish general superiority. The policy costs
more to train, and total work has not broken even in this small campaign.

![Means across three seeds; black dots are individual seed means. Fixed permutations collapse to two effective structural priorities.](examples/repair-policy-v2.png)

## Every control and budget

Each entry averages three seeds, four task families and gains -2/+2 (24 tasks).
Operators: 0=refit, 1=split on a sensor, 2=add delayed action. The common scheduler
always starts with refit, so permutations 012/102/120 and 021/201/210 are equivalent.
They are not six independent pieces of evidence.

| Method | Cap | Accuracy | Composed accuracy | Search work |
| --- | --- | --- | --- | --- |
'''
    for cap in data['caps']:
        for method in data['methods']:
            text += f"| {labels[method]} | {cap:,} | {100*mean(method,'accuracy',cap):.2f}% | {100*mean(method,'accuracy',cap,family='gated-delay'):.2f}% | {mean(method,'work',cap):.0f} |\n"
    text += '''
The old learned policy and the new one share the new scheduler. The legacy
fixed control retains the original scheduler. This separates improvement from
avoiding redundant search from improvement in choosing structural repairs.
At the smallest cap, all methods stop before a complete structural candidate;
there is no accuracy advantage. At the middle cap the new learned policy helps
overall but still fails some composed cases; 100% is limited to the larger cap.

## Individual seeds

| Seed | Best fixed at 6,000 | Learned v2 at 6,000 | Fixed work at 20,000 | Learned v2 work at 20,000 |
| --- | --- | --- | --- | --- |
'''
    for seed in [101, 102, 103]:
        text += f"| {seed} | {100*max(mean(m,'accuracy',6000,seed) for m in fixed):.2f}% | {100*mean('learned-v2','accuracy',6000,seed):.2f}% | {min(mean(m,'work',20000,seed) for m in fixed):.0f} | {mean('learned-v2','work',20000,seed):.0f} |\n"
    text += f'''
## Training cost and break-even

The new policy costs a mean **{training:,.0f}** candidate/example evaluations per
seed, versus **{v1_training:,.0f}** for the old learned policy and no policy-training
cost for a fixed order. Both training procedures are included in the recorded
campaign resource measurements. Candidate generation to collect intermediate
training states is charged as well as each state's operator panel.

With a measured average online saving of {saving:.0f} evaluations at the largest
cap, comparison with the cheapest fixed order would require approximately
**{math.ceil(training/saving):,} tasks per reused policy** to amortize its training.
This is an extrapolation assuming the same task distribution and unchanged
savings; it was not tested over that many tasks. Incremental training over v1
would require about {math.ceil((training-v1_training)/saving):,} such tasks.
The experiment therefore demonstrates better bounded online decisions, not a
measured reduction in total training-plus-use cost or guaranteed wall-clock speed.

The proxy counts candidate/example predictions. Descriptor calculation,
neighbour search and graph-size-dependent arithmetic are not identical unit
costs. Wall/CPU measurements include this overhead, but the run is too short
and the host too shared for a reliable latency comparison.

## What changed and why

The old utility divided immediate validation improvement by operator work. On
a policy-training delay task, seed 42 assigned refit 0.0008516 versus delay
0.0002906: a cheap partial repair outranked a complete solution. Three-neighbour
averaging mixed preferences, and training contained only the initial state.

The v2 policy is still a transparent empirical selector. It uses one nearest
training descriptor, including residual correlations and whether the graph
already has delay/split dependencies. It stores normalized remaining-error
reduction, minus a small declared cost penalty. Training includes initial,
refitted and accepted intermediate states on single-mechanism tasks. Neither
family labels nor simulator gains are policy inputs. The policy is frozen on
final evaluation instances.

The common scheduler does not refit an accepted candidate that already fitted
all its coefficients on the same evidence. It skips an already-present delay
or split, and retries a rejected edit only after the graph changes. Thus a
split and delayed dependency can both be discovered before exhausting the
same declared budget. All priority controls use this scheduler; no method gets
additional simulator data or special graph primitives.

The graph and fitting grammar are unchanged from v1. Nodes still represent
predefined arithmetic, a binary conditional and an observed previous action.
The system learns their coefficients and connections, not new primitives.

## Data independence and limits

The protocol was frozen before viewing the v2 final outcomes. Seeds 101/102/103
and disjoint initial/policy/final RNG domains replace the first pilot's streams.
Each task retains 64 training, 32 validation and 128 test episodes, with disjoint
position ranges and the same supplied x+delta scaffold. Other input combinations
may recur across these partitions.

Policy training uses parameter/gate/delay at gains -1/+1, four episode sets each,
and intermediate states. Final evaluation uses gains -2/+2 and includes gated
delay. The composed family is absent from policy training but was already
examined in v1 research. This is transfer to fresh instances of known research
tasks, not validation on a family never seen during system design. No reward,
neighbor or cap sweep followed final evaluation. The middle cap was fixed from
the analytically known refit+delay versus refit+split+delay work requirements.

## Retention remains a separate problem

'''
    text += f'''For the new selector at the larger cap, old-world mean accuracy falls from
{100*mean('learned-v2','old_before'):.2f}% to {100*mean('learned-v2','old_after'):.2f}%
in the active graph. Explicit retrieval of the saved old graph retains
{100*mean('learned-v2','archived_old'):.2f}%. This remains version preservation,
not simultaneous skill retention or automatic task routing. The repair policy's
success does not resolve that research gate.

## Native checkpoints and resources

The 648-condition campaign took {data['wall_seconds']:.4f} seconds wall time,
{data['cpu_seconds']:.4f} seconds CPU time, and {data['peak_rss_kib']/1024:.2f} MiB
peak RSS in one observed run. It uses one C++ thread and the existing 512 MiB
address-space and 15-minute launch limits. No library/model download was needed.
These tiny finite tasks are not a throughput benchmark against language models.

Schema 2 stores seven descriptor values and bounded repair memory. The new
policy's checkpoints span {min(r['checkpoint_bytes'] for r in select('learned-v2')):,}
to {max(r['checkpoint_bytes'] for r in select('learned-v2')):,} bytes, including
current/archived graphs, observed episodes, policy, history and RNG state.
Schema 1 remains readable and is serialized exactly as before. The native C++
runtime executes these data; they are not GGUF or natural-language models.

'''
    text += '''## Verification and use

Focused native checks passed for schema-2 round trips, identical next repair
following restore, frozen policy memory, absence of final-label leakage, budget
rollback, eligibility rules and a composed held-out fixture. The v2 campaign
reproduced all 648 scientific rows and all 648 checkpoints exactly. The original
v1 campaign also reproduced all 264 rows and 198 checkpoint files byte for byte;
wall/CPU/RSS are excluded from these deterministic comparisons.

The existing read-only graph viewer now defaults to the v2 campaign at
http://127.0.0.1:8091/. It displays actual before/candidate/selected expressions
and their measured edit history. The local prediction form evaluates the
selected frozen graph. HTTP/offline tests check native/browser parity, rollback
visibility, playback, mobile layout and restricted routes. No boot/login startup
or background learning daemon is installed. The earlier pilot remains available
offline and through an explicit run-directory argument.

```bash
./experiments/alternative-architecture/tools/build.sh
./experiments/alternative-architecture/tools/run.sh test-v2
./experiments/alternative-architecture/tools/run.sh campaign-v2
./experiments/alternative-architecture/tools/start-viewer.sh
```

The default viewer uses artifacts/alternative-architecture/pilot-02; a new
campaign prints a new directory and must be passed explicitly to the viewer.
Existing output directories are never overwritten. The README records the exact
active configuration, working examples and regeneration commands. The first
pilot and its English PDF remain historical evidence.

## Next research gate

This follow-up completes the first measured improvement to repair selection and
bounded composition. Next comes retention and automatic context selection,
then active experiment choice with a random-exploration/noise control, and
learning reusable abstractions. Those capabilities remain unimplemented; no
claim of open-ended intelligence follows from this small positive result.
'''
    source = root / 'docs/REPAIR_POLICY_RESULTS.md'
    source.write_text(text)
    subprocess.run([sys.executable, str(root/'tools/build-guide.py'), str(source), '--cover', str(root/'docs/brand/repair-policy-cover.json'), '--title', 'Repair Graph v2 — Repair Selection and Composition', '--footer', 'AI Pilot | Repair Graph v2 | 11 October 2026'], check=True)
    print('PASS: v2 648 conditions, budgets, acceptance, rollback and retention accounting; report generated.')
