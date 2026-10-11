"""Validate and render the native continual/inquiry measurements, without training."""
import json
import statistics
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def render(root, run, data):
    streams, inquiries = data['streams'], data['inquiries']
    assert len(streams) == 12 and len(inquiries) == 288
    assert len({r['id'] for r in streams}) == 12
    assert len({r['id'] for r in inquiries}) == 288
    for r in streams:
        assert len(r['windows']) == 48 and r['bank_size'] <= 16
        assert (run / (r['id'] + '.model')).stat().st_size == r['checkpoint_bytes']
        assert 0 <= r['accuracy'] <= 1
        for key in ['search_work', 'routing_work', 'retention_work', 'extraction_work', 'matching_work']:
            assert r[key] == sum(w[key] for w in r['windows'])
        for w in r['windows']:
            assert w['search_work'] <= 20000
            assert w['search_work'] == sum(e['work'] for e in w['repairs'])
            if w['reused']:
                assert not w['repairs'] and w['search_work'] == 0 and w['route'] >= 0
            for e in w['repairs']:
                if e['accepted']:
                    assert not e['exhausted']
                    assert e['candidate_mse'] + .001*e['candidate']['complexity'] < e['before_mse'] + .001*e['before']['complexity'] - 1e-9
                else:
                    assert e['selected'] == e['before']
            for template in w['library']:
                assert len(template['supports']) >= 2
                assert all(0 <= i < w['bank_size'] for i in template['supports'])
    for r in inquiries:
        assert r['queries'] == len(r['events']) <= r['cap']
        assert r['noisy_queries'] == sum(e['unpredictable'] for e in r['events'])
        assert r['evaluations'] >= sum(e['evaluations'] for e in r['events'])
        assert (run / (r['id'] + '.inquiry')).stat().st_size == r['checkpoint_bytes']
        assert all(e['after_hypotheses'] <= e['before_hypotheses'] for e in r['events'])
        assert 0 <= r['accuracy'] <= 1
    def group(method, cap=None, seed=None):
        source = streams if cap is None else inquiries
        return [r for r in source if r['method'] == method and (cap is None or r['cap'] == cap) and (seed is None or r['seed'] == seed)]
    def mean(method, key, cap=None, seed=None):
        return statistics.mean(r[key] for r in group(method, cap, seed))
    def work(r):
        return sum(r[k] for k in ['search_work','routing_work','retention_work','extraction_work','matching_work'])
    search_saving = 1 - mean('bank-macros','search_work')/mean('bank','search_work')
    total_saving = 1 - statistics.mean(work(r) for r in group('bank-macros'))/statistics.mean(work(r) for r in group('bank'))
    return_search = {m: statistics.mean(sum(w['search_work'] for w in r['windows'] if w['phase']=='return') for r in group(m)) for m in ['single','bank','bank-macros']}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    labels = ['Single','Replay','Bank','Bank + templates']
    axes[0].bar(labels, [mean(m,'search_work') for m in ['single','replay','bank','bank-macros']], color=['#7b8a99','#6787b5','#33a397','#9556bd'])
    axes[0].set_ylabel('Mean search evaluations / 48 windows'); axes[0].tick_params(axis='x',labelrotation=15)
    axes[0].set_title('Routing and reusable repairs')
    for method in ['random','error','active']:
        axes[1].plot([4,8,16,32], [100*mean(method,'accuracy',cap) for cap in [4,8,16,32]], marker='o', label=method)
    axes[1].set_xlabel('Query cap'); axes[1].set_ylabel('Held-out transition accuracy (%)'); axes[1].set_ylim(0,105);axes[1].legend();axes[1].set_title('Interaction efficiency, not compute efficiency')
    for ax in axes: ax.grid(axis='y',alpha=.2)
    figure = root/'guides/publication/reports/repair-graph/examples/continual-inquiry-v3.png';fig.savefig(figure,dpi=160);plt.close(fig)
    examples=root/'experiments/alternative-architecture/examples'
    (examples/'pilot-v3.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
    html=(root/'experiments/alternative-architecture/ui/report.html').read_text().replace('__REPORT_JSON__',json.dumps(data).replace('<',r'\u003c'))
    (examples/'pilot-v3.html').write_text(html);(run/'index.html').write_text(html)
    text=f'''# Repair Graph v3: memory, reusable repairs and active inquiry

Measured 11 October 2026. The remaining bounded prototype gates now have native
implementations, controls, persisted state, tests and a recorded viewer. This
completes this experimental implementation; it does not demonstrate open-ended
self-improvement, a frontier model, or invention of arbitrary mathematics.

## What changed

Previously, the learner kept one current graph and one explicitly retrieved
archive. Now it retains up to 16 learned graphs and selects among them from
observations without a task identifier. Two accepted graphs with a shared shape
can produce a reusable parameterized template. A separate controlled inquiry
experiment chooses which source and action to observe using model disagreement.
All learning and query decisions run in the same C++17 executable, using the
existing graph semantics, fitting, simulator and checkpoint helpers. Python
only renders this document and plot.

## Main outcomes

- Both bank methods retained 16 distinct graphs and recovered earlier behavior
  without new search on the return phase. A single graph also reached 100% after
  re-learning, but spent {return_search['single']:,.0f} search evaluations on return
  versus {return_search['bank']:,.0f} for the bank. Retention saves relearning here;
  it does not improve the already-perfect post-window single-model accuracy.
- Learned templates reduced mean search work by **{100*search_saving:.2f}%** versus
  the same bank without templates. Including the declared routing, retention,
  matching and extraction work proxies, the reduction is **{100*total_saving:.2f}%**.
  Template extraction/support learning is charged; no pretrained library is given.
- At eight queries, active inquiry reached **{100*mean('active','accuracy',8):.2f}%**
  mean held-out accuracy, versus **{100*mean('random','accuracy',8):.2f}%** for uniform
  random inquiry and **{100*mean('error','accuracy',8):.2f}%** for raw error seeking.
  Active inquiry used about **{mean('active','evaluations',8)/mean('random','evaluations',8):.1f} times**
  as many hypothesis evaluations as random inquiry. It saves interactions at a
  substantial compute cost; this is not a general efficiency or speed victory.

![Measured search cost and query-limited prediction accuracy.](examples/continual-inquiry-v3.png)

## Continual experiment: all controls

Three seeds, four methods, 48 observation windows each: **12 streams / 576 windows**.
Each window supplies 64 new training and 32 validation examples; 128 separate
held-out examples are used only for reporting. Eight +/-1 tasks are followed by
eight +/-2 tasks and a return to the first eight, with two windows per segment.
Task order is shuffled per seed. All methods see identical new evidence.

| Method | After-window accuracy | Before-window accuracy | Return accuracy | Search work | All declared work |
| --- | --- | --- | --- | --- | --- |
'''
    for m in ['single','replay','bank','bank-macros']:
        text+=f"| {m} | {100*mean(m,'accuracy'):.2f}% | {100*mean(m,'pre_accuracy'):.2f}% | {100*mean(m,'return_accuracy'):.2f}% | {mean(m,'search_work'):,.0f} | {statistics.mean(work(r) for r in group(m)):,.0f} |\n"
    text+='''
The bank without templates costs more total declared work than the single graph
over the complete sequence, because routing and witness checks have overhead.
The template-equipped bank is cheaper on this workload; none of these operation
sums is a substitute for a matched wall-clock benchmark.

The observation window is fixed and observable, but no change/segment signal,
family name or gain is passed to the learner. Recognition happens after the
window's outcomes are available. Before-window errors remain; the bank is not
an oracle capable of identifying an unobservable change before any evidence.
The last active graph alone still forgets other contexts. Simultaneous retained
knowledge is in the graph bank and its evidence-based routing, not in one graph.

Single-graph replay mixes 32 new and 32 old training examples and 16 new and
16 old validation examples from a 256-example FIFO. Conflicting unlabelled
worlds can require different answers to the same input; mixing them damages
this deterministic single predictor. This result does not disprove replay in
architectures with suitable context representations.

| Seed | Bank return accuracy | Template return accuracy | Bank search work | Template search work | Accepted template uses |
| --- | --- | --- | --- | --- | --- |
'''
    for seed in [201,202,203]:
        text+=f"| {seed} | {100*mean('bank','return_accuracy',seed=seed):.2f}% | {100*mean('bank-macros','return_accuracy',seed=seed):.2f}% | {mean('bank','search_work',seed=seed):,.0f} | {mean('bank-macros','search_work',seed=seed):,.0f} | {mean('bank-macros','macro_uses',seed=seed):.0f} |\n"
    text+='''
## What the learned abstraction contains

A library entry records an actual shared graph topology, its supporting graph
IDs, common coefficient values, and holes at coefficients that differed between
the supports. This is anti-unification of two or more different learned programs.
A candidate repair fills the holes from new training evidence with the existing
fitter; current validation evidence decides acceptance. It must predict new
coefficient values, rather than merely return a cached old answer. The nearest
observed residual descriptor ranks templates. A rejected template spends budget
and falls back to ordinary bounded search with only the remaining budget.

Templates use the existing arithmetic, gate and delayed-action primitives.
Learning a reusable combination is implemented; inventing unrestricted new
primitives is not claimed. Transfer is within the declared grammar and known
task families. The +/-2 instances are fresh, but the families are known research
fixtures from earlier pilots. Common constants may restrict transfer to other
kinds of changes; failed proposals remain visible.

A pre-measurement native fixture exposed a composition barrier: a small gated
and delayed effect can require two changes before either is individually useful.
The common v3 fallback can test their composed topology using the existing
proposer and remaining cap. This is a designed lookahead available to every
control. It is separate from learned-template benefits and does not alter v2.

## Active inquiry: every method and query cap

The learner starts with {count:,} graph hypotheses for each of two anonymous
sources. One source is the world transition, the other independent random
output. Both sources initially have the same model class; their identities are
not supplied to the learner. There are 36 possible probes of action, previous
action and binary sensors. The active rule selects maximum Gini disagreement
among predicted outputs and removes hypotheses contradicted by observations.
A source with an empty version space is unsupported by the current grammar.

| Method | Query cap | Accuracy | Used queries | Queries to unpredictable output | Hypothesis evaluations |
| --- | --- | --- | --- | --- | --- |
'''.replace('{count:,}',f"{data['hypothesis_count']:,}")
    for cap in [4,8,16,32]:
        for m in ['random','error','active']:
            text+=f"| {m} | {cap} | {100*mean(m,'accuracy',cap):.2f}% | {mean(m,'queries',cap):.2f} | {mean(m,'noisy_queries',cap):.2f} | {mean(m,'evaluations',cap):,.0f} |\n"
    text+='''
These are **288 conditions**: three seeds, four families, two gains, four caps
and three query rules. At the same interaction cap, query-selection computation
is not matched. Raw error seeking tends to revisit the unpredictable source;
active inquiry stops when no supported discriminating probe remains. Random
inquiry also reaches 100% at the largest budget. An empty version space can
mean missing model primitives or measurement corruption, not just randomness.

This prototype implements uncertainty-driven experiment selection and a designed
contradiction guard. It does not learn its own curiosity objective, model
stochastic transitions correctly, provide calibrated probabilities, or prove
resistance to every noisy environment. Source IDs are permuted in tests; the
noise branch is distinguished by evidence, not a hard-coded source name.

| Seed | Random accuracy at 8 | Error seeking at 8 | Active accuracy at 8 | Active queries at 32 |
| --- | --- | --- | --- | --- |
'''
    for seed in [201,202,203]:
        text+=f"| {seed} | {100*mean('random','accuracy',8,seed):.2f}% | {100*mean('error','accuracy',8,seed):.2f}% | {100*mean('active','accuracy',8,seed):.2f}% | {mean('active','queries',32,seed):.2f} |\n"
    sizes=[r['checkpoint_bytes'] for r in group('bank-macros')]
    inquiry_sizes=[r['checkpoint_bytes'] for r in inquiries]
    text+=f'''
## Persistence, work and resources

The bank/template checkpoint is **{min(sizes):,}–{max(sizes):,} bytes**. It contains
the active learner, up to 16 graphs, witnesses, descriptors and learned template
definitions. Query checkpoints are **{min(inquiry_sizes):,}–{max(inquiry_sizes):,}
bytes**, including remaining hypothesis IDs, last predictions, error windows,
query counter, compute counter and random state. The grammar itself is supplied
by the C++ runtime. These files are data, not source code or GGUF; stock llama.cpp
cannot execute them. The restored state reproduces the next decision exactly.

The complete native campaign took **{data['wall_seconds']:.4f} s** wall and
**{data['cpu_seconds']:.4f} s** CPU, with **{data['peak_rss_kib']/1024:.2f} MiB** peak
RSS. This is a short shared-host observation, not a latency benchmark. One CPU
thread, 512 MiB address-space limit and 15-minute timeout apply. No model or
framework download was needed. Learning is manual, with no startup daemon.

Search work counts candidate/example evaluations, using the existing proxy.
Routing/retention count graph/example evaluations, matching counts descriptor
components, and extraction counts topology comparisons. Their sum is a declared
operation proxy, not FLOPs or interchangeable wall-clock costs. The cap applies
to search, not these other operations. Inquiry work counts hypothesis prediction
evaluations, including selecting the next query and a final stop decision.
Probability-count bookkeeping, sorting, allocation and report-only scoring are
included in CPU time but not these proxies. State is bounded in memory; writes
occur once per completed stream/condition, not inside candidate scoring loops.

## Reproduce and inspect

```bash
./experiments/alternative-architecture/tools/build.sh
./experiments/alternative-architecture/tools/run.sh test-v3
./experiments/alternative-architecture/tools/run.sh campaign-v3 NEW_DIRECTORY
.venv/bin/python experiments/alternative-architecture/tools/report.py NEW_DIRECTORY
./experiments/alternative-architecture/tools/start-viewer.sh
```

The configured recorded run is artifacts/alternative-architecture/pilot-03.
Open http://127.0.0.1:8091/ and choose memory/templates or active inquiry. The
slider and play button show real recorded decisions and their graphs. The input
form evaluates the displayed graph locally; it does not learn in the background.
The offline copy is experiments/alternative-architecture/examples/pilot-v3.html.
The experiment README documents exact selections and native prediction calls.

## Verification and scope closure

Native tests cover immutable retained evidence, automatic context return,
finite-memory rejection, unseen-window selection without task IDs, accepted
parameter-hole transfer, budget rollback, held-out-label isolation, both new
checkpoint formats and the same next decision after restoration. Inquiry tests
cover contradictory evidence, source permutation, RNG continuation and an
independently seeded composed-world fixture. Existing v1/v2 checks remain.

A separate repeat verifies the scientific rows, event histories and checkpoint
bytes. Viewer checks cover actual recorded transitions, library support,
query histories, playback, native/browser prediction agreement, offline use,
mobile layout and read-only route restrictions. Exact checks and hashes are
recorded in the v3 verification manifest. The four gates now have bounded
implementations and evidence: retention, context routing, active inquiry and
reusable graph abstractions. Broader intelligence claims remain unproven; they
are not hidden unfinished implementation items in this completed pilot.
'''
    source=root/'guides/publication/reports/repair-graph/CONTINUAL_INQUIRY_RESULTS.md';source.write_text(text)
    subprocess.run([sys.executable,str(root/'tools/build-guide.py'),str(source),'--cover',str(root/'guides/publication/reports/repair-graph/continual-inquiry-cover.json'),'--title','Repair Graph v3 — Continual Memory and Active Inquiry','--footer','AI Pilot | Repair Graph v3 | 11 October 2026'],check=True)
    print('PASS: 576 continual windows / 288 inquiries, work accounting, acceptance, library support and checkpoint sizes; report generated.')
