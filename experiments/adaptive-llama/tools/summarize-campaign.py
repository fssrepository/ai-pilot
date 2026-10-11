"""Summarize native measurements and render the English report. No model training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=Path)
args = parser.parse_args()
run = args.run.resolve()
data = json.loads((run / 'status.json').read_text())
assert data['status'] == 'complete' and (run / 'COMPLETE').exists()
rows = data['results']
conditions = ['fixed4-no-replay', 'fixed4-replay', 'fixed5-replay', 'adaptive-replay']
labels = ['Fixed 4, no replay', 'Fixed 4, replay', 'Fixed 5, replay', 'Adaptive, replay']
assert len(rows) == 12
assert {(r['seed'], r['condition']) for r in rows} == {(s, c) for s in (42, 43, 44) for c in conditions}
for r in rows:
    assert 0 < r['adaptation_proxy'] <= r['adaptation_proxy_budget']
    assert r['pretrain_proxy'] <= r['pretrain_proxy_budget']
    for metric in ('old', 'new'):
        assert abs(r[f'gguf_{metric}_accuracy'] - r[f'{metric}_test']['exact_accuracy']) < 1e-8
    assert abs(r['gguf_changed_accuracy'] - r['changed_new_test']['exact_accuracy']) < 1e-8
    assert abs(r['forgetting_pp'] - 100*(r['initial_old_test']['exact_accuracy']-r['old_test']['exact_accuracy'])) < 1e-6
    assert r['export']['f32_max_logit_error'] < .0003 and r['export']['f16_max_logit_error'] < .02
records = [json.loads(line) for line in (run/'episodes.jsonl').read_text().splitlines()]
assert len(records) == len({r['prompt'] for r in records}) == 320
split_by_combination = {}
for r in records:
    key = r['prompt'][4:]
    assert split_by_combination.setdefault(key, r['split']) == r['split']
assert [sum(r['split']==s for r in records) for s in (0,1,2)] == [224,48,48]

def group(c): return [r for r in rows if r['condition']==c]
def mean(c, metric): return statistics.mean(metric(r) for r in group(c))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
old = lambda r: 100*r['old_test']['exact_accuracy']
new = lambda r: 100*r['new_test']['exact_accuracy']
changed = lambda r: 100*r['changed_new_test']['exact_accuracy']
forget = lambda r: r['forgetting_pp']

assets = ROOT/'docs/examples'; assets.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
x = list(range(4))
for off, metric, color, name in [(-.18, old, '#1685ad', 'Old world A'),(.18,new,'#d59418','New world B')]:
    averages = [mean(c,metric) for c in conditions]
    axes[0].bar([i+off for i in x],averages,.34,color=color,label=name)
    for i,c in enumerate(conditions):
        axes[0].scatter([i+off]*3,[metric(r) for r in group(c)],s=18,color='#202c36',zorder=3)
axes[0].set(ylabel='Exact held-out accuracy (%)',ylim=(0,105),title='Task acquisition and retention')
axes[0].legend(loc='lower right',fontsize=8)
axes[1].bar(x,[mean(c,forget) for c in conditions],color=['#a84d47','#228f86','#228f86','#8364ad'])
for i,c in enumerate(conditions): axes[1].scatter([i]*3,[forget(r) for r in group(c)],s=22,color='#202c36',zorder=3)
axes[1].axhline(0,color='#495869',linewidth=.8)
axes[1].set(ylabel='Old-task forgetting (percentage points)',title='Lower is better; dots show all seeds')
for ax in axes:
    ax.set_xticks(x,['4 / no replay','4 / replay','5 / replay','adaptive']);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.savefig(assets/'continual-comparison.png',dpi=180);plt.close(fig)

examples = ROOT/'experiments/adaptive-llama/examples'
template=(ROOT/'experiments/adaptive-llama/ui/campaign.html').read_text()
(examples/'continual.html').write_text(template.replace('__REPORT_JSON__',json.dumps(data,ensure_ascii=True).replace('<',r'\u003c')))
shutil.copyfile(run/'status.json',examples/'continual.json')
resources = json.loads((run/'resources.json').read_text())
accepted = sum(r['growth_accepted'] for r in group('adaptive-replay'))
proposed = sum(r['growth_proposed'] for r in group('adaptive-replay'))
replay_gain=mean('fixed4-no-replay',forget)-mean('fixed4-replay',forget)
adaptive_gap=mean('adaptive-replay',new)-mean('fixed4-replay',new)
text = f'''# Adaptive Llama: changing rules, memory and structural growth

Measured 11 October 2026. Native C++ / LibTorch CPU experiment, with independent
llama.cpp F16 deployment checks. This report completes the requested two-world
experiment, forgetting measurement, fixed/adaptive comparison and live viewer.
The model predicts coordinates; it is not a conversational assistant.

Development and verification of this bounded demonstrator are complete and the
experiment is closed. Native resume/export and browser checks passed. The
documented model errors and lack of adaptive advantage are measured outcomes;
they were not removed by post-hoc retuning. Alternative-architecture development
is a separate next task.

## Result

Under this declared toy-task protocol, adding 25% old-world replay to the fixed
four-layer learner reduced mean forgetting by **{replay_gain:.2f} percentage points**.
The adaptive method proposed growth in {proposed}/3 seeds and accepted it in
{accepted}/3. Its mean new-world test accuracy differed from the fixed four-layer
replay control by **{adaptive_gap:+.2f} percentage points**. These results do not
establish a useful accuracy advantage from structural growth.

![Mean held-out accuracy and forgetting; dots show the three individual seeds.](examples/continual-comparison.png)

## Mean results across the three seeds

| Condition | Old A accuracy | New B accuracy | Changed B only | Forgetting |
| --- | --- | --- | --- | --- |
'''
for c,label in zip(conditions,labels):
    text += f'| {label} | {mean(c,old):.2f}% | {mean(c,new):.2f}% | {mean(c,changed):.2f}% | {mean(c,forget):.2f} pp |\n'
text+='''
Each task has 24 held-out test combinations. The changed-answer subset contains
'''+str(rows[0]['changed_test_count'])+''' combinations per world. Old accuracy is after adaptation;
forgetting is old accuracy before minus after the switch. Negative forgetting
means improvement. The dots and per-seed table expose variation; three seeds
are not enough for a general superiority claim.

## All measured runs

| Seed / condition | A before | A after | B before | B after | Forgetting |
| --- | --- | --- | --- | --- | --- |
'''
for seed in (42,43,44):
    for c,label in zip(conditions,labels):
        r=next(r for r in rows if r['seed']==seed and r['condition']==c)
        text+=f"| {seed} / {label} | {100*r['initial_old_test']['exact_accuracy']:.1f}% | {old(r):.1f}% | {100*r['initial_new_test']['exact_accuracy']:.1f}% | {new(r):.1f}% | {forget(r):.1f} pp |\n"
text+='''
## What the model was asked to learn

In world A, R/L/U/D move right/left/up/down on a 4x4 grid. World B reverses those
four directions. Wait, obstacles and boundary clamping remain unchanged. For
example, `w=0,x=1,y=2,a=R,b=0>` maps to `2,2;`, while the same input under `w=1`
maps to `0,2;`. The world ID is supplied explicitly. This tests retaining and
selecting contextual rules; it does not test discovering an unannounced change.

The original 160 combinations keep their fixed 112/24/24 train/validation/test
partition in both worlds. No held-out combination is trained under the other
world ID. Blocked and wait cases often have unchanged answers, so changed-answer
accuracy is reported separately. All output generation is unconstrained; wrong
or malformed coordinates count as errors.

The model uses 128 hidden dimensions, four attention heads, a fixed byte
vocabulary, RMSNorm, RoPE and SwiGLU. Four layers contain 787,584 parameters;
five contain 951,680. Training is in C++ using the existing LibTorch installation.
The replay buffer contains only old training examples, never old test examples.

## Comparison and compute budget

Four conditions use seeds 42, 43 and 44. Four-layer variants share exactly the
same pretrained checkpoint per seed. Fixed five layers are trained from the
start under the same matrix-work budget. Replay batches contain exactly 12 new
and four old examples; the no-replay control uses 16 new examples.

Pretraining has a 600-four-layer-update budget; adaptation has 400. The declared
compute proxy is three times the forward dense-matrix multiply-add count:

`batch * sequence * (width*vocabulary + layers*(4*width^2 + 3*width*FF + 2*sequence*width))`.

The factor of three approximates forward plus backward computation. It excludes
nonlinearities, normalization, Adam, validation, checkpointing and export. Thus
these are approximately matched matrix-work budgets, not exact FLOPs or equal
wall-clock budgets. Integer updates can leave a small unused remainder.

| Condition | Pretrain updates | Adaptation updates, all branches | Proxy used | Training CPU time |
| --- | --- | --- | --- | --- |
'''
for c,label in zip(conditions,labels):
    rs=group(c)
    counts='/'.join(str(v) for v in sorted({r['adaptation_updates_total'] for r in rs}))
    text+=f"| {label} | {rs[0]['pretrain_updates']} | {counts} | {mean(c,lambda r:100*r['adaptation_proxy']/r['adaptation_proxy_budget']):.2f}% | {mean(c,lambda r:r['adaptation_cpu_seconds']):.2f} s |\n"
text+='''
CPU times above include all adaptation training branches, including rejected
candidates; they exclude validation/export and pretraining. Measurements ran
serially with two CPU threads on a shared Intel i5-1335U laptop. Other workloads
remained active. Wall time and CPU time differ; no isolated speed advantage is
claimed. Full per-run timing and matrix-work values are in the JSON evidence.
'''
text+=f"\nThe final campaign invocation took {resources['invocation_wall_seconds']:.2f} s wall time and {resources['invocation_cpu_seconds']:.2f} s process CPU time, with {resources['peak_rss_mib']:.2f} MiB peak RSS. It resumed after a deliberate pause at initial update 100; these invocation totals exclude that earlier process. The segment counters include its training. Shared pretraining is performed once for the three four-layer conditions, so summing attributed per-condition pretraining times overcounts actual campaign work.\n"
text+='''
## Structural decision and its cost

The adaptive learner first receives 100 new-world updates with replay. If new
validation accuracy is below 98%, loss above 0.02, or old validation accuracy has
dropped more than two points, it proposes a fifth identity-initialized block.
Otherwise it spends the remaining budget continuing four layers.

When proposed, the remaining matrix-work budget is split between the continued
four-layer control and the five-layer candidate. Both start from the same warm
checkpoint and sampling state. The larger candidate gets fewer updates; both
branches are charged to the adaptive method. As a result, the selected branch
receives fewer updates than a fixed model that spends its entire budget on one
trajectory. This is an explicit search cost, not hidden extra compute.

Acceptance requires at least 1% lower new-world validation loss, non-regressing
new accuracy, and old validation accuracy no worse than the control and within
two percentage points of the pre-switch reference. Final-test metrics never
enter this decision. The rule is written by us, not learned by the model.

| Seed | Proposed / accepted | Control B loss | Candidate B loss | Control / candidate A accuracy |
| --- | --- | --- | --- | --- |
'''
for r in group('adaptive-replay'):
    if r['growth_proposed']:
        text+=f"| {r['seed']} | yes / {'yes' if r['growth_accepted'] else 'no'} | {r['control_new_validation']['loss']:.5f} | {r['candidate_new_validation']['loss']:.5f} | {100*r['control_old_validation']['exact_accuracy']:.1f}% / {100*r['candidate_old_validation']['exact_accuracy']:.1f}% |\n"
    else:text+=f"| {r['seed']} | no / no | Not compared | No candidate | Not compared |\n"
text+='''
## What can be concluded

Replay and structural growth are separate mechanisms. In this setup, retaining
old examples has a measurable benefit, but replay does not guarantee zero
forgetting in every seed. The fixed-five-layer control helps separate extra
capacity from an adaptive search procedure. Extra capacity and a successful
layer insertion do not by themselves establish a useful adaptation policy.

A concrete limitation appeared in seed 42. The five-layer candidate improved
new-world validation loss from 0.27067 to 0.10128 and old-world validation
accuracy from 58.33% to 87.50% relative to its continued control. However, the
pre-switch old-world validation reference was 91.67%; a 4.17-point loss exceeds
the predeclared two-point guard. The rule therefore rejected this candidate
and chose a control with worse retention. **The fallback is not a retention
guarantee.** With only 24 validation examples, even one error costs 4.17 points,
so the two-point threshold effectively allows no lost correct example. This
shows a limitation of the selection rule, not proof that every growth candidate
is worse. We did not relax the rule after seeing the results or evaluate
rejected candidates on the test set.

This experiment is small, uses a supplied task ID, and has only six changed-answer
test combinations and three random seeds. Results are sensitive to initialization,
training budget, replay ratio and the decision rule. The study was not tuned
against the final test outcomes. It does not establish a new learning paradigm,
open-ended self-improvement, natural-language ability or frontier intelligence.

## Live and offline inspection

The live viewer is at http://127.0.0.1:8090/ while its Node server runs. It reads
atomic status snapshots from the C++ trainer, checks once per second and stops
polling when the campaign is complete. The trainer measures after each 50-update
chunk. A short highlight means a new measurement arrived; it is not an animation
of individual neuron activation. The history slider can inspect a real candidate
with five layers and compare its old/new validation curves.

The versioned offline report is `experiments/adaptive-llama/examples/continual.html`.
Open it directly in a browser; no login, model or server is required. Final test
comparisons are separated from the validation timeline. The machine-readable
snapshot is `continual.json` alongside it.

From the repository root, start a new run manually:

```bash
./experiments/adaptive-llama/tools/run.sh campaign \\
  artifacts/adaptive-llama/my-campaign
```

In a second terminal, serve its live measurements:

```bash
node experiments/adaptive-llama/tools/live.cjs \\
  artifacts/adaptive-llama/my-campaign
```

Resume an interrupted campaign with `run.sh campaign-resume DIRECTORY`. Keep all
run files together. The native learner, Adam, random state and segment timing
are checkpointed atomically. Completed conditions and baseline checkpoints are
reused. Stop the viewer separately with Ctrl+C. No boot/login service is created.

## Try the model in the official llama.cpp chat

Start the existing llama.cpp built-in web UI with the trained snapshot:

```bash
./experiments/adaptive-llama/tools/start-chat.sh
```

Open http://127.0.0.1:8088/ and start a new conversation. The model name is
`Adaptive-Llama-Toy`; the default snapshot is seed 42, adaptive with replay.
This is the same llama.cpp interface used for Qwen, now serving the tiny model.
The separate measurement viewer stays at port 8090.

Send the following strings exactly, without Markdown backticks:

| Message | Expected | Observed model answer |
| --- | --- | --- |
| `w=0,x=1,y=2,a=R,b=0>` | `2,2;` | `0,2;` (wrong) |
| `w=1,x=1,y=2,a=R,b=0>` | `0,2;` | `0,2;` (correct) |
| `w=1,x=1,y=2,a=R,b=1>` | `1,2;` | `1,2;` (correct) |

The first error illustrates old-world forgetting. The expected answers come
from the world rules; the chat displays only the model's generated answer,
without a simulator replacing it. Checks use greedy generation and four tokens.

The custom chat template sends only the latest user message, stripped of outer
whitespace. System messages and previous turns are absent from the predictor's
input. This provides no conversational memory, chat learning or natural-language
competence. Use a text-only coordinate prompt. The visible chat history is a UI
record. Do not attach files.

Defaults are temperature 0, top-k 1, repeat penalty 1 and max tokens 4. Previous
browser preferences may override UI defaults: reset those fields if necessary.
The model is held in RAM until Ctrl+C stops the server. This manual server holds
the adaptive experiment lock; stop it before training. Qwen uses the same port:
stop the existing server before switching. An optional first launcher argument
selects another generated GGUF snapshot. No boot/login service is installed.

Configuration authority: the experiment README records the exact selected GGUF,
its SHA-256, model alias, local addresses, runtime bounds and observed responses.
The versioned launcher is `experiments/adaptive-llama/tools/start-chat.sh`; the
chat template and UI defaults are `ui/last-user.jinja` and `ui/llama-settings.json`
in the same experiment directory. Changes to these settings must update that
record. The source files live in Git; generated models remain in ignored artifacts.

## Deployment and verification

All 12 selected models were exported in F32 and F16 GGUF. Native/llama.cpp logits
passed the existing 0.0003 F32 and 0.02 F16 maximum-error thresholds. F16 llama.cpp
exact accuracy matched native accuracy on both test worlds and the changed-answer
subset. A four-layer F16 file is 1,592,576 bytes; five layers are 1,921,824 bytes.

The GGUF is a frozen predictor. The adaptation rule, replay and optimizer are
in the C++ learner/native checkpoints, not executable logic inside GGUF.
The complete campaign retains approximately 408 MiB of local checkpoints and
exports, despite the small size of each inference model. These are generated
artifacts, not extra model downloads, and are excluded from Git.

The separate verification manifest records source/artifact hashes and native,
resume, UI and document checks. A live test observed an actual 150-to-200 update
transition, and the browser supports recorded four/five-layer history inspection.

Efficiency review: task examples remain in memory; metrics are measured only at
bounded checkpoints; status JSON is written once per observation. The browser
uses one read-only status request per second and stops after completion. The
viewer serves only its page and status, on loopback. Export checks recompute
short prefixes for correctness; they are not an optimized serving benchmark.

## Next research boundary

The separate alternative-architecture project has not been implemented by this
campaign. The user supplied TypeSafe AI's Jev/System One Models as a reference
for that next discussion. Its architecture, access and claims need separate
assessment; Jev is not the JEPA research line. No Jev API or model was used here.
'''
# Keep changed-subset size truthful if the declared split is updated in a future protocol.
text=text.replace('only six changed-answer',f"only {rows[0]['changed_test_count']} changed-answer")
source=ROOT/'docs/ADAPTIVE_LLAMA_RESULTS.md';source.write_text(text)
subprocess.run([sys.executable,str(ROOT/'tools/build-guide.py'),str(source),'--cover',str(ROOT/'docs/brand/adaptive-cover.json'),'--title','Adaptive Llama — Continual Learning Results','--footer','AI Pilot | Adaptive Llama results | 11 October 2026'],check=True)
print('PASS: 12 unique conditions, split integrity, work budgets, forgetting arithmetic and GGUF parity.')
print(source)
