"""Focused end-to-end check using real, local Qwen weights (no network)."""
import json
import os
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ['MPLBACKEND'] = 'Agg'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
import matplotlib.pyplot as plt
from playground import Playground, bertviz_html, show_activations, show_attention

destination = ROOT / 'artifacts'
destination.mkdir(exist_ok=True)
lab = Playground()
try:
    lab.encode('hello ' * 200)
except ValueError:
    pass
else:
    raise AssertionError('Input limit not enforced')
assert lab.model is None, 'Rejected prompt must not load the model'

result = lab.analyze('The capital of Hungary is')
assert list(result['attention']) == [3, 7, 11, 15, 19, 23]
assert len(result['activation']) == 24
assert all(torch.isfinite(value).all() for value in result['activation'].values())
for values in result['attention'].values():
    assert torch.isfinite(values).all()
    assert values.shape[1] == 8
    assert torch.allclose(values.sum(-1), torch.ones_like(values.sum(-1)), atol=0.02)
    assert float(values.triu(diagonal=1).abs().max()) < 1e-6, 'Causal mask must hide future tokens'
assert torch.isfinite(result['logits']).all()
assert len(result['top_tokens']) == 10
show_activations(result)
plt.savefig(destination / 'activations.png', dpi=140)
plt.close('all')
show_attention(result)
plt.savefig(destination / 'attention.png', dpi=140)
plt.close('all')
(destination / 'bertviz.html').write_text(bertviz_html(result))
result['tokens'].to_csv(destination / 'tokens.csv', index=False)
result['top_tokens'].to_csv(destination / 'next-token.csv', index=False)

ablation = lab.ablate_mlp('The capital of Hungary is', 0)
assert 0 <= ablation['total_variation'] <= 1
assert all(not layer.mlp._forward_hooks for layer in lab.layers), 'Intervention hook leaked'
generated = lab.generate('What is the capital of Hungary? Answer briefly.', 16)
assert generated['tokens'] > 0 and generated['text'].strip()
evidence = {
    'model': 'Qwen/Qwen3.5-0.8B', 'revision': (ROOT / 'models/Qwen3.5-0.8B/REVISION').read_text().strip(),
    'input_tokens': len(result['tokens']), 'attention_layers': list(result['attention']),
    'decoder_layers': len(result['activation']), 'threads': torch.get_num_threads(),
    'dtype': str(lab.model.dtype), 'analysis_seconds': result['seconds'],
    'ablation_total_variation': ablation['total_variation'], 'generation': generated,
    'peak_process_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    'checks': ['input limit before model load', '24 decoder outputs', 'six attention layers / eight heads',
               'finite activations/logits', 'attention normalization', 'causal mask', 'top-token probabilities',
               'MLP ablation / hook cleanup', 'chat generation', 'BertViz HTML export'],
}
(destination / 'analysis-check.json').write_text(json.dumps(evidence, indent=2) + '\n')
print(json.dumps(evidence, indent=2))
lab.unload()
