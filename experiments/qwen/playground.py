"""Small, bounded analysis helpers for the pinned Qwen3.5-0.8B checkpoint.

JupyterLab, ipywidgets and BertViz own the UI. Transformers owns inference.
These helpers only collect measurements and connect those existing tools.
"""
import gc
import html
import json
import os
import time
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT.parents[1]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))

import pandas as pd
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

MODEL_PATH = ROOT / 'models/Qwen3.5-0.8B'
MAX_INPUT_TOKENS = 96
torch.set_num_threads(2)


class Playground:
    def __init__(self):
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    def load(self):
        if self.model is not None:
            return
        available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        if available < 3 * 1024 * 1024:
            raise RuntimeError('Less than 3 GiB RAM available. Close another application before loading Qwen.')
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_PATH, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='eager', device_map='cpu',
        ).eval()

    def unload(self):
        self.model = None
        gc.collect()

    @property
    def layers(self):
        return self.model.model.language_model.layers

    def encode(self, prompt, chat=False):
        if not prompt.strip():
            raise ValueError('Enter a non-empty prompt.')
        if chat:
            prompt = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
        inputs = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        length = inputs['input_ids'].shape[1]
        if length > MAX_INPUT_TOKENS:
            raise ValueError(f'{length} tokens; this CPU playground allows at most {MAX_INPUT_TOKENS}. Shorten the prompt.')
        return inputs

    def token_table(self, inputs):
        ids = inputs['input_ids'][0].tolist()
        return pd.DataFrame({
            'position': range(len(ids)), 'token_id': ids,
            'token': self.tokenizer.convert_ids_to_tokens(ids),
            'decoded_piece': [repr(self.tokenizer.decode([i])) for i in ids],
        })

    def top_tokens(self, logits, k=10):
        probabilities = logits.float().softmax(-1)
        values, ids = probabilities.topk(k)
        return pd.DataFrame({
            'token_id': ids.tolist(),
            'token': [repr(self.tokenizer.decode([i])) for i in ids.tolist()],
            'probability': values.tolist(),
        })

    @torch.inference_mode()
    def analyze(self, prompt, chat=False):
        inputs = self.encode(prompt, chat)
        self.load()
        attention = {}
        activation = {}
        representations = {}
        handles = []

        def snapshot(hidden):
            values = hidden[0].detach().float().cpu()
            return {'width': values.shape[-1], 'first16': values[:, :16].tolist(),
                    'rms': values.square().mean(-1).sqrt().tolist()}

        def representation_hook(name):
            def capture(module, args, output):
                representations[name] = snapshot(output)
            return capture

        def decoder_hook(index):
            def capture(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                activation[index] = hidden[0].float().pow(2).mean(-1).sqrt().cpu()
                representations[f'layer_{index}'] = snapshot(hidden)
            return capture

        def attention_hook(index):
            def capture(module, args, output):
                if isinstance(output, tuple) and output[1] is not None:
                    attention[index] = output[1].detach().float().cpu()
            return capture

        handles.append(self.model.get_input_embeddings().register_forward_hook(representation_hook('embedding')))
        handles.append(self.model.model.language_model.norm.register_forward_hook(representation_hook('final_norm')))
        for index, layer in enumerate(self.layers):
            handles.append(layer.register_forward_hook(decoder_hook(index)))
            if layer.block_type == 'full_attention':
                handles.append(layer.self_attn.register_forward_hook(attention_hook(index)))
        started = time.perf_counter()
        try:
            output = self.model(**inputs, use_cache=False, output_attentions=True, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        logits = output.logits[0, -1].float().cpu()
        if len(attention) != 6 or len(activation) != 24:
            raise RuntimeError(f'Unexpected capture: {len(attention)} attention and {len(activation)} decoder layers.')
        return {
            'inputs': inputs, 'tokens': self.token_table(inputs),
            'attention': attention, 'activation': activation, 'logits': logits,
            'top_tokens': self.top_tokens(logits),
            'seconds': time.perf_counter() - started,
            'layer_types': [layer.block_type for layer in self.layers],
            'representations': representations, 'prompt': prompt, 'chat': chat,
        }

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=32):
        if not 1 <= max_new_tokens <= 64:
            raise ValueError('Choose between 1 and 64 output tokens.')
        inputs = self.encode(prompt, chat=True)
        self.load()
        started = time.perf_counter()
        output = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            use_cache=True, pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = output[0, inputs['input_ids'].shape[1]:]
        return {'text': self.tokenizer.decode(generated, skip_special_tokens=True),
                'tokens': len(generated), 'seconds': time.perf_counter() - started}

    @torch.inference_mode()
    def ablate_mlp(self, prompt, layer_index=0, chat=False):
        """Zero one MLP's output for this forward pass; never modify weights."""
        if not 0 <= layer_index < 24:
            raise ValueError('Layer must be 0..23.')
        inputs = self.encode(prompt, chat)
        self.load()
        baseline = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        handle = self.layers[layer_index].mlp.register_forward_hook(
            lambda module, args, output: torch.zeros_like(output))
        try:
            changed = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        finally:
            handle.remove()
        p, q = baseline.softmax(-1), changed.softmax(-1)
        return {'baseline': self.top_tokens(baseline), 'ablated': self.top_tokens(changed),
                'total_variation': float((p - q).abs().sum() / 2), 'layer': layer_index}


def show_attention(result, layer=3, head=0):
    import matplotlib.pyplot as plt
    weights = result['attention'][layer][0, head].numpy()
    labels = result['tokens']['token'].tolist()
    fig, ax = plt.subplots(figsize=(9, 7))
    chart = ax.imshow(weights, vmin=0, vmax=1, cmap='viridis', aspect='auto')
    if len(labels) <= 40:
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set(xlabel='Attended-to token (key)', ylabel='Query token', title=f'Full-attention layer {layer}, head {head}')
    fig.colorbar(chart, ax=ax, label='Attention weight')
    fig.tight_layout()
    plt.show()


def show_activations(result):
    import matplotlib.pyplot as plt
    values = torch.stack(list(result['activation'].values())).numpy()
    labels = [f'{i}: {"DeltaNet" if t == "linear_attention" else "attention"}'
              for i, t in enumerate(result['layer_types'])]
    fig, ax = plt.subplots(figsize=(10, 8))
    chart = ax.imshow(values, aspect='auto', cmap='magma')
    ax.set_yticks(range(24), labels, fontsize=8)
    ax.set(xlabel='Token position', title='Decoder output RMS across 1024 hidden dimensions')
    fig.colorbar(chart, ax=ax, label='Activation RMS (not importance)')
    fig.tight_layout()
    plt.show()


def bertviz_html(result):
    from bertviz import head_view
    # BertViz's displayed layers 0..5 map to actual layers 3,7,11,15,19,23.
    source = head_view(tuple(result['attention'].values()),
                       result['tokens']['token'].tolist(), html_action='return').data
    # Reuse BertViz unchanged, but provide its dependencies locally inside an
    # isolated document. No CDN requests or external inference are needed.
    scripts = ''.join('<script>' + (EXPERIMENT / 'assets/bertviz' / name).read_text() + '</script>'
                      for name in ('jquery.min.js', 'd3.min.js'))
    shim = '<script>window.require={config:function(){}};window.requirejs=function(d,cb){cb(window.jQuery,window.d3);};</script>'
    source = source.replace('<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>', scripts + shim)
    return '<!doctype html><html><meta charset="utf-8"><body>' + source + '</body></html>'


def token_flow_data(result):
    """JSON-safe measured slices, never a projection or invented semantic axes."""
    stages = [{'kind': 'embedding', **result['representations']['embedding']}]
    stages.extend({'kind': kind, 'layer': index, **result['representations'][f'layer_{index}']}
                  for index, kind in enumerate(result['layer_types']))
    stages.append({'kind': 'final_norm', **result['representations']['final_norm']})
    config = json.loads((MODEL_PATH / 'config.json').read_text())['text_config']
    geometry = {key: config[key] for key in ('vocab_size', 'hidden_size', 'intermediate_size',
                'num_attention_heads', 'num_key_value_heads', 'head_dim')}
    return {'model': 'Qwen3.5-0.8B', 'revision': (MODEL_PATH / 'REVISION').read_text().strip(),
            'prompt': result['prompt'], 'chat': result['chat'], 'dtype': 'bfloat16',
            'geometry': geometry,
            'tokens': result['tokens'].to_dict(orient='records'), 'stages': stages,
            'next_tokens': result['top_tokens'].to_dict(orient='records')}


def token_flow_html(result):
    data = json.dumps(token_flow_data(result), ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    return (EXPERIMENT / 'assets/token-flow.html').read_text().replace('__TOKEN_FLOW_DATA__', data)


def dashboard():
    """Construct a notebook UI without loading the model or running inference."""
    import ipywidgets as widgets
    from IPython.display import HTML, display, clear_output
    lab = Playground()
    prompt = widgets.Textarea(value='The capital of Hungary is', description='Prompt:',
                              layout=widgets.Layout(width='95%', height='85px'))
    mode = widgets.Checkbox(value=False, description='Chat template (otherwise raw text)',
                            layout=widgets.Layout(width='auto'), style={'description_width': 'initial'})
    inspect = widgets.Button(description='Analyze prompt', button_style='primary')
    generate = widgets.Button(description='Generate reply')
    ablate = widgets.Button(description='Compare MLP ablation', layout=widgets.Layout(width='200px'))
    unload = widgets.Button(description='Unload model')
    layer = widgets.IntSlider(value=0, min=0, max=23, description='MLP layer:')
    length = widgets.IntSlider(value=32, min=1, max=64, description='New tokens:')
    status = widgets.HTML('Ready. Model loads only when you press Analyze, Generate or Compare.')
    output = widgets.Output()
    buttons = [inspect, generate, ablate, unload]

    def run(action):
        def callback(_):
            for button in buttons:
                button.disabled = True
            status.value = 'Working on CPU (2 threads)…'
            with output:
                clear_output(wait=True)
                try:
                    action()
                    status.value = ('Done. The model stays in RAM until Unload or kernel shutdown.'
                                    if lab.model is not None else 'Ready. Model unloaded.')
                except Exception as error:
                    status.value = 'Action failed; see details below.'
                    print(f'{type(error).__name__}: {error}')
                finally:
                    for button in buttons:
                        button.disabled = False
        return callback

    def inspect_action():
        result = lab.analyze(prompt.value, mode.value)
        print(f"{len(result['tokens'])} tokens | {result['seconds']:.2f} seconds | Qwen3.5-0.8B")
        tabs = [widgets.Output() for _ in range(6)]
        with tabs[0]:
            display(result['tokens'])
        with tabs[1]:
            display(result['top_tokens'])
            print('Full-vocabulary softmax probabilities for the NEXT token, before sampling.')
        with tabs[2]:
            display(widgets.interactive(show_attention, result=widgets.fixed(result),
                layer=widgets.Dropdown(options=list(result['attention']), value=3),
                head=widgets.IntSlider(min=0, max=7, value=0, continuous_update=False)))
        with tabs[3]:
            show_activations(result)
        with tabs[4]:
            print('BertViz layers 0–5 correspond to Qwen layers 3, 7, 11, 15, 19, 23.')
            print('Only full attention has this token-to-token matrix; DeltaNet is not shown here.')
            document = html.escape(bertviz_html(result), quote=True)
            display(HTML(f'<iframe title="BertViz attention" sandbox="allow-scripts" srcdoc="{document}" style="width:100%;height:700px;border:0"></iframe>'))
        with tabs[5]:
            document = html.escape(token_flow_html(result), quote=True)
            display(HTML(f'<iframe title="Token representation flow" sandbox="allow-scripts" srcdoc="{document}" style="width:100%;height:1050px;border:0"></iframe>'))
        panel = widgets.Tab(children=tabs)
        for index, name in enumerate(['Tokens', 'Next token', 'Attention map', 'All 24 layers', 'BertViz', 'Token flow']):
            panel.set_title(index, name)
        display(panel)

    def generation_action():
        result = lab.generate(prompt.value, length.value)
        print(result['text'])
        print(f"\n{result['tokens']} generated tokens / {result['seconds']:.2f} s (always chat mode, thinking disabled)")

    def ablation_action():
        result = lab.ablate_mlp(prompt.value, layer.value, mode.value)
        print(f"MLP layer {result['layer']} zeroed for one forward pass. Weights unchanged.")
        print(f"Next-token distribution total variation: {result['total_variation']:.6f} (0–1)")
        print('Baseline:'); display(result['baseline'])
        print('Ablated:'); display(result['ablated'])

    def unload_action():
        lab.unload()
        print('Model reference released. Restart/shut down the kernel to fully release Python allocator memory.')

    inspect.on_click(run(inspect_action))
    generate.on_click(run(generation_action))
    ablate.on_click(run(ablation_action))
    unload.on_click(run(unload_action))
    display(widgets.VBox([prompt, mode, widgets.HBox([inspect, generate, unload]),
                         length, widgets.HBox([layer, ablate]), status, output]))
    return lab
