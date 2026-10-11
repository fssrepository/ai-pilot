"""Download one pinned model into this project; never run on server startup."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'

from huggingface_hub import snapshot_download

MODEL_ID = 'Qwen/Qwen3.5-0.8B'
REVISION = '2fc06364715b967f1860aea9cf38778875588b17'

if __name__ == '__main__':
    destination = snapshot_download(
        MODEL_ID, revision=REVISION,
        local_dir=ROOT / 'models/Qwen3.5-0.8B', max_workers=2,
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.md', '*.jinja', 'LICENSE*'],
    )
    (ROOT / 'models/Qwen3.5-0.8B/REVISION').write_text(REVISION + '\n')
    print(f'Model ready: {destination}\nRevision: {REVISION}')
