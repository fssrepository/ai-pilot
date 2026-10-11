#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
mkdir -p .tools .cache models
python3 - <<'PY'
import hashlib
import pathlib
import tarfile
import urllib.request

def download(url, destination, expected):
    destination = pathlib.Path(destination)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + '.part')
        print('Downloading', destination, flush=True)
        urllib.request.urlretrieve(url, temporary)
        temporary.rename(destination)
    digest = hashlib.sha256()
    with destination.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError(f'Checksum mismatch: {destination}')
    print('SHA256 verified:', destination, flush=True)

archive = '.cache/llama-b11429-bin-ubuntu-x64.tar.gz'
download('https://github.com/ggml-org/llama.cpp/releases/download/b11429/llama-b11429-bin-ubuntu-x64.tar.gz',
         archive, 'f6d25dde8f51133143d1453da4fd5f73b145127177612a283bf7995957af3392')
with tarfile.open(archive) as bundle:
    bundle.extractall('.tools', filter='data')
download('https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-Q4_K_M.gguf',
         'models/Qwen3.5-0.8B-Q4_K_M.gguf', 'bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517')
PY
printf 'Ready. Start manually: ./tools/start-chat.sh\n'

