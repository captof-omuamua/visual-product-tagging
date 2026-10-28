#!/usr/bin/env bash
# Stage 1: create two isolated Python environments with uv.
#   .venv-serve  vLLM, pinned, for serving
#   .venv-train  Unsloth, for LoRA training (plus open_clip, for Marqo's tokenizer)
#   .venv-api    google-genai only, for the Gemini mini-pilot
# Why two: vLLM 0.32 requires transformers >=5.16.1,<5.20, and Unsloth pins its own
# torch and transformers ranges. One shared environment would force one of them onto
# versions its authors never tested. See docs/adr/0001-separate-envs.md.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VLLM_VERSION="${VLLM_VERSION:-0.32.0}"
# "auto" lets uv pick the PyTorch build that matches your driver. If Unsloth fails to
# import, rerun with TRAIN_TORCH_BACKEND=cu128 (the build Unsloth's Blackwell guide uses).
TRAIN_TORCH_BACKEND="${TRAIN_TORCH_BACKEND:-auto}"
mkdir -p reports requirements

echo "== Serving environment: vLLM ${VLLM_VERSION} =="
[ -x .venv-serve/bin/python ] || uv venv .venv-serve --python "${PYTHON_VERSION}"
uv pip install --python .venv-serve/bin/python "vllm==${VLLM_VERSION}" --torch-backend=auto
uv pip install --python .venv-serve/bin/python -e .

echo "== Training environment: Unsloth (torch backend: ${TRAIN_TORCH_BACKEND}) =="
[ -x .venv-train/bin/python ] || uv venv .venv-train --python "${PYTHON_VERSION}"
uv pip install --python .venv-train/bin/python unsloth torchvision \
  --torch-backend="${TRAIN_TORCH_BACKEND}"
uv pip install --python .venv-train/bin/python -e . open_clip_torch

echo "== API environment: google-genai =="
[ -x .venv-api/bin/python ] || uv venv .venv-api --python "${PYTHON_VERSION}"
uv pip install --python .venv-api/bin/python google-genai

echo "== Recording exact versions (commit these after a PASS) =="
uv pip freeze --python .venv-serve/bin/python > requirements/serve-freeze.txt
uv pip freeze --python .venv-train/bin/python > requirements/train-freeze.txt
uv pip freeze --python .venv-api/bin/python > requirements/api-freeze.txt
.venv-serve/bin/python - > reports/env_versions_serve.json <<'EOF'
import json

import torch
import vllm

cuda = torch.cuda.is_available()
print(json.dumps({
    "vllm": vllm.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": cuda,
    "arch_list": torch.cuda.get_arch_list() if cuda else [],
    "device": torch.cuda.get_device_name(0) if cuda else None,
    "capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
}, indent=2))
EOF
cat reports/env_versions_serve.json
echo "Setup done."
