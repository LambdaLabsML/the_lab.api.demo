#!/usr/bin/env bash
# install_local.sh — build a self-contained local venv with vLLM 0.19.0 +
# upstream transformers (with native Gemma 4 support). Replaces the symlink to
# the shared NFS venv. Re-runnable; uses uv for fast resolution.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
UV="$HOME/.local/bin/uv"

log() { echo "[install_local] $*"; }

if [ ! -f "$PY" ]; then
    log "creating venv at $VENV ..."
    "$UV" venv "$VENV" --python 3.12
fi

# Pinned to match install.sh (Slurm base venv) so behavior is consistent.
TORCH=2.10.0
TRITON=3.6.0
VLLM=0.19.0
TOKENIZERS=0.21.4

log "torch $TORCH (cu126) ..."
"$UV" pip install --python "$PY" \
    "torch==$TORCH" \
    --index-url https://download.pytorch.org/whl/cu126

log "triton $TRITON ..."
"$UV" pip install --python "$PY" "triton==$TRITON"

log "tokenizers $TOKENIZERS ..."
"$UV" pip install --python "$PY" "tokenizers==$TOKENIZERS"

log "vllm $VLLM ..."
"$UV" pip install --python "$PY" "vllm==$VLLM"

# Transformers: latest release that ships gemma4 modeling + AutoModel registration
# (added 2026-04-01 to main). Pin >=4.58 to ensure gemma4 vision/audio modeling.
log "transformers (>=4.58, native gemma4) ..."
"$UV" pip install --python "$PY" --upgrade "transformers>=4.58"

# Quant tooling — installed for completeness so 31B-AWQ remains an option.
log "autoawq + llm-compressor ..."
"$UV" pip install --python "$PY" autoawq llm-compressor || \
    log "WARN: autoawq/llm-compressor install failed (continuing — bf16 runs unaffected)"

# Client + server deps.
log "openai + pillow ..."
"$UV" pip install --python "$PY" openai pillow

log "verifying gemma4 modeling is importable ..."
"$PY" - << 'PYEOF'
import torch, transformers, vllm
print(f"torch        {torch.__version__}")
print(f"transformers {transformers.__version__}")
print(f"vllm         {vllm.__version__}")
from transformers import AutoConfig, AutoModel
from transformers.models.gemma4 import modeling_gemma4
print("gemma4 modeling import OK")
# Check AutoModel knows Gemma4VisionConfig
from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
try:
    cls = AutoModel._model_mapping[Gemma4VisionConfig]
    print(f"AutoModel[Gemma4VisionConfig] = {cls.__name__}")
except KeyError:
    print("WARN: Gemma4VisionConfig not in AutoModel mapping")
PYEOF

log "done"
