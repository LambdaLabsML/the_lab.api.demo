#!/usr/bin/env bash
# install_local.sh — build a self-contained local venv with the pinned vLLM +
# upstream transformers (with native Gemma 4 support). Replaces the symlink to
# the shared NFS venv. Re-runnable; uses uv for fast resolution.
# All version pins come from versions.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
UV="$HOME/.local/bin/uv"

# Version pins live only in versions.sh — same source as install_user_venv.sh,
# so the local venv and the Slurm base venv can never drift apart.
# shellcheck source=versions.sh
source "$ROOT/versions.sh"

log() { echo "[install_local] $*"; }

if [ ! -f "$PY" ]; then
    log "creating venv at $VENV ..."
    "$UV" venv "$VENV" --python 3.12
fi

# torch + torchvision + torchaudio in ONE command from ONE index, so all three
# are the same CUDA build. --reinstall-package torch is what makes this
# re-runnable: uv considers an already-installed 2.10.0+cu126 to satisfy
# "torch==2.10.0", so without it a venv built by an older revision of this
# script keeps its mismatched cu126 wheel and reports "would make no changes".
log "torch $TORCH_VERSION + torchvision $TORCHVISION_VERSION + torchaudio $TORCHAUDIO_VERSION (cu128, from $TORCH_INDEX_URL) ..."
"$UV" pip install --python "$PY" \
    --index-url "$TORCH_INDEX_URL" \
    --reinstall-package torch \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION"

log "triton $TRITON_VERSION ..."
"$UV" pip install --python "$PY" "triton==$TRITON_VERSION"

log "tokenizers $TOKENIZERS_VERSION ..."
"$UV" pip install --python "$PY" "tokenizers==$TOKENIZERS_VERSION"

log "vllm $VLLM_VERSION ..."
"$UV" pip install --python "$PY" "vllm==$VLLM_VERSION"

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

log "verifying CUDA builds agree and gemma4 modeling is importable ..."
"$PY" - << 'PYEOF'
import torch, transformers, vllm

# Import the torch companions FIRST and on their own. torchaudio raises a
# RuntimeError at import time when its CUDA build differs from torch's, but
# transformers imports lazily and re-raises it as a bogus
# "Could not import module 'modeling_gemma4'" — checking here keeps the real
# cause visible instead of hiding it behind the gemma4 import below.
import torchvision, torchaudio

cuda = {
    "torch": torch.version.cuda,
    "torchvision": torchvision.__version__.partition("+")[2] or "?",
    "torchaudio": torchaudio.__version__.partition("+")[2] or "?",
}
print(f"torch        {torch.__version__} (CUDA {cuda['torch']})")
print(f"torchvision  {torchvision.__version__}")
print(f"torchaudio   {torchaudio.__version__}")
print(f"transformers {transformers.__version__}")
print(f"vllm         {vllm.__version__}")

tag = "cu" + cuda["torch"].replace(".", "")
mismatched = {k: v for k, v in cuda.items() if k != "torch" and v != tag}
if mismatched:
    raise SystemExit(
        f"CUDA build mismatch: torch is {tag} but {mismatched} — "
        "all three must come from the same index in one uv command "
        "(see TORCH_INDEX_URL in versions.sh)"
    )
print(f"CUDA builds consistent ({tag})")

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
