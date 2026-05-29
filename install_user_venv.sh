#!/usr/bin/env bash
# install.sh — build or reuse a versioned base venv on the Slurm cluster.
#
# Each unique set of version pins (from versions.sh) gets its own venv
# subdirectory under BASE_VENV_ROOT, named by the hash of the pins.  Parallel
# experiments with different versions never share or clobber each other's venv.
#
# Fast path: if the versioned venv directory already exists and is valid,
# the script exits in under a second — no pip, no Python import checks.
#
# Concurrent safety: if two jobs race to create the same versioned venv,
# flock (with mkdir spin-lock fallback) ensures only one installs; the others
# wait, see the venv is ready, and proceed immediately.
#
# Usage:
#   bash install.sh           # SSH to slurm head node and install there
#   bash install.sh --local   # install directly on the current machine

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=versions.sh
source "$ROOT/versions.sh"

SLURM_HOST="slurm"
LOCAL="${1:-}"

log() { echo "[install.sh] $*"; }

# ── install logic (runs on target machine) ───────────────────────────────────
# Passed via env so the heredoc body can reference them without quoting pain.
INSTALL_BODY=$(cat << 'INSTALL_BODY_EOF'
set -euo pipefail

_log() { echo "[install] $*"; }

# Versioned venv path: BASE_VENV_ROOT/<hash>/
VERSIONS_HASH="$(printf '%s\n' \
    "torch=$TORCH_VERSION" \
    "torchvision=$TORCHVISION_VERSION" \
    "torchaudio=$TORCHAUDIO_VERSION" \
    "triton=$TRITON_VERSION" \
    "vllm=$VLLM_VERSION" \
    "tokenizers=$TOKENIZERS_VERSION" \
    "index=$TORCH_INDEX_URL" \
    "packages=$PACKAGES_HASH" \
    "patches=$PATCHES_HASH" \
  | md5sum | cut -d' ' -f1)"

BASE_VENV="$BASE_VENV_ROOT/$VERSIONS_HASH"
PYTHON="$BASE_VENV/bin/python"
UV="$BASE_VENV/bin/uv"
LOCKFILE="$BASE_VENV_ROOT/.install-${VERSIONS_HASH}.lock"
READY_SENTINEL="$BASE_VENV/.ready"

mkdir -p "$BASE_VENV_ROOT"

# Fast path: venv exists, is marked ready, and passes a basic import check.
# The torchvision CUDA-check patch that used to live here (sed 'raise
# RuntimeError(' → 'warnings.warn(') is gone: torchvision/torchaudio are now
# pinned to the torch-matching releases and installed from TORCH_INDEX_URL in
# the same command as torch, so their build CUDA can no longer diverge. The
# patch also only covered torchvision — torchaudio's identical check still
# fired, and transformers re-raised it as "Could not import module
# 'modeling_gemma4'". Suppressing the check is the wrong layer to fix it at.

if [ -f "$READY_SENTINEL" ] && [ -x "$PYTHON" ]; then
  # torchvision/torchaudio are imported here on purpose: each raises at import
  # time on a CUDA build mismatch, so a venv that predates the pins above fails
  # this probe and gets rebuilt instead of failing later inside vLLM.
  if "$PYTHON" -c "
import torch, torchvision, torchaudio, vllm
assert torch.__version__.startswith('$TORCH_VERSION'), \
    f'torch version mismatch: {torch.__version__!r} != $TORCH_VERSION'
" 2>/dev/null; then
    _log "venv already ready at $BASE_VENV — skipping install"
    echo "$BASE_VENV"   # caller can capture this
    exit 0
  else
    _log "venv sentinel exists but import/version check failed — rebuilding"
    rm -f "$READY_SENTINEL"
  fi
fi

_log "Building versioned venv at $BASE_VENV (hash $VERSIONS_HASH)..."

_do_install() {
  # ── venv bootstrap ─────────────────────────────────────────────────────────
  if [ ! -f "$PYTHON" ]; then
    _log "Creating venv ..."
    python3 -m venv "$BASE_VENV"
  fi
  if [ ! -f "$UV" ]; then
    _log "Bootstrapping uv ..."
    "$PYTHON" -m pip install -q uv 2>/dev/null || true
    # pip may install uv to a non-standard location; search common fallbacks.
    if [ ! -f "$UV" ]; then
      for _uv_try in \
          "$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("scripts"))' 2>/dev/null)/uv" \
          "$HOME/.local/bin/uv" \
          "$(which uv 2>/dev/null)"; do
        [ -x "$_uv_try" ] || continue
        _log "uv not at $UV — symlinking from $_uv_try"
        mkdir -p "$(dirname "$UV")"
        ln -sf "$_uv_try" "$UV"
        break
      done
    fi
  fi
  _log "uv $($UV --version)"

  # ── torch trio ─────────────────────────────────────────────────────────────
  # One command, one index, so torch/torchvision/torchaudio are all the same
  # CUDA build. Leaving torchvision/torchaudio to vllm's resolution pulls PyPI
  # (+cu128) wheels alongside whatever torch build this index serves, and
  # torchaudio then hard-fails at import. --reinstall-package torch forces the
  # swap in a venv that already has a differently-built torch of the same
  # version number (uv treats 2.10.0+cu126 as satisfying "torch==2.10.0").
  _log "Installing torch==$TORCH_VERSION + torchvision==$TORCHVISION_VERSION + torchaudio==$TORCHAUDIO_VERSION ..."
  "$UV" pip install --python "$PYTHON" \
      --index-url "$TORCH_INDEX_URL" \
      --reinstall-package torch \
      "torch==$TORCH_VERSION" \
      "torchvision==$TORCHVISION_VERSION" \
      "torchaudio==$TORCHAUDIO_VERSION"

  # ── triton ─────────────────────────────────────────────────────────────────
  _log "Installing triton==$TRITON_VERSION ..."
  "$UV" pip install --python "$PYTHON" "triton==$TRITON_VERSION"

  # ── tokenizers ─────────────────────────────────────────────────────────────
  _log "Installing tokenizers==$TOKENIZERS_VERSION ..."
  "$UV" pip install --python "$PYTHON" "tokenizers==$TOKENIZERS_VERSION"

  # ── vllm ───────────────────────────────────────────────────────────────────
  _log "Installing vllm==$VLLM_VERSION ..."
  "$UV" pip install --python "$PYTHON" "vllm==$VLLM_VERSION"

  # ── ninja — required by vllm's JIT kernel compilation ─────────────────────
  "$UV" pip install --python "$PYTHON" ninja

  # ── mark ready (written last so a partial install is never seen as valid) ──
  echo "$VERSIONS_HASH" > "$READY_SENTINEL"
  _log "Done. Venv ready at $BASE_VENV"

  "$PYTHON" -c "
import torch, torchvision, torchaudio, triton, vllm
print(f'  torch       {torch.__version__} (CUDA {torch.version.cuda})')
print(f'  torchvision {torchvision.__version__}')
print(f'  torchaudio  {torchaudio.__version__}')
print(f'  triton      {triton.__version__}')
print(f'  vllm        {vllm.__version__}')
"
}

# Acquire lock — only one process builds this specific venv at a time.
if command -v flock &>/dev/null; then
  (
    flock -x 200
    if [ -f "$READY_SENTINEL" ] && [ -x "$PYTHON" ]; then
      _log "Built by another process — skipping"
    else
      _do_install
    fi
  ) 200>"$LOCKFILE"
else
  LOCKDIR="${LOCKFILE}.d"
  while ! mkdir "$LOCKDIR" 2>/dev/null; do
    _log "Waiting for lock ..."
    sleep 3
  done
  trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT INT TERM
  if [ -f "$READY_SENTINEL" ] && [ -x "$PYTHON" ]; then
    _log "Built by another process — skipping"
  else
    _do_install
  fi
  rmdir "$LOCKDIR" 2>/dev/null || true
fi

echo "$BASE_VENV"
INSTALL_BODY_EOF
)

# Hash the packages/ and patches/ directories so any change triggers a fresh venv.
PACKAGES_HASH=""
if [ -d "$ROOT/packages" ]; then
    PACKAGES_HASH="$(find "$ROOT/packages" -type f \
        ! -path '*/.git/*' ! -path '*/build/*' ! -path '*/__pycache__/*' \
        ! -name '*.pyc' ! -name '.build_hash' \
        | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1)"
fi
PATCHES_HASH=""
if [ -d "$ROOT/patches" ]; then
    PATCHES_HASH="$(find "$ROOT/patches" -type f | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1)"
fi

EXPORTS="BASE_VENV_ROOT=$BASE_VENV_ROOT TORCH_VERSION=$TORCH_VERSION TORCHVISION_VERSION=$TORCHVISION_VERSION TORCHAUDIO_VERSION=$TORCHAUDIO_VERSION TRITON_VERSION=$TRITON_VERSION VLLM_VERSION=$VLLM_VERSION TOKENIZERS_VERSION=$TOKENIZERS_VERSION TORCH_INDEX_URL=$TORCH_INDEX_URL PACKAGES_HASH=$PACKAGES_HASH PATCHES_HASH=$PATCHES_HASH"

if [ "$LOCAL" = "--local" ]; then
  log "Running install locally"
  env $EXPORTS bash -c "$INSTALL_BODY"
else
  log "Target: $SLURM_HOST  root: $BASE_VENV_ROOT"
  log "Versions: vllm=$VLLM_VERSION  torch=$TORCH_VERSION  triton=$TRITON_VERSION"
  ssh "$SLURM_HOST" env $EXPORTS bash -s <<< "$INSTALL_BODY"
fi

log "Done."
