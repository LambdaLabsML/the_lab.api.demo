#!/usr/bin/env bash
# versions.sh — pinned dependency versions for the shared base venv.
#
# This is the ONLY place version pins live. install_user_venv.sh, launch_gemma.sh
# and run_experiment.sh all source this file. A hash of these values is used as
# the venv directory name under BASE_VENV_ROOT, so parallel experiments with
# different version pins each get their own isolated venv and never interfere
# with each other.

# Re-pinned to a CUDA 12.8 stack: this node's driver is 570.148.08, which caps
# out at CUDA 12.8. The previous pins (vllm 0.24.0 / torch 2.11.0+cu130) are
# internally consistent but cu130 needs driver >= 580.x, so every run died with
# "The NVIDIA driver on your system is too old (found version 12080)" during
# EngineCore init — see idea 1 notes.
#
# torch 2.10.0 is the newest release whose wheels are still CUDA 12
# (nvidia-cuda-runtime-cu12==12.8.90 — exactly 12.8); torch 2.11.0 moved to
# cu13 nvidia deps. vllm 0.19.1 is the newest vllm that pins torch==2.10.0,
# so this is the closest working stack to the original pins.
TORCH_VERSION="2.10.0"
TRITON_VERSION="3.6.0"
VLLM_VERSION="0.19.1"
TOKENIZERS_VERSION="0.21.4"

# torchvision/torchaudio must be pinned and installed from TORCH_INDEX_URL in
# the SAME uv command as torch, never left to vllm's dependency resolution.
# torchaudio hard-fails at import time if its build CUDA differs from torch's
# ("PyTorch has CUDA version 12.6 whereas TorchAudio has CUDA version 12.8"),
# and because transformers imports lazily, that RuntimeError surfaces as the
# very misleading "Could not import module 'modeling_gemma4'". These are the
# releases built against torch 2.10.0.
TORCHVISION_VERSION="0.25.0"
TORCHAUDIO_VERSION="2.10.0"

# CUDA index URL for the whole torch trio (must match the cluster's CUDA version).
# PyPI rather than download.pytorch.org/whl/cu128: the plain PyPI torch 2.10.0
# wheel is ALREADY a CUDA 12.8 build (it resolves to 2.10.0+cu128 and pulls
# nvidia-cuda-runtime-cu12==12.8.90), so it is exactly what this driver needs,
# and the pytorch index 302-redirects to download-r2.pytorch.org which the
# sandbox proxy does not allow.
#
# Do NOT point this at .../whl/cu126: that yields torch 2.10.0+cu126 while
# torchvision/torchaudio still come from PyPI as +cu128 builds, which is exactly
# the mismatch described above.
TORCH_INDEX_URL="https://pypi.org/simple"

# Main-repo root, resolved from this file's own location — never hardcoded, so
# the scripts work from any clone path, any CWD, and inside the experiment
# sandbox. For a per-experiment worktree, --git-common-dir points back at the
# main clone, which is where venvs must live: a venv is not relocatable, and a
# worktree is deleted when its experiment ends.
_VERSIONS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_REPO_ROOT="$(git -C "$_VERSIONS_DIR" rev-parse --path-format=absolute \
    --git-common-dir 2>/dev/null)" \
  && MAIN_REPO_ROOT="$(dirname "$MAIN_REPO_ROOT")" \
  || MAIN_REPO_ROOT="$_VERSIONS_DIR"
# Not a git checkout, or the main clone isn't visible from here (e.g. a sandbox
# that bind-mounts only the worktree): fall back to this file's directory.
[ -d "$MAIN_REPO_ROOT" ] || MAIN_REPO_ROOT="$_VERSIONS_DIR"
unset _VERSIONS_DIR

# Root directory that holds one subdirectory per unique version-set.
# Each subdirectory is named after the hash of the pins above.
#
# Must live under the repo: experiments run inside a bwrap sandbox where the
# home directory is an empty --dir and only the repo tree is bind-mounted, so a
# venv root anywhere else under $HOME is invisible to them. The previous value
# pointed into another user's home directory, which does not exist on this node.
BASE_VENV_ROOT="$MAIN_REPO_ROOT/.venvs"
