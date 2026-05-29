#!/usr/bin/env bash
# versions.sh — pinned dependency versions for the shared base venv.
#
# This is the ONLY place version pins live. Both install.sh and launch_gemma.sh
# source this file. A hash of these values is used as the venv directory name
# under BASE_VENV_ROOT, so parallel experiments with different version pins
# each get their own isolated venv and never interfere with each other.

TORCH_VERSION="2.11.0"
TRITON_VERSION="3.6.0"
VLLM_VERSION="0.22.0"
TOKENIZERS_VERSION="0.21.4"

# CUDA index URL for torch (must match the cluster's CUDA version).
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"

# Root directory that holds one subdirectory per unique version-set.
# Each subdirectory is named after the hash of the pins above.
BASE_VENV_ROOT="/home/davidh/.thelab/venvs"
