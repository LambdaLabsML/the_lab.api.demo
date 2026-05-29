#!/usr/bin/env bash
# launch_gemma.sh — start a single Gemma-4 vLLM server (port $PORT, default
# 8008) on 1× H100 80GB (vLLM 0.22.0). Selected via the MODEL env var.
#
# Defaults: 31B-AWQ + the gemma4_assistant MTP drafter at NUM_SPECULATIVE_TOKENS=12,
# tuned for the "think then call a tool" workload. The MTP drafter is incompatible
# with vLLM 0.19's prefix-cache alignment, so prefix caching is enabled iff the
# drafter is off.
#
# Usage:
#   ./launch_gemma.sh                            # default config (31B-AWQ + drafter)
#   MODEL=google/gemma-4-E2B-it ./launch_gemma.sh
#       DTYPE=bfloat16 QUANTIZATION= KV_CACHE_DTYPE=auto \
#       ASSISTANT_MODEL=google/gemma-4-E2B-it-assistant \
#       ./launch_gemma.sh                        # E2B with its own drafter
#   PORT=8002 ./launch_gemma.sh                  # override port
#   HOST=0.0.0.0 ./launch_gemma.sh               # expose beyond localhost
#   WATCHDOG_ENABLED=0 ./launch_gemma.sh         # run vLLM directly, no restart loop
#   GEMMA4_ASSISTANT_ENABLED=0 ./launch_gemma.sh # disable MTP drafter, enable prefix cache
#   EXTRA_ARGS="--foo=bar" ./launch_gemma.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── version pins ───────────────────────────────────────────────────────────
# shellcheck source=versions.sh
source "$ROOT/versions.sh"

# ── API key — auto-generated on first run, persisted in secret.key ─────────
SECRET_KEY_FILE="${SECRET_KEY_FILE:-$ROOT/secret.key}"
if [ -z "${API_KEY:-}" ]; then
  if [ ! -f "$SECRET_KEY_FILE" ]; then
    openssl rand -hex 32 > "$SECRET_KEY_FILE"
    chmod 600 "$SECRET_KEY_FILE"
    echo "Generated new API key -> $SECRET_KEY_FILE"
  fi
  API_KEY="$(cat "$SECRET_KEY_FILE")"
fi

# ── model + serving config ─────────────────────────────────────────────────
MODEL="${MODEL:-QuantTrio/gemma-4-31B-it-AWQ}"
PORT="${PORT:-8008}"
HOST="${HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
DTYPE="${DTYPE:-float16}"
QUANTIZATION="${QUANTIZATION-awq_marlin}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9636}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 16}}"
GEMMA4_ASSISTANT_ENABLED="${GEMMA4_ASSISTANT_ENABLED:-1}"
ASSISTANT_MODEL="${ASSISTANT_MODEL:-google/gemma-4-31B-it-assistant}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-12}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ── watchdog config ────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
DRY_RUN="${DRY_RUN:-0}"
WATCHDOG_ENABLED="${WATCHDOG_ENABLED:-0}"
WATCHDOG_POLL_S="${WATCHDOG_POLL_S:-10}"
WATCHDOG_STALL_S="${WATCHDOG_STALL_S:-180}"
WATCHDOG_GRACE_S="${WATCHDOG_GRACE_S:-30}"
WATCHDOG_RESTART_DELAY_S="${WATCHDOG_RESTART_DELAY_S:-5}"
WATCHDOG_HTTP_TIMEOUT_S="${WATCHDOG_HTTP_TIMEOUT_S:-5}"
WATCHDOG_HEALTH_WAIT_S="${WATCHDOG_HEALTH_WAIT_S:-900}"
# How many consecutive /metrics failures before declaring the server stuck.
# Default 6 × WATCHDOG_POLL_S = 60 s of unresponsive HTTP → restart.
WATCHDOG_METRICS_FAIL_MAX="${WATCHDOG_METRICS_FAIL_MAX:-6}"
WATCHDOG_HEALTH_HOST="${WATCHDOG_HEALTH_HOST:-127.0.0.1}"
mkdir -p "$LOG_DIR"

# ── venv discovery ─────────────────────────────────────────────────────────
# Ensure $ROOT/.venv exists so patch scripts and `.venv/bin/vllm` invocations
# below work. Resolution order:
#   1. VENV_DIR env var (set by Slurm job wrapper for per-job isolated venv)
#   2. Shared model_inference venv on NFS (present on the lab head node)
#   3. Slurm base-venv built by install.sh (present on Slurm compute nodes)
if [ ! -e "$ROOT/.venv" ]; then
  _VENV_SRC="${VENV_DIR:-}"
  if [ -z "$_VENV_SRC" ] || [ ! -d "$_VENV_SRC" ]; then
    for _try in \
        "/lambda/nfs/architects-us-south-2/model_inference/.venv" \
        "/home/davidh/.thelab/base-venv"; do
      if [ -d "$_try" ] && [ -x "$_try/bin/python3" ]; then
        _VENV_SRC="$_try"
        break
      fi
    done
  fi
  if [ -n "$_VENV_SRC" ] && [ -d "$_VENV_SRC" ]; then
    ln -sfn "$_VENV_SRC" "$ROOT/.venv"
    echo "launch_gemma: linked $ROOT/.venv -> $_VENV_SRC"
  else
    echo "ERROR: Cannot find a usable venv. Set VENV_DIR or run install.sh." >&2
    exit 1
  fi
fi

# ── versioned venv resolution ──────────────────────────────────────────────
# Compute the hash of the pinned versions and resolve (or build) the matching
# venv under BASE_VENV_ROOT.  Each unique version set lives in its own
# subdirectory so parallel experiments with different pins never conflict.
_PACKAGES_HASH=""
if [ -d "$ROOT/packages" ]; then
    _PACKAGES_HASH="$(find "$ROOT/packages" -type f \
        ! -path '*/.git/*' ! -path '*/build/*' ! -path '*/__pycache__/*' \
        ! -name '*.pyc' ! -name '.build_hash' \
        | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1)"
fi
_PATCHES_HASH=""
if [ -d "$ROOT/patches" ]; then
    _PATCHES_HASH="$(find "$ROOT/patches" -type f | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1)"
fi
_VERSIONS_HASH="$(printf '%s\n' \
    "torch=$TORCH_VERSION" \
    "triton=$TRITON_VERSION" \
    "vllm=$VLLM_VERSION" \
    "tokenizers=$TOKENIZERS_VERSION" \
    "index=$TORCH_INDEX_URL" \
    "packages=$_PACKAGES_HASH" \
    "patches=$_PATCHES_HASH" \
  | md5sum | cut -d' ' -f1)"
_VERSIONED_VENV="$BASE_VENV_ROOT/$_VERSIONS_HASH"

if [ -f "$_VERSIONED_VENV/.ready" ] && [ -x "$_VERSIONED_VENV/bin/python" ]; then
  echo "launch_gemma: versioned venv ready ($_VERSIONS_HASH)"
else
  echo "launch_gemma: versioned venv missing — building (first use for this version set)"
  bash "$ROOT/install.sh" --local
fi

# Override VENV_DIR with the versioned path and update the .venv symlink so
# all .venv/bin/vllm invocations below use the correct version.
export VENV_DIR="$_VERSIONED_VENV"
ln -sfn "$VENV_DIR" "$ROOT/.venv"
unset _VERSIONS_HASH _VERSIONED_VENV

export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export GEMMA4_ASSISTANT_ENABLED

# Chat template is PINNED — env override is intentionally disabled. The
# optimization process previously discovered it could outsource Tetris
# move-search to a Turing-complete chat template by swapping this path.
CHAT_TEMPLATE="$ROOT/templates/tool_chat_template_gemma4.jinja"
if [ ! -f "$CHAT_TEMPLATE" ]; then
  echo "ERROR: $CHAT_TEMPLATE missing" >&2
  exit 1
fi

# ── compose vLLM arguments ─────────────────────────────────────────────────
ARGS=(
  serve "$MODEL"
  --host "$HOST" --port "$PORT"
  --api-key "$API_KEY"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --enable-auto-tool-choice
  --tool-call-parser gemma4
  --reasoning-parser gemma4
  --chat-template "$CHAT_TEMPLATE"
  ${QUANTIZATION:+--quantization "$QUANTIZATION"}
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT"
)

if [ "$GEMMA4_ASSISTANT_ENABLED" = "1" ] || [ "$GEMMA4_ASSISTANT_ENABLED" = "true" ]; then
  if [ -z "$SPECULATIVE_CONFIG" ]; then
    SPECULATIVE_CONFIG="{\"model\":\"$ASSISTANT_MODEL\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS}"
  fi
  # vLLM 0.19 defaults enable_prefix_caching=True; turn it off explicitly
  # since the MTP drafter trips an alignment_tokens assert in the cache hit path.
  ARGS+=(--no-enable-prefix-caching)
else
  ARGS+=(--enable-prefix-caching)
fi

if [ -n "$SPECULATIVE_CONFIG" ]; then
  ARGS+=(--speculative-config "$SPECULATIVE_CONFIG")
fi

# shellcheck disable=SC2206
EXTRA=( $EXTRA_ARGS )
if [ ${#EXTRA[@]} -gt 0 ]; then
  ARGS+=("${EXTRA[@]}")
fi

echo "launch args: ${ARGS[*]}"

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

# Ensure venv binaries (including ninja) are on PATH for vLLM subprocesses.
export PATH="$ROOT/.venv/bin:$PATH"

# ── unsupervised path (WATCHDOG_ENABLED=0) ─────────────────────────────────
if [ "$WATCHDOG_ENABLED" = "0" ] || [ "$WATCHDOG_ENABLED" = "false" ]; then
  exec .venv/bin/vllm "${ARGS[@]}" 2>&1 | tee -a "$LOG_DIR/vllm.log"
fi

# ── supervised path: restart on crash / wedge ──────────────────────────────
VLLM_PID=""

metric_value() {
  local metric="$1"
  awk -v metric="$metric" '$1 ~ "^" metric "\\{" { print $2; exit }'
}

start_vllm() {
  echo "watchdog: starting vLLM at $(date -Is)" | tee -a "$LOG_DIR/vllm.log"
  .venv/bin/vllm "${ARGS[@]}" > >(tee -a "$LOG_DIR/vllm.log") 2>&1 &
  VLLM_PID="$!"
  # Export actual vLLM PID so run_experiment.sh can detect crashes fast.
  [ -n "${VLLM_PID_FILE:-}" ] && echo "$VLLM_PID" > "$VLLM_PID_FILE"
}

stop_vllm() {
  if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "watchdog: stopping vLLM pid=$VLLM_PID at $(date -Is)" | tee -a "$LOG_DIR/vllm.log"
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    local deadline=$(( $(date +%s) + WATCHDOG_GRACE_S ))
    while kill -0 "$VLLM_PID" 2>/dev/null && [ "$(date +%s)" -lt "$deadline" ]; do
      sleep 1
    done
    if kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "watchdog: pid=$VLLM_PID ignored TERM; sending KILL" | tee -a "$LOG_DIR/vllm.log"
      kill -KILL "$VLLM_PID" 2>/dev/null || true
    fi
  fi
}

wait_for_health() {
  local deadline=$(( $(date +%s) + WATCHDOG_HEALTH_WAIT_S ))
  local url="http://${WATCHDOG_HEALTH_HOST}:${PORT}/health"
  while kill -0 "$VLLM_PID" 2>/dev/null; do
    if curl -fsS --max-time "$WATCHDOG_HTTP_TIMEOUT_S" "$url" >/dev/null 2>&1; then
      echo "watchdog: vLLM healthy at $url" | tee -a "$LOG_DIR/vllm.log"
      return 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "watchdog: vLLM did not become healthy within ${WATCHDOG_HEALTH_WAIT_S}s" | tee -a "$LOG_DIR/vllm.log"
      return 1
    fi
    sleep "$WATCHDOG_POLL_S"
  done
  return 1
}

is_scheduler_wedged() {
  local metrics="$1"
  local waiting running kv
  waiting="$(printf '%s\n' "$metrics" | metric_value 'vllm:num_requests_waiting')"
  running="$(printf '%s\n' "$metrics" | metric_value 'vllm:num_requests_running')"
  kv="$(printf '%s\n' "$metrics" | metric_value 'vllm:kv_cache_usage_perc')"
  waiting="${waiting:-0}"
  running="${running:-0}"
  kv="${kv:-0}"
  awk -v w="$waiting" -v r="$running" -v kv="$kv" \
    'BEGIN { exit !((w + 0) > 0 && (r + 0) == 0 && (kv + 0) == 0) }'
}

trap 'stop_vllm; exit 130' INT TERM

while true; do
  start_vllm
  wait_for_health || true
  bad_since=0
  metrics_fail_streak=0
  while kill -0 "$VLLM_PID" 2>/dev/null; do
    sleep "$WATCHDOG_POLL_S"
    metrics="$(curl -fsS --max-time "$WATCHDOG_HTTP_TIMEOUT_S" \
      "http://${WATCHDOG_HEALTH_HOST}:${PORT}/metrics" 2>/dev/null || true)"
    if [ -z "$metrics" ]; then
      metrics_fail_streak=$(( metrics_fail_streak + 1 ))
      if [ "$metrics_fail_streak" -ge "$WATCHDOG_METRICS_FAIL_MAX" ]; then
        echo "watchdog: /metrics unresponsive for ${metrics_fail_streak} consecutive polls (${WATCHDOG_POLL_S}s each); restarting" \
          | tee -a "$LOG_DIR/vllm.log"
        stop_vllm
        break
      fi
      continue
    fi
    metrics_fail_streak=0
    if is_scheduler_wedged "$metrics"; then
      now="$(date +%s)"
      if [ "$bad_since" -eq 0 ]; then
        bad_since="$now"
      fi
      elapsed=$(( now - bad_since ))
      if [ "$elapsed" -ge "$WATCHDOG_STALL_S" ]; then
        echo "watchdog: scheduler wedged for ${elapsed}s (waiting>0, running=0, kv=0); restarting" \
          | tee -a "$LOG_DIR/vllm.log"
        stop_vllm
        break
      fi
    else
      bad_since=0
    fi
  done
  wait "$VLLM_PID" || true
  echo "watchdog: vLLM exited; restarting after ${WATCHDOG_RESTART_DELAY_S}s" | tee -a "$LOG_DIR/vllm.log"
  sleep "$WATCHDOG_RESTART_DELAY_S"
done
