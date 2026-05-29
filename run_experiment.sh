#!/usr/bin/env bash
# run_experiment.sh — install local packages, launch vLLM, run benchmark.
#
# Flow:
#   1. uv install local packages (if present) — no-op once optimization
#      platform manages them
#   2. launch vLLM
#   3. benchmark: run a Tetris game with the supplied args
#
# Usage:
#   ./run_experiment.sh [run_tetris_experiment.py args...]
#
# Environment:
#   VLLM_PORT        vLLM port            (default random free port)
#   SKIP_INSTALL     skip uv install step
#   SKIP_VLLM        assume vLLM already running
#   API_KEY          override vLLM API key

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load default env vars. Experiment scripts may override any of these.
# shellcheck source=.env
[ -f "$ROOT/.env" ] && source "$ROOT/.env"
# Allow VENV_DIR / UV to be overridden by the environment (e.g. from the
# Slurm job wrapper which sets up a per-job isolated venv).  The lab-server
# defaults are preserved when these vars are not already exported.
export VENV_DIR="${VENV_DIR:-/lambda/nfs/architects-us-south-2/model_inference/.venv}"
PYTHON="${PYTHON:-$VENV_DIR/bin/python}"
UV="${UV:-$HOME/.local/bin/uv}"
LAUNCH_SH="$ROOT/launch_gemma.sh"
SECRET_KEY="${SECRET_KEY:-$ROOT/arc3_lab/model_inference/secret.key}"
# On slurm, launch_gemma.sh writes the key to $ROOT/secret.key — check both
[ -f "$SECRET_KEY" ] || SECRET_KEY="$ROOT/secret.key"
PACKAGES_DIR="$ROOT/packages"

# Pick a random free port if not explicitly set — avoids collisions when
# multiple jobs run on the same node (each job gets its own vLLM instance).
_pick_free_port() {
    python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)"
}
VLLM_PORT="${VLLM_PORT:-$(_pick_free_port)}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_VLLM="${SKIP_VLLM:-0}"
RESULTS_FILE="${RESULTS_FILE:-$ROOT/tetris_results.json}"

# ── helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[run_experiment] $*"; }
log_section() { echo; echo "════════════════════════════════════════════════════════════════════════"; echo "[run_experiment] $*"; echo "════════════════════════════════════════════════════════════════════════"; }

wait_for_health() {
    local url="$1" timeout="${2:-300}" label="${3:-server}"
    local deadline=$(( $(date +%s) + timeout ))
    log "waiting for $label at $url ..."
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then
            log "$label is healthy"; return 0
        fi
        # Exit early if the background process is dead — no point waiting.
        if [ -n "${VLLM_PID:-}" ] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
            log "ERROR: $label process (pid $VLLM_PID) died — aborting health wait"; return 1
        fi
        # Also check the actual vLLM pid if launch_gemma.sh exported it.
        local vllm_pid_file="${VLLM_PID_FILE:-}"
        if [ -n "$vllm_pid_file" ] && [ -f "$vllm_pid_file" ]; then
            local inner_pid; inner_pid=$(cat "$vllm_pid_file" 2>/dev/null)
            if [ -n "$inner_pid" ] && ! kill -0 "$inner_pid" 2>/dev/null; then
                log "ERROR: $label inner process (pid $inner_pid) died — aborting health wait"; return 1
            fi
        fi
        sleep 3
    done
    log "ERROR: $label did not become healthy within ${timeout}s"; return 1
}

# ── package build helpers ─────────────────────────────────────────────────────

# Files that require a native recompile when changed.
# Pure-Python packages have no matches → always treated as "no recompile needed".
NATIVE_GLOB='*.cu *.cuh *.cpp *.c *.h *.hip CMakeLists.txt setup.py setup.cfg pyproject.toml'

pkg_src_hash() {
    # Hash all native source files in a package dir.
    # Returns empty string if no native files exist (pure Python).
    local dir="$1"
    local files
    files=$(find "$dir" \
        \( -path "$dir/.git" -o -path "$dir/build" -o -path "$dir/dist" \
           -o -name "*.egg-info" \) -prune \
        -o \( -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.c" \
              -o -name "*.h"  -o -name "*.hip" \
              -o -name "CMakeLists.txt" -o -name "setup.py" \
              -o -name "setup.cfg"     -o -name "pyproject.toml" \) \
        -print 2>/dev/null | sort)
    if [ -z "$files" ]; then
        echo ""
        return
    fi
    echo "$files" | xargs md5sum 2>/dev/null | md5sum | awk '{print $1}'
}

pkg_needs_rebuild() {
    local dir="$1"
    local sentinel="$dir/.build_hash"
    local current_hash
    current_hash="$(pkg_src_hash "$dir")"
    [ -z "$current_hash" ] && { echo "pure-python"; return 1; }   # no native files → no rebuild
    [ ! -f "$sentinel" ]   && { echo "no-sentinel"; return 0; }   # never built
    local stored_hash
    stored_hash="$(cat "$sentinel" 2>/dev/null || true)"
    [ "$current_hash" != "$stored_hash" ] && { echo "src-changed"; return 0; }
    return 1   # up to date
}

pkg_record_build() {
    local dir="$1"
    local hash
    hash="$(pkg_src_hash "$dir")"
    [ -n "$hash" ] && echo "$hash" > "$dir/.build_hash"
}

pkg_install() {
    local pkg_dir="$1" import_name="$2"
    local reason
    if reason="$(pkg_needs_rebuild "$pkg_dir")"; then
        log "  $(basename "$pkg_dir") needs build ($reason) — compiling..."
        "$UV" pip install --python "$PYTHON" --no-build-isolation -e "$pkg_dir" 2>&1 \
            | grep -v "^Audited\|^Resolved\|already installed" || true
        pkg_record_build "$pkg_dir"
        log "  $(basename "$pkg_dir") built and installed"
    else
        # Check editable install exists even if no recompile needed
        local installed
        installed=$("$PYTHON" -c "
import importlib.util
spec = importlib.util.find_spec('$import_name')
if spec and spec.origin: print(spec.origin)
" 2>/dev/null || true)
        if echo "$installed" | grep -q "$pkg_dir"; then
            log "  $(basename "$pkg_dir") up to date (pure-python editable)"
        else
            log "  $(basename "$pkg_dir") editable install missing — installing..."
            "$UV" pip install --python "$PYTHON" --no-build-isolation -e "$pkg_dir" 2>&1 \
                | grep -v "^Audited\|^Resolved\|already installed" || true
            pkg_record_build "$pkg_dir"
        fi
    fi
}

tetris_run() {
    # tetris_run <results_file> <extra args...>
    local results="$1"; shift
    # Auto-detect served model name from vLLM API if not explicitly set
    local model_arg=""
    if [ -z "${VLLM_MODEL:-}" ]; then
        local served_model
        served_model=$(curl -fsS --max-time 5 \
            ${API_KEY:+-H "Authorization: Bearer ${API_KEY}"} \
            "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null \
            | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || true)
        [ -n "$served_model" ] && model_arg="--model $served_model"
    fi
    "$PYTHON" "$ROOT/run_tetris_experiment.py" \
        --vllm       "http://localhost:${VLLM_PORT}/v1" \
        --api-key    "${API_KEY:-}" \
        --results    "$results" \
        ${model_arg} \
        "$@"
}

# ── 1. install local packages ─────────────────────────────────────────────────
log_section "1/3  package install"
if [ "$SKIP_INSTALL" != "1" ]; then
    # pure-Python packages only — no native compilation needed
    for entry in \
        transformers:transformers \
        autoawq:awq \
        llm-compressor:llmcompressor; do
        # eagle has no setup.py — source-browse only, skip install

        pkg_dir="$PACKAGES_DIR/${entry%%:*}"
        import_name="${entry##*:}"
        [ -d "$pkg_dir" ] || { log "  $pkg_dir not found — skipping"; continue; }
        pkg_install "$pkg_dir" "$import_name"
    done
else
    log "SKIP_INSTALL=1 — skipped"
fi

# ── 2. launch vLLM ────────────────────────────────────────────────────────────
log_section "2/3  vLLM launch"
VLLM_PID=""
cleanup() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        log "stopping vLLM (pid $VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        local deadline=$(( $(date +%s) + 30 ))
        while kill -0 "$VLLM_PID" 2>/dev/null && [ "$(date +%s)" -lt "$deadline" ]; do sleep 1; done
        kill -KILL "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [ "$SKIP_VLLM" != "1" ]; then
    [ -f "$LAUNCH_SH" ] || { log "ERROR: $LAUNCH_SH not found"; exit 1; }
    log "launching vLLM on port $VLLM_PORT ..."
    export VLLM_PID_FILE="$(mktemp)"
    PORT="$VLLM_PORT" \
        bash "$LAUNCH_SH" &
    VLLM_PID="$!"
    if ! wait_for_health "http://localhost:${VLLM_PORT}/health" 900 "vLLM"; then
        log "FATAL: vLLM never became healthy — aborting experiment"
        exit 2
    fi
else
    log "SKIP_VLLM=1 — using existing vLLM on :$VLLM_PORT"
fi

if [ -z "${API_KEY:-}" ] && [ -f "$SECRET_KEY" ]; then
    API_KEY="$(cat "$SECRET_KEY")"
fi
log "API_KEY: ${API_KEY:+set(${#API_KEY} chars)} ${API_KEY:-EMPTY}  SECRET_KEY=$SECRET_KEY (exists=$([ -f "$SECRET_KEY" ] && echo yes || echo no))"

# ── 3. benchmark run ──────────────────────────────────────────────────────────
log_section "3/3  benchmark run"
_tetris_rc=0
tetris_run "$RESULTS_FILE" \
    --gif "$(dirname "$RESULTS_FILE")/tetris_replay.gif" \
    "$@" || _tetris_rc=$?
if [ "$_tetris_rc" -ne 0 ]; then
    log "WARN: tetris benchmark exited with rc=$_tetris_rc (server/client crashed?)"
fi

# If the client crashed before writing results (e.g. ValueError at game end),
# construct a minimal results JSON from the log output so metrics are captured.
if [ ! -f "$RESULTS_FILE" ]; then
    log "results file missing — constructing from log output"
    # Find the job log file (it's in the job dir, parent of worktree)
    # THE_LAB_PROGRESS is exported by the wrapper as {job_dir}/script.progress
    _job_log="$ROOT/script.log"  # fallback to worktree (unlikely to exist but harmless)
    if [ -n "${THE_LAB_PROGRESS:-}" ]; then
        _job_log="$(dirname "$THE_LAB_PROGRESS")/script.log"
    fi
    # Parse turn stats from log lines like "[  496] move=... latency=554ms ..."
    _model="${MODEL:-QuantTrio/gemma-4-31B-it-AWQ}"
    _duration="${DURATION:-600}"
    # Python reads log from file (passed as arg) and writes JSON to RESULTS_FILE
    python3 /dev/stdin "$RESULTS_FILE" "$_model" "$_duration" \
            "$(dirname "$RESULTS_FILE")/tetris_replay.gif" "$_job_log" 2>/dev/null << 'PYEOF' || true
import sys, re, json, statistics
results_file, model, duration, gif, log_path = \
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
turns=[]; latencies=[]; max_score=0; total_score=0; restarts=0
try:
    src = open(log_path)
except OSError:
    sys.exit(1)
for line in src:
    m=re.match(r'^\[piece\s*(\d+)\] moves=\S+\s+max_score=(\d+)\s+total_score=(\d+)\s+restarts=(\d+).*latency=(\d+)ms',line)
    if m:
        t,ms,ts,rs,lat=(int(m.group(i)) for i in range(1,6))
        turns.append(t);latencies.append(lat);max_score=ms;total_score=ts;restarts=rs
if not turns: sys.exit(1)
lat=sorted(latencies)
def pct(s,p): return s[int(len(s)*p/100)] if s else 0
result={'model':model,'think':False,'max_score':max_score,'total_score':total_score,
        'max_lines':0,'total_lines':0,'level':1,
        'restarts':restarts,'turns':max(turns),
        'latency_mean_ms':int(statistics.mean(lat)) if lat else 0,
        'latency_p50_ms':pct(lat,50),'latency_p90_ms':pct(lat,90),
        'latency_p95_ms':pct(lat,95),'latency_p99_ms':pct(lat,99),
        'latency_min_ms':lat[0] if lat else 0,'latency_max_ms':lat[-1] if lat else 0,
        'prompt_tokens_mean':0,'completion_tokens_mean':0,
        'reasoning_chars_mean':0,'reasoning_chars_p95':0,
        'seed':42,'duration':duration,'gif':gif}
print(json.dumps(result))
with open(results_file,'w') as f: json.dump(result,f)
PYEOF
fi

# Wait for GIF files to finish writing (size stable for two consecutive checks).
# run_tetris_experiment.py waits for the server to exit after SIGTERM, but very
# large GIFs can take a moment; this ensures we copy a complete file.
_wait_gif_stable() {
    local gif="$1" timeout="${2:-60}"
    local deadline=$(( $(date +%s) + timeout ))
    local prev="-1" cur
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if [ -f "$gif" ]; then
            cur=$(stat -c%s "$gif" 2>/dev/null || echo "0")
            if [ "$cur" = "$prev" ] && [ "$cur" != "0" ]; then
                return 0
            fi
            prev="$cur"
        fi
        sleep 2
    done
    return 0  # best-effort: proceed even if timeout
}

# Resolve NFS experiment directory (where labapi stores persistent artifacts).
# THE_LAB_PROGRESS is set by the wrapper to {nfs_exp_dir}/script.progress.
_EXP_DIR=""
if [ -n "${THE_LAB_PROGRESS:-}" ]; then
    _EXP_DIR="$(dirname "$THE_LAB_PROGRESS")"
fi

# Copy GIF to the NFS experiment dir so it persists after the worktree is cleaned.
_gif_src="$(dirname "$RESULTS_FILE")/tetris_replay.gif"
if [ -f "$_gif_src" ]; then
    _wait_gif_stable "$_gif_src" 60
    if [ -n "$_EXP_DIR" ]; then
        cp "$_gif_src" "$_EXP_DIR/tetris_replay.gif" 2>/dev/null \
            && log "copied tetris_replay.gif to experiment dir" || true
    fi
fi

# Write script.output.md so the labapi dashboard can render metrics + GIF replay.
# The table mirrors the keys the client writes to script.progress (sans
# pct_complete, which is always 100 at this point) so the dashboard's
# live view and post-run summary line up.
if [ -n "$_EXP_DIR" ] && [ -f "$RESULTS_FILE" ]; then
    python3 - "$RESULTS_FILE" "$_EXP_DIR/script.output.md" << 'OUTMD_EOF' || true
import json, sys

results_file, output_md = sys.argv[1], sys.argv[2]
with open(results_file) as f:
    r = json.load(f)

def fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)

# Progress-file keys (live during the run) — keep this list in sync with
# the json.dump(...) block in tetris_client.py.
progress_keys = [
    ("elapsed_s",          "Elapsed (s)"),
    ("duration_s",          "Duration (s)"),
    ("model",               "Model"),
    ("turns",               "Turns (LLM calls)"),
    ("moves_applied",       "Moves applied"),
    ("max_score",           "Max score (best single game)"),
    ("total_score",         "Total score (sum across restarts)"),
    ("max_lines",           "Max lines (best single game)"),
    ("total_lines",         "Total lines (sum across restarts)"),
    ("restarts",            "Restarts"),
    ("stale_responses",     "Stale responses"),
    ("invalid_tool_calls",  "Invalid tool calls"),
    ("latency_p50_ms",      "Latency p50 (ms)"),
    ("latency_last_ms",     "Latency last (ms)"),
]
# Extra end-of-run keys present only in the final results JSON.
extra_keys = [
    ("level",                  "Level"),
    ("think",                  "Thinking enabled"),
    ("latency_mean_ms",        "Latency mean (ms)"),
    ("latency_p90_ms",         "Latency p90 (ms)"),
    ("latency_p95_ms",         "Latency p95 (ms)"),
    ("latency_p99_ms",         "Latency p99 (ms)"),
    ("latency_min_ms",         "Latency min (ms)"),
    ("latency_max_ms",         "Latency max (ms)"),
    ("prompt_tokens_mean",     "Prompt tokens (mean)"),
    ("completion_tokens_mean", "Completion tokens (mean)"),
    ("reasoning_chars_mean",   "Reasoning chars (mean)"),
    ("reasoning_chars_p95",    "Reasoning chars (p95)"),
]

total_score = r.get("total_score")
total_score_fmt = fmt(total_score)

rows = []
# Headline: total_score in bold up top.
rows.append(f"| **Total score** | **{total_score_fmt}** |")
for key, label in progress_keys:
    if key == "total_score":
        continue  # already shown as headline
    rows.append(f"| {label} | {fmt(r.get(key))} |")
for key, label in extra_keys:
    if key not in r:
        continue
    rows.append(f"| {label} | {fmt(r.get(key))} |")

md = "# Experiment Results\n\n| Metric | Value |\n|--------|-------|\n"
md += "\n".join(rows) + "\n\n## Replay\n\n![Tetris Replay](tetris_replay.gif)\n"
with open(output_md, "w") as f:
    f.write(md)
OUTMD_EOF
    log "wrote script.output.md"
fi

# ── print result JSON + metrics ({"metrics"} must be the last print) ──────────
echo
echo "════════════════════════════════════════════════════════════════════════"
if [ -f "$RESULTS_FILE" ]; then
    echo "RESULT JSON:"
    cat "$RESULTS_FILE"
    echo
    echo "════════════════════════════════════════════════════════════════════════"
    log "done — results at $RESULTS_FILE"
    # {"metrics"} is the very last line printed — nothing may follow it.
    python3 - "$RESULTS_FILE" << 'METRICS_EOF' || true
import json, sys
with open(sys.argv[1]) as f:
    r = json.load(f)
metrics = {k: r.get(k, 0) for k in (
    "max_score", "total_score", "max_lines", "total_lines",
    "restarts", "turns", "moves_applied",
    "invalid_tool_calls", "stale_responses",
    "latency_p50_ms", "latency_p95_ms", "latency_last_ms",
    "elapsed_s", "duration_s",
)}
import sys as _sys
_sys.stdout.write(
    "SUMMARY: "
    f"max_score={metrics['max_score']} total_score={metrics['total_score']} "
    f"max_lines={metrics['max_lines']} total_lines={metrics['total_lines']} "
    f"restarts={metrics['restarts']} turns={metrics['turns']} "
    f"latency_p50={metrics['latency_p50_ms']}ms latency_p95={metrics['latency_p95_ms']}ms "
    f"invalid={metrics['invalid_tool_calls']} stale={metrics['stale_responses']}\n"
)
print(json.dumps({"metrics": metrics}))
METRICS_EOF
else
    echo "NO RESULT JSON — results file missing at $RESULTS_FILE"
    echo "════════════════════════════════════════════════════════════════════════"
    log "FAILED — no results file produced"
    # No results file means the benchmark genuinely failed. Force a non-zero
    # exit even if the tetris subprocess somehow exited 0, so the wrapper
    # (and labapi) marks the experiment as failed instead of completed.
    [ "${_tetris_rc:-0}" -eq 0 ] && _tetris_rc=1
fi

# Propagate any non-zero exit from the tetris subprocesses so the wrapper
# (and labapi) sees the failure instead of treating an aborted run as OK.
exit "${_tetris_rc:-0}"
