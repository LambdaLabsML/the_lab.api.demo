#!/usr/bin/env python3
"""Launch tetris_server + tetris_client, wait for completion, emit one JSON result line.

Usage:
  python run_tetris_experiment.py [options]

  # quick test against vLLM on :8008
  python run_tetris_experiment.py --vllm http://localhost:8008/v1 \
      --model QuantTrio/gemma-4-31B-it-AWQ

  # E2B with thinking
  python run_tetris_experiment.py --think --max-tokens 2048

  The tetris server binds to a random free port so multiple games can run
  on the same machine without collisions.

  --out appends the JSON line to a file (one experiment per line, easy to grep/jq).
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _pick_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_url(url, timeout_s=60, poll=0.5):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status < 400:
                    return True
        except Exception:
            pass
        time.sleep(poll)
    return False


def main():
    parser = argparse.ArgumentParser(description="Run one Tetris+Gemma experiment")
    # Server args
    parser.add_argument("--port",       type=int,   default=0,                             help="Tetris server port (0 = pick a free port)")
    # --seed intentionally NOT exposed. Board RNG seed is a PINNED
    # benchmark constant (see GAME_SEED in tetris_server.py).
    parser.add_argument("--gif-speed",  type=float, default=1.0,                           help="GIF playback speed multiplier")
    parser.add_argument("--gif-fps",    type=float, default=6.5,                           help="Fixed GIF playback fps (default 6.5 ≈ 0.15s/frame; game always runs in wait-for-turn mode)")
    # NOTE: --duration intentionally NOT exposed. Game duration is a
    # PINNED benchmark constant (see GAME_DURATION_S in tetris_server.py).
    # Client / model args
    parser.add_argument("--vllm",       default=os.environ.get("VLLM_BASE",   "http://localhost:8009/v1"))
    parser.add_argument("--model",      default=os.environ.get("VLLM_MODEL",  "google/gemma-4-E2B-it"))
    parser.add_argument("--api-key",    default=os.environ.get("VLLM_API_KEY", ""))
    parser.add_argument("--max-tokens",      type=int,   default=512,                      help="Max tokens per model call")
    parser.add_argument("--think",           action="store_true",                          help="Enable force_thinking (use --max-tokens >=2048)")
    parser.add_argument("--max-soft-tokens", type=int,   default=1120,                    help="Max image soft tokens (lower=smaller image, faster)")
    parser.add_argument("--image",           action="store_true",                         help="Send PNG screenshot alongside the ASCII board (default: text-only, much faster prefill)")
    parser.add_argument("--temperature",     type=float, default=0.0,                     help="Sampling temperature (0.0=greedy, 0.3-0.7 for diverse actions)")
    # Output
    parser.add_argument("--out",        default=None,                                      help="Append JSON result line to this file")
    parser.add_argument("--gif",        default="tetris_replay.gif",                       help="Path to save replay GIF")
    parser.add_argument("--results",    default="tetris_results.json",                     help="Intermediate results file from client")
    parser.add_argument("--log",        default=None,                                      help="Path for the per-turn thinking log (default: <results-dir>/tetris_thinking.log)")
    args = parser.parse_args()

    # Default per-run log next to the results file so each experiment dir gets
    # its own thinking log (otherwise concurrent or sequential runs overwrite
    # each other's logs at the cwd default).
    if args.log is None:
        args.log = str(Path(args.results).parent / "tetris_thinking.log")

    if args.port == 0:
        args.port = _pick_free_port()
    server_url = f"http://localhost:{args.port}"

    # ------------------------------------------------------------------
    # Build subprocess commands
    # ------------------------------------------------------------------
    python = sys.executable

    server_cmd = [
        python, str(HERE / "tetris_server.py"),
        "--port",     str(args.port),
        "--gif-out",  args.gif,
        *(["--gif-speed", str(args.gif_speed)] if args.gif_speed != 1.0      else []),
        "--wait-for-turn",   # PINNED — always turn-based; cannot be disabled via run_experiment.sh
        *(["--gif-fps", str(args.gif_fps)]     if args.gif_fps is not None    else []),
        # --duration and --seed are PINNED constants in tetris_server.py
        # (GAME_DURATION_S, GAME_SEED) — they are NOT optimization knobs.
    ]

    client_cmd = [
        python, str(HERE / "tetris_client.py"),
        "--server",      server_url,
        "--vllm",        args.vllm,
        "--model",       args.model,
        "--max-tokens",  str(args.max_tokens),
        "--gif-save",    args.gif,
        "--results-out", args.results,
        "--log",         args.log,
    ]
    if args.api_key:
        client_cmd += ["--api-key", args.api_key]
    if args.think:
        client_cmd += ["--think"]
    if args.max_soft_tokens != 1120:
        client_cmd += ["--max-soft-tokens", str(args.max_soft_tokens)]
    if args.image:
        client_cmd += ["--image"]
    if args.temperature != 0.0:
        client_cmd += ["--temperature", str(args.temperature)]

    # ------------------------------------------------------------------
    # Launch server
    # ------------------------------------------------------------------
    print(f"[run] starting server on :{args.port}  (seed/duration pinned by tetris_server.py)")
    server_proc = subprocess.Popen(server_cmd, stdout=sys.stdout, stderr=sys.stderr)

    if not _wait_url(f"{server_url}/state", timeout_s=15):
        print("[run] server did not start in time — aborting", file=sys.stderr)
        server_proc.terminate()
        sys.exit(1)
    print("[run] server ready")

    # ------------------------------------------------------------------
    # Launch client
    # ------------------------------------------------------------------
    print(f"[run] starting client  model={args.model}  think={args.think}")
    client_proc = subprocess.Popen(client_cmd, stdout=sys.stdout, stderr=sys.stderr)

    # ------------------------------------------------------------------
    # Wait — server drives the game duration, client exits when server stops
    # ------------------------------------------------------------------
    def _shutdown(sig, frame):
        print("\n[run] interrupted — stopping both processes")
        client_proc.terminate()
        server_proc.terminate()
        sys.exit(1)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Wait for the client to finish naturally. The server can exit on its
    # own when the game duration elapses (rc=0, "Game over. Saving GIF…");
    # that's expected and means the client will see alive=False on its next
    # /state poll and shut down cleanly. We only escalate if the server
    # exits NON-ZERO (vLLM crash propagated, server traceback, etc.) — in
    # that case there's no point letting the client spin.
    while True:
        client_rc = client_proc.poll()
        if client_rc is not None:
            break
        server_rc = server_proc.poll()
        if server_rc is not None and server_rc != 0:
            print(f"[run] server crashed (rc={server_rc}) — terminating client",
                  file=sys.stderr)
            client_proc.terminate()
            try:
                client_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                client_proc.kill()
            sys.exit(server_rc)
        time.sleep(0.5)

    # Give server a moment to flush its GIF/output, then stop it if still
    # alive (the natural-end path already exited it).
    time.sleep(2)
    if server_proc.poll() is None:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()

    # Surface a non-zero client exit so the bash wrapper can fail loudly.
    if client_rc != 0:
        print(f"[run] client exited with rc={client_rc} — failing", file=sys.stderr)
        sys.exit(client_rc)

    # ------------------------------------------------------------------
    # Read client results and emit final JSON line
    # ------------------------------------------------------------------
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"[run] results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    # Augment with experiment-level metadata and write back so the results
    # file (read by run_experiment.sh) also contains the gif path.
    results["gif"]      = args.gif
    with open(results_path, "w") as f:
        json.dump(results, f)

    line = json.dumps(results)
    print("\n" + "=" * 72)
    print("RESULT:", line)
    print("=" * 72)

    if args.out:
        out_path = Path(args.out)
        with open(out_path, "a") as f:
            f.write(line + "\n")
        print(f"[run] appended to {out_path}")


if __name__ == "__main__":
    main()
