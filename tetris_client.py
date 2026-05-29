#!/usr/bin/env python3
"""Gemma-4 E2B Tetris client — drives the tetris_server.py via vision + tool use.

The model sees a PNG screenshot of the board and calls one of five action tools.
We collect per-call latency and print a summary when the game ends.

Usage:
  python tetris_client.py [--server http://localhost:7777] [--vllm http://localhost:8009]
                          [--model google/gemma-4-E2B-it] [--api-key KEY]
                          [--max-tokens 512] [--think]

Environment variables (override CLI defaults):
  TETRIS_SERVER   Tetris server base URL
  VLLM_BASE       vLLM base URL
  VLLM_MODEL      model name
  VLLM_API_KEY    API key
"""
import argparse
import base64
import json
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(__file__))
from tetris_server import BOARD_W, BOARD_H
import os
import re
import sys
import time
import urllib.request

from openai import OpenAI

# Prompt files live in ./prompts/ as plain text with $VAR$ placeholders.
# Deliberately NOT Jinja or any other code-capable templating engine — the
# optimization process previously discovered it could outsource the entire
# Tetris move-search to a Turing-complete chat template, bypassing the model.
# Keep this renderer dumb on purpose.
_PROMPTS_DIR = _os.path.join(_os.path.dirname(__file__), "prompts")
_VAR_RE = re.compile(r"\$([A-Z][A-Z0-9_]*)\$")


def _render_prompt(name, **vars):
    """Substitute $UPPERCASE_VAR$ tokens. No conditionals, loops, or filters.

    Unknown placeholders raise KeyError so a missing var is loud, not silent.
    Values are str()-cast; callers compute any conditional lines beforehand.
    """
    path = _os.path.join(_PROMPTS_DIR, f"{name}.txt")
    with open(path) as f:
        text = f.read()

    def _sub(m):
        key = m.group(1)
        if key not in vars:
            raise KeyError(f"Unknown placeholder ${key}$ in prompts/{name}.txt")
        return str(vars[key])

    return _VAR_RE.sub(_sub, text).rstrip()

# ---------------------------------------------------------------------------
# Action names — loaded from prompts/action_names.json so individual idea
# branches can rename actions without touching client code.  Keys are the
# internal names the server understands; values are what the model sees.
# ---------------------------------------------------------------------------
import json as _json
_action_names_file = _os.path.join(_PROMPTS_DIR, "action_names.json")
try:
    with open(_action_names_file) as _f:
        _action_names = _json.load(_f)
except (FileNotFoundError, ValueError):
    _action_names = {}
for _k in ("LEFT", "RIGHT", "ROT_LEFT", "ROT_RIGHT", "WAIT", "COMMIT"):
    _action_names.setdefault(_k, _k)
# Actions whose display name is "disabled" are removed from the model's action
# space — they won't appear in the tool enum or be sent to the server.
_action_names = {k: v for k, v in _action_names.items() if v != "disabled"}
# Reverse map: model-visible name → internal name (for send_action)
_name_to_internal = {v: k for k, v in _action_names.items()}

# Convenience dict passed to every _render_prompt call so templates can
# reference $ACTION_LEFT$, $ACTION_COMMIT$, etc.
_ACTION_VARS = {f"ACTION_{k}": v for k, v in _action_names.items()}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SERVER = os.environ.get("TETRIS_SERVER", "http://localhost:7777")
DEFAULT_VLLM   = os.environ.get("VLLM_BASE",    "http://localhost:8009/v1")
DEFAULT_MODEL  = os.environ.get("VLLM_MODEL",   "google/gemma-4-E2B-it")
DEFAULT_APIKEY = os.environ.get("VLLM_API_KEY",  "")

# ACTIONS uses the model-visible names from action_names.json, excluding disabled ones.
_ALL_ACTION_KEYS = ("LEFT", "RIGHT", "ROT_RIGHT", "ROT_LEFT", "COMMIT", "WAIT")
ACTIONS = [_action_names[k] for k in _ALL_ACTION_KEYS if k in _action_names]
SERVER_ACTIONS = ACTIONS
MAX_MOVES_PER_CALL = 30

# ---------------------------------------------------------------------------
# Fallback regex — applied to the reasoning trace when no tool call is returned.
# ---------------------------------------------------------------------------
_FALLBACK_REGEX_FILE = _os.path.join(_PROMPTS_DIR, "fallback.regex")

def _load_fallback_spec():
    """Return (compiled_pattern, fallback_internal_name, mode) from prompts/fallback.regex.

    mode is 'last' (default) or 'all'.
    """
    pattern_str = None
    fallback = "WAIT"
    mode = "last"
    try:
        with open(_FALLBACK_REGEX_FILE) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line.startswith("#") or not _line:
                    continue
                if "=" in _line:
                    _k, _, _v = _line.partition("=")
                    _k, _v = _k.strip(), _v.strip()
                    if _k == "pattern":
                        pattern_str = _v
                    elif _k == "fallback":
                        fallback = _v
                    elif _k == "mode":
                        mode = _v.lower()
    except OSError:
        pass
    compiled = None
    if pattern_str:
        try:
            compiled = re.compile(pattern_str, re.IGNORECASE)
        except re.error:
            pass
    return compiled, fallback, mode


def _apply_fallback_regex(reasoning_trace):
    """Extract moves from reasoning when the model returned no valid tool call.

    Matches the pattern from prompts/fallback.regex against the reasoning text.
    Group 1 may capture a single action name OR a comma/space-separated list —
    both are split and resolved to internal action names.
    Returns the fallback action when nothing matches.
    """
    compiled, fallback_internal, mode = _load_fallback_spec()
    moves = []
    if compiled and reasoning_trace:
        all_matches = list(compiled.finditer(reasoning_trace))
        # mode=last: use only the final match (model's final decision).
        # mode=all:  concatenate all matches in order.
        matches = all_matches[-1:] if (mode == "last" and all_matches) else all_matches
        for m in matches:
            raw = (m.group(1) if m.lastindex else m.group(0)).strip()
            # Split on comma or whitespace to support both list formats.
            tokens = re.split(r'[\s,]+', raw)
            for token in tokens:
                name = token.strip().upper()
                if not name:
                    continue
                if name in _ALL_ACTION_KEYS:
                    moves.append(name)
                else:
                    internal = _name_to_internal.get(name)
                    if internal:
                        moves.append(internal)
        if moves:
            return moves[:MAX_MOVES_PER_CALL]
    return [fallback_internal]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tetris_action",
            "description": _render_prompt("tool_description", **_ACTION_VARS),
            "parameters": {
                "type": "object",
                "properties": {
                    "moves": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_MOVES_PER_CALL,
                        "items": {
                            "type": "string",
                            "enum": ACTIONS,
                        },
                        "description": (
                            f"Ordered list of inputs to apply. At most {MAX_MOVES_PER_CALL} "
                            f"entries. Use {_action_names['COMMIT']} to lock the current piece "
                            "mid-list and continue planning for the next piece (see 'Next' field)."
                        ),
                    }
                },
                "required": ["moves"],
            },
        },
    }
]

SYSTEM_PROMPT = _render_prompt(
    "system",
    BOARD_W=BOARD_W,
    BOARD_H=BOARD_H,
    BOARD_W_LAST=BOARD_W - 1,
    BOARD_H_LAST=BOARD_H - 1,
    BOARD_H_LAST_PAD=f"{BOARD_H - 1:02d}",
    MAX_MOVES_PER_CALL=MAX_MOVES_PER_CALL,
    **_ACTION_VARS,
)


# ---------------------------------------------------------------------------
# Board-derived features (exposed to the chat template via chat_template_kwargs)
# ---------------------------------------------------------------------------

def _parse_board(board_str):
    """Parse the server's ASCII board into a 2D array of 0/1.

    1 marks locked blocks (█); falling piece (▓) and ghost (░) cells stay 0
    so heights reflect the settled stack, not transient piece positions.
    """
    if not board_str:
        return []
    raw = []
    for line in board_str.split("\n"):
        if not line.startswith("R"):
            continue  # skip the column ruler "    0123456789"
        # Row format: "R<idx> <cells>". Split off the header.
        _, _, cells = line.partition(" ")
        raw.append([1 if c == "█" else 0 for c in cells])
    return raw


def _compute_heights(raw):
    """Per-column stack height: rows above the lowest empty cell in each col."""
    if not raw:
        return []
    h, w = len(raw), len(raw[0])
    heights = [0] * w
    for c in range(w):
        for r in range(h):
            if raw[r][c]:
                heights[c] = h - r
                break
    return heights


def _piece_bbox(piece_cells):
    """(width, height) bounding box of the absolute piece cells."""
    if not piece_cells:
        return 0, 0
    xs = [c[0] for c in piece_cells]
    ys = [c[1] for c in piece_cells]
    return max(xs) - min(xs) + 1, max(ys) - min(ys) + 1


def _compute_holes(raw, heights):
    """Empty cells located beneath the top of their column.

    A "hole" is a buried gap — empty under a filled cell — and is the
    standard Tetris stack-quality metric (lower is better).
    """
    if not raw:
        return 0
    h, w = len(raw), len(raw[0])
    total = 0
    for c in range(w):
        col_top = h - heights[c]  # row index of the topmost filled cell
        for r in range(col_top + 1, h):
            if not raw[r][c]:
                total += 1
    return total


def _compute_bumpiness(heights):
    """Sum of |Δheight| between adjacent columns. Lower = flatter stack."""
    return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))


def _compute_near_full(raw, min_filled=7):
    """Rows that are at least `min_filled`/10 filled with locked blocks.

    Returns a list of {"row": r, "filled": n, "gaps": [c, ...]} sorted by row,
    so user.jinja can render them as line-clear targets.
    """
    if not raw:
        return []
    out = []
    for r, row in enumerate(raw):
        gaps = [c for c, cell in enumerate(row) if not cell]
        filled = len(row) - len(gaps)
        if filled >= min_filled:
            out.append({"row": r, "filled": filled, "gaps": gaps})
    return out


# ---------------------------------------------------------------------------
# HTTP helpers (no third-party deps)
# ---------------------------------------------------------------------------

def _http_get_bytes(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _http_post_json(url, body, headers=None, timeout=60):
    data = json.dumps(body).encode()
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _http_get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------------------
# Tetris server helpers
# ---------------------------------------------------------------------------

def get_screen_b64(server):
    png = _http_get_bytes(f"{server}/getscreen")
    return base64.b64encode(png).decode()


def mark_snapshot(server, latency_ms=None):
    """Tell the server to flag the current frame as model-seen (for the GIF
    outline) without fetching the PNG."""
    url = f"{server}/marksnap"
    if latency_ms is not None:
        url += f"?latency_ms={int(latency_ms)}"
    _http_get_json(url)


def send_action(server, move, latency_ms=None):
    # Model-visible name → internal name → server API name
    internal = _name_to_internal.get(move, move)
    server_move = "DROP" if internal == "COMMIT" else internal
    url = f"{server}/action?move={server_move}"
    if latency_ms is not None:
        url += f"&latency_ms={int(latency_ms)}"
    return _http_get_json(url)


def get_state(server):
    return _http_get_json(f"{server}/state")


# ---------------------------------------------------------------------------
# Anti-cheat: chat template validation
# ---------------------------------------------------------------------------

_CHAT_TEMPLATE_PATH = _os.path.join(_os.path.dirname(__file__),
                                    "templates", "tool_chat_template_gemma4.jinja")

def _anticheat_check(messages, tools, system_text, user_text, template_kwargs):
    """Render the chat template locally and verify it hasn't injected move names.

    Two checks:
    1. The static prompt texts (system, user, tool descriptions) must appear
       verbatim in the rendered output — ensures the template didn't suppress them.
    2. After removing all known static content, the remainder must not contain
       any action display name — ensures the template didn't inject move hints.
    """
    try:
        import jinja2
    except ImportError:
        return  # jinja2 not available — skip check

    try:
        with open(_CHAT_TEMPLATE_PATH) as f:
            template_src = f.read()
    except OSError:
        return  # template file not found — skip check

    # Build a minimal Jinja env matching vLLM's setup.
    env = jinja2.Environment(loader=jinja2.BaseLoader(), keep_trailing_newline=True)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    env.filters["tojson"] = _json.dumps

    try:
        rendered = env.from_string(template_src).render(
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
            **template_kwargs,
        )
    except Exception as e:
        # Template rendering failed — flag it so the agent investigates.
        print(f"ANTI-CHEAT ERROR: chat template failed to render: {e}", file=sys.stderr)
        sys.exit(2)

    # ── Check 1: static texts must be present verbatim ──────────────────────
    static_texts = [system_text]
    if isinstance(messages[-1]["content"], str):
        static_texts.append(messages[-1]["content"])
    else:
        # Multimodal: check the text part
        for part in messages[-1]["content"]:
            if part.get("type") == "text":
                static_texts.append(part["text"])

    for text in static_texts:
        if text and text not in rendered:
            print(
                "ANTI-CHEAT ERROR: chat template suppressed or altered a required "
                f"prompt (first 60 chars: {text[:60]!r}). "
                "The template MUST include system and user prompts verbatim.",
                file=sys.stderr,
            )
            sys.exit(2)

    # ── Check 2: strip static content, scan remainder for action names ───────
    remainder = rendered
    for text in static_texts:
        if text:
            remainder = remainder.replace(text, "")

    # Also strip the tool schema JSON (contains action names in the enum).
    # Remove each action name that appears as part of the tools JSON literal.
    tool_desc = _render_prompt("tool_description", **_ACTION_VARS)
    remainder = remainder.replace(tool_desc, "")
    # Strip the tools JSON blob (rendered by the template from the tools parameter).
    tools_json = _json.dumps(tools, ensure_ascii=False)
    remainder = remainder.replace(tools_json, "")
    # Strip string values from the tools structure, except bare action names
    # (we need those to survive so we can detect injection in the remainder).
    # The template serializes the tool schema in various forms; strip them all.
    action_set = set(ACTIONS)
    def _extract_strings(obj):
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from _extract_strings(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from _extract_strings(v)
    for s in _extract_strings(tools):
        if s and s not in action_set:
            remainder = remainder.replace(s, "")
    # Strip action names in all quote/token forms the template may produce —
    # these are the legitimate schema occurrences; bare un-quoted occurrences
    # (injected hints) are left in place for detection below.
    for name in ACTIONS:
        remainder = remainder.replace(f'"{name}"', "")
        remainder = remainder.replace(f"<|\"|>{name}<|\"|>", "")

    injected = [name for name in ACTIONS if name in remainder]
    if injected:
        print(
            f"ANTI-CHEAT ERROR: chat template injected action name(s) {injected} "
            "outside the static prompts. This is not allowed — the template must "
            "not pre-compute or hint at moves.",
            file=sys.stderr,
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

def call_model(screen_b64, vllm_url, model, api_key, max_tokens, think,
               max_soft_tokens=1120, no_image=False,
               last_latency_ms=None, state=None,
               temperature=0.0,
               server=None, piece_id_at_call=None,
               state_poll_interval=0.2,
               carryover_moves=None):
    thinking_kwargs = {"enable_thinking": think, "force_thinking": think}

    # Board-derived features. We compute once and surface them in two places:
    # (a) user.jinja, so the per-turn observation lives next to the board it
    #     describes, and
    # (b) chat_template_kwargs, so existing chat templates that read `heights`
    #     / `raw_board` directly keep working.
    board_str = state.get("board") if state else None
    raw_board = _parse_board(board_str)
    heights = _compute_heights(raw_board)
    near_full = _compute_near_full(raw_board)

    # Piece / shape / landing info — forwarded from the server's state_dict.
    piece_cells = state.get("piece_cells", []) if state else []
    piece_width, piece_height = _piece_bbox(piece_cells)

    # Stack-quality metrics derived from the board.
    holes = _compute_holes(raw_board, heights)
    bumpiness = _compute_bumpiness(heights)
    max_height = max(heights) if heights else 0

    extra_body = {
        "skip_special_tokens": False,
        "top_k": 64,
        "chat_template_kwargs": {
            **thinking_kwargs,
            "heights":      heights,
            "raw_board":    raw_board,
            "piece":        state.get("piece") if state else None,
            "next_piece":   state.get("next_piece") if state else None,
            "piece_x":      state.get("piece_x") if state else None,
            "piece_y":      state.get("piece_y") if state else None,
            "piece_cells":  piece_cells,
            "ghost_y":      state.get("ghost_y") if state else None,
            "piece_width":  piece_width,
            "piece_height": piece_height,
            "holes":        holes,
            "bumpiness":    bumpiness,
            "max_height":   max_height,
        },
    }
    if not no_image:
        extra_body["mm_processor_kwargs"] = {"max_soft_tokens": max_soft_tokens}

    # Build each optional block as plain text. The user prompt is a flat
    # template with $VAR$ slots — no conditional logic on the prompt side.
    piece = state.get("piece") if state else None
    piece_line = (
        f"\nPiece: {piece} at col {state.get('piece_x')}, row {state.get('piece_y')}. "
        f"Next: {state.get('next_piece')}. Tick: {state.get('tick')}s. "
        f"Score: {state.get('score', 0)}. Lines: {state.get('lines', 0)}."
        if piece else ""
    )
    board_str_val = state.get("board") if state else None
    board_block = (
        "\n\nBoard (▓ = falling piece, ░ = ghost / drop position, █ = locked, "
        f". = empty; columns 0–9):\n{board_str_val}"
        if board_str_val else ""
    )
    heights_line = (
        "\n\nHeights[C0..C9] (locked cells per column, computed from the "
        f"board above): [{', '.join(str(h) for h in heights)}]"
        if heights and len(heights) == 10 else ""
    )
    if near_full:
        _parts = [
            f"R{r['row']}={r['filled']}/10 gaps=C{',C'.join(str(g) for g in r['gaps'])}"
            for r in near_full
        ]
        near_full_line = "\nNear-full rows (≥7/10 locked): " + "; ".join(_parts)
    else:
        near_full_line = ""
    co = list(carryover_moves) if carryover_moves else []
    carryover_block = (
        f"\n\nNote: you previously planned across pieces and the actions "
        f"[{', '.join(co)}] were already AUTO-APPLIED to the current piece "
        "(they followed your COMMIT last turn). The piece's current "
        "position reflects them — plan from here."
        if co else ""
    )
    latency_line = (
        f"\n Your last response took {last_latency_ms:.0f}ms."
        if last_latency_ms is not None else ""
    )

    user_text = _render_prompt(
        "user",
        MAX_MOVES_PER_CALL=MAX_MOVES_PER_CALL,
        PIECE_LINE=piece_line,
        BOARD_BLOCK=board_block,
        HEIGHTS_LINE=heights_line,
        NEAR_FULL_LINE=near_full_line,
        CARRYOVER_BLOCK=carryover_block,
        LATENCY_LINE=latency_line,
        **_ACTION_VARS,
    )

    if no_image or screen_b64 is None:
        user_content = user_text
    else:
        user_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screen_b64}"}},
        ]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    _anticheat_check(messages, TOOLS, SYSTEM_PROMPT, user_text, extra_body.get("chat_template_kwargs", {}))

    client = OpenAI(base_url=vllm_url, api_key=api_key or "none", timeout=300.0)

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="required",
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        extra_body=extra_body,
    )
    latency = time.perf_counter() - t0

    msg = response.choices[0].message
    finish_reason = response.choices[0].finish_reason or ""
    usage = response.usage
    prompt_tokens     = usage.prompt_tokens     if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    reasoning_trace = ""
    # vLLM 0.22.0 exposes the thinking trace as msg.reasoning (non-standard field,
    # lands in model_extra via the OpenAI SDK). Older builds used reasoning_content.
    if msg.model_extra:
        reasoning_trace = (msg.model_extra.get("reasoning")
                           or msg.model_extra.get("reasoning_content")
                           or "")
    if not reasoning_trace:
        reasoning_trace = msg.content or ""

    moves = None
    if msg.tool_calls:
        try:
            args = json.loads(msg.tool_calls[0].function.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        candidates = args.get("moves")
        if isinstance(candidates, list) and candidates:
            cleaned = [c for c in candidates if c in ACTIONS][:MAX_MOVES_PER_CALL]
            if cleaned:
                moves = cleaned

    invalid = moves is None
    if invalid:
        # No valid tool call — try to extract moves from the reasoning trace
        # using the pattern in prompts/fallback.regex.
        moves = _apply_fallback_regex(reasoning_trace)

    input_chars = len(user_text)
    output_chars = len(reasoning_trace) + len(msg.tool_calls[0].function.arguments if msg.tool_calls else "")
    return moves, latency, {
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "input_chars":       input_chars,
        "reasoning_chars":   len(reasoning_trace),
        "output_chars":      output_chars,
        "reasoning":         reasoning_trace,
        "finish_reason":     finish_reason,
        "invalid_tool_call": invalid,
        "aborted":           False,
        "user_text":         user_text,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gemma-4 E2B Tetris agent")
    parser.add_argument("--server",     default=DEFAULT_SERVER, help="Tetris server base URL")
    parser.add_argument("--vllm",       default=DEFAULT_VLLM,   help="vLLM base URL (with /v1)")
    parser.add_argument("--model",      default=DEFAULT_MODEL,  help="Model name")
    parser.add_argument("--api-key",    default=DEFAULT_APIKEY, help="API key")
    parser.add_argument("--max-tokens",      type=int, default=512,   help="Max tokens per call (use >=2048 with --think)")
    parser.add_argument("--think",           action="store_true",     help="Enable force_thinking — requires larger --max-tokens (>=2048)")
    parser.add_argument("--max-soft-tokens", type=int, default=1120,  help="Max image soft tokens (lower=smaller image, faster inference)")
    parser.add_argument("--image",           action="store_true",     help="Send PNG screenshot alongside the ASCII board (default: text-only, much faster prefill)")
    parser.add_argument("--temperature",     type=float, default=0.0, help="Sampling temperature (0.0=greedy, 0.3-0.7 for diverse actions)")
    parser.add_argument("--gif-save",    default="tetris_replay.gif",    help="Path to save GIF after game")
    parser.add_argument("--results-out", default="tetris_results.json",  help="Path to write JSON results")
    parser.add_argument("--log",         default="tetris_thinking.log",  help="Path to write per-turn thinking log")
    args = parser.parse_args()

    print(f"Tetris agent starting")
    print(f"  server : {args.server}")
    print(f"  vllm   : {args.vllm}")
    print(f"  model  : {args.model}")

    # Wait for server to be ready
    for attempt in range(30):
        try:
            state = get_state(args.server)
            print(f"  server ready — {state}")
            break
        except Exception as e:
            if attempt == 29:
                print(f"Server not reachable after 30s: {e}", file=sys.stderr)
                sys.exit(1)
            time.sleep(1)

    latencies = []
    prompt_tokens_list = []
    completion_tokens_list = []
    input_chars_list = []
    reasoning_chars_list = []
    output_chars_list = []
    invalid_tool_calls = 0
    stale_responses = 0
    total_moves_applied = 0
    turn = 0
    last_latency_ms = None
    state = {}
    _wall_start = time.time()  # wall-clock start for pct_complete in wait-for-turn mode
    # Set to time.time() the first time the opening countdown clears.  The
    # server sets its start_time to countdown_end (not client launch), so
    # t_remaining must be measured from the same reference or it reaches 0
    # ~4 s before the server actually ends the game.
    _game_wall_start = None
    # Moves the previous LLM call planned for the NEW piece after a COMMIT
    # in its list. Those have already been applied by the apply loop; we tell
    # the next LLM call about them so it knows the piece is pre-positioned.
    carryover_moves = []

    log_f = open(args.log, "w", buffering=1)  # line-buffered so it's readable live
    def log(text):
        # Mirror the per-turn detail to both the side-file (tetris_thinking.log)
        # AND stdout, so the labapi script.log captures the full reasoning trace.
        log_f.write(text + "\n")
        print(text, flush=True)

    # Running highs from every state we observe. These are the source of truth
    # for the final results JSON because the server shuts down moments after
    # the run-duration expires, so the post-loop get_state() often races and
    # returns {}. The server-side accumulators are monotonic, so tracking the
    # max of what we saw is equivalent to "the final value at end of run."
    observed = {
        "max_score":   0,
        "total_score": 0,
        "max_lines":   0,
        "total_lines": 0,
        "restarts":    0,
        "level":       1,
    }

    _consecutive_state_errors = 0
    while True:
        # Refresh state — normally carried from previous send_action response,
        # but fetch explicitly on first turn or after an error left state empty.
        if not state:
            try:
                state = get_state(args.server)
                _consecutive_state_errors = 0
            except Exception as e:
                _consecutive_state_errors += 1
                print(f"[turn {turn}] state fetch error: {e}", file=sys.stderr)
                if _consecutive_state_errors >= 20:
                    print(f"[turn {turn}] server unreachable after 20 tries — game ended", file=sys.stderr)
                    break
                time.sleep(0.1)
                continue

        if not state.get("alive", True):
            print(f"\nGame over after {turn} turns. (alive=False)")
            break

        # Detect timer reset (piece landing causes t_remaining jump — don't exit)
        t_rem = state.get("duration", 600) - state.get("elapsed", 0)
        if t_rem < 0:
            print(f"[turn {turn}] timer anomaly t_rem={t_rem:.1f}, continuing", file=sys.stderr)

        # If an intro countdown is active, capture frames while waiting so the
        # GIF shows the "NEW GAME / 3 / 2 / 1" beat (in wait-for-turn mode the
        # gravity thread doesn't snapshot, so the client must do it here).
        while state.get("countdown_active"):
            try:
                mark_snapshot(args.server)
            except Exception:
                pass
            time.sleep(min(0.2, max(0.05, state.get("countdown_remaining", 0.2))))
            try:
                state = get_state(args.server)
            except Exception:
                break
        if not state.get("alive", True):
            print(f"\nGame over during countdown wait.")
            break
        # Latch the wall-clock reference the first time the opening countdown
        # clears — the server's start_time is set to countdown_end, so
        # t_remaining must be anchored here, not at client launch.
        if _game_wall_start is None:
            _game_wall_start = time.time()

        piece_id_at_call = state.get("piece_id")

        # Mark this moment in the frame stream so the GIF gets the 1px outline
        # at each LLM-decision point. In --image mode this happens implicitly
        # via /getscreen; in text-only mode we mark via /marksnap.
        screen_b64 = None
        if args.image:
            try:
                screen_b64 = get_screen_b64(args.server)
            except Exception as e:
                print(f"[turn {turn}] screenshot error: {e}", file=sys.stderr)
                state = {}
                continue
        else:
            try:
                mark_snapshot(args.server, latency_ms=last_latency_ms)
            except Exception:
                pass  # best-effort; an outline is cosmetic

        # Ask model — one call per piece, returns an ordered move list.
        # Streams the response and aborts mid-flight if the server reports a
        # new piece_id (gravity bottomed the current piece while we were
        # thinking) so we don't waste compute on a stale plan.
        try:
            moves, latency, call_stats = call_model(
                screen_b64,
                args.vllm,
                args.model,
                args.api_key,
                args.max_tokens,
                args.think,
                max_soft_tokens=args.max_soft_tokens,
                no_image=not args.image,
                last_latency_ms=last_latency_ms,
                state=state,
                temperature=args.temperature,
                server=args.server,
                piece_id_at_call=piece_id_at_call,
                carryover_moves=carryover_moves,
            )
        except Exception as e:
            print(f"[turn {turn}] FATAL: model error — aborting so the agent investigates: {e}",
                  file=sys.stderr)
            sys.exit(1)

        last_latency_ms = latency * 1000
        latencies.append(latency)
        prompt_tokens_list.append(call_stats["prompt_tokens"])
        completion_tokens_list.append(call_stats["completion_tokens"])
        input_chars_list.append(call_stats.get("input_chars", 0))
        reasoning_chars_list.append(call_stats["reasoning_chars"])
        output_chars_list.append(call_stats.get("output_chars", 0))
        if call_stats.get("invalid_tool_call"):
            invalid_tool_calls += 1
            print(f"[turn {turn}] WARN: model returned invalid tool call — "
                  "fell back to regex extraction from reasoning trace.",
                  file=sys.stderr)
        turn += 1

        # ── stale-response handling ────────────────────────────────────
        # call_model() streams and aborts itself when the server's piece_id
        # advances mid-call (gravity bottomed the piece). When that happens,
        # moves is None and call_stats["aborted"] is True — discard, refetch
        # state, and loop to re-prompt for the new piece.
        if call_stats.get("aborted"):
            stale_responses += 1
            try:
                live_state = get_state(args.server)
            except Exception:
                live_state = state
            log(f"{'='*72}")
            log(f"ABORTED: piece_id {piece_id_at_call} changed during LLM call "
                f"({latency*1000:.0f}ms, after ~{call_stats.get('completion_tokens', 0)} completion tok). "
                f"Stream closed mid-flight; re-prompting on the new piece.")
            print(f"[piece {turn:4d}] ABORTED — piece changed mid-stream "
                  f"({latency*1000:.0f}ms, {call_stats.get('completion_tokens', 0)} tok generated)",
                  flush=True)
            state = live_state
            # The previous turn's carryover_moves applied to a piece that's
            # now gone. They are NOT pre-applied to the new piece — clear.
            carryover_moves = []
            continue

        # ── log entry ──────────────────────────────────────────────────
        log(f"{'='*72}")
        log(f"TURN {turn:4d}  piece={state.get('piece','?')}  "
            f"col={state.get('piece_x','?')}  row={state.get('piece_y','?')}  "
            f"score={state.get('score',0)}  lines={state.get('lines',0)}  "
            f"tick={state.get('tick','?')}s  latency={latency*1000:.0f}ms  "
            f"finish={call_stats.get('finish_reason','?')}  "
            f"prompt_tok={call_stats.get('prompt_tokens',0)}  "
            f"completion_tok={call_stats.get('completion_tokens',0)}")
        user_text = call_stats.get("user_text", "")
        if user_text:
            log("INPUT (user message):")
            log(user_text)
        reasoning = call_stats.get("reasoning", "").strip()
        if reasoning:
            log("THINKING:")
            log(reasoning)
        else:
            log("THINKING: (none)")
        log(f"MOVES: {', '.join(moves)}")

        # ── apply the batch ────────────────────────────────────────────
        # Server enforces one player action per 100 ms (bypassed in
        # wait-for-turn mode). Moves apply to whichever piece is currently
        # falling. COMMIT (server alias: DROP) locks the current piece and
        # spawns the next — subsequent moves in this batch then apply to the
        # freshly spawned piece. WAIT fires one gravity tick; if the piece
        # was at the bottom it also locks and spawns the next (same cross-
        # piece planning semantics as COMMIT). An UNEXPECTED piece_id change
        # on any other move means gravity bottomed the piece without us
        # issuing COMMIT/WAIT, so we abandon the remaining planned moves.
        applied = 0
        last_applied_move = None
        expected_piece_id = piece_id_at_call
        last_piece_advance_idx = -1   # index of last COMMIT/WAIT that locked a piece
        _batch_latency = int(latency * 1000) if latency else None
        for mv in moves:
            try:
                # Send latency with the first action so the GIF stat panel
                # updates before the piece moves are rendered.
                state = send_action(args.server, mv,
                                    latency_ms=_batch_latency if applied == 0 else None)
            except Exception as e:
                print(f"[turn {turn}] action error ({mv}): {e}", file=sys.stderr)
                state = {}
                break
            applied += 1
            last_applied_move = mv
            new_piece_id = state.get("piece_id")
            if new_piece_id != expected_piece_id:
                # Normalize to internal name: moves may contain display names
                # (e.g. "HARD_DROP") or internal names (e.g. "COMMIT").
                _internal_mv = _name_to_internal.get(mv, mv)
                if _internal_mv in ("COMMIT", "WAIT"):
                    expected_piece_id = new_piece_id
                    last_piece_advance_idx = applied - 1
                else:
                    break

        applied_moves = moves[:applied]
        saw_piece_advance = last_piece_advance_idx >= 0

        # Carryover: moves applied AFTER the last piece-locking event landed
        # on a freshly spawned piece — surface them in the NEXT LLM call so
        # the model knows the piece is already pre-positioned.
        carryover_moves = applied_moves[last_piece_advance_idx + 1:] if saw_piece_advance else []

        total_moves_applied += applied
        log(f"APPLIED: {applied}/{len(moves)} moves "
            f"(last={last_applied_move or 'none'}; "
            f"carryover_to_next={carryover_moves or 'none'})")

        max_score   = state.get("max_score", 0)
        total_score = state.get("total_score", 0)
        elapsed     = state.get("elapsed", "?")
        restarts    = state.get("restarts", 0)
        tick        = state.get("tick", "?")
        # Track running highs so the end-of-run summary survives the server
        # shutting down before we can call get_state() one last time.
        for _k, _v in (
            ("max_score",   max_score),
            ("total_score", total_score),
            ("max_lines",   state.get("max_lines", 0)),
            ("total_lines", state.get("total_lines", 0)),
            ("restarts",    restarts),
            ("level",       state.get("level", 1)),
        ):
            if isinstance(_v, int) and _v > observed[_k]:
                observed[_k] = _v
        try:
            _duration = state.get("duration", 600) or 600
            _ref = _game_wall_start if _game_wall_start is not None else _wall_start
            remaining = max(0, _duration - (time.time() - _ref))
        except (TypeError, ValueError):
            remaining = 0
        print(
            f"[piece {turn:4d}] moves={applied}/{len(moves):<2} "
            f"max_score={max_score:<6} total_score={total_score:<6} restarts={restarts}"
            f"  tick={tick}s  latency={latency*1000:.0f}ms  t_remaining={remaining:.0f}s",
            flush=True,
        )

        # Write a labapi-compatible progress file if the wrapper exported one.
        # The wrapper's background watcher rsyncs this to the API whenever it
        # changes, giving the dashboard live pct_complete + score updates.
        _progress_path = os.environ.get("THE_LAB_PROGRESS")
        if _progress_path:
            try:
                _duration = state.get("duration", 600) or 600
                _elapsed = float(elapsed) if isinstance(elapsed, (int, float)) else 0.0
                # Use wall-clock time for pct_complete — in wait-for-turn mode,
                # state["elapsed"] is virtual time (WAIT moves only) and stays
                # near 0 even though the wall-clock deadline is approaching.
                # Anchor to _game_wall_start (post-countdown) to match the
                # server's actual start_time reference.
                _ref = _game_wall_start if _game_wall_start is not None else _wall_start
                _wall_elapsed = time.time() - _ref
                _pct = int(100 * _wall_elapsed / _duration) if _duration else 0
                _pct = max(0, min(100, _pct))
                with open(_progress_path, "w") as _pf:
                    json.dump({
                        "pct_complete":       _pct,
                        "elapsed_s":          round(_elapsed, 1),
                        "duration_s":         _duration,
                        "model":              args.model,
                        "turns":              turn,
                        "moves_applied":      total_moves_applied,
                        "max_score":          max_score if isinstance(max_score, int) else 0,
                        "total_score":        total_score if isinstance(total_score, int) else 0,
                        "max_lines":          state.get("max_lines", 0),
                        "total_lines":        state.get("total_lines", 0),
                        "restarts":           restarts,
                        "stale_responses":    stale_responses,
                        "invalid_tool_calls": invalid_tool_calls,
                        "latency_p50_ms":     int(latencies[len(latencies)//2] * 1000) if latencies else 0,
                        "latency_last_ms":    int(latency * 1000),
                    }, _pf)
            except Exception:
                pass  # best-effort — progress reporting must not break the run

    # ---------------------------------------------------------------------------
    # Summary + JSON results
    # ---------------------------------------------------------------------------
    def _percentile(sorted_vals, p):
        if not sorted_vals:
            return 0
        idx = min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)
        return sorted_vals[idx]

    def _mean(vals):
        return int(sum(vals) / len(vals)) if vals else 0

    # Try one more state fetch, but fall back to the running highs from the
    # main loop — the server shuts down ~immediately after the duration
    # expires, so this call usually races and returns {}.
    final_state = {}
    try:
        final_state = get_state(args.server)
    except Exception:
        pass
    # Take the max across both sources for safety (server-side accumulators are
    # monotonic, so this can't underreport).
    def _best(key, default=0):
        a = observed.get(key, default)
        b = final_state.get(key, default)
        if isinstance(a, int) and isinstance(b, int):
            return max(a, b)
        return b if isinstance(b, int) else a

    lat_ms = sorted(l * 1000 for l in latencies)
    rc_sorted = sorted(reasoning_chars_list)

    # Mirror the progress file's run-time fields so the post-run results JSON
    # is a strict superset of script.progress (script.output.md reads from
    # here and re-renders the same keys).
    _final_elapsed = 0.0
    _final_duration = 0
    try:
        _final_elapsed = float(final_state.get("elapsed", 0) or 0)
        _final_duration = int(final_state.get("duration", 0) or 0)
    except (TypeError, ValueError):
        pass
    results = {
        "model":                  args.model,
        "think":                  args.think,
        "elapsed_s":              round(_final_elapsed, 1),
        "duration_s":             _final_duration,
        "max_score":              _best("max_score"),
        "total_score":            _best("total_score"),
        "max_lines":              _best("max_lines"),
        "total_lines":            _best("total_lines"),
        "level":                  _best("level", 1),
        "restarts":               _best("restarts"),
        "turns":                  turn,
        "moves_applied":          total_moves_applied,
        "invalid_tool_calls":     invalid_tool_calls,
        "stale_responses":        stale_responses,
        "latency_last_ms":        int(latencies[-1] * 1000) if latencies else 0,
        "latency_mean_ms":        _mean(lat_ms),
        "latency_p50_ms":         int(_percentile(lat_ms, 50)),
        "latency_p90_ms":         int(_percentile(lat_ms, 90)),
        "latency_p95_ms":         int(_percentile(lat_ms, 95)),
        "latency_p99_ms":         int(_percentile(lat_ms, 99)),
        "latency_min_ms":         int(lat_ms[0])  if lat_ms else 0,
        "latency_max_ms":         int(lat_ms[-1]) if lat_ms else 0,
        "prompt_tokens_mean":     _mean(prompt_tokens_list),
        "completion_tokens_mean": _mean(completion_tokens_list),
        "input_chars_mean":       _mean(input_chars_list),
        "reasoning_chars_mean":   _mean(rc_sorted),
        "reasoning_chars_p95":    int(_percentile(rc_sorted, 95)),
        "output_chars_mean":      _mean(output_chars_list),
    }

    print("\n" + "=" * 60)
    print(f"Pieces (LLM calls): {turn}  moves_applied: {total_moves_applied}  max_score: {results['max_score']}  total_score: {results['total_score']}  max_lines: {results['max_lines']}  total_lines: {results['total_lines']}  restarts: {results['restarts']}  invalid_tool_calls: {invalid_tool_calls}  stale_responses: {stale_responses}")
    if lat_ms:
        print(f"Latency (ms): mean={results['latency_mean_ms']}  p50={results['latency_p50_ms']}  p90={results['latency_p90_ms']}  p95={results['latency_p95_ms']}  p99={results['latency_p99_ms']}")
        print(f"Tokens:  prompt_mean={results['prompt_tokens_mean']}  completion_mean={results['completion_tokens_mean']}")
        print(f"Chars (mean): input={results['input_chars_mean']}  reasoning={results['reasoning_chars_mean']}  output={results['output_chars_mean']}")

    with open(args.results_out, "w") as f:
        json.dump(results, f)
    print(f"Results written to {args.results_out}")

    # Fetch and save GIF — long games (500+ turns) can take many minutes to
    # encode in pure Python, so give it generous headroom.
    print(f"Fetching replay GIF…")
    try:
        gif_bytes = _http_get_bytes(f"{args.server}/gif", timeout=900)
        with open(args.gif_save, "wb") as f:
            f.write(gif_bytes)
        print(f"Saved {args.gif_save} ({len(gif_bytes)//1024} KB)")
    except Exception as e:
        print(f"Could not fetch GIF: {e}")

    print("=" * 60)
    log_f.close()
    print(f"Thinking log: {args.log}")


if __name__ == "__main__":
    main()
