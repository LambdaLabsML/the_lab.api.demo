#!/usr/bin/env python3
"""Tetris game server with seeded deterministic physics.

Endpoints:
  GET /getscreen          → PNG image of current board
  POST /action?move=LEFT|RIGHT|FALL|ROT_RIGHT|ROT_LEFT → apply move, return JSON state
  GET /state              → JSON game state (score, level, alive, elapsed)
  GET /gif                → animated GIF of the full game (only after game ends)

Run:
  python tetris_server.py [--port 0] [--seed 42]   # --port 0 = OS picks a free port
"""
import argparse
import base64
import io
import json
import math
import random
import struct
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# Colour palette (GitHub dark theme)
# ---------------------------------------------------------------------------
C_BG       = (13,  17,  23)   # #0d1117
C_TEXT     = (201, 209, 217)  # #c9d1d9
C_MUTED    = (139, 148, 158)  # #8b949e
C_FAINT    = (72,  79,  88)   # #484f58  — empty cells
C_BORDER   = (48,  54,  61)   # #30363d
C_BLUE     = (88,  166, 255)  # #58a6ff
C_PURPLE   = (188, 140, 255)  # #bc8cff
C_GOLD     = (242, 204, 96)   # #f2cc60
C_GREEN    = (63,  185, 80)   # #3fb950
C_RED      = (248, 81,  73)   # #f85149
C_CYAN     = (56,  189, 248)  # #38bdf8  — I-piece (not in palette, close to blue)
C_ORANGE   = (251, 146, 60)   # #fb923c  — L-piece (not in palette, warm accent)

# ---------------------------------------------------------------------------
# Tetromino definitions — (cells relative to pivot, colour RGB)
# ---------------------------------------------------------------------------
TETROMINOES = {
    "I": {"cells": [(0, 0), (1, 0), (2, 0), (3, 0)], "color": C_CYAN},
    "O": {"cells": [(0, 0), (1, 0), (0, 1), (1, 1)], "color": C_GOLD},
    "T": {"cells": [(1, 0), (0, 1), (1, 1), (2, 1)], "color": C_PURPLE},
    "S": {"cells": [(1, 0), (2, 0), (0, 1), (1, 1)], "color": C_GREEN},
    "Z": {"cells": [(0, 0), (1, 0), (1, 1), (2, 1)], "color": C_RED},
    "J": {"cells": [(0, 0), (0, 1), (1, 1), (2, 1)], "color": C_BLUE},
    "L": {"cells": [(2, 0), (0, 1), (1, 1), (2, 1)], "color": C_ORANGE},
}
PIECE_NAMES = list(TETROMINOES.keys())

BOARD_W = 10
BOARD_H = 20
VOXEL = 16          # pixels per cell

TICK_BASE = 1.0       # seconds at 0 lines
TICK_DECAY = 0.5      # multiplied every 10 lines cleared
TICK_FLOOR = 0.1      # minimum tick (10 rows/s)
# Curve: 1.0 → 0.5 → 0.25 → 0.125 → 0.1 (floor at level 5, i.e. 40 lines).
# Same 5-level shape as the previous 0.5/0.5/0.05 curve, just shifted ×2 so
# the model gets twice as long per gravity-tick at every level.

ACTION_MIN_INTERVAL = 0.1   # min wall-clock seconds between player actions


def _rotate_cells(cells, times=1):
    for _ in range(times % 4):
        cells = [(y, -x) for x, y in cells]
    # normalise so min x/y == 0
    min_x = min(c[0] for c in cells)
    min_y = min(c[1] for c in cells)
    cells = [(x - min_x, y - min_y) for x, y in cells]
    return cells


class Piece:
    def __init__(self, name, rng):
        self.name = name
        d = TETROMINOES[name]
        self.cells = list(d["cells"])
        self.color = d["color"]
        self.x = BOARD_W // 2 - 2
        self.y = 0

    def rotated(self, times=1):
        return _rotate_cells(self.cells, times)

    def absolute(self, cells=None, dx=0, dy=0):
        cells = cells or self.cells
        return [(self.x + x + dx, self.y + y + dy) for x, y in cells]


class TetrisGame:
    def __init__(self, seed=42, duration=600, wait_for_turn=False):
        self.rng = random.Random(seed)
        self.duration = duration
        # In wait-for-turn mode gravity only fires on explicit WAIT actions and
        # the game clock is virtual (advances by one gravity interval per WAIT).
        self.wait_for_turn = wait_for_turn
        self.virtual_elapsed = 0.0   # only used when wait_for_turn=True
        self.last_latency_ms = 0     # ms for the most recent LLM decision
        # Game clock and gravity stay frozen until the client snaps the first
        # frame (via /getscreen or /marksnap). That moment is the true "game
        # starts" — otherwise vLLM warmup / HTTP startup latency would burn
        # game time before the model has even seen the board.
        self.start_time = None
        self.frames = []
        self.lock = threading.Lock()
        self.alive = True
        self.end_reason = None   # "topout" | "timeout" — set when alive→False
        self.restarts = 0
        # Score accounting across restarts. self.score (set in _init_board) is
        # the *current game* score and resets on top-out; max_score tracks the
        # best single-game score, total_score the cumulative across restarts.
        self.max_score = 0
        self.total_score = 0
        # Same pattern for cleared lines.
        self.max_lines = 0
        self.total_lines = 0
        self.piece_id = 0
        self.last_action = "WAIT"
        # Recent player actions for the on-screen history strip. Bounded —
        # we only render the tail that fits in the stats panel.
        self.action_history = []
        # apply_move() rate-limits player actions: at most one action every
        # ACTION_MIN_INTERVAL seconds (wall clock), independent of the
        # gravity tick rate.
        self._last_action_time = 0.0
        # Countdown / banner support. _banners is a list of (start, end, text)
        # tuples drawn over the board for the current wall-clock time.
        # _countdown_end pauses gravity AND player actions until time >= that
        # value; used for the "NEW GAME / 3 / 2 / 1" intro and post-restart.
        self._banners = []
        self._countdown_end = None
        self._init_board()

    def _init_board(self):
        self.board = [[None] * BOARD_W for _ in range(BOARD_H)]
        self.score = 0
        self.lines = 0
        self.level = 1
        self._piece = self._new_piece()
        self._next = self._new_piece()
        self._last_gravity = time.time()
        self._snapshot()

    # ------------------------------------------------------------------
    def _new_piece(self):
        return Piece(self.rng.choice(PIECE_NAMES), self.rng)

    def _gravity_interval(self):
        return max(TICK_FLOOR, TICK_BASE * (TICK_DECAY ** (self.lines // 10)))

    def _valid(self, cells):
        for x, y in cells:
            if x < 0 or x >= BOARD_W:
                return False
            if y >= BOARD_H:
                return False
            if y >= 0 and self.board[y][x] is not None:
                return False
        return True

    def _lock_piece(self):
        for x, y in self._piece.absolute():
            self.board[max(y, 0)][x] = self._piece.color
        self._clear_lines()
        self._piece = self._next
        self._next = self._new_piece()
        self.piece_id += 1
        # Give the freshly spawned piece a full gravity interval before its
        # first drop ("spawn delay"). Without this, gravity may tick between
        # piece spawn and the client's snap, so the model never sees the
        # piece at its spawn row.
        self._last_gravity = time.time()
        # Game over: new piece spawns on top of existing blocks → end game.
        if not self._valid(self._piece.absolute()):
            # Set alive=False first so render_board shows the overlay in every
            # snapshot.  13 frames × 15 cs ≈ 2 s of game-over screen.
            self.end_reason = "topout"
            self.alive = False
            for _ in range(13):
                self._snapshot()

    def _clear_lines(self):
        full = [r for r in range(BOARD_H) if all(c is not None for c in self.board[r])]
        for r in full:
            self.board.pop(r)
            self.board.insert(0, [None] * BOARD_W)
        n = len(full)
        if n:
            self.lines += n
            self.score += [0, 1, 3, 5, 8][min(n, 4)] * self.level
            self.level = max(1, self.lines // 10 + 1)

    def _elapsed_s(self):
        """Seconds of wall-clock game time elapsed (from start_time in both modes)."""
        return max(0.0, time.time() - self.start_time) if self.start_time else 0.0

    def tick(self):
        """Called periodically to advance gravity (no-op in wait-for-turn mode)."""
        with self.lock:
            if not self.alive:
                return
            if self.wait_for_turn:
                # No gravity in wait-for-turn, but enforce the wall-clock deadline
                # so the game ends after `duration` real seconds even if the model
                # never issues WAIT moves (which advance virtual_elapsed).
                if self.start_time and time.time() - self.start_time >= self.duration:
                    self.end_reason = "timeout"
                    self.alive = False
                    for _ in range(13):
                        self._snapshot()
                return
            if self.start_time is None:
                # Game hasn't started yet — waiting for the client to snap
                # the first frame.
                return
            now = time.time()
            # Hold gravity during any active countdown (intro or restart) but
            # keep snapshotting at ~10 fps so the GIF can animate the banner
            # transitions (NEW GAME → 3 → 2 → 1).
            if self._countdown_end is not None and now < self._countdown_end:
                if not hasattr(self, "_last_banner_frame_t") or \
                        now - self._last_banner_frame_t >= 0.1:
                    self._last_banner_frame_t = now
                    self._snapshot()
                return
            if now - self.start_time >= self.duration:
                self.end_reason = "timeout"
                self.alive = False
                for _ in range(13):
                    self._snapshot()
                return
            if now - self._last_gravity >= self._gravity_interval():
                self._last_gravity = now
                self._apply_move_nolock("FALL_GRAVITY")
                self._snapshot()

    def _apply_move_nolock(self, move):
        self.last_action = move
        # Don't pollute the on-screen history with gravity ticks — only
        # player moves go in the strip. Each entry carries the wall-clock
        # timestamp so the strip can scroll on a time axis (blank slots
        # appear when no action fires within a slot window).
        if move != "FALL_GRAVITY":
            self.action_history.append((time.time(), move, self.piece_id))
            if len(self.action_history) > 256:
                self.action_history = self.action_history[-256:]
        p = self._piece
        if move == "LEFT":
            new = p.absolute(dx=-1)
            if self._valid(new):
                p.x -= 1
            if self.wait_for_turn:
                self._snapshot()           # show horizontal move
                new = p.absolute(dy=1)     # then apply one gravity tick
                if self._valid(new):
                    p.y += 1
                else:
                    self._lock_piece()
        elif move == "RIGHT":
            new = p.absolute(dx=1)
            if self._valid(new):
                p.x += 1
            if self.wait_for_turn:
                self._snapshot()           # show horizontal move
                new = p.absolute(dy=1)     # then apply one gravity tick
                if self._valid(new):
                    p.y += 1
                else:
                    self._lock_piece()
        elif move == "FALL_GRAVITY":
            new = p.absolute(dy=1)
            if self._valid(new):
                p.y += 1
            else:
                self._lock_piece()
        elif move == "DROP":
            # Hard drop: animate each row as a fast frame so the fall is visible.
            while self._valid(p.absolute(dy=1)):
                p.y += 1
                self._snapshot(fast=True)
            self._lock_piece()
        elif move == "ROT_RIGHT":
            rc = p.rotated(3)
            new = [(p.x + x, p.y + y) for x, y in rc]
            if self._valid(new):
                p.cells = rc
        elif move == "ROT_LEFT":
            rc = p.rotated(1)
            new = [(p.x + x, p.y + y) for x, y in rc]
            if self._valid(new):
                p.cells = rc
        elif move == "WAIT":
            # Advance virtual clock by one gravity interval, then apply gravity.
            if self.wait_for_turn:
                self.virtual_elapsed += self._gravity_interval()
                if self.virtual_elapsed >= self.duration:
                    self.end_reason = "timeout"
                    self.alive = False
                    for _ in range(13):
                        self._snapshot()
                    return
            new = p.absolute(dy=1)
            if self._valid(new):
                p.y += 1
            else:
                self._lock_piece()

    def apply_move(self, move):
        # Block until ACTION_MIN_INTERVAL has elapsed since the last player
        # action so the model can issue at most ~10 actions per second.
        # DROP (auto-commit) bypasses the rate limit. In wait-for-turn mode
        # the rate limit is also skipped — actions are sequential by design.
        # BOTH paths must wait out any active countdown.
        while True:
            with self.lock:
                if not self.alive:
                    return
                now = time.time()
                countdown_active = (self._countdown_end is not None
                                    and now < self._countdown_end)
                rate_ok = (move == "DROP" or self.wait_for_turn
                           or now - self._last_action_time >= ACTION_MIN_INTERVAL)
                if not countdown_active and rate_ok:
                    self._last_action_time = now
                    self._apply_move_nolock(move)
                    self._snapshot()
                    return
                if countdown_active:
                    wait_s = self._countdown_end - now
                else:
                    wait_s = ACTION_MIN_INTERVAL - (now - self._last_action_time)
            time.sleep(min(0.05, max(0.005, wait_s)))

    def _snapshot(self, model_seen=False, fast=False, delay_cs=None):
        # Frame tuple: (time, rgb, w, h, model_seen, fast, delay_cs)
        # delay_cs: absolute centisecond override (None → use fixed_fps default).
        img_bytes, w, h = render_board(self)
        self.frames.append((time.time(), img_bytes, w, h, model_seen, fast, delay_cs))

    def get_screen_png(self):
        with self.lock:
            self._maybe_start_game_nolock()
            rgb, w, h = render_board(self)
            self.frames.append((time.time(), rgb, w, h, True, False, None))
            return _png_from_raw(rgb, w, h)

    def mark_snapshot(self):
        """Record a model-decision moment in the frame stream (no PNG returned).

        Used by text-only clients that don't fetch /getscreen but still want
        the GIF to highlight when each LLM call happened. Also starts the
        game clock on the first call.

        Countdown deduplication: the client polls every ~0.2 s so each 1-second
        banner would otherwise generate ~5 identical frames.  We skip frames for
        the same banner and assign a fixed 100 cs (1 s) delay to the one frame
        that IS kept — guaranteeing exactly 1 s per digit regardless of polling
        rate.
        """
        with self.lock:
            self._maybe_start_game_nolock()
            now = time.time()
            # Find the banner currently on screen (if any).
            current_banner = None
            if self._countdown_end is not None and now < self._countdown_end:
                for b_start, b_end, b_text in self._banners:
                    if b_start <= now < b_end:
                        current_banner = b_text
                        break
            # Skip if this is a duplicate call for the same banner.
            if current_banner is not None and \
                    current_banner == getattr(self, "_last_snapshot_banner", None):
                return
            self._last_snapshot_banner = current_banner
            delay_cs = 100 if current_banner is not None else None  # 100 cs = 1 s
            rgb, w, h = render_board(self)
            self.frames.append((time.time(), rgb, w, h, True, False, delay_cs))

    def _begin_countdown_nolock(self, include_game_over=False, step_s=1.0):
        """Queue a sequence of full-screen banners and pause gravity/actions
        until they finish. With `include_game_over=True` the sequence is
        'GAME OVER / NEW GAME / 3 / 2 / 1' (used on top-out restart); without
        it the sequence is 'NEW GAME / 3 / 2 / 1' (used at game start)."""
        now = time.time()
        seq = []
        if include_game_over:
            seq.append("GAME OVER")
        seq.extend(["NEW GAME", "3", "2", "1"])
        banners = []
        t = now
        for label in seq:
            banners.append((t, t + step_s, label))
            t += step_s
        self._banners = banners
        self._countdown_end = t

    def _maybe_start_game_nolock(self):
        """Start the game clock + gravity on the first client snap.
        Triggers the opening 'NEW GAME / 3 / 2 / 1' countdown."""
        if self.start_time is None:
            now = time.time()
            self._begin_countdown_nolock(include_game_over=False)
            # The game clock starts NOW; gravity is held off by
            # _countdown_end so no time is "lost" in the countdown.
            self.start_time = self._countdown_end
            self._last_gravity = self._countdown_end

    def _ghost_cells(self):
        """Cells the current falling piece would occupy after a hard drop
        (auto-commit) from its current column and orientation."""
        p = self._piece
        dy = 0
        while True:
            candidate = [(p.x + cx, p.y + cy + dy + 1) for cx, cy in p.cells]
            if not self._valid(candidate):
                break
            dy += 1
        return set((p.x + cx, p.y + cy + dy) for cx, cy in p.cells)

    def _ascii_board(self):
        """Return a Unicode-shaded representation of the board.

        Glyphs:
          ``█`` locked piece (immovable)
          ``▓`` falling piece (current position)
          ``░`` ghost / drop position (where the piece auto-commits to)
          ``.`` empty cell

        Falling-piece cells take precedence over ghost cells when they
        overlap (e.g. immediately after spawn before any drop)."""
        piece_cells = set(self._piece.absolute())
        ghost_cells = self._ghost_cells() - piece_cells
        # Each board line is prefixed with R<rowidx> (zero-padded to the
        # board's row-index width) so the model can reference rows directly.
        pad = len(str(BOARD_H - 1))
        header_pad = " " * (1 + pad + 1)  # "R" + digits + " "
        rows = [header_pad + "0123456789"]
        for r in range(BOARD_H):
            row = f"R{r:0{pad}d} "
            for c in range(BOARD_W):
                if (c, r) in piece_cells:
                    row += "▓"   # ▓ falling piece
                elif (c, r) in ghost_cells:
                    row += "░"   # ░ ghost (drop position)
                elif self.board[r][c] is not None:
                    row += "█"   # █ locked
                else:
                    row += "."
            rows.append(row)
        rows.append(header_pad + "0123456789")
        return "\n".join(rows)

    def _ascii_board_transposed(self):
        """Column-major view: each row = one board column, each char = one row.

        Header shows tens then ones of the row index so columns 0-9 map to
        R00-R09 and columns 10-19 map to R10-R19.
        Glyphs match _ascii_board: █ locked, ▓ falling, ░ ghost, . empty."""
        piece_cells = set(self._piece.absolute())
        ghost_cells = self._ghost_cells() - piece_cells
        rows = [
            "    " + "".join(str(r // 10) for r in range(BOARD_H)),
            "    " + "".join(str(r %  10) for r in range(BOARD_H)),
        ]
        for c in range(BOARD_W):
            row = f"C{c:01d}  "
            for r in range(BOARD_H):
                if (c, r) in piece_cells:
                    row += "▓"
                elif (c, r) in ghost_cells:
                    row += "░"
                elif self.board[r][c] is not None:
                    row += "█"
                else:
                    row += "."
            rows.append(row)
        return "\n".join(rows)

    def state_dict(self):
        with self.lock:
            # Touch the start-game side effect: the first /state call starts
            # the intro countdown, so the model doesn't see the banner in its
            # first snapshot.
            self._maybe_start_game_nolock()
            now = time.time()
            countdown_active = (self._countdown_end is not None
                                and now < self._countdown_end)
            countdown_remaining = max(0.0, (self._countdown_end - now)
                                      if self._countdown_end else 0.0)
            # max_/total_ include the current (live) game so the values stay
            # monotonic across the run, not just across restarts.
            live_max_score   = max(self.max_score, self.score)
            live_total_score = self.total_score + self.score
            live_max_lines   = max(self.max_lines, self.lines)
            live_total_lines = self.total_lines + self.lines
            # Piece geometry — surfaced for the prompt template so it can
            # reason about exact shape and landing row without re-parsing the
            # glyph board. Coords match piece_x / piece_y / piece_y so the
            # template can compare them directly.
            piece_abs = self._piece.absolute()
            ghost = self._ghost_cells()
            local_max_cy = max(cy for _, cy in self._piece.cells)
            ghost_max_y  = max(y for _, y in ghost) if ghost else self._piece.y + local_max_cy
            ghost_y_anchor = ghost_max_y - local_max_cy
            return {
                "score": self.score,
                "max_score": live_max_score,
                "total_score": live_total_score,
                "level": self.level,
                "lines": self.lines,
                "max_lines": live_max_lines,
                "total_lines": live_total_lines,
                "alive": self.alive,
                "restarts": self.restarts,
                "elapsed": round(self._elapsed_s(), 2),
                "duration": self.duration,
                "piece": self._piece.name,
                "piece_x": self._piece.x,
                "piece_y": self._piece.y,
                "piece_id": self.piece_id,
                "piece_cells": [[x, y] for x, y in piece_abs],
                "ghost_y": ghost_y_anchor,
                "next_piece": self._next.name,
                "tick": self._gravity_interval(),
                "board": self._ascii_board(),
                "board_transposed": self._ascii_board_transposed(),
                "countdown_active": countdown_active,
                "countdown_remaining": round(countdown_remaining, 2),
            }


# ---------------------------------------------------------------------------
# Rendering — pure stdlib PNG, no Pillow
# ---------------------------------------------------------------------------

# Tiny 5x7 pixel font for digits and basic chars.
# Each char is a list of 5 rows of 5 bits (MSB left).
_FONT5 = {
    "0": [0b11111, 0b10001, 0b10001, 0b10001, 0b11111],
    "1": [0b00100, 0b01100, 0b00100, 0b00100, 0b01110],
    "2": [0b11110, 0b00001, 0b01110, 0b10000, 0b11111],
    "3": [0b11110, 0b00001, 0b01110, 0b00001, 0b11110],
    "4": [0b10001, 0b10001, 0b11111, 0b00001, 0b00001],
    "5": [0b11111, 0b10000, 0b11110, 0b00001, 0b11110],
    "6": [0b11111, 0b10000, 0b11111, 0b10001, 0b11111],
    "7": [0b11111, 0b00001, 0b00010, 0b00100, 0b00100],
    "8": [0b11111, 0b10001, 0b11111, 0b10001, 0b11111],
    "9": [0b11111, 0b10001, 0b11111, 0b00001, 0b11111],
    ":": [0b00000, 0b00100, 0b00000, 0b00100, 0b00000],
    " ": [0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    "L": [0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
    "V": [0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
    "S": [0b01111, 0b10000, 0b01110, 0b00001, 0b11110],
    "C": [0b01110, 0b10001, 0b10000, 0b10001, 0b01110],
    "O": [0b01110, 0b10001, 0b10001, 0b10001, 0b01110],
    "R": [0b11110, 0b10001, 0b11110, 0b10010, 0b10001],
    "E": [0b11111, 0b10000, 0b11110, 0b10000, 0b11111],
    "N": [0b10001, 0b11001, 0b10101, 0b10011, 0b10001],
    "X": [0b10001, 0b01010, 0b00100, 0b01010, 0b10001],
    "T": [0b11111, 0b00100, 0b00100, 0b00100, 0b00100],
    "I": [0b11111, 0b00100, 0b00100, 0b00100, 0b11111],
    "G": [0b01110, 0b10000, 0b10111, 0b10001, 0b01110],
    "A": [0b01110, 0b10001, 0b11111, 0b10001, 0b10001],
    "M": [0b10001, 0b11011, 0b10101, 0b10001, 0b10001],
    "!": [0b00100, 0b00100, 0b00100, 0b00000, 0b00100],
    ".": [0b00000, 0b00000, 0b00000, 0b00000, 0b00100],
    "<": [0b00010, 0b00100, 0b01000, 0b00100, 0b00010],
    ">": [0b01000, 0b00100, 0b00010, 0b00100, 0b01000],
    "-": [0b00000, 0b00000, 0b11111, 0b00000, 0b00000],
    "Z": [0b11111, 0b00010, 0b00100, 0b01000, 0b11111],
    "U": [0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    "W": [0b10001, 0b10001, 0b10101, 0b10101, 0b01010],
    "[": [0b00110, 0b00100, 0b00100, 0b00100, 0b00110],
    "]": [0b01100, 0b00100, 0b00100, 0b00100, 0b01100],
}


def _draw_char(pixels, x0, y0, ch, color, scale=2, img_w=None):
    rows = _FONT5.get(ch.upper(), _FONT5[" "])
    for row_i, row_bits in enumerate(rows):
        for col_i in range(5):
            if row_bits & (1 << (4 - col_i)):
                for dy in range(scale):
                    for dx in range(scale):
                        px = x0 + col_i * scale + dx
                        py = y0 + row_i * scale + dy
                        if img_w and 0 <= px < img_w:
                            idx = (py * img_w + px) * 3
                            pixels[idx:idx+3] = color


def _draw_text(pixels, x, y, text, color, scale=2, img_w=None):
    cx = x
    for ch in text:
        _draw_char(pixels, cx, y, ch, color, scale=scale, img_w=img_w)
        cx += (5 + 1) * scale


def _draw_rect(pixels, x0, y0, w, h, color, img_w):
    for dy in range(h):
        for dx in range(w):
            px, py = x0 + dx, y0 + dy
            idx = (py * img_w + px) * 3
            pixels[idx:idx+3] = color


def render_board(game):
    """Render to raw RGB bytes and return (bytes, width, height).

    Vertical layout (narrow, single column):
      ┌──────────────┐
      │  NEXT piece  │  next_h
      ├──────────────┤
      │  game board  │  board_h
      ├──────────────┤
      │ SCORE LINES  │  stats_h
      │  TICK  TIME  │
      └──────────────┘
    """
    img_w   = BOARD_W * VOXEL   # 160 — same as board width
    next_h  = VOXEL * 4         # 64  — "NEXT" label + up to 2-cell-tall preview
    board_h = BOARD_H * VOXEL   # 320
    stats_h = 104               # two rows of label+value + action symbol + 6px margin
    img_h   = next_h + board_h + stats_h  # 464

    bg = bytes(C_BG)
    pixels = bytearray(bg * img_w * img_h)

    # ── NEXT strip ──────────────────────────────────────────────────────
    _draw_text(pixels, 4, 4, "NEXT", list(C_MUTED), scale=2, img_w=img_w)
    # Centre the next piece horizontally in img_w
    piece_cells = game._next.cells
    max_cx = max(c[0] for c in piece_cells)
    piece_px_w = (max_cx + 1) * VOXEL
    off_x = (img_w - piece_px_w) // 2
    for cx, cy in piece_cells:
        x0 = off_x + cx * VOXEL
        y0 = 14 + cy * VOXEL
        _draw_rect(pixels, x0, y0, VOXEL - 1, VOXEL - 1, list(game._next.color), img_w)

    # ── board ────────────────────────────────────────────────────────────
    board_y0 = next_h
    for row in range(BOARD_H):
        for col in range(BOARD_W):
            cell = game.board[row][col]
            color = list(cell) if cell else list(C_FAINT)
            _draw_rect(pixels, col * VOXEL, board_y0 + row * VOXEL,
                       VOXEL - 1, VOXEL - 1, color, img_w)

    if game.alive:
        for ax, ay in game._piece.absolute():
            if ay >= 0:
                _draw_rect(pixels, ax * VOXEL, board_y0 + ay * VOXEL,
                           VOXEL - 1, VOXEL - 1, list(game._piece.color), img_w)

    # ── stats strip ──────────────────────────────────────────────────────
    elapsed   = game._elapsed_s()
    remaining = max(0, game.duration - elapsed)

    # Two rows × two columns
    col_w = img_w // 2   # 80px each
    sy0   = next_h + board_h + 4

    def stat(label, value, color, x, y):
        _draw_text(pixels, x, y,       label,      list(C_MUTED), scale=2, img_w=img_w)
        _draw_text(pixels, x, y + 14,  str(value), list(color),   scale=2, img_w=img_w)

    stat("SCORE", game.score,                         C_GOLD,   4,         sy0)
    stat("LINES", game.lines,                         C_BLUE,   col_w + 4, sy0)
    # "LAT" + "MS" drawn separately with a 4 px gap (tighter than a full space).
    _draw_text(pixels, 4,  sy0 + 36,      "LAT", list(C_MUTED),   scale=2, img_w=img_w)
    _draw_text(pixels, 44, sy0 + 36,      "MS",  list(C_MUTED),   scale=2, img_w=img_w)
    _draw_text(pixels, 4,  sy0 + 36 + 14, str(game.last_latency_ms), list(C_PURPLE), scale=2, img_w=img_w)
    stat("TIME",  f"{int(remaining)}S",               C_TEXT,   col_w + 4, sy0 + 36)

    # ── action history strip ─────────────────────────────────────────────
    # Tick-axis view: one slot per action, one empty separator slot between
    # consecutive pieces. Newest action is on the right.
    ACTION_GLYPH = {
        "LEFT":      ("<", C_BLUE),
        "RIGHT":     (">", C_BLUE),
        "DROP":      ("V", C_RED),
        "ROT_RIGHT": ("]", C_PURPLE),
        "ROT_LEFT":  ("[", C_PURPLE),
        "WAIT":      ("-", C_MUTED),
        "COMMIT":    ("V", C_RED),
    }
    hist_scale = 2
    char_w = 6 * hist_scale
    hist_capacity = max(1, img_w // char_w)
    # Build token list newest-to-oldest: each action is a move string,
    # each piece boundary inserts a None separator.
    tokens = []
    prev_pid = None
    for entry in reversed(game.action_history):
        ts, mv, pid = entry[0], entry[1], entry[2] if len(entry) > 2 else 0
        if prev_pid is not None and pid != prev_pid:
            tokens.append(None)   # empty separator between pieces
        tokens.append(mv)
        prev_pid = pid
        if len(tokens) >= hist_capacity:
            break
    hist_y = sy0 + 78
    # Render right-to-left: index 0 = rightmost = most recent.
    for idx, token in enumerate(tokens):
        x = img_w - (idx + 1) * char_w
        if x < 0:
            break
        if token is None:
            continue  # separator — leave blank
        ch, col = ACTION_GLYPH.get(token, (".", C_FAINT))
        _draw_text(pixels, x, hist_y, ch, list(col),
                   scale=hist_scale, img_w=img_w)

    if not game.alive:
        _draw_rect(pixels, 0, board_y0, img_w, board_h, list(C_BG), img_w)
        scale = 4
        if game.end_reason == "timeout":
            lines = ["TIME", "OUT"]
            color = list(C_GOLD)
        else:
            lines = ["GAME", "OVER"]
            color = [255, 255, 255]
        line_h = 7 * scale + scale
        total_h = line_h * len(lines) - scale
        by0 = board_y0 + board_h // 2 - total_h // 2
        for i, line in enumerate(lines):
            text_w = len(line) * 6 * scale
            bx = (img_w - text_w) // 2
            _draw_text(pixels, bx, by0 + i * line_h, line, color,
                       scale=scale, img_w=img_w)

    # ── banner overlay (intro countdown, restart countdown) ─────────────
    # Drawn on top of everything else so the model and the GIF viewer see
    # the same "NEW GAME / 3 / 2 / 1" beat. Single banner active per frame;
    # picked by current wall-clock time.
    now = time.time()
    active_banner = None
    for (b_start, b_end, b_text) in game._banners:
        if b_start <= now < b_end:
            active_banner = b_text
            break
    if active_banner:
        if active_banner == "GAME OVER":
            scale, color = 4, list(C_RED)
        elif active_banner == "NEW GAME":
            scale, color = 4, list(C_GOLD)
        else:
            # countdown digits — biggest
            scale, color = 6, list(C_TEXT)
        # Split multi-word banners onto multiple lines so the text fits
        # the narrow board width and reads big.
        banner_lines = active_banner.split()
        line_h = 7 * scale + 2 * scale  # glyph + small gap
        total_h = line_h * len(banner_lines) - 2 * scale
        by0 = board_y0 + board_h // 2 - total_h // 2
        # Solid background strip behind all lines
        bg_color = list(C_BG)
        strip_top = max(0, by0 - scale)
        strip_h = total_h + 2 * scale
        for yy in range(strip_top, min(img_h, strip_top + strip_h)):
            row_off = yy * img_w * 3
            for xx in range(img_w):
                idx = row_off + xx * 3
                pixels[idx:idx+3] = bg_color
        for i, line in enumerate(banner_lines):
            text_w = len(line) * 6 * scale
            bx = (img_w - text_w) // 2
            by = by0 + i * line_h
            _draw_text(pixels, bx, by, line, color, scale=scale, img_w=img_w)

    return bytes(pixels), img_w, img_h


def _png_from_raw(rgb_bytes, width, height):
    def chunk(tag, data):
        length = struct.pack(">I", len(data))
        body = tag + data
        crc = struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        return length + body + crc

    raw_rows = []
    for row in range(height):
        raw_rows.append(b"\x00")  # filter None
        raw_rows.append(rgb_bytes[row * width * 3: (row + 1) * width * 3])
    compressed = zlib.compress(b"".join(raw_rows), 6)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    return png


def render_board_png(game):
    rgb, w, h = render_board(game)
    return _png_from_raw(rgb, w, h)


# ---------------------------------------------------------------------------
# GIF encoder (pure stdlib, palette-per-frame)
# ---------------------------------------------------------------------------

def _quantize_rgb(rgb_bytes, w, h, max_colors=255):
    """Extremely simple median-cut quantization → (palette list, indexed bytes)."""
    pixels = []
    for i in range(0, len(rgb_bytes), 3):
        pixels.append((rgb_bytes[i], rgb_bytes[i+1], rgb_bytes[i+2]))

    # Build palette from unique colours, truncated to max_colors
    seen = {}
    for p in pixels:
        seen[p] = seen.get(p, 0) + 1
    palette = list(seen.keys())[:max_colors]
    palette_set = {c: i for i, c in enumerate(palette)}
    # Pad to 256
    while len(palette) < 256:
        palette.append((0, 0, 0))

    indexed = bytearray()
    for p in pixels:
        idx = palette_set.get(p)
        if idx is None:
            # nearest
            best = min(range(len(seen)), key=lambda i: sum((a-b)**2 for a,b in zip(p, list(seen.keys())[i])))
            idx = palette_set.setdefault(p, len(palette_set))
        indexed.append(idx)

    return palette, bytes(indexed)


def _lzw_compress(data, min_code_size=8):
    """LZW compress bytes for GIF."""
    clear_code = 1 << min_code_size
    eoi_code = clear_code + 1
    table = {bytes([i]): i for i in range(clear_code)}
    next_code = eoi_code + 1
    code_size = min_code_size + 1
    max_code = 1 << code_size

    bits = 0
    bit_count = 0
    output = bytearray()

    def emit(code):
        nonlocal bits, bit_count
        bits |= code << bit_count
        bit_count += code_size
        while bit_count >= 8:
            output.append(bits & 0xFF)
            bits >>= 8
            bit_count -= 8

    def flush_bits():
        nonlocal bits, bit_count
        if bit_count > 0:
            output.append(bits & 0xFF)
            bits = 0
            bit_count = 0

    emit(clear_code)
    buf = bytes([data[0]])
    for byte in data[1:]:
        ext = buf + bytes([byte])
        if ext in table:
            buf = ext
        else:
            emit(table[buf])
            nonlocal_next = next_code
            if nonlocal_next < 4096:
                table[ext] = nonlocal_next
                next_code += 1
                if next_code > max_code and code_size < 12:
                    code_size += 1
                    max_code = 1 << code_size
            elif nonlocal_next >= 4096:
                emit(clear_code)
                table = {bytes([i]): i for i in range(clear_code)}
                next_code = eoi_code + 1
                code_size = min_code_size + 1
                max_code = 1 << code_size
            buf = bytes([byte])
    emit(table[buf])
    emit(eoi_code)
    flush_bits()
    return bytes(output)


def _gif_sub_blocks(data):
    """Pack data into GIF sub-blocks (max 255 bytes each)."""
    out = bytearray()
    for i in range(0, len(data), 255):
        block = data[i:i+255]
        out.append(len(block))
        out.extend(block)
    out.append(0)
    return bytes(out)


def _add_white_outline(rgb, w, h):
    """Draw a 1px white border around the perimeter of a raw RGB frame.

    Applied only to GIF frames so the viewer can clearly see the exact
    rectangle Gemma is looking at; /getscreen still returns the unmodified
    image so the model's input is unchanged.
    """
    buf = bytearray(rgb)
    white = b"\xff\xff\xff"
    # top and bottom rows
    buf[0:3 * w] = white * w
    buf[3 * w * (h - 1):3 * w * h] = white * w
    # left and right columns
    for row in range(h):
        off = 3 * w * row
        buf[off:off + 3] = white
        buf[off + 3 * (w - 1):off + 3 * w] = white
    return bytes(buf)


def _frames_to_gif(stamped_frames, target_fps=10, speed=1.0, fixed_fps=None):
    """Build a GIF from captured frames.

    fixed_fps: use a fixed delay per frame (ideal for wait-for-turn mode).
               All frames are included without subsampling. Frames marked
               fast=True use a 3 cs (30 ms) delay for drop animations.
    No fixed_fps: subsample to target_fps from wall-clock timestamps.
    """
    if not stamped_frames:
        return b""

    if fixed_fps is not None:
        # Include every frame — no subsampling.
        sampled = stamped_frames
        default_cs = max(2, int(100 / fixed_fps))
        frames = []
        delays = []
        for f in sampled:
            _, rgb, w, h, outlined = f[0], f[1], f[2], f[3], f[4]
            fast     = f[5] if len(f) > 5 else False
            delay_cs = f[6] if len(f) > 6 else None
            frames.append((_add_white_outline(rgb, w, h) if outlined else rgb, w, h))
            if fast:
                delays.append(3)
            elif delay_cs is not None:
                delays.append(max(2, delay_cs))   # absolute override
            else:
                delays.append(default_cs)
    else:
        interval = 1.0 / target_fps
        sampled = [stamped_frames[0]]
        for frame in stamped_frames[1:]:
            if frame[4] or frame[0] - sampled[-1][0] >= interval:
                sampled.append(frame)
        if sampled[-1] is not stamped_frames[-1]:
            sampled.append(stamped_frames[-1])
        frames = [
            (_add_white_outline(rgb, w, h) if outlined else rgb, w, h)
            for _, rgb, w, h, outlined, *_ in sampled
        ]
        timestamps = [t for t, *_ in sampled]
        delays = []
        for i in range(len(timestamps)):
            if i + 1 < len(timestamps):
                dt_cs = (timestamps[i + 1] - timestamps[i]) * 100 / speed
            else:
                dt_cs = 150 / speed
            delays.append(max(2, min(500, int(dt_cs))))
    return build_gif(frames, delays)


def build_gif(frames, delays_cs):
    """Build an animated GIF from list of (rgb_bytes, width, height) frames.
    delays_cs: list of per-frame delays in centiseconds, same length as frames.
    """
    if not frames:
        return b""

    _, w, h = frames[0]

    out = bytearray()
    # Header
    out += b"GIF89a"
    out += struct.pack("<HH", w, h)
    out += bytes([0xF7, 0, 0])  # global colour table flag + size (256 colours), background=0, aspect=0
    # Fake global colour table (all black — we use local per frame)
    out += b"\x00" * (3 * 256)

    # Netscape loop extension
    out += b"\x21\xFF\x0B" + b"NETSCAPE2.0" + b"\x03\x01" + struct.pack("<H", 0) + b"\x00"

    for (rgb_bytes, fw, fh), delay_cs in zip(frames, delays_cs):
        palette, indexed = _quantize_rgb(rgb_bytes, fw, fh)
        # Graphic control extension (delay)
        out += b"\x21\xF9\x04"
        out += bytes([0x00])  # disposal method 0
        out += struct.pack("<H", max(1, delay_cs))
        out += b"\x00\x00"

        # Image descriptor
        out += b"\x2C"
        out += struct.pack("<HHHHB", 0, 0, fw, fh, 0x87)  # local colour table, 256 colours

        # Local colour table
        for r, g, b in palette:
            out += bytes([r, g, b])

        # Image data
        lzw = _lzw_compress(indexed, min_code_size=8)
        out += bytes([8])  # LZW min code size
        out += _gif_sub_blocks(lzw)

    out += b"\x3B"  # trailer
    return bytes(out)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

_game: TetrisGame = None
_gif_speed: float = 1.0
_gif_fixed_fps: float = None


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress access log noise

    def _send(self, code, ctype, body):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/getscreen":
            png = _game.get_screen_png()
            self._send(200, "image/png", png)

        elif path == "/marksnap":
            # Record a model-decision moment in the frame stream without
            # returning the PNG. Used by text-only clients so the GIF still
            # shows a 1px white outline at each decision tick.
            lat = qs.get("latency_ms", [None])[0]
            if lat is not None:
                try:
                    _game.last_latency_ms = int(lat)
                except (ValueError, TypeError):
                    pass
            _game.mark_snapshot()
            self._send(200, "application/json", json.dumps({"ok": True}))

        elif path == "/state":
            self._send(200, "application/json", json.dumps(_game.state_dict()))

        elif path == "/action":
            move = qs.get("move", ["WAIT"])[0].upper()
            if move not in ("LEFT", "RIGHT", "DROP", "WAIT", "ROT_RIGHT", "ROT_LEFT"):
                self._send(400, "application/json", json.dumps({"error": "unknown move"}))
                return
            lat = qs.get("latency_ms", [None])[0]
            if lat is not None:
                try:
                    _game.last_latency_ms = int(lat)
                except (ValueError, TypeError):
                    pass
            _game.apply_move(move)
            self._send(200, "application/json", json.dumps(_game.state_dict()))

        elif path == "/gif":
            with _game.lock:
                stamped = list(_game.frames)
            gif = _frames_to_gif(stamped, speed=_gif_speed, fixed_fps=_gif_fixed_fps)
            self._send(200, "image/gif", gif)

        else:
            self._send(404, "text/plain", "not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/action":
            move = qs.get("move", ["WAIT"])[0].upper()
            if move not in ("LEFT", "RIGHT", "DROP", "WAIT", "ROT_RIGHT", "ROT_LEFT"):
                self._send(400, "application/json", json.dumps({"error": "unknown move"}))
                return
            lat = qs.get("latency_ms", [None])[0]
            if lat is not None:
                try:
                    _game.last_latency_ms = int(lat)
                except (ValueError, TypeError):
                    pass
            _game.apply_move(move)
            self._send(200, "application/json", json.dumps(_game.state_dict()))
        else:
            self._send(404, "text/plain", "not found")


def gravity_loop(game, poll=0.05):
    while game.alive:
        game.tick()
        time.sleep(poll)


# Benchmark invariants: the game duration and RNG seed are PINNED so the
# optimizer can't cheat the score by giving Gemma more time or an easier
# board. Do NOT expose these as CLI flags.
GAME_DURATION_S = 1800  # 30-minute game — benchmark constant
GAME_SEED = 42         # deterministic board RNG — benchmark constant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0,
                        help="TCP port to bind (0 = OS picks a free port; the chosen port is printed on startup)")
    parser.add_argument("--gif-speed", type=float, default=1.0,
                        help="GIF playback speed multiplier (default 1.0 = real-time, "
                             "matches actual tick rate so you see the game as it happened)")
    parser.add_argument("--gif-out",   default="tetris_replay.gif")
    parser.add_argument("--wait-for-turn", action="store_true",
                        help="Turn-based mode: gravity only fires on explicit WAIT actions; "
                             "game time is virtual (advances by one gravity interval per WAIT). "
                             "Disables the real-time gravity thread.")
    parser.add_argument("--gif-fps", type=float, default=None,
                        help="Fixed GIF playback fps (ignores wall-clock timestamps). "
                             "Recommended with --wait-for-turn so the replay plays at a "
                             "consistent speed instead of matching LLM call latency.")
    args = parser.parse_args()

    global _game, _gif_speed, _gif_fixed_fps
    _gif_speed = args.gif_speed
    _gif_fixed_fps = args.gif_fps
    _game = TetrisGame(seed=GAME_SEED, duration=GAME_DURATION_S,
                       wait_for_turn=args.wait_for_turn)

    # Always start the gravity/tick thread. In wait-for-turn mode tick() skips
    # gravity but still enforces the wall-clock deadline (sets alive=False after
    # duration real seconds), which is the only way the server exits cleanly.
    gthread = threading.Thread(target=gravity_loop, args=(_game,), daemon=True)
    gthread.start()

    server = ThreadedHTTPServer(("0.0.0.0", args.port), Handler)
    bound_port = server.server_address[1]
    mode = "wait-for-turn" if args.wait_for_turn else "real-time"
    print(f"Tetris server on :{bound_port}  seed={GAME_SEED}  mode={mode}", flush=True)
    print(f"  GET  /getscreen          → PNG screenshot")
    print(f"  GET  /action?move=LEFT   → apply move (LEFT|RIGHT|ROT_RIGHT|ROT_LEFT|DROP|WAIT) (also POST)")
    print(f"  GET  /state              → JSON state")
    print(f"  GET  /gif                → animated GIF (live or after game ends)")

    sthread = threading.Thread(target=server.serve_forever, daemon=True)
    sthread.start()

    try:
        while _game.alive:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        _game.alive = False
        server.shutdown()

    print("\nGame over. Saving GIF…")
    gif = _frames_to_gif(_game.frames, speed=_gif_speed, fixed_fps=_gif_fixed_fps)
    with open(args.gif_out, "wb") as f:
        f.write(gif)
    print(f"Saved {args.gif_out} ({len(gif)//1024} KB)")
    print(f"Final score: {_game.score}  level: {_game.level}  lines: {_game.lines}")


if __name__ == "__main__":
    main()
