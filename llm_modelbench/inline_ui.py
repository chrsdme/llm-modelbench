"""Inline live UI for benchmark runs.

The detached `watch` command remains the robust way to attach from another terminal or
another SSH session. This module provides the same dashboard inside the runner process when
`--live-ui compact` or `--live-ui full` is used. It is intentionally best-effort: if stdin or
stdout is not a TTY, it degrades to normal log printing.
"""
from __future__ import annotations

import os
import atexit
import select
import sys
import time

try:
    import termios
    import tty
except ImportError:  # Windows: termios/tty are POSIX-only.
    termios = None
    tty = None
from collections import deque
from pathlib import Path
from typing import Deque

from . import watch
from .hardware import live_snapshot


class InlineUI:
    """Small terminal controller for in-process dashboard/log toggling.

    Keys, checked between tasks:
      d  dashboard view
      l  log view
      q  request graceful stop after the current task
    """

    def __init__(self, run_dir: Path, layout: str = "compact", *, enabled: bool = False,
                 log_lines: int = 200, refresh_interval: float = 0.15):
        self.run_dir = run_dir
        self.layout = layout if layout in watch.RENDERERS else "compact"
        self.enabled = bool(
            enabled
            and termios is not None
            and tty is not None
            and sys.stdout.isatty()
            and sys.stdin.isatty()
        )
        self.mode = "dashboard"
        self.logs: Deque[str] = deque(maxlen=log_lines)
        self.prev_cpu = None
        self.stop_requested = False
        self._old_term = None
        self._started = False
        self._last_render = 0.0
        self.refresh_interval = max(0.05, float(refresh_interval))

    def start(self) -> None:
        if not self.enabled or self._started:
            return
        try:
            self._old_term = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            sys.stdout.write("\033[?1049h\033[?25l\033[2J\033[H")
            sys.stdout.flush()
            self._started = True
            atexit.register(self.close)
            self.render(force=True)
        except Exception:
            self.enabled = False
            self._restore_terminal()

    def close(self) -> None:
        if self._started:
            self._restore_terminal()
            self._started = False

    def _restore_terminal(self) -> None:
        try:
            if self._old_term is not None:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_term)
        except Exception:
            pass
        try:
            sys.stdout.write("\033[?25h\033[?1049l")
            sys.stdout.flush()
        except Exception:
            pass

    def poll_keys(self) -> None:
        if not self.enabled:
            return
        try:
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = os.read(sys.stdin.fileno(), 1).decode(errors="ignore").lower()
                if ch == "d":
                    self.mode = "dashboard"
                    self.render(force=True)
                elif ch == "l":
                    self.mode = "log"
                    self.render(force=True)
                elif ch == "q":
                    self.stop_requested = True
                    self.log("[operator] graceful stop requested, will stop after current task")
        except Exception:
            pass

    def log(self, line: str) -> None:
        line = str(line).rstrip("\n")
        if line:
            self.logs.append(line)
        if self.enabled and self.mode == "log":
            self.render(force=True)
        elif not self.enabled:
            print(line)

    def render(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        # Do not redraw too aggressively when called repeatedly between quick events.
        now = time.monotonic()
        if not force and now - self._last_render < self.refresh_interval:
            return
        self._last_render = now
        self.poll_keys()
        if self.mode == "log":
            frame = self._render_log()
        else:
            frame = self._render_dashboard()
        width, height = watch._terminal_size()
        frame = watch._fit_screen(frame, width, height)
        sys.stdout.write("\033[H\033[J" + frame)
        sys.stdout.flush()

    def _render_dashboard(self) -> str:
        repair_status = watch._load_repair_status_for_run(self.run_dir)
        hw, self.prev_cpu = live_snapshot(self.prev_cpu)
        if repair_status is not None:
            hw = repair_status.get("simulated_hardware") or hw
            body = watch.render_repair(repair_status, hw)
        else:
            st = watch._load_json(self.run_dir / "status.json")
            if "error" in st:
                body = st["error"] + "\n" + f"Waiting for {self.run_dir / 'status.json'} ..."
            elif st.get("status_type") == "context_profile":
                body = watch.render_context_profile(st, hw)
            else:
                body = watch.RENDERERS.get(self.layout, watch.render_compact)(st, hw)
        return body + "\n\nKEYS: d dashboard | l log | q graceful stop after current task"

    def _render_log(self) -> str:
        lines = [
            f"LLM MODELBENCH inline log | run {self.run_dir.name}",
            "=" * 80,
            "KEYS: d dashboard | l log | q graceful stop after current task",
            "-" * 80,
        ]
        lines.extend(list(self.logs))
        return "\n".join(lines)
