"""Pure state machine and TTY wrapper for interactive ``watch`` navigation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class InteractiveState:
    rows: List[Dict[str, Any]] = field(default_factory=list)
    selected: int = 0
    search: str = ""
    searching: bool = False
    sort_key: str = "order"
    sort_desc: bool = True
    detail_open: bool = False
    quit: bool = False
    SORT_CYCLE = ("order", "quality_avg", "tps_avg", "failures", "weak")

    def refresh_rows(self, status: Dict[str, Any]) -> None:
        self.rows = list(status.get("completed_models") or [])
        self.clamp_selection()

    def visible_rows(self) -> List[Dict[str, Any]]:
        rows = self.rows
        if self.search:
            rows = [row for row in rows if self.search.lower() in str(row.get("model", "")).lower()]
        if self.sort_key != "order":
            rows = sorted(rows, key=lambda row: (row.get(self.sort_key) is None, row.get(self.sort_key, 0)), reverse=self.sort_desc)
        return rows

    def clamp_selection(self) -> None:
        self.selected = max(0, min(self.selected, max(0, len(self.visible_rows()) - 1)))

    def handle_key(self, key: str) -> None:
        if self.searching:
            if key in ("\r", "\n"):
                self.searching = False
            elif key == "\x1b":
                self.searching, self.search = False, ""
            elif key in ("\x7f", "\b"):
                self.search = self.search[:-1]
            elif key.isprintable():
                self.search += key
            self.clamp_selection()
            return
        if key == "q": self.quit = True
        elif key == "j": self.selected += 1
        elif key == "k": self.selected -= 1
        elif key == "/": self.searching, self.search = True, ""
        elif key == "s": self.sort_key = self.SORT_CYCLE[(self.SORT_CYCLE.index(self.sort_key) + 1) % len(self.SORT_CYCLE)]
        elif key == "S": self.sort_desc = not self.sort_desc
        elif key in ("\r", "\n"): self.detail_open = not self.detail_open
        elif key == "\x1b" and self.detail_open: self.detail_open = False
        self.clamp_selection()


def watch_interactive(run_dir, *, refresh: float = 1.0) -> int:
    import select
    import sys
    import termios
    import tty
    from .hardware import live_snapshot
    from .watch import _enter_alt_screen, _fit_screen, _leave_alt_screen, _load_json, _terminal_size

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("interactive mode needs a real terminal (stdin/stdout are not a TTY)")
        return 1
    state, previous_cpu, status_path = InteractiveState(), None, run_dir / "status.json"
    fd, old = sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno())
    _enter_alt_screen()
    try:
        tty.setraw(fd)
        while not state.quit:
            status = _load_json(status_path)
            hardware, previous_cpu = live_snapshot(previous_cpu)
            if "error" not in status:
                state.refresh_rows(status)
            rows = state.visible_rows()
            body = [f"LLM WATCH {status.get('run_id', '?')} sort={state.sort_key}", "-" * 70]
            body.extend(f"{'>' if i == state.selected else ' '} {row.get('model')} q={row.get('quality_avg')} tps={row.get('tps_avg')}" for i, row in enumerate(rows))
            if state.detail_open and rows:
                body.append(f"DETAIL: {rows[state.selected].get('model')}")
            body.append("j/k select  / search  s sort  S reverse  Enter detail  q quit")
            width, height = _terminal_size()
            sys.stdout.write("\033[H\033[J" + _fit_screen("\n".join(body), width, height))
            sys.stdout.flush()
            ready, _, _ = select.select([sys.stdin], [], [], max(0.05, float(refresh)))
            if ready:
                state.handle_key(sys.stdin.read(1))
        return 0
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        _leave_alt_screen()
