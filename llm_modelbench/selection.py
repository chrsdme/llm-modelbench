"""Model selection helpers and a dependency-free terminal selector."""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Iterable, List, Optional, Sequence


def parse_models_spec(spec: Optional[str]) -> Optional[List[str]]:
    """Parse the documented semicolon-delimited model list."""
    if spec is None:
        return None
    values = [part.strip() for part in spec.split(";") if part.strip()]
    if not values:
        raise ValueError("--models did not contain any model names")
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def resolve_exact_models(requested: Optional[Iterable[str]], installed: Sequence[str]) -> Optional[List[str]]:
    if requested is None:
        return None
    installed_list = list(installed)
    lower = {}
    for name in installed_list:
        lower.setdefault(name.lower(), []).append(name)
    resolved: List[str] = []
    errors: List[str] = []
    for raw in requested:
        if raw in installed_list:
            resolved.append(raw)
            continue
        matches = lower.get(str(raw).lower(), [])
        if len(matches) == 1:
            resolved.append(matches[0])
            continue
        suggestions = get_close_matches(str(raw), installed_list, n=3, cutoff=0.45)
        hint = f"; closest: {', '.join(suggestions)}" if suggestions else ""
        errors.append(f"{raw!r} is not an installed model{hint}")
    if errors:
        raise ValueError("invalid --models selection: " + " | ".join(errors))
    return list(dict.fromkeys(resolved))


@dataclass
class SelectorState:
    models: List[str]
    cursor: int = 0
    selected: set[str] = field(default_factory=set)
    search: str = ""
    searching: bool = False
    cancelled: bool = False
    accepted: bool = False

    def visible(self) -> List[str]:
        if not self.search:
            return list(self.models)
        q = self.search.lower()
        return [m for m in self.models if q in m.lower()]

    def clamp(self) -> None:
        n = len(self.visible())
        self.cursor = 0 if n == 0 else max(0, min(self.cursor, n - 1))

    def handle(self, key: str) -> None:
        if self.searching:
            if key in ("\r", "\n"):
                self.searching = False
            elif key == "\x1b":
                self.searching = False
                self.search = ""
            elif key in ("\x7f", "\b"):
                self.search = self.search[:-1]
            elif key.isprintable():
                self.search += key
            self.clamp()
            return
        visible = self.visible()
        if key in ("j", "\x1b[B"):
            self.cursor += 1
        elif key in ("k", "\x1b[A"):
            self.cursor -= 1
        elif key == " ":
            if visible:
                model = visible[self.cursor]
                if model in self.selected:
                    self.selected.remove(model)
                else:
                    self.selected.add(model)
        elif key == "a":
            self.selected.update(visible)
        elif key == "n":
            self.selected.difference_update(visible)
        elif key == "/":
            self.searching = True
            self.search = ""
        elif key in ("\r", "\n"):
            self.accepted = True
        elif key in ("q", "\x1b"):
            self.cancelled = True
        self.clamp()


def render_selector(state: SelectorState, *, height: int = 20) -> str:
    visible = state.visible()
    start = max(0, state.cursor - max(1, height // 2))
    end = min(len(visible), start + height)
    lines = [
        "LLM ModelBench - select models",
        f"selected={len(state.selected)} visible={len(visible)}/{len(state.models)}",
        (f"/{state.search}_" if state.searching else (f"filter={state.search}" if state.search else "")),
        "-" * 78,
    ]
    for index in range(start, end):
        model = visible[index]
        cursor = ">" if index == state.cursor else " "
        mark = "x" if model in state.selected else " "
        lines.append(f"{cursor} [{mark}] {model}")
    if not visible:
        lines.append("  no models match the current filter")
    lines.extend(["-" * 78, "j/k or arrows move | Space toggle | a all | n none | / search | Enter accept | q cancel"])
    return "\n".join(lines)


def _read_key(stream) -> str:
    ch = stream.read(1)
    if ch == "\x1b":
        # Decode common arrow escape sequences without blocking indefinitely.
        import select
        seq = ch
        for _ in range(2):
            ready, _, _ = select.select([stream], [], [], 0.02)
            if not ready:
                break
            seq += stream.read(1)
        return seq
    return ch


def select_models(models: Sequence[str], *, preselected: Optional[Iterable[str]] = None) -> List[str]:
    """Interactively select models only.

    Test scope is intentionally not edited here. Use the wizard for combined
    model and test/category editing.
    """
    import shutil
    import sys

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError("--select requires an interactive terminal; use --models or --all in scripts")
    state = SelectorState(list(models), selected=set(preselected or []))
    if not state.selected:
        state.selected.update(models)

    try:
        import termios
        import tty
    except ImportError as exc:
        raise RuntimeError("interactive selector is supported on POSIX terminals; use --models on this platform") from exc

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    sys.stdout.write("\033[?1049h\033[H")
    sys.stdout.flush()
    try:
        tty.setraw(fd)
        while not state.accepted and not state.cancelled:
            height = max(6, shutil.get_terminal_size((100, 28)).lines - 7)
            sys.stdout.write("\033[H\033[2J" + render_selector(state, height=height))
            sys.stdout.flush()
            state.handle(_read_key(sys.stdin))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\033[?1049l")
        sys.stdout.flush()
    if state.cancelled:
        raise SystemExit("model selection cancelled")
    if not state.selected:
        raise SystemExit("no models selected")
    # Preserve Ollama inventory order, not selection toggle order.
    return [model for model in models if model in state.selected]
