#!/usr/bin/env python3
"""Fast repository release-hygiene checks used locally and in CI."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
EXECUTABLES = [
    "bootstrap.sh", "install.sh", "update.sh", "uninstall.sh",
    "llmb", "llmb-run", "llmb-watch", "llmb-read-kv-env.sh",
]
FORBIDDEN_TRACKED = re.compile(
    r"(^|/)(__pycache__/|build/|dist/|[^/]+\.egg-info/)|\.(?:pyc|pyo)$"
)
PRIVATE_PATH = re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/")
WATERMARK = re.compile(("generated" + r" by (?:clau" + "de|chat" + r"gpt|gpt)|as an ai " + "language model"), re.I)

_EXCLUDED_DIR_NAMES = {
    "__pycache__", ".venv", "venv", "build", "dist", "runs", "rankings",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov", "model_cards",
    "rankings-separate", "snapshots", ".git",
}
_EXCLUDED_FILE_NAMES = {".coverage", "coverage.xml", "_last_summary.json"}
_EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".review.zip"}


def fail(message: str) -> None:
    print(f"RELEASE CHECK FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def _is_local_generated(relative: Path) -> bool:
    if _EXCLUDED_DIR_NAMES.intersection(relative.parts):
        return True
    if any(part.endswith(".egg-info") for part in relative.parts):
        return True
    if relative.name in _EXCLUDED_FILE_NAMES:
        return True
    return any(str(relative).endswith(suffix) for suffix in _EXCLUDED_SUFFIXES)


def repository_files() -> list[str]:
    """Return files that are publishable from the current working tree.

    In a Git checkout, include tracked and untracked, non-ignored files. Before
    ``git init``, mirror the repository's generated-artifact exclusions so a
    normal local virtualenv or editable-install metadata is not misreported as
    source that would be published.
    """
    if (ROOT / ".git").exists():
        output = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            text=True,
        )
        return sorted({line for line in output.splitlines() if line})

    files: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if _is_local_generated(relative):
            continue
        files.append(str(relative))
    return sorted(files)


def check_version() -> None:
    sys.path.insert(0, str(ROOT))
    from llm_modelbench import __version__

    readme = (ROOT / "README.md").read_text()
    changelog = (ROOT / "CHANGELOG.md").read_text()
    first = re.search(r"^##\s+([^\n]+)", changelog, re.MULTILINE)
    if not first or first.group(1).strip() != __version__:
        fail("newest changelog heading does not match runtime version")
    if f"`{__version__}`" not in readme:
        fail("README does not state the runtime version")
    pyproject = (ROOT / "pyproject.toml").read_text()
    if 'version = {attr = "llm_modelbench.__version__"}' not in pyproject:
        fail("pyproject is not reading the package version dynamically")


def check_tracked_files(files: list[str]) -> None:
    bad = [path for path in files if FORBIDDEN_TRACKED.search(path)]
    if bad:
        fail("generated artifacts would be published: " + ", ".join(bad))


def check_permissions() -> None:
    if os.name != "posix":
        return
    bad = [path for path in EXECUTABLES if not os.access(ROOT / path, os.X_OK)]
    if bad:
        fail("documented wrappers are not executable: " + ", ".join(bad))


def check_text_spill(files: list[str]) -> None:
    for relative in files:
        path = ROOT / relative
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".zip", ".pyc"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if PRIVATE_PATH.search(text):
            fail(f"absolute user-home path found in {relative}")
        if WATERMARK.search(text):
            fail(f"AI watermark phrase found in {relative}")


def check_resources() -> None:
    sys.path.insert(0, str(ROOT))
    from llm_modelbench import media
    from llm_modelbench.tasks import TASKS

    for task in TASKS:
        resource = task.meta.get("image_path")
        if resource:
            media.load_image_file(resource)


def main() -> None:
    files = repository_files()
    check_tracked_files(files)
    check_version()
    check_permissions()
    check_text_spill(files)
    check_resources()
    print("RELEASE CHECK: PASS")


if __name__ == "__main__":
    main()
