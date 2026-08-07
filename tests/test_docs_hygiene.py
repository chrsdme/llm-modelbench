"""Public documentation and local-material separation invariants."""
from pathlib import Path

from llm_modelbench import __version__

ROOT = Path(__file__).resolve().parents[1]


_EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    "runs",
    "campaigns",
    "rankings",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    "model_cards",
    "rankings-separate",
    "snapshots",
    "local_only",
    ".git",
}


def _is_local_generated_path(path: Path) -> bool:
    if path.name == "CODEX_RC20_COMPLETION_REMEDIATION.md":
        return True
    if _EXCLUDED_DIR_NAMES.intersection(path.parts):
        return True
    return any(part.endswith(".egg-info") for part in path.parts)


def test_readme_states_current_version_and_public_https_clone():
    readme = (ROOT / "README.md").read_text()
    assert f"`{__version__}`" in readme
    assert "git clone https://github.com/chrsdme/llm-modelbench.git" in readme
    assert "git clone git@github.com" not in readme


def test_current_docs_are_separate_from_local_development_material():
    assert (ROOT / "docs" / "README.md").is_file()
    assert (ROOT / "docs" / "RUNTIMES.md").is_file()
    tracked = __import__("subprocess").check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    assert "/handovers/" not in "\n".join(tracked)
    assert "/prompts/" not in "\n".join(tracked)
    assert not any(path.startswith(("docs/history/", "docs/rc21/", "local_only/")) for path in tracked)


def test_no_absolute_user_home_paths_in_text_sources():
    import re

    private_home = re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/")
    for path in ROOT.rglob("*"):
        if not path.is_file() or _is_local_generated_path(path.relative_to(ROOT)):
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".pyc", ".zip"}:
            continue
        text = path.read_text(errors="ignore")
        assert not private_home.search(text), f"{path} contains an absolute user-home path"


def test_private_workspaces_are_ignored_and_not_publicly_tracked():
    import subprocess

    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    assert not any(path == "AGENTS.md" or path == "REPO_MAP_PRIVATE.md" for path in tracked)
    assert not any(path.startswith("local_only/") for path in tracked)
    for path in ("local_only/", "AGENTS.md", "REPO_MAP_PRIVATE.md"):
        assert subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0


def test_known_host_hardware_identifiers_are_not_publicly_tracked():
    known_identifiers = {
        "GPU-78077308-" + "f4b2-3330-6d4e-19581d7b1511",
        "GPU-5b99bce2-" + "35ab-f6db-857b-72162069fa72",
    }
    import subprocess

    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    for relative in tracked:
        path = ROOT / relative
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".pyc", ".zip"}:
            continue
        text = path.read_text(errors="ignore")
        assert not any(identifier in text for identifier in known_identifiers), relative
