"""Public documentation and historical-material separation invariants."""
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
    "rankings",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    "model_cards",
    "rankings-separate",
    "snapshots",
    ".git",
}


def _is_local_generated_path(path: Path) -> bool:
    if _EXCLUDED_DIR_NAMES.intersection(path.parts):
        return True
    return any(part.endswith(".egg-info") for part in path.parts)


def test_readme_states_current_version_and_public_https_clone():
    readme = (ROOT / "README.md").read_text()
    assert f"`{__version__}`" in readme
    assert "git clone https://github.com/chrsdme/llm-modelbench.git" in readme
    assert "git clone git@github.com" not in readme


def test_current_docs_are_separate_from_release_history():
    history = ROOT / "docs" / "history"
    assert history.is_dir()
    assert (ROOT / "docs" / "README.md").is_file()
    assert list(history.glob("RC*_AUDIT.md"))
    assert not list((ROOT / "docs").glob("RC*_AUDIT.md"))
    assert not list((ROOT / "docs").glob("APPLY_RC*.md"))


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
