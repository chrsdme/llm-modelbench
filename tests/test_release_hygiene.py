import os
import re
import subprocess
from pathlib import Path

from llm_modelbench import __version__, media
from llm_modelbench.tasks import TASKS
from tools import release_check

ROOT = Path(__file__).resolve().parents[1]
EXECUTABLES = [
    "bootstrap.sh", "install.sh", "update.sh", "uninstall.sh",
    "llmb", "llmb-run", "llmb-watch", "scripts/libexec/llmb-ollama-kv-control",
]


def test_version_identity_is_synchronized():
    changelog = (ROOT / "CHANGELOG.md").read_text()
    readme = (ROOT / "README.md").read_text()
    first_heading = re.search(r"^##\s+([^\n]+)", changelog, re.MULTILINE)
    assert first_heading and first_heading.group(1).strip() == __version__
    assert f"`{__version__}`" in readme
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'version = {attr = "llm_modelbench.__version__"}' in pyproject


def test_no_generated_python_or_build_artifacts_are_tracked():
    if not (ROOT / ".git").exists():
        return
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    bad = [
        path for path in tracked
        if "__pycache__/" in path or path.endswith((".pyc", ".pyo"))
        or ".egg-info/" in path or path.startswith(("build/", "dist/"))
    ]
    assert not bad, f"generated artifacts tracked: {bad}"


def test_documented_wrappers_are_executable_on_unix():
    if os.name != "posix":
        return
    for relative in EXECUTABLES:
        assert os.access(ROOT / relative, os.X_OK), f"{relative} is not executable"


def test_all_static_task_images_load_from_package_resources():
    image_tasks = [task for task in TASKS if task.meta.get("image_path")]
    assert image_tasks
    for task in image_tasks:
        payload = media.load_image_file(task.meta["image_path"])
        assert Path(payload["path"]).is_file()
        assert "llm_modelbench/fixtures/" in payload["path"].replace("\\", "/")
        assert payload["data"]


def test_release_check_scans_tracked_codex_logs_for_private_paths(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    tracked = tmp_path / "codex_logs" / "test.md"
    tracked.parent.mkdir()
    private_path = "/home/" + "example/path"
    tracked.write_text(f"private {private_path}\n", encoding="utf-8")

    def fake_check_output(command, **kwargs):
        if command == ["git", "ls-files", "--cached"]:
            return "codex_logs/test.md\n"
        if command == ["git", "ls-files", "--others", "--exclude-standard"]:
            return ""
        raise AssertionError(command)

    monkeypatch.setattr(release_check, "ROOT", tmp_path)
    monkeypatch.setattr(release_check.subprocess, "check_output", fake_check_output)
    files = release_check.repository_files()
    assert files == ["codex_logs/test.md"]
    try:
        release_check.check_text_spill(files)
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("tracked codex log private path was not rejected")


def test_release_check_may_ignore_untracked_local_codex_logs(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    untracked = tmp_path / "codex_logs" / "local.md"
    untracked.parent.mkdir()
    private_path = "/home/" + "example/path"
    untracked.write_text(f"private {private_path}\n", encoding="utf-8")

    def fake_check_output(command, **kwargs):
        if command == ["git", "ls-files", "--cached"]:
            return ""
        if command == ["git", "ls-files", "--others", "--exclude-standard"]:
            return "codex_logs/local.md\n"
        raise AssertionError(command)

    monkeypatch.setattr(release_check, "ROOT", tmp_path)
    monkeypatch.setattr(release_check.subprocess, "check_output", fake_check_output)
    assert release_check.repository_files() == []
