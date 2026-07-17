import os
import re
import subprocess
from pathlib import Path

from llm_modelbench import __version__, media
from llm_modelbench.tasks import TASKS

ROOT = Path(__file__).resolve().parents[1]
EXECUTABLES = [
    "bootstrap.sh", "install.sh", "update.sh", "uninstall.sh",
    "llmb", "llmb-run", "llmb-watch", "llmb-read-kv-env.sh",
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
