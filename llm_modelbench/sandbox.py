"""Guarded host execution for deterministic code scorers.

The runner uses isolated interpreter flags, a scrubbed environment, a throwaway working
folder, Linux resource limits, process-group termination, a static reject list, and a
wall-clock timeout. These controls reduce accidental damage but are not a complete security
boundary. Evaluate untrusted models inside a container, VM, or disposable host.
"""
from __future__ import annotations

import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

BLOCKLIST = [
    r"\brm\s+-rf\b", r"shutil\.rmtree\s*\(\s*[\"']?/", r"os\.remove\s*\(\s*[\"']?/",
    r"subprocess", r"os\.system", r"__import__\s*\(\s*[\"']os", r"socket\.",
    r"open\s*\(\s*[\"']/(etc|root|home|srv|var)", r"Path\s*\(\s*[\"']/[\"']\s*\)",
    r"\.\./\.\.", r"requests\.", r"urllib",
]


def is_safe(code: str) -> Tuple[bool, str]:
    for pattern in BLOCKLIST:
        if re.search(pattern, code or ""):
            return False, f"blocked pattern: {pattern}"
    return True, "ok"


def _scrubbed_env(work_dir: str) -> dict[str, str]:
    """Return a minimal environment that does not pass API keys or user paths to candidates."""
    env = {
        "HOME": work_dir,
        "TMPDIR": work_dir,
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    return env


def _linux_limits(timeout: int, *, limit_processes: bool = True):
    """Create a Linux pre-exec limiter. Cross-platform support is intentionally out of scope."""
    def apply_limits() -> None:
        try:
            import resource

            cpu_seconds = max(1, int(timeout) + 1)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            if limit_processes and hasattr(resource, "RLIMIT_NPROC"):
                resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
        except (ImportError, OSError, ValueError):
            # The caller still has timeout, process-group handling, isolated mode, and env scrubbing.
            pass

    return apply_limits


def _run_guarded_process(
    args: List[str], *, cwd: str, timeout: int, env: dict[str, str] | None = None,
    limit_processes: bool = True,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        args,
        cwd=cwd,
        env=env or _scrubbed_env(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        preexec_fn=_linux_limits(timeout, limit_processes=limit_processes) if os.name == "posix" else None,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        stdout, stderr = process.communicate()
        return subprocess.CompletedProcess(process.args, 124, stdout, stderr or "execution timed out")
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def _run_python(path: Path, *, cwd: str, timeout: int) -> subprocess.CompletedProcess[str]:
    return _run_guarded_process(
        [sys.executable, "-I", "-S", "-B", str(path)], cwd=cwd, timeout=timeout,
    )


def run_node_harness(
    harness: str, candidate_code: str, timeout: int = 3,
) -> subprocess.CompletedProcess[str] | None:
    """Run a fixed Node harness around candidate code with the same host guards as Python."""
    node = shutil.which("node")
    if node is None:
        return None
    with tempfile.TemporaryDirectory(prefix="llmb-js-score-") as temp_dir:
        path = Path(temp_dir) / "harness.js"
        path.write_text(harness)
        env = _scrubbed_env(temp_dir)
        env["LLM_MODELBENCH_JS_CODE"] = candidate_code
        return _run_guarded_process(
            [node, "--max-old-space-size=128", str(path)],
            cwd=temp_dir,
            timeout=timeout,
            env=env,
            limit_processes=False,
        )


def run_python_checks(code: str, checks: List[str], timeout: int = 10) -> Tuple[float, str]:
    """Run each check independently and score the fraction that exits successfully."""
    ok, why = is_safe(code)
    if not ok:
        return 0.0, why
    passed = 0
    for check in checks:
        with tempfile.TemporaryDirectory(prefix="llmb-score-") as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(code + "\n\n" + check)
            try:
                result = _run_python(candidate, cwd=temp_dir, timeout=timeout)
                passed += int(result.returncode == 0)
            except (OSError, ValueError):
                pass
    return (100.0 * passed / len(checks) if checks else 0.0), f"{passed}/{len(checks)} checks"


def run_script_in_fixture(code: str, fixture: dict, timeout: int = 10) -> Tuple[bool, dict]:
    """Run a script in a seeded temporary directory and return the resulting file layout."""
    ok, _ = is_safe(code)
    if not ok:
        return False, {}
    with tempfile.TemporaryDirectory(prefix="llmb-fixture-") as temp_dir:
        for relative_path, content in fixture.items():
            fixture_path = Path(temp_dir) / relative_path
            fixture_path.parent.mkdir(parents=True, exist_ok=True)
            fixture_path.write_text(content)
        runner = Path(temp_dir) / "_run.py"
        runner.write_text(code)
        try:
            result = _run_python(runner, cwd=temp_dir, timeout=timeout)
        except (OSError, ValueError):
            return False, {}
        if result.returncode != 0:
            return False, {}
        runner.unlink(missing_ok=True)
        layout: dict = {}
        for directory, _, files in os.walk(temp_dir):
            relative = os.path.relpath(directory, temp_dir)
            for filename in files:
                layout.setdefault(relative, set()).add(filename)
        return True, layout
