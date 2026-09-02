"""Stage 3B.1C — read-only backend executable availability discovery.

These prove that ModelBench can answer "is a backend executable installed
here?" (so a later stage can decide whether it *could* launch an ephemeral
llama-server) without spawning any process, reading no live endpoint, and
mutating nothing.
"""
from llm_modelbench.runtime_profiles import (
    BackendExecutable,
    discover_backend_executables,
)


def _which(mapping):
    """Injected ``shutil.which`` stand-in — never touches the real PATH."""
    return lambda name: mapping.get(name)


def test_reports_installed_and_absent_backends_from_injected_path():
    found = discover_backend_executables(
        which_fn=_which({"ollama": "/usr/bin/ollama"}),
    )
    by_id = {item.backend: item for item in found}

    assert by_id["ollama"].executable_available is True
    assert by_id["ollama"].executable_path == "/usr/bin/ollama"
    assert by_id["ollama"].state == "installed"

    assert by_id["llama_cpp"].executable_available is False
    assert by_id["llama_cpp"].executable_path is None
    assert by_id["llama_cpp"].state == "not_installed"


def test_llama_cpp_availability_is_discovered_without_spawning():
    calls = []

    def spy_which(name):
        calls.append(name)
        return "/opt/llama.cpp/llama-server" if name == "llama-server" else None

    found = discover_backend_executables(which_fn=spy_which)
    llama = next(item for item in found if item.backend == "llama_cpp")

    assert llama.executable_available is True
    assert llama.executable_path == "/opt/llama.cpp/llama-server"
    assert llama.state == "installed"
    # The probe resolved availability purely by a PATH lookup for the
    # executable name -- it never ran a subprocess.
    assert "llama-server" in calls


def test_explicit_configured_path_overrides_path_lookup_and_is_verified():
    found = discover_backend_executables(
        which_fn=_which({}),  # nothing on PATH
        configured_paths={"llama_cpp": "/custom/llama-server"},
        path_exists=lambda p: p == "/custom/llama-server",
    )
    llama = next(item for item in found if item.backend == "llama_cpp")

    assert llama.executable_available is True
    assert llama.executable_path == "/custom/llama-server"
    assert llama.state == "installed"
    assert llama.source == "configured"


def test_configured_path_that_does_not_exist_is_not_configured_not_a_crash():
    found = discover_backend_executables(
        which_fn=_which({}),
        configured_paths={"llama_cpp": "/custom/missing"},
        path_exists=lambda p: False,
    )
    llama = next(item for item in found if item.backend == "llama_cpp")

    assert llama.executable_available is False
    assert llama.executable_path is None
    assert llama.state == "not_configured"


def test_discovery_runs_no_subprocess_and_touches_no_real_path(monkeypatch):
    import subprocess

    def _forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("backend executable discovery spawned a process")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(subprocess, "Popen", _forbidden)
    monkeypatch.setattr(subprocess, "check_output", _forbidden)

    # Default call: real shutil.which over the real PATH is allowed (it is a
    # pure lookup), but nothing may be executed and nothing may be written.
    result = discover_backend_executables()
    assert {item.backend for item in result} == {"ollama", "llama_cpp"}
    assert all(isinstance(item, BackendExecutable) for item in result)
    assert all(item.state in {"installed", "not_installed", "not_configured"} for item in result)


def test_result_is_json_safe_and_stable_order():
    first = discover_backend_executables(which_fn=_which({"ollama": "/x/ollama"}))
    second = discover_backend_executables(which_fn=_which({"ollama": "/x/ollama"}))

    assert [i.backend for i in first] == ["ollama", "llama_cpp"]
    assert [i.to_dict() for i in first] == [i.to_dict() for i in second]
    for item in first:
        d = item.to_dict()
        assert set(d) == {"backend", "executable_available", "executable_path", "state", "source"}
