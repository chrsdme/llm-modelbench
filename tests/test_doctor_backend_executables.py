"""Stage 3B.1C -- doctor surfaces backend executable availability, additively."""
from llm_modelbench import doctor


def _base_data(**overrides):
    data = {
        "llm_version": "test", "python": "3", "sys_executable": "python",
        "imported_from": "package", "entrypoint": None, "venv": None,
        "pythonpath": None, "ollama_url": "http://127.0.0.1:11434",
        "ollama_model_count": 0, "ollama_loaded_count": 0, "nvidia_smi": None,
        "node": None, "node_version": None, "gpu": {}, "gpus": [],
        "disk_free_gb": 1, "disk_total_gb": 2,
    }
    data.update(overrides)
    return data


def test_render_lists_backend_executables_when_present():
    text = doctor.render(_base_data(backend_executables=[
        {"backend": "ollama", "executable_available": True,
         "executable_path": "/usr/bin/ollama", "state": "installed", "source": "path"},
        {"backend": "llama_cpp", "executable_available": False,
         "executable_path": None, "state": "not_installed", "source": "path"},
    ]))
    assert "Backend executables:" in text
    assert "ollama: installed (/usr/bin/ollama)" in text
    assert "llama_cpp: not_installed" in text


def test_render_omits_the_section_for_pre_3b1c_artifacts():
    text = doctor.render(_base_data())
    assert "Backend executables:" not in text


def test_collect_includes_backend_executables(monkeypatch):
    import llm_modelbench.doctor as d

    monkeypatch.setattr(d, "detect_gpu", lambda: type("G", (), {"__dict__": {}})())
    monkeypatch.setattr(d, "detect_gpus", lambda: [])
    monkeypatch.setattr(d, "live_snapshot", lambda _prev: ({}, None))
    monkeypatch.setattr(d, "_url_json", lambda *a, **k: {"error": "offline"})
    monkeypatch.setattr(
        d, "discover_backend_executables",
        lambda: [type("B", (), {"to_dict": lambda self: {"backend": "ollama", "state": "installed"}})()],
    )

    data = d.collect(type("Cfg", (), {"ollama_url": "http://127.0.0.1:11434"})())
    assert data["backend_executables"] == [{"backend": "ollama", "state": "installed"}]
