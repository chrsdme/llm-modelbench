import json
from types import SimpleNamespace

import pytest

from llm_modelbench import cli
from llm_modelbench.runtime_profiles import (
    RuntimeCandidate,
    RuntimeProfile,
    RuntimeProfileError,
    RuntimeSelectionError,
    MAX_HEALTH_RESPONSE_BYTES,
    delete_profile,
    discover_runtimes,
    implicit_ollama_profile,
    load_profiles,
    probe_profile,
    save_profile,
    select_runtime,
)


def _cfg(url="http://127.0.0.1:11434"):
    return SimpleNamespace(ollama_url=url, seed=42, temperature=0.0, request_timeout=30)


def _candidate(name, backend, endpoint, health="healthy", recommended=False):
    return RuntimeCandidate(RuntimeProfile(name, backend, endpoint), health, ("fixture",), "fixture", recommended)


def _healthy_probe(endpoint, path):
    if "ollama" in endpoint or endpoint.endswith("11434"):
        return (200, '{"models": []}') if path == "/api/tags" else (200, '{"version": "test"}')
    return 200, "ok"


def test_profile_serialization_and_atomic_write_round_trip(tmp_path):
    store = tmp_path / "config" / "runtime_profiles.json"
    profile = RuntimeProfile("local", "ollama", "http://127.0.0.1:11434", physical_gpu_uuids=("GPU-a",))
    save_profile(profile, path=store, set_default=True)

    profiles, default = load_profiles(store)
    assert profiles == [profile]
    assert default == "local"
    assert not list(store.parent.glob("*.tmp"))

    replacement = RuntimeProfile("local", "ollama", "http://127.0.0.1:11435", description="replacement")
    save_profile(replacement, path=store, replace=True)
    assert load_profiles(store)[0] == [replacement]


def test_legacy_implicit_ollama_profile_preserves_config_url():
    profile = implicit_ollama_profile(_cfg("http://127.0.0.1:11435"))
    assert profile.name == "legacy-ollama"
    assert profile.backend == "ollama"
    assert profile.endpoint == "http://127.0.0.1:11435"
    assert profile.provenance == "legacy-default"


def test_explicit_and_default_profile_precedence():
    candidates = [_candidate("ollama", "ollama", "http://127.0.0.1:11434"),
                  _candidate("llama", "llama_cpp", "http://127.0.0.1:8081")]
    assert select_runtime(candidates, explicit_profile="llama", default_profile="ollama").profile.name == "llama"
    assert select_runtime(candidates, default_profile="ollama").profile.name == "ollama"


def test_exactly_one_and_interactive_multiple_selection():
    only = [_candidate("ollama", "ollama", "http://127.0.0.1:11434")]
    assert select_runtime(only).profile.name == "ollama"

    output = []
    selected = select_runtime(
        [_candidate("ollama", "ollama", "http://127.0.0.1:11434"),
         _candidate("llama", "llama_cpp", "http://127.0.0.1:8081", recommended=True)],
        interactive=True, input_fn=lambda _: "2", output_fn=output.append,
    )
    assert selected.profile.name == "llama"
    assert "recommended" in output[1]


def test_unattended_ambiguity_and_unhealthy_saved_default_fail_closed():
    candidates = [_candidate("ollama", "ollama", "http://127.0.0.1:11434"),
                  _candidate("llama", "llama_cpp", "http://127.0.0.1:8081")]
    with pytest.raises(RuntimeSelectionError, match="--runtime-profile"):
        select_runtime(candidates)
    unhealthy = [_candidate("saved", "ollama", "http://127.0.0.1:11434", health="unhealthy")]
    with pytest.raises(RuntimeSelectionError, match="unhealthy"):
        select_runtime(unhealthy, default_profile="saved")


def test_recommendation_uses_real_inventory_count_and_deduplicates_candidates(tmp_path):
    saved = RuntimeProfile("saved-ollama", "ollama", "http://127.0.0.1:11434")
    save_profile(saved, path=tmp_path / "profiles.json")
    process = [RuntimeProfile("process-llama", "llama_cpp", "http://127.0.0.1:8081", provenance="discovered"),
               RuntimeProfile("duplicate-ollama", "ollama", "http://127.0.0.1:11434", provenance="discovered")]

    one = discover_runtimes(_cfg(), store_path=tmp_path / "profiles.json", process_profiles=process,
                            http_probe=_healthy_probe, gpu_devices=[object()])
    assert len(one) == 2
    assert next(item for item in one if item.profile.backend == "ollama").recommended is True

    many = discover_runtimes(_cfg(), store_path=tmp_path / "profiles.json", process_profiles=process,
                             http_probe=_healthy_probe, gpu_devices=[object(), object()])
    assert next(item for item in many if item.profile.backend == "llama_cpp").recommended is True
    ollama = next(item for item in many if item.profile.backend == "ollama")
    assert set(ollama.source) == {"saved_profile", "legacy_default", "process"}


def test_profile_delete_changes_only_the_profile_store(tmp_path):
    store = tmp_path / "runtime_profiles.json"
    sentinel = tmp_path / "runtime-owned-file"
    sentinel.write_text("leave me alone")
    save_profile(RuntimeProfile("remove", "ollama", "http://127.0.0.1:11434"), path=store)
    delete_profile("remove", path=store)
    assert load_profiles(store)[0] == []
    assert sentinel.read_text() == "leave me alone"


def test_discovery_is_read_only_and_does_not_probe_nonlocal_or_create_store(tmp_path):
    calls = []
    profile = RuntimeProfile("remote", "ollama", "http://example.invalid:11434")

    candidates = discover_runtimes(
        _cfg(), store_path=tmp_path / "missing.json", process_profiles=[profile],
        http_probe=lambda endpoint, path: calls.append((endpoint, path)) or (200, "{}"), gpu_devices=[],
    )
    assert not (tmp_path / "missing.json").exists()
    assert calls == [("http://127.0.0.1:11434", "/api/version"), ("http://127.0.0.1:11434", "/api/tags")]
    assert next(item for item in candidates if item.profile.name == "remote").health == "unsupported"


def test_selected_llama_cpp_creates_stage_five_client(monkeypatch):
    profile = RuntimeProfile("llama", "llama_cpp", "http://127.0.0.1:8081")
    candidate = RuntimeCandidate(profile, "healthy", ("fixture",), "fixture")
    monkeypatch.setattr(cli, "load_profiles", lambda path: ([profile], None))
    monkeypatch.setattr(cli, "discover_runtimes", lambda cfg, store_path: [candidate])
    args = SimpleNamespace(mock=False, runtime_profile="llama", runtime_profiles_file=None)

    from llm_modelbench.llama_cpp import LlamaCppBackendAdapter
    assert isinstance(cli._client(args, _cfg()), LlamaCppBackendAdapter)


def test_invalid_profile_is_rejected_without_store_mutation(tmp_path):
    with pytest.raises(RuntimeProfileError):
        RuntimeProfile("bad name", "ollama", "http://127.0.0.1:11434")
    assert not (tmp_path / "profiles.json").exists()


class _ProbeResponse:
    status = 200

    def __init__(self, body):
        self.body = body
        self.read_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        self.read_sizes.append(size)
        return self.body


def test_ollama_large_tags_response_is_read_completely_and_healthy(monkeypatch):
    tags = json.dumps({"models": [{"name": f"model-{index}-" + "x" * 100} for index in range(60)]})
    assert len(tags.encode()) > 4096
    version = _ProbeResponse(b'{"version":"test"}')
    inventory = _ProbeResponse(tags.encode())
    responses = [version, inventory]
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: responses.pop(0))

    profile = RuntimeProfile("ollama", "ollama", "http://127.0.0.1:11434")
    assert probe_profile(profile) == ("healthy", "Ollama version/tags available")
    assert responses == []
    assert inventory.read_sizes == [MAX_HEALTH_RESPONSE_BYTES + 1]


def test_health_response_above_limit_is_rejected_explicitly(monkeypatch):
    version = _ProbeResponse(b'{"version":"test"}')
    response = _ProbeResponse(b"x" * (MAX_HEALTH_RESPONSE_BYTES + 1))
    responses = [version, response]
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: responses.pop(0))

    health, detail = probe_profile(RuntimeProfile("ollama", "ollama", "http://127.0.0.1:11434"))
    assert health == "unhealthy"
    assert "bounded-size limit" in detail
    assert response.read_sizes == [MAX_HEALTH_RESPONSE_BYTES + 1]


@pytest.mark.parametrize(
    ("tags_body", "detail"),
    [
        ("[]", "did not return an object"),
        ("{}", "did not return models"),
    ],
)
def test_ollama_tags_invalid_shape_is_unhealthy_without_traceback(tags_body, detail):
    profile = RuntimeProfile("ollama", "ollama", "http://127.0.0.1:11434")

    health, result = probe_profile(
        profile,
        http_probe=lambda endpoint, path: (200, '{"version":"test"}') if path == "/api/version" else (200, tags_body),
    )

    assert health == "unhealthy"
    assert detail in result


def test_runtime_delete_unknown_profile_fails_before_confirmation(tmp_path, monkeypatch):
    confirmed = []
    monkeypatch.setattr(cli, "_confirm_profile_change", lambda *args, **kwargs: confirmed.append(args))
    args = SimpleNamespace(runtime_cmd="delete", name="missing", yes=False, runtime_profiles_file=str(tmp_path / "profiles.json"))

    with pytest.raises(SystemExit, match="unknown runtime profile: missing"):
        cli.cmd_runtime(args, _cfg())
    assert confirmed == []


def test_runtime_cli_profile_errors_exit_cleanly(tmp_path, monkeypatch):
    store = str(tmp_path / "profiles.json")
    save_args = SimpleNamespace(
        runtime_cmd="save", name="bad", backend="ollama", endpoint="not-a-url", gpu_uuid=None,
        description=None, provenance="configured", replace=False, set_default=False, yes=True,
        runtime_profiles_file=store,
    )
    with pytest.raises(SystemExit, match="runtime endpoint"):
        cli.cmd_runtime(save_args, _cfg())

    show_args = SimpleNamespace(runtime_cmd="show", name="missing", runtime_profiles_file=store)
    with pytest.raises(SystemExit, match="unknown runtime profile: missing"):
        cli.cmd_runtime(show_args, _cfg())

    monkeypatch.setattr(cli, "discover_runtimes", lambda cfg, store_path: [_candidate("only", "ollama", "http://127.0.0.1:11434")])
    select_args = SimpleNamespace(
        runtime_cmd="select", runtime_profile="missing", save_name=None, set_default=False,
        replace=False, yes=True, runtime_profiles_file=store,
    )
    with pytest.raises(SystemExit, match="unknown runtime profile: missing"):
        cli.cmd_runtime(select_args, _cfg())


def test_client_ambiguity_does_not_fall_back_to_ollama(monkeypatch):
    candidates = [
        _candidate("ollama", "ollama", "http://127.0.0.1:11434"),
        _candidate("llama", "llama_cpp", "http://127.0.0.1:8081"),
    ]
    monkeypatch.setattr(cli, "load_profiles", lambda path: ([], None))
    monkeypatch.setattr(cli, "discover_runtimes", lambda cfg, store_path: candidates)
    args = SimpleNamespace(mock=False, runtime_profile=None, runtime_profiles_file=None)

    with pytest.raises(SystemExit, match="--runtime-profile"):
        cli._client(args, _cfg())


def test_client_legacy_fallback_only_applies_with_no_healthy_candidates(monkeypatch):
    monkeypatch.setattr(cli, "load_profiles", lambda path: ([], None))
    monkeypatch.setattr(
        cli, "discover_runtimes",
        lambda cfg, store_path: [_candidate("legacy-ollama", "ollama", cfg.ollama_url, health="unreachable")],
    )
    cfg = _cfg()
    client = cli._client(SimpleNamespace(mock=False, runtime_profile=None, runtime_profiles_file=None), cfg)

    assert isinstance(client, cli.OllamaBackendAdapter)
    assert client.backend_identity().endpoint == "http://127.0.0.1:11434"


@pytest.mark.parametrize("explicit, default", [("saved", None), (None, "saved")])
def test_client_explicit_or_default_unhealthy_profile_remains_fail_closed(monkeypatch, explicit, default):
    profile = RuntimeProfile("saved", "ollama", "http://127.0.0.1:11434")
    candidate = RuntimeCandidate(profile, "unhealthy", ("fixture",), "fixture")
    monkeypatch.setattr(cli, "load_profiles", lambda path: ([profile], default))
    monkeypatch.setattr(cli, "discover_runtimes", lambda cfg, store_path: [candidate])

    with pytest.raises(SystemExit, match="is unhealthy"):
        cli._client(SimpleNamespace(mock=False, runtime_profile=explicit, runtime_profiles_file=None), _cfg())
