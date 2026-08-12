from types import SimpleNamespace

import pytest

from llm_modelbench.backend import (
    BackendCapabilities,
    BackendCapability,
    BackendCapabilityError,
    BackendIdentity,
    CapabilityStatus,
    InferenceClient,
    MockBackendAdapter,
    OllamaBackendAdapter,
    require_capability,
    supports_capability,
)
from llm_modelbench.config import Config
from llm_modelbench.identity import RuntimeProfileIdentity
from llm_modelbench.ollama import MockClient, OllamaClient
from llm_modelbench.runtime_profiles import RuntimeCandidate, RuntimeProfile


def test_ollama_adapter_conforms_and_delegates_without_network(monkeypatch):
    client = OllamaClient("http://127.0.0.1:11434")
    adapter = OllamaBackendAdapter(client)
    monkeypatch.setattr(client, "tags", lambda: [{"name": "delegated"}])
    monkeypatch.setattr(client, "chat", lambda model, prompt, **kwargs: {
        "ok": True, "model": model, "prompt": prompt, "kwargs": kwargs,
    })

    assert isinstance(adapter, InferenceClient)
    assert adapter.backend_identity() == BackendIdentity(
        "ollama", "ollama", "http://127.0.0.1:11434"
    )
    assert adapter.tags() == [{"name": "delegated"}]
    assert adapter.chat("model", "prompt", think="off") == {
        "ok": True, "model": "model", "prompt": "prompt", "kwargs": {"think": "off"},
    }


def test_mock_adapter_preserves_mock_behavior_and_identity():
    direct = MockClient()
    adapter = MockBackendAdapter(direct)

    assert isinstance(adapter, InferenceClient)
    assert adapter.backend_identity() == BackendIdentity("mock", "llm-modelbench-mock")
    assert adapter.tags() == direct.tags()
    assert adapter.chat("qwen2.5-coder:14b", "Return exactly AIW_TEXT_OK and nothing else.") == direct.chat(
        "qwen2.5-coder:14b", "Return exactly AIW_TEXT_OK and nothing else."
    )


def test_capability_statuses_and_ollama_only_guards_are_explicit():
    capabilities = BackendCapabilities({
        BackendCapability.CHAT: CapabilityStatus.UNAVAILABLE,
        BackendCapability.EMBEDDINGS: CapabilityStatus.UNSUPPORTED,
    })
    assert capabilities.state(BackendCapability.CHAT) is CapabilityStatus.UNAVAILABLE
    assert capabilities.state(BackendCapability.EMBEDDINGS) is CapabilityStatus.UNSUPPORTED
    assert capabilities.state(BackendCapability.NATIVE_TOOLS) is CapabilityStatus.UNKNOWN

    adapter = MockBackendAdapter(MockClient())
    assert adapter.backend_capabilities().supports(BackendCapability.CHAT)
    with pytest.raises(BackendCapabilityError, match="ollama_service_repair.*unsupported"):
        require_capability(adapter, BackendCapability.OLLAMA_SERVICE_REPAIR)
    with pytest.raises(BackendCapabilityError, match="ollama_kv_repair.*unsupported"):
        require_capability(adapter, BackendCapability.OLLAMA_KV_REPAIR)
    assert supports_capability(adapter, BackendCapability.OFFLOAD_FRACTION)


def test_adapter_preserves_underlying_exception_behavior(monkeypatch):
    client = OllamaClient("http://127.0.0.1:11434")
    adapter = OllamaBackendAdapter(client)

    def fail(*args, **kwargs):
        raise RuntimeError("existing client failure")

    monkeypatch.setattr(client, "embed", fail)
    with pytest.raises(RuntimeError, match="existing client failure"):
        adapter.embed("model", ["text"])


def test_cli_client_construction_uses_neutral_adapters(monkeypatch):
    from llm_modelbench import cli, runtime_profiles

    cfg = Config()
    candidate = RuntimeCandidate(
        RuntimeProfile("test-ollama", "ollama", "http://127.0.0.1:11434"),
        "healthy", ("fixture",), "fixture",
    )
    discovery_calls = []
    def unexpected_host_access(*args, **kwargs):
        raise AssertionError("adapter construction must not consult host runtime state")

    monkeypatch.setattr(runtime_profiles, "_process_profiles", unexpected_host_access)
    monkeypatch.setattr("urllib.request.urlopen", unexpected_host_access)
    monkeypatch.setattr(cli, "load_profiles", lambda path: ([], None))
    monkeypatch.setattr(
        cli,
        "discover_runtimes",
        lambda received_cfg, store_path, gpu_devices=None: discovery_calls.append((received_cfg, store_path)) or [candidate],
    )

    mock = cli._client(SimpleNamespace(mock=True), cfg)
    real = cli._client(SimpleNamespace(mock=False), cfg)

    assert isinstance(mock, MockBackendAdapter)
    assert isinstance(real, OllamaBackendAdapter)
    assert mock.backend_identity().backend == "mock"
    assert real.backend_identity().backend == "ollama"
    assert discovery_calls == [(cfg, cli._runtime_store(SimpleNamespace()))]


# ---------------------------------------------------------------------------
# Anvil Stage 1.2 additions: health(), runtime_profile_identity(), and the
# Stage 3B-dependent managed-lifecycle stubs.
# ---------------------------------------------------------------------------


def test_mock_adapter_health_true_when_version_available():
    adapter = MockBackendAdapter(MockClient())
    assert adapter.health() is True


def test_ollama_adapter_health_false_on_any_failure(monkeypatch):
    client = OllamaClient("http://127.0.0.1:11434")
    adapter = OllamaBackendAdapter(client)
    monkeypatch.setattr(client, "version", lambda: (_ for _ in ()).throw(RuntimeError("unreachable")))
    assert adapter.health() is False


def test_ollama_adapter_health_false_when_version_is_none(monkeypatch):
    client = OllamaClient("http://127.0.0.1:11434")
    adapter = OllamaBackendAdapter(client)
    monkeypatch.setattr(client, "version", lambda: None)
    assert adapter.health() is False


def test_runtime_profile_identity_reflects_backend_and_version():
    adapter = MockBackendAdapter(MockClient())
    identity = adapter.runtime_profile_identity()
    assert isinstance(identity, RuntimeProfileIdentity)
    assert identity.backend == "mock"
    # MockClient's version() is deterministic -- confirm it round-trips,
    # not just that some value is present.
    assert identity.backend_version == MockClient().version()


def test_runtime_profile_identity_survives_version_failure(monkeypatch):
    client = OllamaClient("http://127.0.0.1:11434")
    adapter = OllamaBackendAdapter(client)
    monkeypatch.setattr(client, "version", lambda: (_ for _ in ()).throw(RuntimeError("unreachable")))
    identity = adapter.runtime_profile_identity()
    assert identity.backend == "ollama"
    assert identity.backend_version is None


@pytest.mark.parametrize(
    "method_name,args",
    [
        ("start_model", ("some-model",)),
        ("stop_model", ("some-model",)),
        ("load_model", ("some-model",)),
        ("launch_managed_runtime", ()),
        ("stop_managed_runtime", ()),
        ("switch_model", ("some-model",)),
    ],
)
def test_stage3b_dependent_methods_fail_closed_on_every_current_adapter(method_name, args):
    """Not built yet (Stage 3B), and every current adapter must say so
    clearly -- not raise AttributeError, not silently no-op."""
    for adapter in (OllamaBackendAdapter(OllamaClient("http://127.0.0.1:11434")), MockBackendAdapter(MockClient())):
        method = getattr(adapter, method_name)
        with pytest.raises(BackendCapabilityError):
            method(*args)


def test_stage3b_capabilities_are_explicitly_declared_unsupported_not_unknown():
    """Distinguishes "not built yet" from "nobody said" -- an UNKNOWN
    status would silently pass some fail-open checks that UNSUPPORTED
    correctly blocks."""
    for capability in (
        BackendCapability.START_MODEL,
        BackendCapability.STOP_MODEL,
        BackendCapability.LOAD_MODEL,
        BackendCapability.MANAGED_RUNTIME_LAUNCH,
        BackendCapability.MANAGED_RUNTIME_STOP,
        BackendCapability.MODEL_SWITCH,
    ):
        for adapter in (
            OllamaBackendAdapter(OllamaClient("http://127.0.0.1:11434")),
            MockBackendAdapter(MockClient()),
        ):
            assert adapter.backend_capabilities().state(capability) is CapabilityStatus.UNSUPPORTED


def test_health_check_capability_is_supported_on_both_current_adapters():
    for adapter in (OllamaBackendAdapter(OllamaClient("http://127.0.0.1:11434")), MockBackendAdapter(MockClient())):
        assert supports_capability(adapter, BackendCapability.HEALTH_CHECK)
