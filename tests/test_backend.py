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
        lambda received_cfg, store_path: discovery_calls.append((received_cfg, store_path)) or [candidate],
    )

    mock = cli._client(SimpleNamespace(mock=True), cfg)
    real = cli._client(SimpleNamespace(mock=False), cfg)

    assert isinstance(mock, MockBackendAdapter)
    assert isinstance(real, OllamaBackendAdapter)
    assert mock.backend_identity().backend == "mock"
    assert real.backend_identity().backend == "ollama"
    assert discovery_calls == [(cfg, cli._runtime_store(SimpleNamespace()))]
