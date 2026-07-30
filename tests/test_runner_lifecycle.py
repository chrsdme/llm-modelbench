import json

import pytest

from llm_modelbench.backend import BackendCapabilities, BackendCapability, CapabilityStatus
from llm_modelbench.config import Config
from llm_modelbench.llama_cpp import LlamaCppBackendAdapter, LlamaCppClient
from llm_modelbench.ollama import MockClient
from llm_modelbench import runner


MODEL = "qwen2.5-coder:14b"


def _profile():
    return {MODEL: {"declared_capabilities": ["completion"], "supported_families": ["text"]}}


def _run(client, tmp_path):
    return runner.run(client, Config(fingerprint=False), level="smoke", out_dir=tmp_path,
                      include=None, exclude=None, skip_offload=False, categories=None,
                      task_ids=["py_anagram"], resume=False, live_ui="off", fingerprint_enabled=False,
                      selected_models=[MODEL], capability_profiles=_profile(), auto_probe=False)


class _CountingMock(MockClient):
    def __init__(self, states=None, *, fail_flush=False, fail_unload=False):
        super().__init__(); self.flushes = 0; self.unloads = 0; self.chats = 0
        self.states = states; self.fail_flush = fail_flush; self.fail_unload = fail_unload

    def backend_capabilities(self):
        return BackendCapabilities(self.states) if self.states is not None else None

    def flush_all(self):
        self.flushes += 1
        if self.fail_flush: raise RuntimeError("flush failure")

    def unload(self, model):
        self.unloads += 1
        if self.fail_unload: raise RuntimeError("unload failure")

    def chat(self, *args, **kwargs):
        self.chats += 1
        return super().chat(*args, **kwargs)


def test_runner_skips_unsupported_lifecycle_and_still_generates(tmp_path):
    states = {BackendCapability.FLUSH_ALL: CapabilityStatus.UNSUPPORTED, BackendCapability.MODEL_UNLOAD: CapabilityStatus.UNSUPPORTED}
    client = _CountingMock(states)
    _run(client, tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "raw_results.jsonl").read_text().splitlines()]
    assert client.flushes == client.unloads == 0
    assert client.chats > 0
    assert rows[0].get("error_kind") != "harness_error"


def test_runner_preserves_supported_and_legacy_lifecycle_calls(tmp_path):
    supported = _CountingMock({BackendCapability.FLUSH_ALL: CapabilityStatus.SUPPORTED, BackendCapability.MODEL_UNLOAD: CapabilityStatus.SUPPORTED})
    _run(supported, tmp_path / "supported")
    assert (supported.flushes, supported.unloads) == (1, 1)

    class Legacy(_CountingMock):
        backend_capabilities = None
    legacy = Legacy()
    _run(legacy, tmp_path / "legacy")
    assert (legacy.flushes, legacy.unloads) == (1, 1)


@pytest.mark.parametrize("failure", ["flush", "unload"])
def test_runner_preserves_supported_lifecycle_failures(tmp_path, failure):
    client = _CountingMock({BackendCapability.FLUSH_ALL: CapabilityStatus.SUPPORTED, BackendCapability.MODEL_UNLOAD: CapabilityStatus.SUPPORTED},
                           fail_flush=failure == "flush", fail_unload=failure == "unload")
    with pytest.raises(RuntimeError, match=f"{failure} failure"):
        _run(client, tmp_path)


def test_fixture_llama_cpp_adapter_runs_without_lifecycle_calls(tmp_path):
    served = "served-model"
    calls = []
    def transport(base, method, path, payload, timeout):
        calls.append((method, path, payload))
        if path == "/v1/models": return {"data": [{"id": served, "aliases": [], "meta": {"n_ctx": 4096, "size": 1}}]}
        if path == "/props": return {"build_info": "fixture", "default_generation_settings": {"n_ctx": 4096}, "modalities": {}, "chat_template_caps": {}}
        if path == "/v1/chat/completions": return {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}], "usage": {"completion_tokens": 1}}
        raise AssertionError(path)
    adapter = LlamaCppBackendAdapter(LlamaCppClient("http://127.0.0.1:8081", transport=transport))
    profile = {served: {"declared_capabilities": ["completion"], "supported_families": ["text"]}}
    runner.run(adapter, Config(fingerprint=False), level="smoke", out_dir=tmp_path,
               include=None, exclude=None, skip_offload=False, categories=None, task_ids=["py_anagram"],
               resume=False, live_ui="off", fingerprint_enabled=False, selected_models=[served],
               capability_profiles=profile, auto_probe=False)
    assert all(path not in {"/api/ps", "/api/chat"} for _, path, _ in calls)
    assert not hasattr(adapter.client, "flushes")
