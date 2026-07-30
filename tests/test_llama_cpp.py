import json
import urllib.error

import pytest

from llm_modelbench.backend import BackendCapability, BackendCapabilityError, InferenceClient
from llm_modelbench.llama_cpp import LlamaCppBackendAdapter, LlamaCppClient, LlamaCppError, _default_transport
from llm_modelbench.runtime_profiles import RuntimeCandidate, RuntimeProfile


MODEL = "/models/sha256-b0651e28555bde7d2459ce99f091319b1a547143463e8d49f2aa7f572675fe67"


def _props(**extra):
    value = {
        "build_info": "b10086-66e4bf7e5",
        "default_generation_settings": {"n_ctx": 65536},
        "modalities": {"vision": False, "audio": False, "video": False},
        "chat_template": "{% if enable_thinking is defined %}{% endif %}",
        "chat_template_caps": {"supports_tools": True, "supports_tool_calls": True,
                               "supports_system_role": True, "supports_preserve_reasoning": True},
    }
    value.update(extra)
    return value


def _models(rows=None, legacy=True):
    data = rows if rows is not None else [{"id": MODEL, "aliases": ["served-alias"], "object": "model",
        "meta": {"n_ctx": 65536, "n_ctx_train": 262144, "n_params": 27320697856,
                 "size": 16799719424, "ftype": "Q4_K - Medium"}}]
    value = {"object": "list", "data": data}
    if legacy:
        value["models"] = [{"name": MODEL, "model": MODEL, "capabilities": ["completion"]}]
    return value


class _Transport:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, base, method, path, payload, timeout):
        self.calls.append((method, path, payload))
        value = self.responses[path]
        if callable(value):
            return value(payload)
        return value


def _client(responses=None):
    transport = _Transport(responses if responses is not None else {"/health": {"status": "ok"}, "/props": _props(), "/v1/models": _models(),
                                         "/slots": [{"id": 0, "n_ctx": 65536, "is_processing": False}],
                                         "/tokenize": {"tokens": [1, 2, 3]}})
    return LlamaCppClient("http://127.0.0.1:8081", transport=transport), transport


def test_protocol_identity_inventory_and_observed_metadata():
    client, _ = _client()
    adapter = LlamaCppBackendAdapter(client)
    assert isinstance(adapter, InferenceClient)
    assert adapter.backend_identity().endpoint == "http://127.0.0.1:8081"
    assert adapter.version() == "b10086-66e4bf7e5"
    assert adapter.tags()[0]["name"] == MODEL
    assert adapter.tags()[0]["digest"] == "sha256-b0651e28555bde7d2459ce99f091319b1a547143463e8d49f2aa7f572675fe67"
    assert adapter.context_length(MODEL) == 65536
    assert adapter.show(MODEL)["model_info"]["general.training_context_length"] == 262144
    assert adapter.model_size_bytes(MODEL) == 16799719424
    assert "tools" in adapter.capabilities(MODEL)
    assert adapter.backend_capabilities().supports(BackendCapability.CHAT)
    assert not adapter.backend_capabilities().supports(BackendCapability.EMBEDDINGS)


def test_inventory_deduplicates_dual_arrays_and_rejects_zero_or_router_mode():
    client, _ = _client()
    assert len(client.tags()) == 1
    zero, _ = _client({"/v1/models": {"data": []}})
    with pytest.raises(LlamaCppError, match="no served model"):
        zero.tags()
    multi, _ = _client({"/v1/models": _models(rows=[{"id": "one"}, {"id": "two"}], legacy=False)})
    with pytest.raises(LlamaCppError, match="multi-model"):
        multi.tags()


def test_authoritative_data_ignores_differently_named_compatibility_models_and_validates_rows():
    payload = _models()
    payload["models"] = [{"name": "different-compatibility-name", "tags": []}]
    client, _ = _client({"/v1/models": payload})
    assert client.tags()[0]["name"] == MODEL
    fallback, _ = _client({"/v1/models": {"models": [{"name": "fallback", "tags": [], "details": {}}]}})
    assert fallback.tags()[0]["name"] == "fallback"
    bad_alias, _ = _client({"/v1/models": {"data": [{"id": "one", "aliases": "not-a-list", "meta": {}}]}})
    with pytest.raises(LlamaCppError, match="aliases"):
        bad_alias.tags()
    bad_meta, _ = _client({"/v1/models": {"data": [{"id": "one", "aliases": [], "meta": []}]}})
    with pytest.raises(LlamaCppError, match="metadata"):
        bad_meta.tags()
    for field in ("size", "n_params", "n_ctx", "n_ctx_train"):
        malformed, _ = _client({"/v1/models": {"data": [{"id": "one", "aliases": [], "meta": {field: True}}]}})
        with pytest.raises(LlamaCppError, match="must be an integer"):
            malformed.tags()
    compatibility = _models(rows=[{"id": MODEL, "aliases": [], "meta": {}}])
    compatibility["models"] = [{"name": MODEL, "tags": [], "details": {"size": True}}]
    malformed, _ = _client({"/v1/models": compatibility})
    with pytest.raises(LlamaCppError, match="must be an integer"):
        malformed.tags()


def test_tags_do_not_invent_format_or_vision_capability():
    client, _ = _client({"/v1/models": {"data": [{"id": "plain", "aliases": [], "meta": {}}]}, "/props": _props(modalities={"vision": True})})
    item = client.tags()[0]
    assert "details" not in item
    assert "digest" not in item
    assert "vision" not in client.capabilities("plain")


def test_model_alias_validation_context_limit_slots_and_tokenization():
    client, transport = _client()
    assert client.model_size_bytes("served-alias") == 16799719424
    with pytest.raises(LlamaCppError, match="not served"):
        client.model_info("other")
    assert client.slots()[0]["id"] == 0
    assert client.tokenize("hello", add_special=True)["tokens"] == [1, 2, 3]
    assert transport.calls[-1] == ("POST", "/tokenize", {"content": "hello", "add_special": True, "parse_special": False, "with_pieces": False})
    result = client.chat(MODEL, "hi", num_ctx=65537)
    assert result["ok"] is False and "exceeds served context" in result["error"]


@pytest.mark.parametrize("kwargs", [{"num_ctx": True}, {"num_predict": True}])
def test_boolean_chat_integer_requests_fail_before_chat_request(kwargs):
    client, transport = _client()
    result = client.chat(MODEL, "hi", **kwargs)
    assert result["ok"] is False and "must be an integer" in result["error"]
    assert not any(path == "/v1/chat/completions" for _, path, _ in transport.calls)


def test_boolean_seed_fails_before_chat_request():
    client, transport = _client()
    client.seed = True
    result = client.chat(MODEL, "hi")
    assert result["ok"] is False and "request seed must be an integer" in result["error"]
    assert not any(path == "/v1/chat/completions" for _, path, _ in transport.calls)


def test_chat_normalizes_reasoning_usage_timings_and_request_mapping():
    def completion(payload):
        assert payload["stream"] is False and payload["max_tokens"] == 12
        assert "enable_thinking" not in payload
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["messages"][0]["role"] == "system"
        return {"choices": [{"finish_reason": "stop", "message": {"content": "answer", "reasoning_content": "reason"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                "timings": {"prompt_per_second": 100, "predicted_per_second": 25, "total_ms": 80}}
    client, _ = _client({"/health": {"status": "ok"}, "/props": _props(), "/v1/models": _models(),
                         "/v1/chat/completions": completion})
    result = client.chat(MODEL, "prompt", system="system", num_predict=12, think="off", format="json")
    assert result["ok"] and result["thinking"] == "reason"
    assert result["tokens"] == 2 and result["prompt_eval_count"] == 4 and result["tps"] == 25.0


@pytest.mark.parametrize(("think", "expected"), [("on", True), ("off", False), ("auto", None)])
def test_thinking_uses_chat_template_kwargs_only(think, expected):
    def completion(payload):
        assert "enable_thinking" not in payload
        if expected is None:
            assert payload["chat_template_kwargs"] == {"preserve_thinking": True}
        else:
            assert payload["chat_template_kwargs"] == {"enable_thinking": expected, "preserve_thinking": True}
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}
    client, _ = _client({"/props": _props(), "/v1/models": _models(), "/v1/chat/completions": completion})
    assert client.chat(MODEL, "x", think=think, chat_template_kwargs={"preserve_thinking": True})["ok"]


def test_thinking_kwargs_reject_malformed_or_conflicting_values_before_chat():
    client, transport = _client()
    assert client.chat(MODEL, "x", think="on", chat_template_kwargs=[])["ok"] is False
    assert client.chat(MODEL, "x", think="off", chat_template_kwargs={"enable_thinking": True})["ok"] is False
    assert not any(path == "/v1/chat/completions" for _, path, _ in transport.calls)


def test_chat_schema_stop_and_message_system_policy():
    def completion(payload):
        assert payload["stop"] == ["END"]
        assert payload["messages"] == [{"role": "system", "content": "listed"}, {"role": "user", "content": "turn two"}]
        assert payload["response_format"] == {"type": "json_schema", "json_schema": {"name": "response", "schema": {"type": "object"}}}
        return {"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}], "usage": {"completion_tokens": 1}}
    client, _ = _client({"/props": _props(), "/v1/models": _models(), "/v1/chat/completions": completion})
    assert client.chat(MODEL, "ignored", system="argument", messages=[{"role": "system", "content": "listed"}, {"role": "user", "content": "turn two"}], stop=["END"], format={"type": "object"})["text"] == '{"ok":true}'
    with pytest.raises(LlamaCppError, match="messages"):
        client._messages("x", None, ["bad"])


def test_messages_validate_roles_content_and_system_placement():
    client, _ = _client()
    assert client._messages("p", "system", [{"role": "user", "content": "u"}]) == [{"role": "system", "content": "system"}, {"role": "user", "content": "u"}]
    assert client._messages("p", "ignored", [{"role": "system", "content": "listed"}, {"role": "assistant", "content": None, "tool_calls": []}])[0]["content"] == "listed"
    for messages in (
        [{"role": "invalid", "content": "x"}], [{"role": "user", "content": ["x"]}],
        [{"role": "user", "content": "x"}, {"role": "system", "content": "late"}], [{"role": "user", "content": None}],
    ):
        with pytest.raises(LlamaCppError):
            client._messages("p", None, messages)


@pytest.mark.parametrize("response", [[], {"choices": "bad"}, {"choices": [{}]}, {"choices": [{"message": {"content": 3}}]}, {"choices": [{"message": {"content": "x"}}], "usage": []}, {"choices": [{"message": {"content": "x"}}], "timings": []}])
def test_malformed_chat_shapes_are_clean_errors(response):
    client, _ = _client()
    with pytest.raises(LlamaCppError):
        client._normalise_chat(response, num_predict=1, num_ctx=None, think="auto")


@pytest.mark.parametrize("response", [
    {"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": True}},
    {"choices": [{"message": {"content": "x"}}], "usage": {"completion_tokens": True}},
    {"choices": [{"message": {"content": "x"}}], "usage": {"total_tokens": True}},
    {"choices": [{"message": {"content": "x"}}], "usage": {}, "timings": {"prompt_n": True}},
    {"choices": [{"message": {"content": "x"}}], "usage": {}, "timings": {"predicted_n": False}},
])
def test_boolean_usage_and_timing_integer_fields_are_clean_errors(response):
    client, _ = _client()
    with pytest.raises(LlamaCppError, match="must be an integer"):
        client._normalise_chat(response, num_predict=1, num_ctx=None, think="auto")


def test_think_tags_tools_and_unsupported_operations_are_explicit():
    def completion(payload):
        assert payload["tools"][0]["function"]["name"] == "lookup"
        return {"choices": [{"finish_reason": "tool_calls", "message": {
            "content": "<think>private</think>calling", "tool_calls": [{"function": {"name": "lookup", "arguments": '{"city":"Paris"}'}}]}}],
                "usage": {"completion_tokens": 1}}
    client, _ = _client({"/health": {"status": "ok"}, "/props": _props(), "/v1/models": _models(),
                         "/v1/chat/completions": completion})
    result = client.chat_tools(MODEL, "weather", tools=[{"type": "function", "function": {"name": "lookup"}}])
    assert result["thinking"] == "private"
    assert result["tool_calls"] == [{"function": {"name": "lookup", "arguments": {"city": "Paris"}}}]
    assert client.generate_suffix(MODEL, "x", suffix="y")["ok"] is False
    with pytest.raises(BackendCapabilityError):
        client.embed(MODEL, ["x"])
    with pytest.raises(BackendCapabilityError):
        client.unload(MODEL)
    with pytest.raises(BackendCapabilityError):
        client.flush_all()


def test_tool_call_shape_and_optional_messages_are_validated():
    def completion(payload):
        assert payload["messages"] == [{"role": "user", "content": "turn"}]
        return {"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": {"city": "Paris"}}}
        ]}}], "usage": {}}
    client, _ = _client({"/props": _props(), "/v1/models": _models(), "/v1/chat/completions": completion})
    result = client.chat_tools(MODEL, "ignored", messages=[{"role": "user", "content": "turn"}], tools=[{"type": "function", "function": {"name": "lookup"}}])
    assert result["text"] == "" and result["tool_calls"][0]["id"] == "call-1"
    bad, _ = _client({"/props": _props(), "/v1/models": _models(), "/v1/chat/completions": {"choices": [{"message": {"content": None, "tool_calls": [{"function": {"name": "lookup", "arguments": 1}}]}}]}})
    assert bad.chat_tools(MODEL, "x", tools=[])["ok"] is False


@pytest.mark.parametrize("call", [
    {"id": 1, "function": {"name": "lookup", "arguments": {}}},
    {"type": 1, "function": {"name": "lookup", "arguments": {}}},
    {"function": {"name": "", "arguments": {}}},
    {"function": {"name": "lookup", "arguments": []}},
])
def test_malformed_tool_call_fields_fail_cleanly(call):
    response = {"choices": [{"message": {"content": None, "tool_calls": [call]}}], "usage": {}}
    client, _ = _client({"/props": _props(), "/v1/models": _models(), "/v1/chat/completions": response})
    assert client.chat_tools(MODEL, "x", tools=[])["ok"] is False


def test_invalid_token_shape_and_no_mutating_endpoint_calls():
    client, transport = _client({"/health": {"status": "ok"}, "/props": _props(), "/v1/models": _models(),
                                 "/tokenize": {"tokens": ["bad"]}})
    with pytest.raises(LlamaCppError, match="invalid token array"):
        client.tokenize("x")
    assert all(path not in {"/api/delete", "/v1/models/load", "/api/ps", "/props"} for _, path, _ in transport.calls)


def test_slots_metrics_and_lifecycle_allow_list():
    client, transport = _client({"/slots": [{"id": 0, "is_processing": False}], "/metrics": {"text": "llama_requests_total 1\n"}})
    assert client.slots()[0]["id"] == 0
    assert client.metrics()["text"] == "llama_requests_total 1\n"
    assert {(method, path) for method, path, _ in transport.calls} == {("GET", "/slots"), ("GET", "/metrics")}
    malformed, _ = _client({"/slots": ["bad"]})
    with pytest.raises(LlamaCppError, match="entries"):
        malformed.slots()
    for row in ([{"id": "bad"}], [{"id": True}], [{"is_processing": 1}], [{"n_ctx": False}], [{"speculative": "no"}]):
        malformed, _ = _client({"/slots": row})
        with pytest.raises(LlamaCppError): malformed.slots()


def test_props_and_response_format_shapes_fail_before_chat_request():
    for props in (_props(build_info=1), _props(total_slots=-1), _props(total_slots="one"), _props(total_slots=True),
                  _props(default_generation_settings={"n_ctx": True})):
        client, _ = _client({"/props": props})
        with pytest.raises(LlamaCppError): client.props()
    client, transport = _client()
    for value in ("bad", {"type": "json_object", "schema": []}, {"type": "json_schema"}, {"type": "json_schema", "json_schema": {"schema": []}}):
        assert client.chat(MODEL, "x", response_format=value)["ok"] is False
    assert not any(path == "/v1/chat/completions" for _, path, _ in transport.calls)


def test_complete_read_only_endpoint_allow_list_and_unsupported_zero_requests():
    response = {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}], "usage": {"completion_tokens": 1}}
    client, transport = _client({"/health": {"status": "ok"}, "/props": _props(), "/v1/models": _models(),
                                 "/slots": [{"id": 0, "is_processing": False}], "/metrics": {"text": "metric 1\n"},
                                 "/tokenize": {"tokens": [1]}, "/v1/chat/completions": response})
    client.health(); client.tags(); client.props(); client.slots(); client.metrics(); client.tokenize("x"); client.chat(MODEL, "x")
    assert {(method, path) for method, path, _ in transport.calls} <= {
        ("GET", "/health"), ("GET", "/v1/models"), ("GET", "/props"), ("GET", "/slots"), ("GET", "/metrics"),
        ("POST", "/tokenize"), ("POST", "/v1/chat/completions"),
    }
    calls_before = list(transport.calls)
    with pytest.raises(BackendCapabilityError): client.embed(MODEL, ["x"])
    with pytest.raises(BackendCapabilityError): client.unload(MODEL)
    with pytest.raises(BackendCapabilityError): client.flush_all()
    assert transport.calls == calls_before


@pytest.mark.parametrize("operation", [
    lambda client: client.generate_suffix("unknown", "x", suffix="y"),
    lambda client: client.embed("unknown", ["x"]),
    lambda client: client.offload_fraction("unknown"),
    lambda client: client.unload("unknown"),
    lambda client: client.flush_all(),
])
def test_unsupported_operations_are_cold_cache_transport_free(operation):
    client, transport = _client({})
    try:
        operation(client)
    except BackendCapabilityError:
        pass
    assert transport.calls == []


def test_http_redirect_error_body_timeout_and_connection_are_bounded(monkeypatch):
    class Error(urllib.error.HTTPError):
        def __init__(self): super().__init__("http://127.0.0.1:8081/health", 302, "redirect", {}, None)
        def read(self, amount=None): return b"x" * 9000
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: type("Opener", (), {"open": lambda *a, **k: (_ for _ in ()).throw(Error())})())
    with pytest.raises(LlamaCppError, match="redirect rejected"):
        _default_transport("http://127.0.0.1:8081", "GET", "/health", None, 1)
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: type("Opener", (), {"open": lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("refused"))})())
    with pytest.raises(LlamaCppError, match="connection failure"):
        _default_transport("http://127.0.0.1:8081", "GET", "/health", None, 1)


def test_http_error_body_is_capped(monkeypatch):
    class Error(urllib.error.HTTPError):
        def __init__(self): super().__init__("http://127.0.0.1:8081/health", 500, "bad", {}, None)
        def read(self, amount=None): return b"z" * 9000
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: type("Opener", (), {"open": lambda *a, **k: (_ for _ in ()).throw(Error())})())
    with pytest.raises(LlamaCppError) as caught:
        _default_transport("http://127.0.0.1:8081", "GET", "/health", None, 1)
    assert len(str(caught.value)) <= 8300


def test_http_success_limit_error_limit_and_timeout_are_bounded(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, amount): return b"x" * amount
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: type("Opener", (), {"open": lambda *a, **k: Response()})())
    with pytest.raises(LlamaCppError, match="bounded-size limit"):
        _default_transport("http://127.0.0.1:8081", "GET", "/health", None, 1)
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: type("Opener", (), {"open": lambda *a, **k: (_ for _ in ()).throw(TimeoutError())})())
    with pytest.raises(LlamaCppError, match="timed out"):
        _default_transport("http://127.0.0.1:8081", "GET", "/health", None, 1)


def test_selected_llama_cpp_inventory_error_is_clean_and_never_constructs_ollama(monkeypatch):
    from llm_modelbench import cli, llama_cpp, ollama
    profile = RuntimeProfile("llama", "llama_cpp", "http://127.0.0.1:8081")
    monkeypatch.setattr(cli, "load_profiles", lambda path: ([profile], None))
    monkeypatch.setattr(cli, "discover_runtimes", lambda cfg, store_path: [RuntimeCandidate(profile, "healthy", ("fixture",), "fixture")])

    class FailingClient:
        def __init__(self, base, *args): self.base = base
        def tags(self): raise LlamaCppError("/v1/models invalid fixture response")

    monkeypatch.setattr(llama_cpp, "LlamaCppClient", FailingClient)
    monkeypatch.setattr(ollama, "OllamaClient", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no Ollama fallback")))
    with pytest.raises(SystemExit, match="llama.cpp error: /v1/models invalid fixture response"):
        cli.main(["inventory", "--runtime-profile", "llama"])
