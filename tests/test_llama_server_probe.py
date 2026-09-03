"""Anvil Stage 3B.3C -- production readiness / attribution / conformance
adapters (unit).

These wrap the existing ``LlamaCppClient`` and ``discover_runtime_processes``
so a connection-level failure never escapes the materialiser's bounded poll
loop and attribution is three-valued.
"""
from __future__ import annotations

from llm_modelbench import llama_server_probe as probe
from llm_modelbench.llama_cpp import LlamaCppError


class _StubClient:
    def __init__(self, *, health=None, props=None):
        self._health = health
        self._props = props

    def health(self):
        if isinstance(self._health, Exception):
            raise self._health
        return self._health

    def props(self):
        if isinstance(self._props, Exception):
            raise self._props
        return self._props


def _patch_client(monkeypatch, client):
    monkeypatch.setattr(probe, "_client", lambda url, *, timeout_s: client)


# ---------------------------------------------------------------------------
# readiness probe
# ---------------------------------------------------------------------------
def test_ready_when_health_ok(monkeypatch):
    _patch_client(monkeypatch, _StubClient(health={"status": "ok"}))
    assert probe.llama_server_readiness_probe("http://x") == "ready"


def test_not_ready_when_health_reports_loading(monkeypatch):
    _patch_client(
        monkeypatch,
        _StubClient(health=LlamaCppError("/health did not report an available server")),
    )
    assert probe.llama_server_readiness_probe("http://x") == "not_ready"


def test_unreachable_on_connection_error(monkeypatch):
    _patch_client(
        monkeypatch,
        _StubClient(health=LlamaCppError("<urlopen error [Errno 111] Connection refused>")),
    )
    assert probe.llama_server_readiness_probe("http://x") == "unreachable"


def test_wrong_service_requires_positive_evidence_from_props(monkeypatch):
    # /health malformed BUT /props returns a valid object that is not a
    # llama-server shape (no default_generation_settings) -> positively wrong.
    _patch_client(
        monkeypatch,
        _StubClient(
            health=LlamaCppError("/health response is not an object"),
            props={"server": "some-other-thing", "version": "1.0"},
        ),
    )
    assert probe.llama_server_readiness_probe("http://x") == "wrong_service"


def test_props_error_does_not_yield_wrong_service(monkeypatch):
    # /health malformed and /props ALSO errors (not connection-level) -> we
    # cannot positively prove "not a llama-server"; keep polling.
    _patch_client(
        monkeypatch,
        _StubClient(
            health=LlamaCppError("/health response is not an object"),
            props=LlamaCppError("/props build_info must be a string or null"),
        ),
    )
    assert probe.llama_server_readiness_probe("http://x") == "not_ready"


def test_props_with_llama_marker_is_not_ready_not_wrong_service(monkeypatch):
    _patch_client(
        monkeypatch,
        _StubClient(
            health=LlamaCppError("/health response is not an object"),
            props={"default_generation_settings": {"n_ctx": 4096}},
        ),
    )
    assert probe.llama_server_readiness_probe("http://x") == "not_ready"


def test_readiness_probe_never_raises(monkeypatch):
    _patch_client(monkeypatch, _StubClient(health=LlamaCppError("weird")))
    # even an unclassifiable error resolves to a string, never an exception;
    # and an unclassifiable error is the keep-polling verdict, not abandon.
    assert probe.llama_server_readiness_probe("http://x") == "not_ready"


# ---------------------------------------------------------------------------
# port attribution (three-valued)
# ---------------------------------------------------------------------------
class _Proc:
    def __init__(self, pid, ports):
        self.pid = pid
        self.listening_ports = tuple(ports)


class _DiscoveryResult:
    def __init__(self, processes, *, socket_evidence_complete=True):
        self.processes = tuple(processes)
        self.socket_evidence_complete = socket_evidence_complete


def _patch_discovery(monkeypatch, result):
    monkeypatch.setattr(probe, "discover_runtime_processes", lambda **kw: result)


def test_attribution_ours(monkeypatch):
    _patch_discovery(monkeypatch, _DiscoveryResult([_Proc(500, [8080])]))
    assert probe.llama_server_port_attribution(8080, 500) == "ours"


def test_attribution_foreign_only_with_complete_socket_evidence(monkeypatch):
    _patch_discovery(monkeypatch, _DiscoveryResult([_Proc(999, [8080])]))
    assert probe.llama_server_port_attribution(8080, 500) == "foreign"


def test_attribution_unestablished_when_no_holder(monkeypatch):
    _patch_discovery(monkeypatch, _DiscoveryResult([]))
    assert probe.llama_server_port_attribution(8080, 500) == "unestablished"


def test_attribution_unestablished_when_socket_evidence_incomplete(monkeypatch):
    _patch_discovery(
        monkeypatch,
        _DiscoveryResult([_Proc(999, [8080])], socket_evidence_complete=False),
    )
    # a foreign-looking holder but incomplete evidence -> NOT called foreign
    assert probe.llama_server_port_attribution(8080, 500) == "unestablished"


# ---------------------------------------------------------------------------
# context conformance
# ---------------------------------------------------------------------------
def test_conformance_true_when_no_requested_context(monkeypatch):
    assert probe.llama_server_context_conformance("http://x", None) is True


def test_conformance_true_when_props_unreadable(monkeypatch):
    _patch_client(monkeypatch, _StubClient(props=LlamaCppError("boom")))
    assert probe.llama_server_context_conformance("http://x", 8192) is True


def test_conformance_false_when_server_context_is_smaller(monkeypatch):
    _patch_client(
        monkeypatch,
        _StubClient(props={"default_generation_settings": {"n_ctx": 4096}}),
    )
    assert probe.llama_server_context_conformance("http://x", 8192) is False


def test_conformance_true_when_server_context_is_large_enough(monkeypatch):
    _patch_client(
        monkeypatch,
        _StubClient(props={"default_generation_settings": {"n_ctx": 16384}}),
    )
    assert probe.llama_server_context_conformance("http://x", 8192) is True
