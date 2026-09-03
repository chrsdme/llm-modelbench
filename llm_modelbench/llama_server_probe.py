"""Anvil Stage 3B.3C -- production readiness / attribution / conformance
adapters for managed ``llama-server`` materialisation.

These are the concrete callables :func:`llm_modelbench.llama_server_materialisation.spawn_managed_llama_server`
takes as injected seams. They are kept out of the materialiser itself so
that module stays free of HTTP-client and procfs specifics and is fully
testable with fakes.

* :func:`llama_server_readiness_probe` -- reuses the existing
  :class:`~llm_modelbench.llama_cpp.LlamaCppClient` health/props checks. It
  MUST convert every connection-level failure (``LlamaCppError`` from a
  refused connection / timeout) into the string ``"unreachable"`` -- an
  uncaught raise would escape the materialiser's bounded poll loop and
  defeat the timeout guarantee.
* :func:`llama_server_port_attribution` -- three-valued port->PID
  attribution over :func:`~llm_modelbench.telemetry.discover_runtime_processes`
  (``/proc/net/tcp`` + ``/proc/<pid>/fd``). "Attribution could not be
  established" (permission-denied ``fd`` reads, a truncated PID scan) is a
  distinct value from "provably foreign" and must never be read as foreign.
* :func:`llama_server_context_conformance` -- an independent recipe check:
  the running server's reported context (``/props`` ``n_ctx``) vs the
  resolved ``requested_context``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .llama_cpp import LlamaCppClient, LlamaCppError
from .telemetry import PROC_ROOT, discover_runtime_processes

__all__ = [
    "llama_server_readiness_probe",
    "llama_server_port_attribution",
    "llama_server_context_conformance",
]


def _client(base_url: str, *, timeout_s: float) -> LlamaCppClient:
    return LlamaCppClient(base_url, timeout=int(max(1, round(timeout_s))))


def llama_server_readiness_probe(base_url: str, *, timeout_s: float = 2.0) -> str:
    """Return ``"ready"`` / ``"not_ready"`` / ``"unreachable"`` /
    ``"wrong_service"`` for ``base_url``.

    * ``/health`` reports available -> ``"ready"``.
    * ``/health`` reachable but not available (HTTP 5xx / not-ok body) ->
      ``"not_ready"``.
    * connection refused / timeout -> ``"unreachable"``.
    * the endpoint answers ``/props`` with a *structurally valid* response
      that is nonetheless not a llama-server shape -> ``"wrong_service"``.

    Fallback direction: an error we cannot positively classify resolves to
    ``"not_ready"`` (keep polling). ``"wrong_service"`` -- which abandons the
    endpoint and burns a candidate port -- requires *positive* evidence
    (``/props`` returned a valid object lacking the llama-server marker). The
    materialiser's bounded readiness timeout is the fail-safe for "never
    becomes ready"; a misclassified transient error must not short-circuit
    the whole candidate window.
    """
    client = _client(base_url, timeout_s=timeout_s)
    try:
        client.health()
        return "ready"
    except LlamaCppError as exc:
        message = str(exc).lower()
        # _default_transport maps URLError / TimeoutError -> LlamaCppError
        # ("connection failure: <reason>" / "timed out"); those read as
        # connection-level. A structural/HTTP error means the endpoint answered.
        if "did not report an available server" in message:
            # /health answered with a not-ok body -> reachable, still loading.
            return "not_ready"
        if any(tok in message for tok in ("connection", "refused", "timed out", "timeout", "unreachable", "urlopen")):
            return "unreachable"
        # /health answered something malformed. Cross-check /props for
        # *positive* evidence this is not a llama-server at all.
        try:
            props = client.props()
        except LlamaCppError as props_exc:
            props_message = str(props_exc).lower()
            if any(tok in props_message for tok in ("connection", "refused", "timed out", "timeout", "urlopen")):
                return "unreachable"
            # /props also errored but not at the connection level -- we still
            # cannot positively prove "not a llama-server". Keep polling; the
            # readiness timeout will end a genuinely wrong service.
            return "not_ready"
        # /props returned. Positive evidence of a *different* service is a
        # valid object that does not carry default_generation_settings.
        # Anything else (marker present, or a non-object we cannot interpret)
        # is not positive enough -> keep polling.
        if isinstance(props, dict) and "default_generation_settings" not in props:
            return "wrong_service"
        return "not_ready"


def llama_server_port_attribution(
    port: int,
    pid: int,
    *,
    proc_root: Path = PROC_ROOT,
) -> str:
    """Three-valued attribution of the listener on ``port``.

    * ``"ours"``          -- a discovered process bound to ``port`` is ``pid``.
    * ``"foreign"``       -- a discovered process is bound to ``port`` and it
                             is a *different* pid.
    * ``"unestablished"`` -- no process could be attributed to ``port`` (the
                             ``fd`` scan was permission-denied or truncated,
                             or the socket evidence is incomplete). NOT proof
                             of foreignness.
    """
    result = discover_runtime_processes(
        proc_root=proc_root,
        backend=None,
        endpoint_port=port,
        pid_hints=(pid,),
        retain_pid_hints=True,
    )
    holders = [p for p in result.processes if port in p.listening_ports]
    if not holders:
        return "unestablished"
    if any(p.pid == pid for p in holders):
        return "ours"
    # A process holds the port and none of them is ours. Only call this
    # "foreign" if the socket evidence was complete -- otherwise our own
    # process's fd link may simply not have been read.
    if not result.socket_evidence_complete:
        return "unestablished"
    return "foreign"


def llama_server_context_conformance(
    base_url: str,
    requested_context: Optional[int],
    *,
    timeout_s: float = 2.0,
) -> bool:
    """True iff the running server's reported context is consistent with the
    resolved ``requested_context``.

    * ``requested_context is None`` -> not applicable, ``True``.
    * ``/props`` unreadable -> cannot disprove conformance, ``True`` (the
      readiness probe already established it is a llama-server; a transient
      ``/props`` failure must not fail a healthy launch).
    * ``/props`` ``n_ctx`` present and *less than* the requested context ->
      ``False`` (the server cannot serve the resolved workload).
    """
    if requested_context is None:
        return True
    client = _client(base_url, timeout_s=timeout_s)
    try:
        props = client.props()
    except LlamaCppError:
        return True
    settings = props.get("default_generation_settings")
    if not isinstance(settings, dict):
        return True
    n_ctx = settings.get("n_ctx")
    if not isinstance(n_ctx, int) or n_ctx <= 0:
        return True
    return n_ctx >= int(requested_context)
