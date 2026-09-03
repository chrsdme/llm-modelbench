"""Anvil Stage 3B.3C -- fake-process integration.

Spawns a real ``python -m tests._fake_llama_server`` subprocess so the whole
managed-materialisation path runs against real Popen lifecycle, real PID /
``/proc`` reads, real localhost sockets, real bounded readiness polling, and
real graceful / forced termination -- with no CUDA, no llama.cpp binary, no
Ollama, no model.

Every test cleans every child it starts in a ``finally`` so a failure never
leaves a process behind.
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request

import pytest

from llm_modelbench.hardware import GPUDevice
from llm_modelbench.identity import resolve_runtime_profile_identity
from llm_modelbench.runtime_identity import RuntimeExecutionSettings
from llm_modelbench.runtime_lifecycle import CleanupOutcome, RuntimeOwnership
from llm_modelbench.runtime_resolution import (
    ResolvedRuntime,
    RuntimeResolution,
    RuntimeResolutionStatus,
)
from llm_modelbench.runtime_profiles import RuntimeCandidate, RuntimeProfile
from llm_modelbench.runtime_lifecycle import MaterialisationRequest
from llm_modelbench import runtime_process_linux as rpl
from llm_modelbench import llama_server_materialisation as lsm
from llm_modelbench.llama_server_materialisation import (
    MaterialisationStatus,
    lifecycle_controller_for,
    spawn_managed_llama_server,
)

pytestmark = pytest.mark.skipif(
    not rpl.PROC_ROOT.exists(), reason="procfs not available"
)

U_A = "GPU-00000000-1111-2222-3333-444444444444"
SHA = "sha256:" + "c" * 64
INVENTORY = (GPUDevice(0, U_A, "0000:01:00.0", "fixture-A", 16000.0, None, None),)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _recipe(*, requested_context=0, gpu_uuids=(U_A,), model_primary_sha256=SHA):
    settings = RuntimeExecutionSettings(strategy="single_device", context_size=requested_context or None)
    return ResolvedRuntime(
        backend="llama_cpp",
        endpoint="http://127.0.0.1:1",
        runtime_profile_name="llama-local",
        execution_settings=settings,
        runtime_profile_identity=resolve_runtime_profile_identity(
            backend="llama_cpp", execution_settings=settings
        ),
        selected_physical_gpu_uuids=tuple(gpu_uuids),
        placement_class="full_gpu",
        requested_context=requested_context or None,
        allow_ram_spill=False,
        estimated_ram_spill_bytes=None,
        model_primary_sha256=model_primary_sha256,
    )


def _request(**kw):
    recipe = _recipe(**kw)
    cand = RuntimeCandidate(
        profile=RuntimeProfile(
            name="llama-local", backend="llama_cpp",
            endpoint=recipe.endpoint, provenance="configured",
        ),
        health="unreachable", source=("saved_profile",), detail="unreachable fixture",
    )
    return MaterialisationRequest.from_resolution(
        RuntimeResolution(
            status=RuntimeResolutionStatus.RESOLVED, reason="resolved",
            detail="fixture", resolved=recipe, selected_candidate=cand,
        )
    )


def _fake_popen_factory(behaviour: str, ctx: int = 0, *, pids: list | None = None):
    """Popen wrapper that rewrites the executable to our fake-server module
    and injects the behaviour, but keeps the builder's --model / --port /
    --ctx-size / --host so /proc/<pid>/cmdline looks like a real launch.

    Every spawned child pid is appended to ``pids`` (when supplied) so a test
    can assert an abandoned child was really reaped even though a failure
    ``ManagedMaterialisationOutcome`` never carries the process handle.
    """

    def _popen(argv, **kwargs):
        # argv[0] is the (nonexistent) llama-server path; replace with the
        # fake, keep the rest.
        new_argv = [
            sys.executable, "-m", "tests._fake_llama_server",
            "--behaviour", behaviour,
        ]
        if ctx:
            new_argv += ["--ctx", str(ctx)]
        new_argv += list(argv[1:])
        proc = subprocess.Popen(new_argv, **kwargs)
        if pids is not None:
            pids.append(proc.pid)
        return proc

    return _popen


def _http_health_probe(url: str) -> str:
    try:
        with urllib.request.urlopen(url + "/health", timeout=1.0) as resp:
            import json

            body = json.loads(resp.read() or b"{}")
            if not isinstance(body, dict):
                return "not_ready"
            return "ready" if str(body.get("status", "")).lower() == "ok" else "not_ready"
    except urllib.error.HTTPError:
        return "not_ready"
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return "unreachable"
    except (AttributeError, ValueError):
        # a non-object JSON body (e.g. the wrong-service fake's /health array)
        return "not_ready"


def _real_attribution(port: int, pid: int) -> str:
    from llm_modelbench.llama_server_probe import llama_server_port_attribution

    return llama_server_port_attribution(port, pid)


def _spawn(request, *, behaviour="healthy", ctx=0, pids=None, **overrides):
    kw = dict(
        executable_path="/nonexistent/llama-server",
        model_path="/models/fake model.gguf",
        model_primary_sha256=SHA,
        hardware_inventory=INVENTORY,
        observe_identity=lambda pid: rpl.observe_process_identity(pid),
        readiness_probe=_http_health_probe,
        port_attribution=lambda port, pid: "unestablished",
        context_conformance=lambda url, c: True,
        now_iso=lambda: "2026-09-03T12:00:00Z",
        monotonic=time.monotonic,
        sleeper=time.sleep,
        readiness_timeout_s=15.0,
        poll_interval_s=0.05,
        base_port=_free_port(),
        endpoint_window=8,
        # This harness exercises the real Popen / /proc / readiness / cleanup
        # path, not artifact-identity or CLI-contract checks -- stub those so
        # the fake model path / fake executable do not short-circuit the run.
        content_hasher=lambda model_path: SHA,
        cli_contract_probe=lambda exe: frozenset(
            lsm.REQUIRED_LLAMA_SERVER_CLI_OPTIONS
        ),
    )
    kw.update(overrides)
    kw["popen"] = _fake_popen_factory(behaviour, ctx, pids=pids)
    return spawn_managed_llama_server(request, **kw)


def _hard_kill(proc) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    except Exception:
        pass


def _reap_pid(pid: int) -> None:
    """Last-resort cleanup for a child spawned via the popen wrapper whose
    Popen handle the test never held. ``_reap`` in the materialiser normally
    already did this; this only covers an assertion failure mid-test."""
    import os
    import signal as _signal

    try:
        os.kill(pid, _signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


# ==========================================================================
# spawn + readiness (real subprocess)
# ==========================================================================
def test_healthy_fake_server_spawns_ready_and_is_owned():
    out = _spawn(_request(), behaviour="healthy")
    try:
        assert out.status is MaterialisationStatus.SPAWNED_READY
        assert out.result.ownership is RuntimeOwnership.MODELBENCH_OWNED
        owned = out.result.owned_runtime
        assert owned.launch_proof.pid == out.process.pid
        # the proof was really read from /proc
        assert owned.launch_proof.process_start_time_ticks > 0
        assert owned.launch_proof.executable_path is not None
        assert "_fake_llama_server" in " ".join(owned.launch_proof.command_argv)
    finally:
        _hard_kill(out.process)


def test_delayed_ready_succeeds_within_bound():
    # Uses the PRODUCTION readiness probe so the real 503-while-loading ->
    # 200-ready transition (the other state every launch passes through) is
    # proven against LlamaCppClient, not just the hand-rolled probe.
    from llm_modelbench.llama_server_probe import llama_server_readiness_probe

    out = _spawn(
        _request(),
        behaviour="delayed:1.0",
        readiness_probe=llama_server_readiness_probe,
    )
    try:
        assert out.status is MaterialisationStatus.SPAWNED_READY
    finally:
        _hard_kill(out.process)


def test_immediate_exit_is_structured():
    out = _spawn(_request(), behaviour="immediate-exit")
    _hard_kill(out.process)
    assert out.status is MaterialisationStatus.PROCESS_EXITED_BEFORE_READY


def test_never_ready_times_out_and_reaps_the_abandoned_child():
    pids: list[int] = []
    out = _spawn(_request(), behaviour="never-ready", readiness_timeout_s=1.5, pids=pids)
    try:
        assert out.status is MaterialisationStatus.READINESS_TIMEOUT
        # A failure outcome never carries the process handle, so prove the
        # abandoned child was really reaped by re-observing its pid from /proc.
        assert pids, "expected the materialiser to have spawned a child"
        for pid in pids:
            proof = rpl.observe_process_identity(pid)
            # reaped -> pid gone; or (rare) pid reused -> not our fake server
            assert proof is None or "_fake_llama_server" not in " ".join(proof.command_argv)
    finally:
        for pid in pids:
            _reap_pid(pid)


def test_wrong_service_answering_200_is_never_accepted_via_the_real_probe():
    from llm_modelbench.llama_server_probe import llama_server_readiness_probe

    pids: list[int] = []
    out = _spawn(
        _request(),
        behaviour="wrong-service",
        readiness_timeout_s=2.0,
        readiness_probe=llama_server_readiness_probe,
        pids=pids,
    )
    try:
        # The production probe cross-checks /props: the wrong-service body is a
        # valid object without default_generation_settings -> "wrong_service",
        # which the materialiser turns into ENDPOINT_CONFLICT and retries; the
        # window (8) is exhausted with no acceptance.
        assert out.status is MaterialisationStatus.ENDPOINT_CONFLICT
        # every child the materialiser spawned as it burned the window must
        # have been reaped, not leaked.
        for pid in pids:
            proof = rpl.observe_process_identity(pid)
            assert proof is None or "_fake_llama_server" not in " ".join(proof.command_argv)
    finally:
        for pid in pids:
            _reap_pid(pid)


# ==========================================================================
# production readiness probe against a real socket
# ==========================================================================
def test_real_readiness_probe_reports_unreachable_for_a_dead_port():
    from llm_modelbench.llama_server_probe import llama_server_readiness_probe

    # nothing is listening here
    assert llama_server_readiness_probe(f"http://127.0.0.1:{_free_port()}") == "unreachable"


def test_real_readiness_probe_reports_ready_for_the_healthy_fake():
    from llm_modelbench.llama_server_probe import llama_server_readiness_probe

    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests._fake_llama_server", "--behaviour", "healthy",
         "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        verdict = "unreachable"
        while time.monotonic() < deadline:
            verdict = llama_server_readiness_probe(f"http://127.0.0.1:{port}")
            if verdict == "ready":
                break
            time.sleep(0.05)
        assert verdict == "ready"
    finally:
        _hard_kill(proc)


def test_real_readiness_probe_reports_wrong_service_for_a_non_llama_endpoint():
    from llm_modelbench.llama_server_probe import llama_server_readiness_probe

    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests._fake_llama_server", "--behaviour", "wrong-service",
         "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        verdict = "unreachable"
        while time.monotonic() < deadline:
            verdict = llama_server_readiness_probe(f"http://127.0.0.1:{port}")
            if verdict != "unreachable":
                break
            time.sleep(0.05)
        assert verdict == "wrong_service"
    finally:
        _hard_kill(proc)


# ==========================================================================
# process-identity proof (real /proc)
# ==========================================================================
def test_observe_process_identity_reads_real_proc_fields():
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests._fake_llama_server", "--behaviour", "healthy",
         "--port", str(_free_port())],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        proof = rpl.observe_process_identity(proc.pid)
        assert proof is not None
        assert proof.pid == proc.pid
        assert proof.process_start_time_ticks > 0
        assert proof.executable_path and "python" in proof.executable_path.lower()
        assert "_fake_llama_server" in " ".join(proof.command_argv)
    finally:
        _hard_kill(proc)


def test_observe_process_identity_none_for_dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    # extremely unlikely to be reused instantly; if it is, the field mismatch
    # would still make revalidation fail. Here we just assert graceful None.
    result = rpl.observe_process_identity(proc.pid)
    assert result is None or result.pid == proc.pid


def test_revalidation_matches_same_process_and_rejects_different_start_ticks():
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests._fake_llama_server", "--behaviour", "healthy",
         "--port", str(_free_port())],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        a = rpl.observe_process_identity(proc.pid)
        b = rpl.observe_process_identity(proc.pid)
        assert a.revalidation_matches(b) is True
        # forge a different start tick -> must not match
        from llm_modelbench.runtime_lifecycle import LaunchProcessProof

        forged = LaunchProcessProof(
            pid=a.pid,
            process_start_time_ticks=a.process_start_time_ticks + 1,
            executable_path=a.executable_path,
            command_argv=a.command_argv,
        )
        assert a.revalidation_matches(forged) is False
    finally:
        _hard_kill(proc)


# ==========================================================================
# cleanup (real graceful / forced termination)
# ==========================================================================
def test_owned_cleanup_gracefully_terminates_the_real_child():
    out = _spawn(_request(), behaviour="healthy")
    assert out.status is MaterialisationStatus.SPAWNED_READY
    proc = out.process
    try:
        ctrl = lifecycle_controller_for(
            out,
            observe_identity=lambda pid: rpl.observe_process_identity(pid),
            terminate=rpl.terminate_process,
            graceful_timeout_s=5.0,
            forced_timeout_s=3.0,
        )
        res = ctrl.cleanup()
        assert res.outcome is CleanupOutcome.SUCCEEDED
        assert res.destructive_action_performed is True
        assert proc.poll() is not None  # really gone
    finally:
        _hard_kill(proc)


def test_owned_cleanup_forces_a_sigterm_ignoring_child():
    out = _spawn(_request(), behaviour="ignore-sigterm")
    assert out.status is MaterialisationStatus.SPAWNED_READY
    proc = out.process
    try:
        ctrl = lifecycle_controller_for(
            out,
            observe_identity=lambda pid: rpl.observe_process_identity(pid),
            terminate=rpl.terminate_process,
            graceful_timeout_s=1.0,
            forced_timeout_s=3.0,
        )
        res = ctrl.cleanup()
        assert res.outcome is CleanupOutcome.SUCCEEDED
        assert proc.poll() is not None
    finally:
        _hard_kill(proc)


def test_cleanup_refused_when_identity_cannot_be_revalidated():
    out = _spawn(_request(), behaviour="healthy")
    proc = out.process
    try:
        ctrl = lifecycle_controller_for(
            out,
            observe_identity=lambda pid: None,  # cannot revalidate
            terminate=rpl.terminate_process,
        )
        res = ctrl.cleanup()
        assert res.outcome is CleanupOutcome.OWNERSHIP_NOT_REVALIDATED
        assert res.destructive_action_performed is False
        assert proc.poll() is None  # still alive -- no signal was sent
    finally:
        _hard_kill(proc)


def test_diagnostic_sink_is_closed_on_cleanup_no_thread_leak():
    out = _spawn(_request(), behaviour="healthy")
    assert out.status is MaterialisationStatus.SPAWNED_READY
    proc = out.process
    sink = out.diagnostic_sink
    assert sink is not None
    try:
        ctrl = lifecycle_controller_for(
            out,
            observe_identity=lambda pid: rpl.observe_process_identity(pid),
            terminate=rpl.terminate_process,
        )
        ctrl.cleanup()
        # the drain thread has been joined and the stream closed
        assert sink._thread is None or not sink._thread.is_alive()
    finally:
        _hard_kill(proc)


def test_child_output_does_not_deadlock_the_materialiser():
    # The 'flood' fake writes ~256 KiB to stdout -- well past the 64 KiB Linux
    # pipe buffer -- BEFORE it binds its socket. If the materialiser did not
    # drain the pipe, that write blocks, the server never comes up, and this
    # would time out instead of reaching SPAWNED_READY.
    out = _spawn(_request(), behaviour="flood", readiness_timeout_s=15.0)
    try:
        assert out.status is MaterialisationStatus.SPAWNED_READY
        # the child really emitted more than one pipe buffer's worth ...
        assert len(out.diagnostic_tail) > 0
        # ... but the retained tail is bounded to the ring-buffer size.
        assert len(out.diagnostic_tail) <= 64 * 1024
    finally:
        _hard_kill(out.process)


def test_context_manager_cleans_owned_child_on_exception():
    out = _spawn(_request(), behaviour="healthy")
    proc = out.process
    ctrl = lifecycle_controller_for(
        out,
        observe_identity=lambda pid: rpl.observe_process_identity(pid),
        terminate=rpl.terminate_process,
    )
    try:
        with pytest.raises(RuntimeError):
            with ctrl:
                raise RuntimeError("benchmark blew up")
        assert proc.poll() is not None
    finally:
        _hard_kill(proc)


# ==========================================================================
# THE real safety property: only the owned process is terminated
# ==========================================================================
def test_cleanup_terminates_only_the_owned_process_never_a_foreign_one():
    foreign_port = _free_port()
    foreign = subprocess.Popen(
        [sys.executable, "-m", "tests._fake_llama_server", "--behaviour", "healthy",
         "--port", str(foreign_port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    out = None
    try:
        # wait for the foreign server to be up
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if _http_health_probe(f"http://127.0.0.1:{foreign_port}") == "ready":
                break
            time.sleep(0.05)
        assert _http_health_probe(f"http://127.0.0.1:{foreign_port}") == "ready"

        out = _spawn(_request(), behaviour="healthy")
        assert out.status is MaterialisationStatus.SPAWNED_READY
        owned = out.process
        assert owned.pid != foreign.pid

        ctrl = lifecycle_controller_for(
            out,
            observe_identity=lambda pid: rpl.observe_process_identity(pid),
            terminate=rpl.terminate_process,
        )
        res = ctrl.cleanup()
        assert res.outcome is CleanupOutcome.SUCCEEDED
        assert owned.poll() is not None          # owned really terminated
        assert foreign.poll() is None            # foreign untouched
        assert _http_health_probe(f"http://127.0.0.1:{foreign_port}") == "ready"
    finally:
        _hard_kill(out.process if out else None)
        _hard_kill(foreign)


# ==========================================================================
# port-collision safety: a foreign listener on a candidate port is never killed
# ==========================================================================
def test_foreign_listener_on_candidate_port_is_never_signalled():
    base = _free_port()
    # occupy the first candidate with a foreign listener
    foreign = socket.socket()
    foreign.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    foreign.bind(("127.0.0.1", base))
    foreign.listen(1)
    out = None
    try:
        out = _spawn(
            _request(),
            behaviour="healthy",
            base_port=base,
            endpoint_window=4,
        )
        # the materialiser must have skipped the occupied base port and used
        # a later candidate -- and never touched the foreign socket.
        assert out.status is MaterialisationStatus.SPAWNED_READY
        assert out.endpoint != f"http://127.0.0.1:{base}"
        # foreign socket still bound and listening
        probe = socket.socket()
        assert probe.connect_ex(("127.0.0.1", base)) == 0
        probe.close()
    finally:
        _hard_kill(out.process if out else None)
        foreign.close()
