"""Anvil Stage 3B.3C -- managed llama-server materialisation.

Unit + fake-process integration. No CUDA, no llama.cpp binary, no Ollama, no
model. The fake-process integration tests spawn a real
``python -m tests._fake_llama_server`` subprocess so the real Popen / PID /
``/proc`` / readiness / port / cleanup paths are exercised.
"""
from __future__ import annotations

import pytest

from llm_modelbench.hardware import GPUDevice
from llm_modelbench.identity import resolve_runtime_profile_identity
from llm_modelbench.runtime_identity import RuntimeExecutionSettings
from llm_modelbench.runtime_lifecycle import (
    CleanupOutcome,
    LaunchProcessProof,
    MaterialisationRequest,
    RuntimeOwnership,
)
from llm_modelbench.runtime_resolution import (
    ResolvedRuntime,
    RuntimeResolution,
    RuntimeResolutionStatus,
)
from llm_modelbench.runtime_profiles import RuntimeCandidate, RuntimeProfile
from llm_modelbench import runtime_process_linux as rpl
from llm_modelbench import llama_server_materialisation as lsm
from llm_modelbench.llama_server_materialisation import (
    MaterialisationStatus,
    ManagedMaterialisationError,
    build_llama_server_command,
    candidate_endpoints,
    lifecycle_controller_for,
    materialise,
    resolve_cuda_visible_devices,
    spawn_managed_llama_server,
)

U_A = "GPU-00000000-1111-2222-3333-444444444444"
U_B = "GPU-00000000-1111-2222-3333-555555555555"
SHA = "sha256:" + "a" * 64
OTHER_SHA = "sha256:" + "b" * 64

INVENTORY = (
    GPUDevice(0, U_A, "0000:01:00.0", "fixture-A", 16000.0, None, None),
    GPUDevice(1, U_B, "0000:02:00.0", "fixture-B", 12000.0, None, None),
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _recipe(
    *,
    backend="llama_cpp",
    endpoint="http://127.0.0.1:9",
    gpu_uuids=(U_A,),
    strategy="single_device",
    requested_context=8192,
    allow_ram_spill=False,
    model_primary_sha256=SHA,
):
    settings = RuntimeExecutionSettings(strategy=strategy, context_size=requested_context)
    return ResolvedRuntime(
        backend=backend,
        endpoint=endpoint,
        runtime_profile_name="llama-local",
        execution_settings=settings,
        runtime_profile_identity=resolve_runtime_profile_identity(
            backend=backend, execution_settings=settings
        ),
        selected_physical_gpu_uuids=tuple(gpu_uuids),
        placement_class="single_gpu",
        requested_context=requested_context,
        allow_ram_spill=allow_ram_spill,
        estimated_ram_spill_bytes=None,
        model_primary_sha256=model_primary_sha256,
    )


def _resolution(*, candidate_health="unreachable", **recipe_kw):
    recipe = _recipe(**recipe_kw)
    cand = RuntimeCandidate(
        profile=RuntimeProfile(
            name="llama-local",
            backend=recipe.backend,
            endpoint=recipe.endpoint,
            provenance="configured",
        ),
        health=candidate_health,
        source=("saved_profile",),
        detail=f"{candidate_health} fixture",
    )
    return RuntimeResolution(
        status=RuntimeResolutionStatus.RESOLVED,
        reason="resolved",
        detail="resolved fixture",
        resolved=recipe,
        selected_candidate=cand,
    )


def _request(**kw):
    return MaterialisationRequest.from_resolution(_resolution(**kw))


def _iso():
    return "2026-09-03T12:00:00Z"


# ==========================================================================
# command construction
# ==========================================================================
def test_command_is_argv_list_no_shell_no_interpolation():
    cmd, status, _ = build_llama_server_command(
        _request(),
        executable_path="/opt/llama.cpp/llama-server",
        model_path="/models/some model (v2).gguf",
        hardware_inventory=INVENTORY,
        port=8080,
    )
    assert status is None
    argv = cmd.argv
    assert isinstance(argv, tuple) and all(isinstance(a, str) for a in argv)
    # the model path with spaces + parens is a SINGLE argv element
    assert "/models/some model (v2).gguf" in argv
    assert argv[0] == "/opt/llama.cpp/llama-server"
    assert argv[argv.index("--model") + 1] == "/models/some model (v2).gguf"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8080"
    # no shell metacharacters were concatenated anywhere
    assert not any(";" in a or "&&" in a or "|" in a for a in argv)


def test_context_comes_from_the_resolved_recipe():
    cmd, _, _ = build_llama_server_command(
        _request(requested_context=4096),
        executable_path="x",
        model_path="m.gguf",
        hardware_inventory=INVENTORY,
    )
    assert cmd.argv[cmd.argv.index("--ctx-size") + 1] == "4096"


def test_no_context_flag_when_recipe_sets_none():
    cmd, _, _ = build_llama_server_command(
        _request(requested_context=None),
        executable_path="x",
        model_path="m.gguf",
        hardware_inventory=INVENTORY,
    )
    assert "--ctx-size" not in cmd.argv


def test_single_gpu_selection_becomes_cuda_visible_devices_uuid_form():
    cmd, _, _ = build_llama_server_command(
        _request(gpu_uuids=(U_A,)),
        executable_path="x",
        model_path="m.gguf",
        hardware_inventory=INVENTORY,
    )
    assert cmd.env_overlay["CUDA_VISIBLE_DEVICES"] == f"GPU-{U_A[4:].lower()}"
    assert "--split-mode" not in cmd.argv
    assert "--device" not in cmd.argv  # no transient ordinal


def test_multi_gpu_layer_split_emits_split_mode_layer_no_tensor_split():
    cmd, _, _ = build_llama_server_command(
        _request(gpu_uuids=(U_A, U_B), strategy="layer_split"),
        executable_path="x",
        model_path="m.gguf",
        hardware_inventory=INVENTORY,
    )
    assert cmd.argv[cmd.argv.index("--split-mode") + 1] == "layer"
    assert "--tensor-split" not in cmd.argv
    assert cmd.env_overlay["CUDA_VISIBLE_DEVICES"] == (
        f"GPU-{U_A[4:].lower()},GPU-{U_B[4:].lower()}"
    )


def test_gpu_uuid_order_is_preserved_verbatim():
    cmd, _, _ = build_llama_server_command(
        _request(gpu_uuids=(U_B, U_A), strategy="layer_split"),
        executable_path="x",
        model_path="m.gguf",
        hardware_inventory=INVENTORY,
    )
    assert cmd.env_overlay["CUDA_VISIBLE_DEVICES"] == (
        f"GPU-{U_B[4:].lower()},GPU-{U_A[4:].lower()}"
    )


def test_no_extra_gpu_is_added():
    cmd, _, _ = build_llama_server_command(
        _request(gpu_uuids=(U_A,)),
        executable_path="x",
        model_path="m.gguf",
        hardware_inventory=INVENTORY,
    )
    assert cmd.env_overlay["CUDA_VISIBLE_DEVICES"].count("GPU-") == 1


def test_ram_spill_cannot_self_enable_no_spill_flag_when_recipe_forbids():
    cmd, _, _ = build_llama_server_command(
        _request(allow_ram_spill=False),
        executable_path="x",
        model_path="m.gguf",
        hardware_inventory=INVENTORY,
    )
    # no --n-gpu-layers / no host-RAM opt-in flag
    assert "--n-gpu-layers" not in cmd.argv
    assert "-ngl" not in cmd.argv


def test_ram_spill_permitted_still_adds_no_search_flag():
    cmd, _, _ = build_llama_server_command(
        _request(allow_ram_spill=True),
        executable_path="x",
        model_path="m.gguf",
        hardware_inventory=INVENTORY,
    )
    assert "--n-gpu-layers" not in cmd.argv
    assert "--tensor-split" not in cmd.argv


def test_identical_recipe_produces_identical_semantic_argv_and_env():
    a, _, _ = build_llama_server_command(
        _request(gpu_uuids=(U_A, U_B), strategy="layer_split"),
        executable_path="/x/llama-server",
        model_path="m.gguf",
        hardware_inventory=INVENTORY,
        port=8090,
    )
    # different inventory *order* must not change the result
    b, _, _ = build_llama_server_command(
        _request(gpu_uuids=(U_A, U_B), strategy="layer_split"),
        executable_path="/x/llama-server",
        model_path="m.gguf",
        hardware_inventory=tuple(reversed(INVENTORY)),
        port=8090,
    )
    assert a.argv == b.argv
    assert a.env_overlay == b.env_overlay


def test_gpu_identity_untranslatable_when_uuid_absent_from_inventory():
    stale = "GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"
    cmd, status, detail = build_llama_server_command(
        _request(gpu_uuids=(stale,)),
        executable_path="x",
        model_path="m.gguf",
        hardware_inventory=INVENTORY,
    )
    assert cmd is None
    assert status is MaterialisationStatus.GPU_IDENTITY_UNTRANSLATABLE
    assert "not present in the authoritative hardware inventory" in detail


def test_gpu_identity_untranslatable_when_not_nvidia_uuid_shaped():
    cmd, status, _ = build_llama_server_command(
        _request(gpu_uuids=("not-a-uuid",)),
        executable_path="x",
        model_path="m.gguf",
        hardware_inventory=INVENTORY,
    )
    assert cmd is None
    assert status is MaterialisationStatus.GPU_IDENTITY_UNTRANSLATABLE


def test_resolve_cuda_visible_devices_is_inventory_order_independent():
    v1, _ = resolve_cuda_visible_devices((U_A, U_B), hardware_inventory=INVENTORY)
    v2, _ = resolve_cuda_visible_devices(
        (U_A, U_B), hardware_inventory=tuple(reversed(INVENTORY))
    )
    assert v1 == v2


def test_resolve_cuda_visible_devices_none_when_recipe_selects_no_gpu():
    value, reason = resolve_cuda_visible_devices((), hardware_inventory=INVENTORY)
    assert value is None and reason is None


def test_command_builder_rejects_non_request():
    with pytest.raises(ManagedMaterialisationError):
        build_llama_server_command(
            object(), executable_path="x", model_path="m", hardware_inventory=()
        )


# ==========================================================================
# artifact identity
# ==========================================================================
def _spawn(request, **overrides):
    kw = dict(
        executable_path="/opt/llama-server",
        model_path="/models/m.gguf",
        model_primary_sha256=SHA,
        hardware_inventory=INVENTORY,
        observe_identity=lambda pid: _FakeProof(pid),
        readiness_probe=lambda url: "ready",
        port_attribution=lambda port, pid: "ours",
        context_conformance=lambda url, ctx: True,
        now_iso=_iso,
        popen=_FakePopen.factory(),
        monotonic=_clock(),
        sleeper=lambda s: None,
        readiness_timeout_s=1.0,
        poll_interval_s=0.01,
        base_port=18080,
        endpoint_window=4,
        port_bindable=lambda host, port: True,
    )
    kw.update(overrides)
    return spawn_managed_llama_server(request, **kw)


def _FakeProof(pid):
    return LaunchProcessProof(
        pid=pid,
        process_start_time_ticks=1000,
        executable_path="/opt/llama-server",
        command_argv=("/opt/llama-server", "--model", "/models/m.gguf"),
    )


def _clock():
    t = {"v": 0.0}

    def _now():
        t["v"] += 0.001
        return t["v"]

    return _now


class _FakePopen:
    """Minimal Popen-like double: alive until .terminate(), records signals."""

    _next_pid = 40000

    def __init__(self, argv, **kwargs):
        self.argv = list(argv)
        _FakePopen._next_pid += 1
        self.pid = _FakePopen._next_pid
        self.returncode = None
        self._exit_after_polls = kwargs.pop("_exit_after_polls", None)
        self._polls = 0
        self.stdout = None
        self.terminated = False
        self.killed = False
        self.env = kwargs.get("env")

    @classmethod
    def factory(cls, **preset):
        def _make(argv, **kwargs):
            kwargs.update(preset)
            return cls(argv, **kwargs)

        return _make

    def poll(self):
        self._polls += 1
        if self._exit_after_polls is not None and self._polls >= self._exit_after_polls:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_correct_artifact_accepted():
    out = _spawn(_request(), model_primary_sha256=SHA)
    assert out.status is MaterialisationStatus.SPAWNED_READY


def test_mismatched_artifact_identity_rejected_zero_spawn():
    launched = []
    out = _spawn(
        _request(model_primary_sha256=SHA),
        model_primary_sha256=OTHER_SHA,
        popen=_FakePopen.factory(),
    )
    assert out.status is MaterialisationStatus.ARTIFACT_IDENTITY_MISMATCH
    assert out.launched_argv is None
    assert launched == []


def test_missing_artifact_identity_in_recipe_fails_closed():
    out = _spawn(_request(model_primary_sha256=None))
    assert out.status is MaterialisationStatus.RESOLVED_RECIPE_INCOMPLETE
    assert out.launched_argv is None


def test_missing_model_path_fails_closed():
    out = _spawn(_request(), model_path=None)
    assert out.status is MaterialisationStatus.RESOLVED_RECIPE_INCOMPLETE


def test_missing_executable_fails_structurally():
    out = _spawn(_request(), executable_path=None)
    assert out.status is MaterialisationStatus.EXECUTABLE_UNAVAILABLE


# ==========================================================================
# dispatcher -- reuse vs managed vs ollama (the ONE Ollama rule location)
# ==========================================================================
def test_healthy_resolved_candidate_is_reused_zero_spawn_any_backend():
    for backend in ("llama_cpp", "ollama"):
        spawned = {"n": 0}

        def _spawn_managed(req):
            spawned["n"] += 1
            raise AssertionError("must not spawn when reuse is possible")

        out = materialise(
            _request(backend=backend, candidate_health="healthy"),
            spawn_managed=_spawn_managed,
        )
        assert out.status is MaterialisationStatus.REUSED_EXTERNAL
        assert out.result.ownership is RuntimeOwnership.EXTERNAL_REUSED
        assert out.result.cleanup_authority is None
        assert spawned["n"] == 0


def test_ollama_with_no_external_endpoint_returns_structured_no_spawn():
    spawned = {"n": 0}
    out = materialise(
        _request(backend="ollama", candidate_health="unreachable"),
        spawn_managed=lambda req: spawned.__setitem__("n", spawned["n"] + 1),
    )
    assert out.status is MaterialisationStatus.EXTERNAL_RUNTIME_REQUIRED
    assert spawned["n"] == 0
    assert "does not fall back to llama.cpp" in out.detail


def test_llama_cpp_with_no_external_endpoint_spawns_managed():
    seen = {}

    def _spawn_managed(req):
        seen["req"] = req
        return lsm.ManagedMaterialisationOutcome(
            status=MaterialisationStatus.SPAWNED_READY, detail="ok"
        )

    out = materialise(
        _request(backend="llama_cpp", candidate_health="unreachable"),
        spawn_managed=_spawn_managed,
    )
    assert out.status is MaterialisationStatus.SPAWNED_READY
    assert seen["req"].backend == "llama_cpp"


def test_external_still_healthy_probe_can_only_demote_never_promote():
    # candidate was unreachable -> probe is never consulted, still spawns.
    called = {"n": 0}
    materialise(
        _request(backend="llama_cpp", candidate_health="unreachable"),
        spawn_managed=lambda req: lsm.ManagedMaterialisationOutcome(
            status=MaterialisationStatus.SPAWNED_READY, detail="ok"
        ),
        external_still_healthy=lambda req: called.__setitem__("n", called["n"] + 1) or True,
    )
    assert called["n"] == 0

    # candidate was healthy but the refinement probe says it's gone -> demote.
    spawned = {"n": 0}
    out = materialise(
        _request(backend="llama_cpp", candidate_health="healthy"),
        spawn_managed=lambda req: spawned.__setitem__("n", spawned["n"] + 1)
        or lsm.ManagedMaterialisationOutcome(
            status=MaterialisationStatus.SPAWNED_READY, detail="ok"
        ),
        external_still_healthy=lambda req: False,
    )
    assert out.status is MaterialisationStatus.SPAWNED_READY
    assert spawned["n"] == 1


def test_dispatcher_rejects_unknown_backend():
    # the resolver never yields a non-known backend; simulate a forged recipe.
    req = _request(backend="llama_cpp", candidate_health="unreachable")
    object.__setattr__(req.recipe, "backend", "mystery")
    with pytest.raises(ManagedMaterialisationError):
        materialise(req, spawn_managed=lambda r: None)


def test_dispatcher_rejects_non_request():
    with pytest.raises(ManagedMaterialisationError):
        materialise(object(), spawn_managed=lambda r: None)


# ==========================================================================
# spawn / readiness (fake Popen, injected clock)
# ==========================================================================
def test_valid_managed_request_spawns_once_and_returns_owned_runtime():
    calls = {"n": 0}

    def _pf(argv, **kw):
        calls["n"] += 1
        return _FakePopen(argv, **kw)

    out = _spawn(_request(), popen=_pf)
    assert out.status is MaterialisationStatus.SPAWNED_READY
    assert calls["n"] == 1
    assert out.result.ownership is RuntimeOwnership.MODELBENCH_OWNED
    assert out.result.owned_runtime.launch_proof.pid == out.process.pid


def test_spawn_failure_is_structured():
    def _boom(argv, **kw):
        raise OSError("exec format error")

    out = _spawn(_request(), popen=_boom)
    assert out.status is MaterialisationStatus.SPAWN_FAILED
    assert "exec format error" in out.detail


def test_immediate_exit_is_structured_not_ready():
    out = _spawn(
        _request(),
        popen=_FakePopen.factory(_exit_after_polls=1),
        port_bindable=lambda h, p: True,  # port still free -> genuine death
    )
    assert out.status is MaterialisationStatus.PROCESS_EXITED_BEFORE_READY


def test_ownership_proof_failure_is_structured():
    out = _spawn(_request(), observe_identity=lambda pid: None)
    assert out.status is MaterialisationStatus.OWNERSHIP_PROOF_FAILED


def test_delayed_ready_succeeds_within_bound():
    state = {"n": 0}

    def _probe(url):
        state["n"] += 1
        return "ready" if state["n"] >= 3 else "not_ready"

    out = _spawn(_request(), readiness_probe=_probe)
    assert out.status is MaterialisationStatus.SPAWNED_READY


def test_never_ready_times_out_monotonically():
    now = {"v": 0.0}
    # sleeper tripwire so a mutation that disables every readiness bound is
    # caught as a failure, not a hang.
    sleeps = {"n": 0}

    def _mono():
        now["v"] += 0.5
        return now["v"]

    def _sleeper(_s):
        sleeps["n"] += 1
        if sleeps["n"] > 5000:
            raise AssertionError("readiness loop ignored every bound")

    out = _spawn(
        _request(),
        readiness_probe=lambda url: "not_ready",
        monotonic=_mono,
        sleeper=_sleeper,
        readiness_timeout_s=2.0,
    )
    assert out.status is MaterialisationStatus.READINESS_TIMEOUT
    # timeout 2.0 / interval 0.25 default -> ~8 sleeps, never thousands
    assert sleeps["n"] < 100


def test_readiness_loop_is_also_bounded_by_a_hard_attempt_cap():
    # Even if the clock is broken (monotonic never advances), the loop still
    # terminates -- it cannot spin forever waiting on a never-ready probe.
    # The sleeper is a tripwire: if the loop exceeds a generous ceiling the
    # test fails loudly instead of hanging (so a mutation that removes BOTH
    # readiness bounds is caught as a failure, not a hang).
    probes = {"n": 0}
    CEILING = 5000

    def _probe(url):
        probes["n"] += 1
        return "not_ready"

    def _tripwire_sleeper(_s):
        if probes["n"] > CEILING:
            raise AssertionError("readiness loop is unbounded -- exceeded hard ceiling")

    out = _spawn(
        _request(),
        readiness_probe=_probe,
        monotonic=lambda: 0.0,  # frozen clock
        sleeper=_tripwire_sleeper,
        readiness_timeout_s=1.0,
        poll_interval_s=0.01,
    )
    assert out.status is MaterialisationStatus.READINESS_TIMEOUT
    # bounded: ~ timeout/interval + a small margin, never unbounded
    assert probes["n"] <= 1.0 / 0.01 + 5


def test_process_exit_breaks_readiness_poll_immediately():
    out = _spawn(
        _request(),
        popen=_FakePopen.factory(_exit_after_polls=2),
        readiness_probe=lambda url: "not_ready",
        port_bindable=lambda h, p: True,
    )
    assert out.status is MaterialisationStatus.PROCESS_EXITED_BEFORE_READY


def test_wrong_service_on_our_port_is_endpoint_conflict_and_retries():
    # first candidate answers with a foreign service; second is clean.
    calls = {"probe": 0}
    ports_tried = []

    def _pf(argv, **kw):
        ports_tried.append(argv[argv.index("--port") + 1])
        return _FakePopen(argv, **kw)

    def _probe(url):
        calls["probe"] += 1
        # port 18080 -> wrong service, 18081 -> ready
        return "wrong_service" if url.endswith(":18080") else "ready"

    out = _spawn(_request(), popen=_pf, readiness_probe=_probe)
    assert out.status is MaterialisationStatus.SPAWNED_READY
    assert ports_tried == ["18080", "18081"]


def test_recipe_context_mismatch_is_wrong_service_terminal():
    out = _spawn(_request(requested_context=8192), context_conformance=lambda url, ctx: False)
    assert out.status is MaterialisationStatus.WRONG_SERVICE


def test_probe_wrong_service_and_conformance_failure_map_to_distinct_statuses():
    # The brief enumerates "endpoint-occupied/conflicted" and
    # "wrong/incompatible-service" as distinct outcomes. This module's
    # deliberate mapping:
    #   readiness probe returns "wrong_service" (a DIFFERENT service is on the
    #     port we were about to use)         -> ENDPOINT_CONFLICT, retry
    #   our OWN listener comes up but fails the recipe context conformance
    #     check (incompatible service)       -> WRONG_SERVICE, terminal
    exhaust = _spawn(_request(), readiness_probe=lambda url: "wrong_service")
    assert exhaust.status is MaterialisationStatus.ENDPOINT_CONFLICT

    incompatible = _spawn(
        _request(requested_context=8192),
        context_conformance=lambda url, ctx: False,
    )
    assert incompatible.status is MaterialisationStatus.WRONG_SERVICE


def test_foreign_attribution_is_endpoint_conflict_and_retries():
    ports = []

    def _pf(argv, **kw):
        ports.append(argv[argv.index("--port") + 1])
        return _FakePopen(argv, **kw)

    def _attr(port, pid):
        return "foreign" if port == 18080 else "ours"

    out = _spawn(_request(), popen=_pf, port_attribution=_attr)
    assert out.status is MaterialisationStatus.SPAWNED_READY
    assert ports == ["18080", "18081"]


def test_unestablished_attribution_still_proceeds_and_is_preserved():
    out = _spawn(_request(), port_attribution=lambda port, pid: "unestablished")
    assert out.status is MaterialisationStatus.SPAWNED_READY
    assert out.attribution == "unestablished"


# ==========================================================================
# ports
# ==========================================================================
def test_candidate_endpoints_are_bounded_and_deterministic():
    eps = candidate_endpoints(base_port=8080, window=8)
    assert len(eps) == 8
    assert [e.port for e in eps] == list(range(8080, 8088))
    assert all(e.host == "127.0.0.1" for e in eps)


def test_all_candidates_occupied_returns_endpoint_conflict_not_kill():
    out = _spawn(_request(), port_bindable=lambda h, p: False, endpoint_window=4)
    assert out.status is MaterialisationStatus.ENDPOINT_CONFLICT
    assert out.launched_argv is None  # nothing was ever launched


def test_selected_endpoint_is_recorded_on_success():
    out = _spawn(_request())
    assert out.endpoint == "http://127.0.0.1:18080"


def test_lost_bind_race_retries_next_candidate():
    # bind check passes, but the child immediately exits AND the port is now
    # unbindable -> lost race -> ENDPOINT_CONFLICT for that candidate, retry.
    state = {"launch": 0}

    def _pf(argv, **kw):
        state["launch"] += 1
        # first child exits at once; second stays alive
        exits = 1 if state["launch"] == 1 else None
        return _FakePopen(argv, _exit_after_polls=exits, **kw)

    def _bindable(host, port):
        # port 18080 becomes unbindable after the first launch (someone took it)
        if port == 18080 and state["launch"] >= 1:
            return False
        return True

    out = _spawn(_request(), popen=_pf, port_bindable=_bindable)
    assert out.status is MaterialisationStatus.SPAWNED_READY
    assert out.endpoint == "http://127.0.0.1:18081"


# ==========================================================================
# no leaked children across failed candidates
# ==========================================================================
def test_failed_candidates_leave_zero_surviving_children():
    made = []

    def _pf(argv, **kw):
        p = _FakePopen(argv, _exit_after_polls=None, **kw)
        made.append(p)
        return p

    # every candidate: bind ok, but readiness never comes and times out.
    now = {"v": 0.0}

    def _mono():
        now["v"] += 1.0
        return now["v"]

    out = _spawn(
        _request(),
        popen=_pf,
        readiness_probe=lambda url: "not_ready",
        monotonic=_mono,
        readiness_timeout_s=2.0,
        endpoint_window=3,
    )
    # readiness timeout is terminal (not ENDPOINT_CONFLICT) so only 1 child,
    # and it must have been reaped.
    assert out.status is MaterialisationStatus.READINESS_TIMEOUT
    assert len(made) == 1
    assert made[0].terminated or made[0].killed or made[0].returncode is not None


def test_endpoint_conflict_retries_reap_every_failed_child():
    made = []

    def _pf(argv, **kw):
        p = _FakePopen(argv, **kw)
        made.append(p)
        return p

    # every candidate answers wrong_service (foreign) -> retry all, then
    # exhaust the window -> ENDPOINT_CONFLICT; every launched child reaped.
    out = _spawn(
        _request(),
        popen=_pf,
        readiness_probe=lambda url: "wrong_service",
        endpoint_window=3,
    )
    assert out.status is MaterialisationStatus.ENDPOINT_CONFLICT
    assert len(made) == 3
    assert all(p.terminated or p.killed or p.returncode is not None for p in made)


# ==========================================================================
# cleanup wiring (fake process, real controller)
# ==========================================================================
def test_lifecycle_controller_for_reused_external_has_no_cleanup_authority():
    out = materialise(
        _request(candidate_health="healthy"),
        spawn_managed=lambda r: pytest.fail("no spawn"),
    )
    ctrl = lifecycle_controller_for(
        out, observe_identity=lambda pid: None, terminate=lambda *a, **k: "graceful"
    )
    assert ctrl.owns_runtime is False
    res = ctrl.cleanup()
    assert res.outcome is CleanupOutcome.NOT_APPLICABLE_EXTERNAL
    assert res.destructive_action_performed is False


def test_lifecycle_controller_for_owned_revalidates_before_signal():
    out = _spawn(_request())
    seen = {"terminate": 0}

    def _term(proc, *, graceful_timeout_s, forced_timeout_s, revalidate):
        seen["terminate"] += 1
        assert revalidate() is True  # proof still matches
        return "graceful"

    ctrl = lifecycle_controller_for(
        out,
        observe_identity=lambda pid: _FakeProof(pid),
        terminate=_term,
    )
    res = ctrl.cleanup()
    assert res.outcome is CleanupOutcome.SUCCEEDED
    assert seen["terminate"] == 1


def test_owned_cleanup_refused_when_revalidation_fails_no_signal():
    out = _spawn(_request())
    signalled = {"n": 0}

    def _term(proc, **kw):
        signalled["n"] += 1
        return "graceful"

    ctrl = lifecycle_controller_for(
        out, observe_identity=lambda pid: None, terminate=_term
    )
    res = ctrl.cleanup()
    assert res.outcome is CleanupOutcome.OWNERSHIP_NOT_REVALIDATED
    assert signalled["n"] == 0


def test_owned_cleanup_forced_failure_is_structured_forced_failed():
    out = _spawn(_request())

    def _term(proc, **kw):
        return "survived"

    ctrl = lifecycle_controller_for(
        out, observe_identity=lambda pid: _FakeProof(pid), terminate=_term
    )
    res = ctrl.cleanup()
    assert res.outcome is CleanupOutcome.FORCED_FAILED
    assert res.destructive_action_performed is True


def test_owned_cleanup_generic_callback_error_is_graceful_failed():
    out = _spawn(_request())

    def _term(proc, **kw):
        raise RuntimeError("adapter blew up")

    ctrl = lifecycle_controller_for(
        out, observe_identity=lambda pid: _FakeProof(pid), terminate=_term
    )
    res = ctrl.cleanup()
    assert res.outcome is CleanupOutcome.GRACEFUL_FAILED


# ==========================================================================
# authority -- no second planner
# ==========================================================================
def _code_lines(mod) -> str:
    """Module source with comment text and the module docstring removed, but
    code tokens kept verbatim (so ``shell=True`` still reads as
    ``shell=True``). Guard-names mentioned only in prose do not trip the
    'not in source' asserts."""
    import ast
    import io
    import tokenize

    raw = __import__("inspect").getsource(mod)
    tree = ast.parse(raw)
    doc = ast.get_docstring(tree, clean=False)
    if doc is not None:
        raw = raw.replace(doc, "", 1)
    # blank out comments and string literals line-precisely
    result = raw
    for tok in tokenize.generate_tokens(io.StringIO(raw).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            result = result.replace(tok.string, " " * len(tok.string), 1)
    return result


def test_materialiser_never_calls_fit_or_topology_or_recommended():
    src = _code_lines(lsm)
    for banned in (
        "model_placement_fit",
        "evaluate_workload_fit",
        "topology_for_config",
        "resolve_spill_preflight",
        "_recommended",
        "tok_s",
        "tensor_split",  # we never construct --tensor-split ratios
    ):
        assert banned not in src, banned


def test_materialiser_and_process_adapter_have_no_shell_or_broad_kill():
    for mod in (lsm, rpl):
        src = _code_lines(mod)
        assert "shell=True" not in src
        assert "os.system" not in src
        assert "shlex" not in src
        for tool in ("pkill", "killall", "pgrep", "systemctl", "sudo"):
            assert tool not in src, (mod.__name__, tool)


def test_no_signal_is_ever_sent_by_pid_or_port_only_via_the_retained_object():
    # The only ways a process is stopped: Popen.terminate()/kill()/wait() on
    # the retained child object. There is NO os.kill / os.killpg / killpg /
    # send_signal / SIGKILL-by-pid anywhere -- a foreign process discovered by
    # port can never be signalled because there is no code path that signals
    # anything but the object we launched.
    for mod in (lsm, rpl):
        src = _code_lines(mod)
        for banned in ("os.kill", "os.killpg", "killpg", ".send_signal(", "start_new_session"):
            assert banned not in src, (mod.__name__, banned)
