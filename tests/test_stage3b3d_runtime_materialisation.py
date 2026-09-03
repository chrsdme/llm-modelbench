"""Anvil Stage 3B.3D -- composition of resolve_runtime -> materialise ->
lifecycle controller, and the materialisation evidence record.

Pure composition: fake spawn adapters, fake clients, no CUDA / no llama.cpp
binary / no Ollama mutation / no real inference. Proves the pieces compose
with one planning authority, structured pre-row failure dispositions, owned
cleanup on every exit path observed exactly once, Ollama reuse-only, and an
auditable evidence record.
"""
from __future__ import annotations

import pytest

from llm_modelbench.hardware import GPUDevice
from llm_modelbench.identity import resolve_runtime_profile_identity
from llm_modelbench.runtime_identity import RuntimeExecutionSettings
from llm_modelbench.runtime_profiles import RuntimeCandidate, RuntimeProfile
from llm_modelbench.runtime_resolution import (
    ResolvedRuntime,
    RuntimeResolution,
    RuntimeResolutionStatus,
)
from llm_modelbench.runtime_lifecycle import (
    CleanupOutcome,
    LaunchProcessProof,
    MaterialisationRequest,
    RuntimeLifecycleController,
    RuntimeOwnership,
    materialise_owned_runtime,
    reuse_external_runtime,
)
from llm_modelbench.llama_server_materialisation import (
    ManagedMaterialisationOutcome,
    MaterialisationStatus,
)
from llm_modelbench import runtime_materialisation as rm
from llm_modelbench.runtime_materialisation import (
    MaterialisationSeams,
    RuntimeMaterialisationError,
    materialisation_evidence,
    resolve_and_materialise_runtime,
)

U_A = "GPU-00000000-1111-2222-3333-444444444444"
U_B = "GPU-99999999-8888-7777-6666-555555555555"
SHA = "sha256:" + "d" * 64
INV = (
    GPUDevice(0, U_A, "0000:01:00.0", "fx-A", 16000.0, None, None),
    GPUDevice(1, U_B, "0000:02:00.0", "fx-B", 16000.0, None, None),
)


# ---------------------------------------------------------------------------
# fixtures -- recipes / resolutions / requests
# ---------------------------------------------------------------------------
def _recipe(
    *,
    backend="llama_cpp",
    endpoint="http://127.0.0.1:8081",
    gpu_uuids=(U_A,),
    placement_class="full_gpu",
    requested_context=8192,
    allow_ram_spill=False,
    model_primary_sha256=SHA,
    tensor_split_weights=None,
):
    strategy = "single_device" if len(gpu_uuids) <= 1 else "layer_split"
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
        placement_class=placement_class,
        requested_context=requested_context,
        allow_ram_spill=allow_ram_spill,
        estimated_ram_spill_bytes=None,
        model_primary_sha256=model_primary_sha256,
        tensor_split_weights=tensor_split_weights,
    )


def _resolution(status=RuntimeResolutionStatus.RESOLVED, *, recipe=None, health="healthy", backend="llama_cpp", endpoint="http://127.0.0.1:8081"):
    recipe = recipe if recipe is not None else _recipe(backend=backend, endpoint=endpoint)
    cand = RuntimeCandidate(
        profile=RuntimeProfile(name="llama-local", backend=backend, endpoint=endpoint, provenance="configured"),
        health=health, source=("saved_profile",), detail=f"{health} fixture",
    )
    return RuntimeResolution(
        status=status, reason=status.value, detail="fixture",
        resolved=recipe if status is RuntimeResolutionStatus.RESOLVED else None,
        selected_candidate=cand,
    )


def _request(**kw):
    return MaterialisationRequest.from_resolution(_resolution(**kw))


# ---------------------------------------------------------------------------
# fake seams
# ---------------------------------------------------------------------------
def _proof(pid=4321):
    return LaunchProcessProof(
        pid=pid, process_start_time_ticks=99, executable_path="/opt/llama-server",
        command_argv=("/opt/llama-server", "--model", "/m.gguf", "--port", "8090"),
        parent_pid=1,
    )


class _FakeController(RuntimeLifecycleController):
    """A real controller (so owns_runtime / last_cleanup semantics are real)
    with a call-counting injected cleanup so 'exactly once' is provable."""

    def __init__(self, result, *, cleanup_exc=None):
        self.cleanup_calls = 0
        self.revalidate_calls = 0

        def _revalidate(owned):
            self.revalidate_calls += 1
            return owned.launch_proof

        def _cleanup(owned):
            self.cleanup_calls += 1
            if cleanup_exc is not None:
                raise cleanup_exc

        if result.ownership is RuntimeOwnership.MODELBENCH_OWNED:
            super().__init__(result, cleanup_fn=_cleanup, revalidate_fn=_revalidate)
        else:
            super().__init__(result)


def _spawn_ready(request, *, pid=4321):
    result = materialise_owned_runtime(
        request, launch_proof=_proof(pid), launched_at="2026-09-04T00:00:00Z"
    )
    return ManagedMaterialisationOutcome(
        status=MaterialisationStatus.SPAWNED_READY,
        detail="fake spawned ready",
        result=result,
        process=object(),
        endpoint=request.recipe.endpoint.replace("8081", "9099"),
        diagnostic_tail="loaded model; server listening",
        launched_argv=("/opt/llama-server", "--model", "/m.gguf", "--port", "9099", "--ctx-size", "8192"),
        attribution="ours",
    )


def _spawn_fail(request, status=MaterialisationStatus.RESOLVED_RECIPE_INCOMPLETE):
    return ManagedMaterialisationOutcome(
        status=status, detail="fake managed failure",
        diagnostic_tail="child exited 3", launched_argv=None,
    )


def _seams(*, spawn=None, external_still_healthy=None, controller_factory=None, cleanup_exc=None):
    def _default_cf(outcome):
        return _FakeController(outcome.result, cleanup_exc=cleanup_exc)

    return MaterialisationSeams(
        spawn_managed=spawn or (lambda r: _spawn_ready(r)),
        external_still_healthy=external_still_healthy,
        controller_factory=controller_factory or _default_cf,
    )


def _spawn_forbidden(request):  # noqa: ARG001
    raise AssertionError("spawn_managed must not be reached on this path")


# ===========================================================================
# resolver integration
# ===========================================================================
def test_resolve_is_called_before_any_materialisation(monkeypatch):
    order = []

    def _resolve(**kw):
        order.append("resolve")
        return _resolution()

    def _mat(request, **kw):
        order.append("materialise")
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.REUSED_EXTERNAL, detail="reuse",
            result=reuse_external_runtime(request), endpoint=request.recipe.endpoint,
        )

    out = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(), resolve_fn=_resolve, materialise_fn=_mat,
    )
    assert order == ["resolve", "materialise"]
    assert out.ok


@pytest.mark.parametrize("status", [
    RuntimeResolutionStatus.RUNTIME_AMBIGUOUS,
    RuntimeResolutionStatus.UNSUPPORTED_BACKEND_SELECTED,
    RuntimeResolutionStatus.ENVIRONMENT_INFEASIBLE,
    RuntimeResolutionStatus.NO_USABLE_ENDPOINT,
    RuntimeResolutionStatus.FIT_UNKNOWN,
])
def test_unresolved_status_is_a_structured_refusal_not_a_quality_failure(status):
    out = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(spawn=_spawn_forbidden),
        resolve_fn=lambda **kw: _resolution(status),
    )
    assert out.ok is False
    assert out.controller is None and out.endpoint is None
    assert out.refusal_reason.startswith(f"runtime_not_resolved: {status.value}:")
    assert out.resolution_status == status.value


def test_failed_resolver_prevents_materialisation_and_client_construction():
    out = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(spawn=_spawn_forbidden),
        resolve_fn=lambda **kw: _resolution(RuntimeResolutionStatus.RUNTIME_UNAVAILABLE),
    )
    assert out.ok is False
    assert out.materialisation is None  # never attempted


# ===========================================================================
# materialisation integration
# ===========================================================================
def test_successful_external_reuse_reaches_client_construction():
    def _mat(request, **kw):
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.REUSED_EXTERNAL, detail="reuse",
            result=reuse_external_runtime(request), endpoint=request.recipe.endpoint,
        )

    out = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(spawn=_spawn_forbidden), materialise_fn=_mat,
        resolve_fn=lambda **kw: _resolution(),
    )
    assert out.ok
    assert out.endpoint == "http://127.0.0.1:8081"
    assert out.ownership == RuntimeOwnership.EXTERNAL_REUSED.value
    assert out.owns_runtime is False


def test_successful_managed_llama_server_reaches_client_construction():
    # managed path: real resolver never emits RESOLVED with an unhealthy
    # candidate, so drive the managed branch via the demotion probe.
    out = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(external_still_healthy=lambda r: False),
        resolve_fn=lambda **kw: _resolution(health="healthy"),
    )
    assert out.ok
    assert out.ownership == RuntimeOwnership.MODELBENCH_OWNED.value
    assert out.owns_runtime is True
    assert out.endpoint == "http://127.0.0.1:9099"  # managed endpoint, not 8081


def test_materialisation_failure_prevents_client_construction():
    out = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(spawn=lambda r: _spawn_fail(r),
                                      external_still_healthy=lambda r: False),
        resolve_fn=lambda **kw: _resolution(health="healthy"),
    )
    assert out.ok is False
    assert out.controller is None and out.endpoint is None
    assert out.refusal_reason.startswith(
        "runtime_not_materialised: resolved_recipe_incomplete_for_materialisation:"
    )


def test_managed_endpoint_overrides_stale_configured_endpoint():
    # the resolved/configured endpoint is 8081; the materialised managed
    # endpoint is 9099 -- the outcome must carry 9099.
    out = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(external_still_healthy=lambda r: False),
        resolve_fn=lambda **kw: _resolution(health="healthy"),
    )
    assert out.endpoint == "http://127.0.0.1:9099"
    assert out.resolution.resolved.endpoint == "http://127.0.0.1:8081"


# ===========================================================================
# cleanup integration
# ===========================================================================
def test_owned_runtime_cleaned_on_success_exactly_once():
    out = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(external_still_healthy=lambda r: False),
        resolve_fn=lambda **kw: _resolution(health="healthy"),
    )
    ctrl = out.controller
    with ctrl:
        pass  # benchmark "succeeds"
    assert ctrl.cleanup_calls == 1
    assert ctrl.last_cleanup.outcome is CleanupOutcome.SUCCEEDED


def test_owned_runtime_cleaned_on_benchmark_exception_exactly_once():
    out = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(external_still_healthy=lambda r: False),
        resolve_fn=lambda **kw: _resolution(health="healthy"),
    )
    ctrl = out.controller
    with pytest.raises(RuntimeError, match="boom"), ctrl:
        raise RuntimeError("boom")
    assert ctrl.cleanup_calls == 1


def test_external_runtime_is_never_cleaned():
    def _mat(request, **kw):
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.REUSED_EXTERNAL, detail="reuse",
            result=reuse_external_runtime(request), endpoint=request.recipe.endpoint,
        )

    out = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(spawn=_spawn_forbidden), materialise_fn=_mat,
        resolve_fn=lambda **kw: _resolution(),
    )
    ctrl = out.controller
    assert ctrl.owns_runtime is False
    with ctrl:
        pass
    # external exit is a pure no-op: no destructive callback, ever.
    assert ctrl.cleanup_calls == 0
    assert ctrl.last_cleanup is None
    # and an explicit cleanup() call still refuses destructively
    assert ctrl.cleanup().outcome is CleanupOutcome.NOT_APPLICABLE_EXTERNAL
    assert ctrl.cleanup_calls == 0


def test_cleanup_failure_on_successful_benchmark_is_recorded_in_evidence():
    out = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(external_still_healthy=lambda r: False,
                                      cleanup_exc=RuntimeError("teardown kaput")),
        resolve_fn=lambda **kw: _resolution(health="healthy"),
    )
    ctrl = out.controller
    with ctrl:
        pass
    assert ctrl.last_cleanup.outcome is CleanupOutcome.GRACEFUL_FAILED
    ev = materialisation_evidence(out, cleanup_result=ctrl.last_cleanup, benchmark_completed=True)
    assert ev["cleanup"]["observed"] is True
    assert ev["cleanup"]["outcome"] == CleanupOutcome.GRACEFUL_FAILED.value
    assert "cleanup_failed_on_successful_benchmark" in ev["warnings"]


# ===========================================================================
# Ollama -- reuse only
# ===========================================================================
def test_healthy_external_ollama_is_reused():
    def _mat(request, **kw):
        assert request.backend == "ollama"
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.REUSED_EXTERNAL, detail="reuse ollama",
            result=reuse_external_runtime(request), endpoint=request.recipe.endpoint,
        )

    out = resolve_and_materialise_runtime(
        selected_backend="ollama", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(spawn=_spawn_forbidden), materialise_fn=_mat,
        resolve_fn=lambda **kw: _resolution(backend="ollama", endpoint="http://127.0.0.1:11434"),
    )
    assert out.ok and out.ownership == RuntimeOwnership.EXTERNAL_REUSED.value


def test_missing_ollama_endpoint_is_a_structured_failure_no_spawn_no_fallback():
    # real materialise() dispatch: ollama + not reusable -> EXTERNAL_RUNTIME_REQUIRED
    out = resolve_and_materialise_runtime(
        selected_backend="ollama", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(spawn=_spawn_forbidden,
                                      external_still_healthy=lambda r: False),
        resolve_fn=lambda **kw: _resolution(backend="ollama", endpoint="http://127.0.0.1:11434", health="healthy"),
    )
    assert out.ok is False
    assert out.refusal_reason.startswith("runtime_not_materialised: external_runtime_required:")
    assert out.materialisation.result is None  # nothing materialised
    # no fallback: backend echo is still ollama, nothing llama_cpp happened
    assert out.backend == "ollama"


def test_ollama_path_never_touches_the_llama_server_spawn_adapter():
    out = resolve_and_materialise_runtime(
        selected_backend="ollama", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(spawn=_spawn_forbidden,
                                      external_still_healthy=lambda r: False),
        resolve_fn=lambda **kw: _resolution(backend="ollama", endpoint="http://127.0.0.1:11434", health="healthy"),
    )
    assert out.ok is False  # _spawn_forbidden would have raised AssertionError


# ===========================================================================
# llama.cpp
# ===========================================================================
def test_external_llama_server_reused_zero_spawn():
    def _mat(request, **kw):
        # reuse-eligible (healthy candidate) -> materialise() never calls spawn
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.REUSED_EXTERNAL, detail="reuse",
            result=reuse_external_runtime(request), endpoint=request.recipe.endpoint,
        )

    out = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(spawn=_spawn_forbidden), materialise_fn=_mat,
        resolve_fn=lambda **kw: _resolution(),
    )
    assert out.ok and out.ownership == RuntimeOwnership.EXTERNAL_REUSED.value


def test_managed_llama_server_path_uses_the_injected_materialiser_outcome():
    seen = {}

    def _spawn(request):
        seen["called"] = True
        return _spawn_ready(request)

    out = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(spawn=_spawn, external_still_healthy=lambda r: False),
        resolve_fn=lambda **kw: _resolution(health="healthy"),
    )
    assert seen.get("called") is True
    assert out.materialisation.launched_argv[0] == "/opt/llama-server"


def test_no_fallback_from_failed_llama_cpp_materialisation_to_ollama():
    out = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(spawn=lambda r: _spawn_fail(r, MaterialisationStatus.SPAWN_FAILED),
                                      external_still_healthy=lambda r: False),
        resolve_fn=lambda **kw: _resolution(health="healthy"),
    )
    assert out.ok is False
    assert out.backend == "llama_cpp"
    assert out.refusal_reason.startswith("runtime_not_materialised: spawn_failed:")


# ===========================================================================
# evidence / ledger
# ===========================================================================
def _ok_managed_outcome(tensor_split_weights=None, gpu_uuids=(U_A, U_B), placement_class="multi_gpu"):
    recipe = _recipe(gpu_uuids=gpu_uuids, placement_class=placement_class,
                     tensor_split_weights=tensor_split_weights)
    return resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(external_still_healthy=lambda r: False),
        resolve_fn=lambda **kw: _resolution(recipe=recipe, health="healthy"),
    )


def test_evidence_carries_resolved_recipe_verbatim():
    # U_B sorts AFTER U_A -- put U_B first so a stray sort() in evidence
    # serialisation would reorder to [U_A, U_B] and this assertion fails.
    out = _ok_managed_outcome(gpu_uuids=(U_B, U_A), tensor_split_weights=(7, 5))
    ev = materialisation_evidence(out)
    res = ev["resolution"]
    assert res["selected_physical_gpu_uuids"] == [U_B, U_A]  # resolver order, NOT sorted
    assert sorted(res["selected_physical_gpu_uuids"]) != res["selected_physical_gpu_uuids"]
    assert res["tensor_split_weights"] == [7, 5]
    assert res["placement_class"] == "multi_gpu"
    assert res["allow_ram_spill"] is False
    assert res["requested_context"] == 8192
    assert res["model_primary_sha256"] == SHA


def test_evidence_records_status_ownership_endpoint_argv_and_identity_key():
    out = _ok_managed_outcome()
    ev = materialisation_evidence(out)
    m = ev["materialisation"]
    assert m["status"] == "spawned_ready"
    assert m["ownership"] == "modelbench_owned"
    assert m["endpoint_used"] == out.endpoint
    assert m["launched_argv"][0] == "/opt/llama-server"
    assert m["diagnostic_tail"]  # bounded, non-empty
    assert ev["identity_key"] == out.identity_key
    assert ev["identity_key"].startswith("materialisation_request_v2|")
    assert m["launch_proof"]["pid"] == 4321


def test_evidence_for_a_refused_outcome_has_no_client_fields_but_keeps_status():
    out = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(spawn=_spawn_forbidden),
        resolve_fn=lambda **kw: _resolution(RuntimeResolutionStatus.RUNTIME_AMBIGUOUS),
    )
    ev = materialisation_evidence(out)
    assert ev["ok"] is False
    assert ev["resolution_status"] == "runtime_ambiguous"
    assert "cleanup" not in ev


def test_evidence_is_json_serialisable():
    import json

    out = _ok_managed_outcome(tensor_split_weights=(3, 2))
    ctrl = out.controller
    with ctrl:
        pass
    ev = materialisation_evidence(out, cleanup_result=ctrl.last_cleanup, benchmark_completed=True)
    json.dumps(ev)  # must not raise


# ===========================================================================
# identity
# ===========================================================================
def test_identity_key_is_persisted_and_primary_gpu_reorder_does_not_collapse_it():
    a = _ok_managed_outcome(gpu_uuids=(U_A, U_B), tensor_split_weights=(1, 1))
    b = _ok_managed_outcome(gpu_uuids=(U_B, U_A), tensor_split_weights=(1, 1))
    assert a.identity_key != b.identity_key
    assert materialisation_evidence(a)["identity_key"] != materialisation_evidence(b)["identity_key"]


def test_endpoint_and_pid_are_evidence_not_durable_identity():
    # two managed spawns of the same recipe with different pid/endpoint have
    # the SAME identity_key (recipe identity), but different launch evidence.
    r = _recipe()
    o1 = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(spawn=lambda req: _spawn_ready(req, pid=111),
                                      external_still_healthy=lambda req: False),
        resolve_fn=lambda **kw: _resolution(recipe=r, health="healthy"),
    )
    o2 = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(spawn=lambda req: _spawn_ready(req, pid=222),
                                      external_still_healthy=lambda req: False),
        resolve_fn=lambda **kw: _resolution(recipe=r, health="healthy"),
    )
    assert o1.identity_key == o2.identity_key
    assert materialisation_evidence(o1)["materialisation"]["launch_proof"]["pid"] == 111
    assert materialisation_evidence(o2)["materialisation"]["launch_proof"]["pid"] == 222


# ===========================================================================
# authority
# ===========================================================================
def test_recommended_flag_cannot_influence_materialisation():
    # candidate.recommended True vs False -> identical outcome (resolver never
    # consults it; composition never sees it).
    def _res(recommended):
        recipe = _recipe()
        cand = RuntimeCandidate(
            profile=RuntimeProfile(name="llama-local", backend="llama_cpp",
                                   endpoint=recipe.endpoint, provenance="configured"),
            health="healthy", source=("saved_profile",), detail="fx", recommended=recommended,
        )
        return RuntimeResolution(status=RuntimeResolutionStatus.RESOLVED, reason="resolved",
                                 detail="fx", resolved=recipe, selected_candidate=cand)

    def _mat(request, **kw):
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.REUSED_EXTERNAL, detail="reuse",
            result=reuse_external_runtime(request), endpoint=request.recipe.endpoint,
        )

    a = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(spawn=_spawn_forbidden), materialise_fn=_mat,
        resolve_fn=lambda **kw: _res(True),
    )
    b = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(spawn=_spawn_forbidden), materialise_fn=_mat,
        resolve_fn=lambda **kw: _res(False),
    )
    assert a.endpoint == b.endpoint and a.ownership == b.ownership
    assert a.backend == b.backend == "llama_cpp"  # backend echo is the caller's, never derived from `recommended`


def test_backend_echo_is_the_caller_selected_backend_verbatim():
    # Whatever the selected_candidate's `recommended` flag is, the outcome's
    # backend is exactly the caller's selected_backend -- never re-derived.
    for recommended in (True, False):
        cand = RuntimeCandidate(
            profile=RuntimeProfile(name="llama-local", backend="llama_cpp",
                                   endpoint="http://127.0.0.1:8081", provenance="configured"),
            health="healthy", source=("saved_profile",), detail="fx", recommended=recommended,
        )
        res = RuntimeResolution(status=RuntimeResolutionStatus.RESOLVED, reason="resolved",
                                detail="fx", resolved=_recipe(), selected_candidate=cand)
        out = resolve_and_materialise_runtime(
            selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
            host_meminfo={}, seams=_seams(external_still_healthy=lambda r: False),
            resolve_fn=lambda **kw: res,
        )
        assert out.backend == "llama_cpp"


def test_composition_adds_no_planning_authority_bad_seams_type_raises():
    with pytest.raises(RuntimeMaterialisationError):
        resolve_and_materialise_runtime(
            selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
            host_meminfo={}, seams=object(),  # not a MaterialisationSeams
        )


def test_no_direct_subprocess_or_popen_in_the_composition_module():
    import inspect

    src = inspect.getsource(rm)
    for banned in ("subprocess.Popen", "os.system", "shell=True", "Popen("):
        assert banned not in src
