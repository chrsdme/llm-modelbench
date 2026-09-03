"""Anvil Stage 3B.3B -- runtime lifecycle domain / ownership contract.

Pure/testable: no process spawn, no real socket, no real port, no real
backend, no runner integration. Cleanup is an injected fake callback.

Covers the stage prompt's required matrix:
* resolver boundary -- only an accepted RESOLVED resolution yields a request
* ownership -- external reuse has no cleanup authority; owned has explicit
  authority; PID alone is insufficient; proof mismatch refuses cleanup; no
  profile flag confers ownership
* cleanup -- allowed once; external is a structural no-op; idempotent;
  context manager cleans owned on normal and exceptional exit; external
  context exit never destroys; cleanup failure keeps the primary exception;
  result is structured
* determinism -- same recipe -> identical request identity; discovery order
  irrelevant; _recommended() has no influence
* fit authority -- no lifecycle type recomputes topology/fit; no RAM-spill
  permission mintable in materialisation; GPU UUIDs come from the recipe;
  lifecycle cannot append a GPU
"""
import subprocess

import pytest

from llm_modelbench.identity import resolve_runtime_profile_identity
from llm_modelbench.runtime_identity import RuntimeExecutionSettings
from llm_modelbench.runtime_resolution import (
    ResolvedRuntime,
    RuntimeResolution,
    RuntimeResolutionStatus,
)
from llm_modelbench.runtime_lifecycle import (
    CleanupAuthority,
    CleanupOutcome,
    CleanupResult,
    LaunchProcessProof,
    LifecycleState,
    MaterialisationRequest,
    MaterialisationRequestError,
    OwnedRuntime,
    RuntimeLifecycleController,
    RuntimeOwnership,
    materialise_owned_runtime,
    reuse_external_runtime,
)

U_A = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
U_B = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _exec_settings(*, context_size=8192, strategy="single_gpu"):
    return RuntimeExecutionSettings(strategy=strategy, context_size=context_size)


def _resolved_recipe(
    *,
    backend="llama_cpp",
    endpoint="http://127.0.0.1:8081",
    gpu_uuids=(U_A,),
    placement_class="single_gpu",
    requested_context=8192,
    allow_ram_spill=False,
    execution_settings=None,
):
    settings = execution_settings or _exec_settings(context_size=requested_context)
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
    )


def _resolution(status=RuntimeResolutionStatus.RESOLVED, *, recipe=None, **recipe_overrides):
    if status is RuntimeResolutionStatus.RESOLVED and recipe is None:
        recipe = _resolved_recipe(**recipe_overrides)
    return RuntimeResolution(
        status=status,
        reason=status.value,
        detail=f"{status.value} fixture",
        resolved=recipe if status is RuntimeResolutionStatus.RESOLVED else None,
    )


def _request(**recipe_overrides):
    return MaterialisationRequest.from_resolution(_resolution(**recipe_overrides))


def _proof(pid=4242, ticks=99000, exe="/opt/llama/bin/llama-server",
           argv=("llama-server", "-m", "model.gguf", "--port", "8081")):
    return LaunchProcessProof(
        pid=pid, process_start_time_ticks=ticks, executable_path=exe, command_argv=argv
    )


def _owned_result(request=None, proof=None):
    request = request or _request()
    return materialise_owned_runtime(
        request, launch_proof=proof or _proof(), launched_at="2026-09-03T12:00:00Z"
    )


def _ok_reval(owned):
    """A revalidator that proves the process is still the one we launched --
    returns an identical LaunchProcessProof."""
    p = owned.launch_proof
    return LaunchProcessProof(
        pid=p.pid, process_start_time_ticks=p.process_start_time_ticks,
        executable_path=p.executable_path, command_argv=p.command_argv,
    )


def _owned_controller(*, cleanup_fn, revalidate_fn=_ok_reval, result=None):
    return RuntimeLifecycleController(
        result or _owned_result(), cleanup_fn=cleanup_fn, revalidate_fn=revalidate_fn
    )


# ==========================================================================
# resolver boundary
# ==========================================================================
def test_accepted_resolution_produces_a_materialisation_request():
    req = MaterialisationRequest.from_resolution(_resolution())
    assert isinstance(req, MaterialisationRequest)
    assert req.backend == "llama_cpp"
    assert req.endpoint == "http://127.0.0.1:8081"


@pytest.mark.parametrize(
    "status",
    [
        RuntimeResolutionStatus.NO_BACKEND_SELECTED,
        RuntimeResolutionStatus.UNSUPPORTED_BACKEND_SELECTED,
        RuntimeResolutionStatus.RUNTIME_AMBIGUOUS,
        RuntimeResolutionStatus.ENVIRONMENT_INFEASIBLE,
        RuntimeResolutionStatus.FIT_UNKNOWN,
        RuntimeResolutionStatus.CAPABILITY_EVIDENCE_INSUFFICIENT,
        RuntimeResolutionStatus.IDENTITY_INSUFFICIENT,
        RuntimeResolutionStatus.RUNTIME_UNAVAILABLE,
    ],
)
def test_non_accepted_resolution_cannot_produce_a_request(status):
    with pytest.raises(MaterialisationRequestError) as ei:
        MaterialisationRequest.from_resolution(_resolution(status))
    assert status.value in str(ei.value)


def test_resolved_status_without_recipe_is_rejected():
    malformed = RuntimeResolution(
        status=RuntimeResolutionStatus.RESOLVED, reason="resolved", detail="no recipe",
        resolved=None,
    )
    with pytest.raises(MaterialisationRequestError):
        MaterialisationRequest.from_resolution(malformed)


def test_request_cannot_be_hand_constructed_bypassing_the_resolver():
    # A caller with an accepted resolution object still must not skip the
    # authoritative factory -- the private token guards it.
    good = _resolution()
    with pytest.raises(MaterialisationRequestError):
        MaterialisationRequest(resolution=good, _token=object())


def test_request_construction_without_a_token_is_an_error():
    with pytest.raises((MaterialisationRequestError, TypeError)):
        MaterialisationRequest(resolution=_resolution())


def test_owned_runtime_construction_without_a_token_is_an_error():
    with pytest.raises((MaterialisationRequestError, TypeError)):
        OwnedRuntime(
            backend="llama_cpp",
            endpoint="http://127.0.0.1:8081",
            launch_proof=_proof(),
            launched_at="2026-09-03T12:00:00Z",
            runtime_profile_identity=resolve_runtime_profile_identity(backend="llama_cpp"),
            recipe_identity_key="forged",
            selected_physical_gpu_uuids=(U_A,),
        )


def test_request_rejects_non_resolution_input():
    with pytest.raises(MaterialisationRequestError):
        MaterialisationRequest.from_resolution({"status": "resolved"})


# ==========================================================================
# ownership
# ==========================================================================
def test_external_reuse_has_no_cleanup_authority():
    result = reuse_external_runtime(_request())
    assert result.ownership is RuntimeOwnership.EXTERNAL_REUSED
    assert result.usable is True
    assert result.owned_runtime is None
    assert result.cleanup_authority is None


def test_owned_runtime_has_explicit_cleanup_authority():
    result = _owned_result()
    assert result.ownership is RuntimeOwnership.MODELBENCH_OWNED
    assert isinstance(result.owned_runtime, OwnedRuntime)
    authority = result.cleanup_authority
    assert isinstance(authority, CleanupAuthority)
    assert authority.owned_runtime is result.owned_runtime


def test_owned_runtime_cannot_be_hand_constructed():
    with pytest.raises(MaterialisationRequestError):
        OwnedRuntime(
            backend="llama_cpp",
            endpoint="http://127.0.0.1:8081",
            launch_proof=_proof(),
            launched_at="2026-09-03T12:00:00Z",
            runtime_profile_identity=resolve_runtime_profile_identity(backend="llama_cpp"),
            recipe_identity_key="forged",
            selected_physical_gpu_uuids=(U_A,),
            _token=object(),
        )


def test_pid_alone_is_insufficient_ownership_proof():
    owned = _owned_result().owned_runtime
    # Same PID, but the start-time tick differs -> a reused PID, not our
    # process.
    observed = _proof(pid=owned.launch_proof.pid, ticks=owned.launch_proof.process_start_time_ticks + 1)
    assert owned.launch_proof.revalidation_matches(observed) is False


def test_missing_start_time_tick_never_counts_as_a_match():
    owned = _owned_result().owned_runtime
    # Revalidation returns an identity with no start-time tick -- identity
    # could not be proven; must not be treated as a match.
    no_ticks = LaunchProcessProof(
        pid=owned.launch_proof.pid, process_start_time_ticks=None,
        executable_path=owned.launch_proof.executable_path,
        command_argv=owned.launch_proof.command_argv,
    )
    assert owned.launch_proof.revalidation_matches(no_ticks) is False
    # ...and symmetrically, a launch record with no tick can never be matched.
    weak_record = LaunchProcessProof(
        pid=4242, process_start_time_ticks=None, executable_path="/x", command_argv=("x",)
    )
    assert weak_record.revalidation_matches(_proof(pid=4242)) is False


def test_full_process_identity_match_is_accepted():
    owned = _owned_result().owned_runtime
    identical = LaunchProcessProof(
        pid=owned.launch_proof.pid,
        process_start_time_ticks=owned.launch_proof.process_start_time_ticks,
        executable_path=owned.launch_proof.executable_path,
        command_argv=owned.launch_proof.command_argv,
    )
    assert owned.launch_proof.revalidation_matches(identical) is True


def test_executable_or_argv_divergence_breaks_the_match():
    p = _proof()
    assert p.revalidation_matches(_proof(exe="/usr/bin/other")) is False
    assert p.revalidation_matches(_proof(argv=("llama-server", "--port", "9999"))) is False


def test_ownership_proof_mismatch_causes_cleanup_refusal():
    result = _owned_result()
    calls = []

    def _cleanup(owned):
        calls.append(owned)

    def _revalidate(owned):
        # process is gone / replaced -- identity cannot be revalidated
        return _proof(ticks=owned.launch_proof.process_start_time_ticks + 5)

    ctrl = RuntimeLifecycleController(result, cleanup_fn=_cleanup, revalidate_fn=_revalidate)
    outcome = ctrl.cleanup()
    assert outcome.outcome is CleanupOutcome.OWNERSHIP_NOT_REVALIDATED
    assert outcome.destructive_action_performed is False
    assert calls == []
    assert ctrl.state is LifecycleState.CLEANUP_FAILED


def test_revalidation_returning_none_refuses_cleanup():
    result = _owned_result()
    ctrl = RuntimeLifecycleController(
        result, cleanup_fn=lambda o: None, revalidate_fn=lambda o: None
    )
    assert ctrl.cleanup().outcome is CleanupOutcome.OWNERSHIP_NOT_REVALIDATED


def test_no_profile_flag_confers_ownership():
    # An external-reuse result wrapped in a controller stays unowned even
    # though it carries a real runtime_profile_identity and endpoint.
    result = reuse_external_runtime(_request())
    ctrl = RuntimeLifecycleController(result, cleanup_fn=lambda o: pytest.fail("must not run"))
    assert ctrl.owns_runtime is False
    assert ctrl.cleanup().outcome is CleanupOutcome.NOT_APPLICABLE_EXTERNAL


# ==========================================================================
# cleanup
# ==========================================================================
def test_owned_runtime_cleanup_allowed_once():
    calls = []
    ctrl = _owned_controller(cleanup_fn=lambda o: calls.append(o))
    first = ctrl.cleanup()
    assert first.outcome is CleanupOutcome.SUCCEEDED
    assert first.destructive_action_performed is True
    assert len(calls) == 1
    assert ctrl.state is LifecycleState.CLEANED


def test_external_cleanup_is_structural_no_op():
    ctrl = RuntimeLifecycleController(reuse_external_runtime(_request()))
    r = ctrl.cleanup()
    assert r.outcome is CleanupOutcome.NOT_APPLICABLE_EXTERNAL
    assert r.destructive_action_performed is False


def test_owned_controller_requires_cleanup_and_revalidate_fns():
    # The fail-open modes this slice prevents: signalling a PID with no proof,
    # or reporting a destruction that never ran. Constructing an owned
    # controller without both callbacks is refused.
    result = _owned_result()
    with pytest.raises(TypeError):
        RuntimeLifecycleController(result)
    with pytest.raises(TypeError):
        RuntimeLifecycleController(result, cleanup_fn=lambda o: None)
    with pytest.raises(TypeError):
        RuntimeLifecycleController(result, revalidate_fn=_ok_reval)


def test_cleanup_is_idempotent():
    calls = []
    ctrl = _owned_controller(cleanup_fn=lambda o: calls.append(o))
    ctrl.cleanup()
    second = ctrl.cleanup()
    assert second.outcome is CleanupOutcome.ALREADY_COMPLETED
    assert second.destructive_action_performed is False
    assert len(calls) == 1


def test_exactly_one_destructive_action_even_after_failure():
    calls = []

    def _cleanup(owned):
        calls.append(owned)
        raise RuntimeError("SIGTERM ignored")

    ctrl = _owned_controller(cleanup_fn=_cleanup)
    first = ctrl.cleanup()
    assert first.outcome is CleanupOutcome.GRACEFUL_FAILED
    assert first.destructive_action_performed is True
    second = ctrl.cleanup()
    assert second.outcome is CleanupOutcome.ALREADY_COMPLETED
    assert len(calls) == 1  # no second signal


def test_context_manager_cleans_owned_runtime_on_normal_exit():
    calls = []
    with _owned_controller(cleanup_fn=lambda o: calls.append(o)) as ctrl:
        assert ctrl.state is LifecycleState.READY
    assert len(calls) == 1
    assert ctrl.state is LifecycleState.CLEANED


def test_context_manager_cleans_owned_runtime_on_exception():
    calls = []
    with pytest.raises(ValueError, match="boom"):
        with _owned_controller(cleanup_fn=lambda o: calls.append(o)) as ctrl:
            raise ValueError("boom")
    assert len(calls) == 1
    assert ctrl.state is LifecycleState.CLEANED


def test_external_runtime_context_exit_never_destroys():
    calls = []
    with RuntimeLifecycleController(
        reuse_external_runtime(_request()), cleanup_fn=lambda o: calls.append(o)
    ):
        pass
    assert calls == []


def test_cleanup_failure_does_not_replace_the_original_exception():
    def _cleanup(owned):
        raise RuntimeError("cleanup blew up")

    with pytest.raises(ValueError, match="primary"):
        with _owned_controller(cleanup_fn=_cleanup) as ctrl:
            raise ValueError("primary failure")
    assert ctrl.last_cleanup.outcome is CleanupOutcome.GRACEFUL_FAILED
    assert ctrl.state is LifecycleState.CLEANUP_FAILED


def test_cleanup_failure_on_clean_exit_is_structured_not_raised():
    def _cleanup(owned):
        raise RuntimeError("teardown failed")

    with _owned_controller(cleanup_fn=_cleanup) as ctrl:
        pass
    # no exception propagated; outcome is available structurally
    assert ctrl.last_cleanup.outcome is CleanupOutcome.GRACEFUL_FAILED
    assert ctrl.last_cleanup.ok is False


def test_forced_cleanup_failure_is_distinct_from_a_generic_callback_error():
    from llm_modelbench.runtime_lifecycle import ForcedCleanupFailed

    def _cleanup(owned):
        # the adapter ran the full graceful->forced escalation and the
        # process still outlived SIGKILL
        raise ForcedCleanupFailed("survived SIGTERM and SIGKILL")

    ctrl = _owned_controller(cleanup_fn=_cleanup)
    res = ctrl.cleanup()
    assert res.outcome is CleanupOutcome.FORCED_FAILED
    assert res.outcome is not CleanupOutcome.GRACEFUL_FAILED
    assert res.destructive_action_performed is True
    assert ctrl.state is LifecycleState.CLEANUP_FAILED
    # still exactly one destructive attempt
    second = ctrl.cleanup()
    assert second.outcome is CleanupOutcome.ALREADY_COMPLETED


def test_forced_cleanup_failure_does_not_replace_the_primary_exception():
    from llm_modelbench.runtime_lifecycle import ForcedCleanupFailed

    def _cleanup(owned):
        raise ForcedCleanupFailed("refused to die")

    with pytest.raises(ValueError, match="primary"):
        with _owned_controller(cleanup_fn=_cleanup) as ctrl:
            raise ValueError("primary failure")
    assert ctrl.last_cleanup.outcome is CleanupOutcome.FORCED_FAILED


def test_cleanup_result_is_structured():
    ctrl = _owned_controller(cleanup_fn=lambda o: None)
    out = ctrl.cleanup()
    assert isinstance(out, CleanupResult)
    assert isinstance(out.outcome, CleanupOutcome)
    assert isinstance(out.detail, str) and out.detail
    assert isinstance(out.destructive_action_performed, bool)


def test_context_exit_after_explicit_cleanup_does_not_re_clean():
    calls = []
    ctrl = _owned_controller(cleanup_fn=lambda o: calls.append(o))
    with ctrl:
        ctrl.cleanup()
    assert len(calls) == 1


# ==========================================================================
# determinism
# ==========================================================================
def test_same_resolved_recipe_gives_identical_request_identity():
    a = _request()
    b = _request()
    assert a.identity_key() == b.identity_key()


def test_considered_endpoint_order_does_not_enter_request_identity():
    recipe = _resolved_recipe()
    res_a = RuntimeResolution(
        status=RuntimeResolutionStatus.RESOLVED, reason="resolved", detail="a",
        resolved=recipe, considered_candidate_endpoints=("http://a", "http://b"),
    )
    res_b = RuntimeResolution(
        status=RuntimeResolutionStatus.RESOLVED, reason="resolved", detail="b",
        resolved=recipe, considered_candidate_endpoints=("http://b", "http://a", "http://c"),
    )
    key_a = MaterialisationRequest.from_resolution(res_a).identity_key()
    key_b = MaterialisationRequest.from_resolution(res_b).identity_key()
    assert key_a == key_b


def test_recommended_flag_on_selected_candidate_has_no_lifecycle_influence():
    from llm_modelbench.runtime_profiles import RuntimeCandidate, RuntimeProfile

    recipe = _resolved_recipe()
    cand_reco = RuntimeCandidate(
        profile=RuntimeProfile(name="p", backend="llama_cpp", endpoint="http://127.0.0.1:8081",
                               provenance="configured"),
        health="healthy", source=("saved_profile",), detail="x", recommended=True,
    )
    cand_not = RuntimeCandidate(
        profile=RuntimeProfile(name="p", backend="llama_cpp", endpoint="http://127.0.0.1:8081",
                               provenance="configured"),
        health="healthy", source=("saved_profile",), detail="x", recommended=False,
    )
    res_reco = RuntimeResolution(
        status=RuntimeResolutionStatus.RESOLVED, reason="resolved", detail="",
        resolved=recipe, selected_candidate=cand_reco,
    )
    res_not = RuntimeResolution(
        status=RuntimeResolutionStatus.RESOLVED, reason="resolved", detail="",
        resolved=recipe, selected_candidate=cand_not,
    )
    k1 = MaterialisationRequest.from_resolution(res_reco).identity_key()
    k2 = MaterialisationRequest.from_resolution(res_not).identity_key()
    assert k1 == k2


def test_gpu_uuid_order_in_recipe_does_not_change_request_identity():
    k1 = _request(gpu_uuids=(U_A, U_B), placement_class="minimum_multi_gpu").identity_key()
    k2 = _request(gpu_uuids=(U_B, U_A), placement_class="minimum_multi_gpu").identity_key()
    assert k1 == k2


def test_owned_runtime_recipe_identity_key_tracks_the_request():
    req = _request()
    owned = _owned_result(request=req).owned_runtime
    assert owned.recipe_identity_key == req.identity_key()


# ==========================================================================
# fit authority
# ==========================================================================
def test_no_lifecycle_type_recomputes_topology_or_fit():
    import inspect
    import llm_modelbench.runtime_lifecycle as mod

    src = inspect.getsource(mod)
    for forbidden in ("evaluate_workload_fit", "resolve_spill_preflight",
                      "topology_from_inventory", "TopologyBudget", "WorkloadFit"):
        assert forbidden not in src, f"lifecycle module must not reference {forbidden}"


def test_ram_spill_permission_cannot_be_created_in_materialisation():
    # The recipe says spill is NOT permitted. Nothing in the lifecycle path
    # can turn it on -- the request only references the recipe.
    req = _request(allow_ram_spill=False)
    assert req.recipe.allow_ram_spill is False
    owned = _owned_result(request=req).owned_runtime
    # OwnedRuntime carries no spill field at all -- it cannot express a
    # permission the resolver did not grant.
    assert not hasattr(owned, "allow_ram_spill")
    assert not hasattr(owned, "allow_cpu_spill")


def test_spill_permission_flows_through_only_from_the_recipe():
    permit = _request(allow_ram_spill=True)
    deny = _request(allow_ram_spill=False)
    assert permit.identity_key() != deny.identity_key()  # honestly distinguished
    assert permit.recipe.allow_ram_spill is True


def test_selected_gpu_uuids_come_from_the_resolved_recipe():
    req = _request(gpu_uuids=(U_A, U_B), placement_class="minimum_multi_gpu")
    owned = _owned_result(request=req).owned_runtime
    assert owned.selected_physical_gpu_uuids == (U_A, U_B)


def test_lifecycle_code_cannot_append_an_extra_gpu():
    import dataclasses

    req = _request(gpu_uuids=(U_A,))
    owned = _owned_result(request=req).owned_runtime
    # frozen dataclass -- no rebind path
    with pytest.raises(dataclasses.FrozenInstanceError):
        owned.selected_physical_gpu_uuids = (U_A, U_B)
    # tuple -- no in-place append
    with pytest.raises(AttributeError):
        owned.selected_physical_gpu_uuids.append(U_B)
    assert owned.selected_physical_gpu_uuids == (U_A,)


def test_request_recipe_is_the_resolver_object_by_reference():
    res = _resolution()
    req = MaterialisationRequest.from_resolution(res)
    assert req.recipe is res.resolved  # not a copy


# ==========================================================================
# no-spawn / no-IO proof
# ==========================================================================
def test_module_imports_no_spawn_or_network_primitives():
    import llm_modelbench.runtime_lifecycle as mod

    # Catch both `import X` (binds the module) and `from X import name`
    # (binds `name`) by scanning the module namespace directly.
    banned = {
        "subprocess", "socket", "signal", "os", "requests", "httpx", "urllib",
        "Popen", "call", "check_call", "check_output", "getoutput",
        "kill", "killpg", "SIGTERM", "SIGKILL", "urlopen", "connect",
    }
    leaked = banned & set(vars(mod))
    # `run` is a legitimate common name; only flag it if it is the
    # subprocess one.
    if "run" in vars(mod) and getattr(vars(mod)["run"], "__module__", "") == "subprocess":
        leaked.add("run")
    assert not leaked, f"spawn/network primitives leaked into the module: {sorted(leaked)}"


def test_full_owned_lifecycle_never_calls_subprocess(monkeypatch):
    def _boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("subprocess invoked in lifecycle domain")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    monkeypatch.setattr(subprocess, "run", _boom)

    req = _request()
    result = materialise_owned_runtime(
        req, launch_proof=_proof(), launched_at="2026-09-03T12:00:00Z"
    )
    with RuntimeLifecycleController(
        result, cleanup_fn=lambda o: None, revalidate_fn=lambda o: _proof()
    ) as ctrl:
        assert ctrl.state is LifecycleState.READY
    assert ctrl.state is LifecycleState.CLEANED


def test_lifecycle_states_are_only_those_with_a_real_producer():
    # 3B.3C does NOT add SPAWN_PENDING / STARTING / FAILED / NOT_REQUIRED:
    # its spawn is synchronous and never surfaces a mid-flight
    # MaterialisationResult, and every spawn/reuse/failure outcome is carried
    # by llama_server_materialisation.MaterialisationStatus, not by a new
    # LifecycleState. The controller's own state stays READY / CLEANUP_* /
    # CLEANED / REUSED_EXTERNAL.
    for reserved in ("SPAWN_PENDING", "STARTING", "FAILED", "NOT_REQUIRED"):
        assert not hasattr(LifecycleState, reserved), reserved
    assert set(LifecycleState) == {
        LifecycleState.REUSED_EXTERNAL,
        LifecycleState.READY,
        LifecycleState.CLEANUP_PENDING,
        LifecycleState.CLEANED,
        LifecycleState.CLEANUP_FAILED,
    }
