"""Anvil Stage 3B.3D -- production composition seam.

This module is the ONE place that composes the accepted, independently-tested
Stage 3B.2 / 3B.3B / 3B.3C pieces into a single production control-flow step
the runner path (``cli.cmd_run``) can call:

    resolve_runtime()                    [3B.2 -- pure planner]
        -> MaterialisationRequest.from_resolution()   [3B.3B -- domain]
        -> materialise(spawn_managed=..., external_still_healthy=...)  [3B.3C]
        -> lifecycle_controller_for(...)              [3B.3C]
        -> (verified endpoint, controller, outcome)

It adds **no** planning authority of its own:

* it does not choose a backend (the caller passes the explicitly selected one);
* it does not decide GPU pool / RAM spill / context feasibility / tensor
  split / model fit / environment feasibility -- those are ``resolve_runtime``'s
  outputs, read verbatim;
* it never calls ``subprocess`` / ``Popen`` -- only the accepted
  ``spawn_managed_llama_server`` materialiser spawns, and only for
  ``llama_cpp``;
* it never falls back between backends;
* ``_recommended()`` / any speed / performance metric cannot influence it.

Every OS-touching dependency is an injected seam so the whole composition is
exercised with fakes (no CUDA, no llama.cpp binary, no Ollama, no model, no
real inference). Production wiring supplies the real adapters via
:func:`production_seams`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from .llama_server_materialisation import (
    ManagedMaterialisationOutcome,
    lifecycle_controller_for,
    materialise,
)
from .runtime_lifecycle import (
    MaterialisationRequest,
    MaterialisationRequestError,
    RuntimeLifecycleController,
    RuntimeOwnership,
)
from .runtime_resolution import (
    RequiredCapability,
    RuntimeResolution,
    RuntimeResolutionStatus,
    resolve_runtime,
)

__all__ = [
    "RuntimeMaterialisationError",
    "RuntimeMaterialisationOutcome",
    "MaterialisationSeams",
    "resolve_and_materialise_runtime",
    "production_seams",
]


class RuntimeMaterialisationError(RuntimeError):
    """Programming error in composition wiring (bad seam type, a backend the
    resolver never produces reaching the dispatcher). Every *expected*
    outcome -- unresolved runtime, materialisation failure -- is a structured
    :class:`RuntimeMaterialisationOutcome`, not an exception."""


@dataclass(frozen=True)
class RuntimeMaterialisationOutcome:
    """Typed result of the composed resolve->materialise step.

    ``ok`` is the single readiness gate the caller switches on. On ``ok`` the
    caller builds its backend client against :attr:`endpoint` (never a stale
    configured endpoint) and runs the benchmark inside ``with controller:``.
    On not-``ok`` the caller converts :attr:`refusal_reason` into its existing
    structured pre-row disposition (a ``SystemExit`` in ``cmd_run``) -- never a
    benchmark-quality failure.
    """

    ok: bool
    #: The backend the caller selected and asked to resolve (echoed for evidence).
    backend: str
    #: ``RuntimeResolutionStatus`` value. Always present.
    resolution_status: str
    #: The accepted resolution (``None`` only when resolution itself failed).
    resolution: Optional[RuntimeResolution] = None
    #: ``MaterialisationStatus`` value once materialisation was attempted.
    materialisation_status: Optional[str] = None
    #: The full 3B.3C outcome (argv, diagnostic tail, attribution, process) --
    #: the evidence source. ``None`` when resolution failed before materialisation.
    materialisation: Optional[ManagedMaterialisationOutcome] = None
    #: Verified endpoint to point the client at. Present iff ``ok``.
    endpoint: Optional[str] = None
    #: Lifecycle controller wrapping the materialisation result. Present iff
    #: ``ok``. External reuse -> no cleanup authority; managed -> owned.
    controller: Optional[RuntimeLifecycleController] = None
    #: The ``MaterialisationRequest.identity_key()`` this step authorised
    #: (recoverable in evidence). ``None`` when resolution failed.
    identity_key: Optional[str] = None
    #: Structured, enum-keyed refusal reason for a not-``ok`` outcome. The
    #: first ``:``-delimited token is the machine key; downstream never parses
    #: the prose that follows.
    refusal_reason: Optional[str] = None
    #: Anvil Stage 3B.3E -- the local GGUF artifact resolution snapshot for a
    #: managed ``llama_cpp`` run (``LocalArtifactResolution.to_dict()`` plus a
    #: ``blocked_managed_spawn`` bool). ``None`` for external / Ollama reuse
    #: (no local artifact needed) and for a ``--mock`` run. Echoed verbatim
    #: onto every outcome this step returns so the evidence / refusal reason
    #: can report whether artifact resolution blocked the managed spawn.
    artifact_resolution: Optional[Mapping[str, Any]] = None

    @property
    def ownership(self) -> Optional[str]:
        if self.materialisation is None or self.materialisation.result is None:
            return None
        return self.materialisation.result.ownership.value

    @property
    def owns_runtime(self) -> bool:
        return (
            self.controller is not None
            and self.controller.owns_runtime
        )


# ---------------------------------------------------------------------------
# injected seams
# ---------------------------------------------------------------------------
#: ``(request) -> ManagedMaterialisationOutcome`` -- the managed llama-server
#: spawn adapter. Only reached for ``llama_cpp`` with no reusable external.
SpawnManagedFn = Callable[[MaterialisationRequest], ManagedMaterialisationOutcome]
#: ``(request) -> bool`` -- optional refinement probe; may only *demote* a
#: reuse to a fresh materialisation decision, never promote a spawn to a reuse.
ExternalStillHealthyFn = Callable[[MaterialisationRequest], bool]


@dataclass(frozen=True)
class MaterialisationSeams:
    """The OS-touching adapters the composition needs. Production supplies the
    real ones (:func:`production_seams`); tests supply fakes."""

    spawn_managed: SpawnManagedFn
    external_still_healthy: Optional[ExternalStillHealthyFn] = None
    #: ``(outcome) -> RuntimeLifecycleController`` -- builds the controller with
    #: its revalidate/cleanup adapters. Defaults to the accepted 3B.3C factory
    #: wired to the real ``/proc`` + scoped-terminate adapters.
    controller_factory: Callable[[ManagedMaterialisationOutcome], RuntimeLifecycleController] = None  # type: ignore[assignment]


def production_seams(
    *,
    executable_path: Optional[str],
    model_path: Optional[str],
    model_primary_sha256: Optional[str],
    hardware_inventory: Sequence[Any],
    now_iso: Callable[[], str],
) -> MaterialisationSeams:
    """Build the real seams.

    ``model_path`` / ``model_primary_sha256`` are ``None`` in Stage 3B.3D
    production (there is no benchmark-model -> local-GGUF resolution yet -- a
    documented blocker routed to a later stage). The managed spawn adapter
    then fails closed at :attr:`MaterialisationStatus.RESOLVED_RECIPE_INCOMPLETE`
    -- the status Stage 3B.3C built for exactly this gap -- rather than
    launching an unproven artifact. External reuse is unaffected (ModelBench
    does not choose what an already-running server loaded).
    """
    from . import runtime_process_linux as rpl
    from .llama_server_materialisation import spawn_managed_llama_server
    from .llama_server_probe import (
        llama_server_context_conformance,
        llama_server_port_attribution,
        llama_server_readiness_probe,
    )

    def _spawn(request: MaterialisationRequest) -> ManagedMaterialisationOutcome:
        return spawn_managed_llama_server(
            request,
            executable_path=executable_path,
            model_path=model_path,
            model_primary_sha256=model_primary_sha256,
            hardware_inventory=hardware_inventory,
            observe_identity=lambda pid: rpl.observe_process_identity(pid),
            readiness_probe=llama_server_readiness_probe,
            port_attribution=llama_server_port_attribution,
            context_conformance=llama_server_context_conformance,
            now_iso=now_iso,
        )

    def _external_still_healthy(request: MaterialisationRequest) -> bool:
        # Bounded liveness recheck of the resolved external endpoint. A
        # refused/timed-out probe demotes reuse -> materialisation decision;
        # it can never promote a spawn to a reuse.
        from .llama_cpp import LlamaCppClient, LlamaCppError

        endpoint = request.endpoint
        try:
            if request.backend == "llama_cpp":
                LlamaCppClient(endpoint, timeout=2).health()
                return True
            # ollama: a bounded version() check
            from .ollama import OllamaClient

            return OllamaClient(endpoint, 0, 0.0, 2).version() is not None
        except (LlamaCppError, OSError, ValueError):
            return False
        except Exception:  # noqa: BLE001 -- a probe failure is a demotion, never a crash
            return False

    def _controller(outcome: ManagedMaterialisationOutcome) -> RuntimeLifecycleController:
        return lifecycle_controller_for(
            outcome,
            observe_identity=lambda pid: rpl.observe_process_identity(pid),
            terminate=rpl.terminate_process,
        )

    return MaterialisationSeams(
        spawn_managed=_spawn,
        external_still_healthy=_external_still_healthy,
        controller_factory=_controller,
    )


# ---------------------------------------------------------------------------
# the composed step
# ---------------------------------------------------------------------------
def resolve_and_materialise_runtime(
    *,
    artifact_resolution: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> RuntimeMaterialisationOutcome:
    """Resolve the runtime, then reuse-or-materialise it. Pure orchestration.

    Thin wrapper over :func:`_compose_resolve_and_materialise` that stamps the
    Stage 3B.3E ``artifact_resolution`` snapshot onto whatever outcome the
    composition returns (every early-return path included), so the caller sees
    it uniformly. ``artifact_resolution`` is also forwarded so the composition
    can key its own decisions off it if a later stage needs to; today it is
    purely evidence.
    """
    outcome = _compose_resolve_and_materialise(
        artifact_resolution=artifact_resolution, **kwargs
    )
    if artifact_resolution is None or outcome.artifact_resolution is not None:
        return outcome
    return replace(outcome, artifact_resolution=artifact_resolution)


def _compose_resolve_and_materialise(
    *,
    selected_backend: Optional[str],
    discovered_candidates: Iterable[Any],
    topology: Any,
    host_meminfo: Mapping[str, Any],
    seams: MaterialisationSeams,
    weight_bytes: Optional[int] = None,
    kv_cache_bytes: Optional[int] = None,
    requested_context: Optional[int] = None,
    allow_ram_spill: bool = False,
    explicit_profile_name: Optional[str] = None,
    backend_executables: Optional[Iterable[Any]] = None,
    required_capabilities: Sequence[RequiredCapability] = (),
    model_primary_sha256: Optional[str] = None,
    backend_version: Optional[str] = None,
    runtime_overhead_bytes: Optional[int] = None,
    owned_placement_required: bool = True,
    artifact_resolution: Optional[Mapping[str, Any]] = None,
    resolve_fn: Callable[..., RuntimeResolution] = resolve_runtime,
    materialise_fn: Callable[..., ManagedMaterialisationOutcome] = materialise,
) -> RuntimeMaterialisationOutcome:
    """Resolve the runtime, then reuse-or-materialise it. Pure orchestration.

    Returns a :class:`RuntimeMaterialisationOutcome`. Never raises for an
    expected failure (unresolved / ambiguous / unsupported backend /
    environment infeasible / fit unknown / materialisation failure) -- those
    are ``ok=False`` outcomes carrying an enum-keyed ``refusal_reason``.

    ``require_content_addressed_model_identity`` is intentionally left at the
    resolver's ``False`` default: the resolver *plans*, the materialiser
    *proves the artifact*. A missing artifact identity surfaces as
    ``MaterialisationStatus.RESOLVED_RECIPE_INCOMPLETE`` at the layer built for
    it -- not as a generic resolver identity error -- and an external-reuse
    materialisation (which legitimately has no artifact identity) is never
    blocked by a resolver-level requirement.
    """
    if not isinstance(seams, MaterialisationSeams):
        raise RuntimeMaterialisationError("seams must be a MaterialisationSeams")
    if not callable(seams.spawn_managed):
        raise RuntimeMaterialisationError("seams.spawn_managed must be callable")

    resolution = resolve_fn(
        selected_backend=selected_backend,
        discovered_candidates=discovered_candidates,
        topology=topology,
        host_meminfo=host_meminfo,
        weight_bytes=weight_bytes,
        kv_cache_bytes=kv_cache_bytes,
        requested_context=requested_context,
        allow_ram_spill=allow_ram_spill,
        explicit_profile_name=explicit_profile_name,
        backend_executables=backend_executables,
        required_capabilities=required_capabilities,
        model_primary_sha256=model_primary_sha256,
        backend_version=backend_version,
        runtime_overhead_bytes=runtime_overhead_bytes,
        owned_placement_required=owned_placement_required,
    )
    backend_echo = selected_backend or ""

    if resolution.status is not RuntimeResolutionStatus.RESOLVED:
        return RuntimeMaterialisationOutcome(
            ok=False,
            backend=backend_echo,
            resolution_status=resolution.status.value,
            resolution=None,
            refusal_reason=(
                f"runtime_not_resolved: {resolution.status.value}: {resolution.detail}"
            ),
        )

    try:
        request = MaterialisationRequest.from_resolution(resolution)
    except MaterialisationRequestError as exc:  # defensive -- RESOLVED implies a recipe
        return RuntimeMaterialisationOutcome(
            ok=False,
            backend=backend_echo,
            resolution_status=resolution.status.value,
            resolution=resolution,
            refusal_reason=f"runtime_not_resolved: malformed_resolution: {exc}",
        )

    ikey = request.identity_key()
    outcome = materialise_fn(
        request,
        spawn_managed=seams.spawn_managed,
        external_still_healthy=seams.external_still_healthy,
    )

    if not outcome.ok:
        # Ensure no owned child leaked from a failed managed attempt: the
        # accepted spawn adapter reaps its own abandoned children, and a
        # failed outcome never carries a process handle. Nothing to clean.
        return RuntimeMaterialisationOutcome(
            ok=False,
            backend=backend_echo,
            resolution_status=resolution.status.value,
            resolution=resolution,
            materialisation_status=outcome.status.value,
            materialisation=outcome,
            identity_key=ikey,
            refusal_reason=(
                f"runtime_not_materialised: {outcome.status.value}: {outcome.detail}"
            ),
        )

    factory = seams.controller_factory or _default_controller_factory
    controller = factory(outcome)
    endpoint = outcome.endpoint or (
        outcome.result.endpoint if outcome.result is not None else None
    )
    if not endpoint:
        # A ready materialisation with no endpoint is a wiring bug, not an
        # expected outcome -- but fail closed rather than build a client
        # against ``None``. Tear down anything we own first.
        if controller.owns_runtime:
            controller.cleanup()
        return RuntimeMaterialisationOutcome(
            ok=False,
            backend=backend_echo,
            resolution_status=resolution.status.value,
            resolution=resolution,
            materialisation_status=outcome.status.value,
            materialisation=outcome,
            identity_key=ikey,
            refusal_reason=(
                f"runtime_not_materialised: missing_endpoint: a ready "
                f"materialisation ({outcome.status.value}) carried no endpoint"
            ),
        )

    return RuntimeMaterialisationOutcome(
        ok=True,
        backend=backend_echo,
        resolution_status=resolution.status.value,
        resolution=resolution,
        materialisation_status=outcome.status.value,
        materialisation=outcome,
        endpoint=endpoint,
        controller=controller,
        identity_key=ikey,
    )


def _default_controller_factory(
    outcome: ManagedMaterialisationOutcome,
) -> RuntimeLifecycleController:
    from . import runtime_process_linux as rpl

    return lifecycle_controller_for(
        outcome,
        observe_identity=lambda pid: rpl.observe_process_identity(pid),
        terminate=rpl.terminate_process,
    )


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------
def materialisation_evidence(
    outcome: RuntimeMaterialisationOutcome,
    *,
    cleanup_result: Optional[Any] = None,
    benchmark_completed: Optional[bool] = None,
    failure_stage: Optional[str] = None,
    rss_bytes_launch_ready: Optional[int] = None,
    rss_bytes_post_execution: Optional[int] = None,
    runtime_telemetry_ref: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Build the auditable, JSON-serialisable materialisation evidence record.

    Persisted by the caller next to the run's benchmark evidence. Carries the
    resolved recipe verbatim (so a persisted-evidence mutation that sorts the
    GPU UUIDs or drops the tensor split is structurally impossible), the
    materialisation status / ownership / endpoint / sanitised argv / bounded
    diagnostic tail, the request identity key, and -- crucially -- the
    observed cleanup outcome (a clean benchmark with a failed cleanup must not
    disappear).

    ``pid`` / ``process_start_time_ticks`` / ``endpoint`` are recorded as
    launch *evidence*, explicitly not as durable semantic identity: a managed
    runtime restarted later is not the same process merely because these match.

    ``rss_bytes_launch_ready`` / ``rss_bytes_post_execution`` are the caller's
    own bounded ``/proc/<pid>/status`` reads (see
    ``runtime_process_linux.read_process_rss_bytes``), sampled by the caller
    at the two safe inline points -- once the owned runtime is ready and
    again immediately after ``runner.run()`` returns or raises, i.e. *before*
    the lifecycle ``with`` block exits and the owned process is torn down.
    This function never reads ``/proc`` itself: by the time evidence is
    built the owned process may already be gone (``__exit__`` runs cleanup
    before the caller's ``finally`` calls this), so a post-hoc read here
    would silently and permanently be ``None``.

    ``runtime_telemetry_ref`` is an optional cross-reference to the existing
    ``runtime_telemetry.json`` artifact (Stage 6B, unrelated to this stage's
    schema) -- pass ``{"artifact": "runtime_telemetry.json", "status": ...}``
    from the caller's own capture-result if one exists for this run. That
    artifact already carries real per-process GPU-memory attribution
    (``telemetry.attribute_runtime_gpus``) but is captured **once, before
    benchmark work** from inside ``runner.run()`` -- it never exists on a
    pre-benchmark refusal and never reflects post-execution state. This
    function does not re-collect it (a second collection would add
    subprocess/procfs work inside the cleanup-sensitive lifecycle window);
    it only records whether that artifact exists for this run and the
    explicit ``pre_benchmark_only`` capture-window caveat.

    ``benchmark_completed`` is recorded verbatim on an ``ok`` record.
    ``failure_stage`` (e.g. ``"client_construction"`` / ``"pre_run_gates"``)
    names the phase that raised *after* an owned runtime was materialised but
    *before* any benchmark row -- persisted so a client-construction failure is
    never later misread as a model-quality failure, and used to arm the
    client-construction cleanup-failure warning (schema stays version 1: both
    are additive fields).
    """
    record: dict = {
        "schema_version": 1,
        "backend_selected": outcome.backend,
        "resolution_status": outcome.resolution_status,
        "resolution": (
            outcome.resolution.resolved.to_dict()
            if outcome.resolution is not None and outcome.resolution.resolved is not None
            else None
        ),
        "identity_key": outcome.identity_key,
    }
    if outcome.artifact_resolution is not None:
        record["artifact_resolution"] = dict(outcome.artifact_resolution)
    if not outcome.ok:
        record["ok"] = False
        record["refusal_reason"] = outcome.refusal_reason
        record["materialisation_status"] = outcome.materialisation_status
        record["benchmark_completed"] = False
        if failure_stage is not None:
            record["failure_stage"] = failure_stage
        if outcome.materialisation is not None:
            record["materialisation"] = {
                "status": outcome.materialisation.status.value,
                "detail": outcome.materialisation.detail,
                "attempted_endpoint": outcome.materialisation.endpoint,
                "diagnostic_tail": outcome.materialisation.diagnostic_tail,
                "launched_argv": (
                    list(outcome.materialisation.launched_argv)
                    if outcome.materialisation.launched_argv is not None
                    else None
                ),
                "attribution": outcome.materialisation.attribution,
                "env_overlay": (
                    dict(outcome.materialisation.env_overlay)
                    if outcome.materialisation.env_overlay is not None else None
                ),
                "candidate_attempts": [dict(item) for item in outcome.materialisation.candidate_attempts],
            }
        # Anvil Stage 3B.5: an actual-placement block is present on EVERY
        # returned record, refusal included -- an additive block placed only
        # on the ok branch would be silently omitted on every refusal path,
        # which is indistinguishable from "we forgot to look". A refusal
        # never owned a runtime, so every fact here is legitimately
        # not-applicable, not merely not-observed.
        record["actual_placement"] = _actual_placement_evidence(
            owns_runtime=False,
            rss_bytes_launch_ready=None,
            rss_bytes_post_execution=None,
            runtime_telemetry_ref=runtime_telemetry_ref,
        )
        return record

    mat = outcome.materialisation
    assert mat is not None and mat.result is not None  # ok implies both
    result = mat.result
    owned = result.owned_runtime
    owns_runtime = result.ownership is RuntimeOwnership.MODELBENCH_OWNED
    record["ok"] = True
    record["materialisation"] = {
        "status": mat.status.value,
        "ownership": result.ownership.value,
        "endpoint_used": outcome.endpoint,
        "attribution": mat.attribution,
        "readiness_status": mat.status.value,
        "diagnostic_tail": mat.diagnostic_tail,
        "launched_argv": (
            list(mat.launched_argv) if mat.launched_argv is not None else None
        ),
        "cuda_visible_devices": _cuda_visible_devices_evidence(
            mat.env_overlay, owns_runtime=owns_runtime
        ),
        "launch_proof": (
            {
                "pid": owned.launch_proof.pid,
                "process_start_time_ticks": owned.launch_proof.process_start_time_ticks,
            }
            if owned is not None
            else None
        ),
    }
    record["actual_placement"] = _actual_placement_evidence(
        owns_runtime=owns_runtime,
        rss_bytes_launch_ready=rss_bytes_launch_ready,
        rss_bytes_post_execution=rss_bytes_post_execution,
        runtime_telemetry_ref=runtime_telemetry_ref,
    )
    record["cleanup"] = _cleanup_evidence(cleanup_result)
    record["benchmark_completed"] = bool(benchmark_completed)
    if failure_stage is not None:
        record["failure_stage"] = failure_stage
    owned_cleanup_failed = (
        result.ownership is RuntimeOwnership.MODELBENCH_OWNED
        and cleanup_result is not None
        and not getattr(cleanup_result, "ok", True)
    )
    warnings = []
    if benchmark_completed and owned_cleanup_failed:
        warnings.append("cleanup_failed_on_successful_benchmark")
    if (
        not benchmark_completed
        and failure_stage == "client_construction"
        and owned_cleanup_failed
    ):
        warnings.append("cleanup_failed_after_client_construction_failure")
    # Anvil Stage 3B.3E (owner-accepted DEFECT-3B.3D-03 debt): the
    # client-construction key above only fires for that one stage. Generalise
    # the operator-visible warning so ANY non-success owned cleanup on a
    # non-completed run at a pre-benchmark stage other than client construction
    # (``pre_run_gates`` today, any future one) is surfaced too. Distinct key
    # so the existing 3B.3D key/test is untouched; the ``failure_stage`` split
    # keeps the two mutually exclusive -- exactly one message per condition.
    if (
        not benchmark_completed
        and failure_stage is not None
        and failure_stage != "client_construction"
        and owned_cleanup_failed
    ):
        warnings.append("cleanup_failed_before_benchmark")
    # Anvil Stage 3B.5: a `ram_spill` placement is not full-GPU equivalent --
    # part of the model resides in host RAM, with the throughput/latency
    # consequences that implies. Persisting `placement_class: "ram_spill"`
    # alone (already present in `resolution.resolved`) is not the same as an
    # explicit, easy-to-grep operator marker; surface it as a warning too.
    resolved = outcome.resolution.resolved if outcome.resolution is not None else None
    if resolved is not None and resolved.placement_class == "ram_spill":
        warnings.append("placement_ram_spill_not_full_gpu_equivalent")
    if warnings:
        record["warnings"] = warnings
    return record


def _actual_placement_evidence(
    *,
    owns_runtime: bool,
    rss_bytes_launch_ready: Optional[int],
    rss_bytes_post_execution: Optional[int],
    runtime_telemetry_ref: Optional[Mapping[str, Any]],
) -> dict:
    """Anvil Stage 3B.5: what was actually *observed* about where the
    runtime executed, as opposed to what was *planned* (the resolved recipe,
    already recorded verbatim elsewhere in this record).

    Every fact here is either a real inline observation or an explicit
    ``not_observed`` / ``not_applicable`` marker with a reason -- never a
    silent omission (mutation battery MUT-13 exercises this). Nothing here
    is a scoring input; it is evidence only.

    * ``process_rss_bytes``: the caller's own bounded ``/proc`` reads,
      passed in verbatim (see ``materialisation_evidence``'s docstring for
      why this function cannot read them itself).
    * ``gpu_memory_observed``: always ``False`` here -- this function does
      not sample GPU memory (no new subprocess/procfs work inside the
      lifecycle window; see DEFECT-3B.5-01). The real per-process GPU
      attribution lives in the separate, pre-existing ``runtime_telemetry``
      artifact, cross-referenced below, never re-collected.
    * ``device_placement_observed``: always ``False`` -- llama-server
      exposes no endpoint that reports actual device placement (verified by
      reading ``llama_server_probe.py``; ``/props`` reports model/context,
      not devices). Ollama likewise has no such reuse-time reporting used
      here. Not a gap this stage can close without a new probe surface.
    """
    if not owns_runtime:
        return {
            "process_rss_bytes": {
                "launch_ready": None,
                "post_execution": None,
                "reason": "no_owned_process",
            },
            "gpu_memory_observed": False,
            "device_placement_observed": False,
            "runtime_telemetry": _runtime_telemetry_cross_reference(runtime_telemetry_ref),
        }
    return {
        "process_rss_bytes": {
            "launch_ready": rss_bytes_launch_ready,
            "post_execution": rss_bytes_post_execution,
            "reason": (
                "observed" if rss_bytes_launch_ready is not None or rss_bytes_post_execution is not None
                else "not_observed"
            ),
        },
        "gpu_memory_observed": False,
        "device_placement_observed": False,
        "runtime_telemetry": _runtime_telemetry_cross_reference(runtime_telemetry_ref),
    }


def _runtime_telemetry_cross_reference(
    runtime_telemetry_ref: Optional[Mapping[str, Any]],
) -> dict:
    """Cross-reference (never re-collect) the separate, pre-existing
    ``runtime_telemetry.json`` artifact -- see DEFECT-3B.5-01. That artifact
    already carries real per-process GPU-memory attribution, but is captured
    once, before benchmark work, from inside ``runner.run()``: it does not
    exist at all on a pre-benchmark refusal, and never reflects
    post-execution state. Both facts are recorded explicitly so a reader
    cannot mistake a present cross-reference for a post-execution or
    refusal-time observation."""
    if runtime_telemetry_ref is None:
        return {"present": False, "artifact": None, "captured": None}
    return {
        "present": True,
        "artifact": runtime_telemetry_ref.get("artifact"),
        "status": runtime_telemetry_ref.get("status"),
        "captured": "pre_benchmark_only",
    }


def _cleanup_evidence(cleanup_result: Optional[Any]) -> dict:
    if cleanup_result is None:
        return {"observed": False, "outcome": None}
    return {
        "observed": True,
        "outcome": getattr(getattr(cleanup_result, "outcome", None), "value", None),
        "detail": getattr(cleanup_result, "detail", None),
        "destructive_action_performed": bool(
            getattr(cleanup_result, "destructive_action_performed", False)
        ),
        "ok": bool(getattr(cleanup_result, "ok", False)),
    }


def _cuda_visible_devices_evidence(
    env_overlay: Optional[Mapping[str, str]], *, owns_runtime: bool
) -> dict:
    """Anvil Stage 3B.5: report the *real* ``CUDA_VISIBLE_DEVICES`` value the
    managed spawn actually applied (``ManagedMaterialisationOutcome.env_overlay``,
    plumbed from the real ``cmd.env_overlay`` used at launch) -- never
    re-derived from the resolved GPU UUID order. Re-deriving would silently
    fabricate a value on any placement where the spawn adapter legitimately
    sets no override (e.g. no GPU pinning in the recipe): the real overlay is
    the only truthful source.

    Three distinct states, each with a machine-readable reason so a missing
    value is never ambiguous with "not applicable":

    * external/Ollama reuse (``owns_runtime`` false) -- ModelBench never
      launched anything, so it never set an env overlay for the runtime.
    * a managed launch that legitimately pinned no GPU (``env_overlay == {}``,
      e.g. no GPU selected in the recipe).
    * a managed launch whose overlay carried no ``CUDA_VISIBLE_DEVICES`` key.
    """
    if not owns_runtime:
        return {"value": None, "reason": "reuse_only_no_managed_launch"}
    if env_overlay is None:
        return {"value": None, "reason": "no_env_overlay_recorded"}
    value = env_overlay.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return {"value": None, "reason": "managed_launch_set_no_gpu_pinning"}
    return {"value": value, "reason": "observed_at_launch"}
