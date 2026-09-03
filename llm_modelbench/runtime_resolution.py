"""Anvil Stage 3B.2 -- deterministic runtime resolution + fit/preflight.

The resolver is **policy and planning only**. It does not launch, spawn,
connect to, or mutate anything. Every environmental fact it needs
(discovered runtime candidates, GPU topology, host-memory snapshot, backend
executable availability, model weight / KV estimates, capability decisions)
is passed **in** as an already-collected value; the resolver is a pure
function over its inputs.

What it does (amended architecture §26 Stage 3B.2):

1. Takes the **explicitly selected backend** as its backend authority. It
   never uses ``runtime_profiles._recommended`` (gpu_count -> backend) or
   the ``RuntimeCandidate.recommended`` flag -- changing those cannot change
   an authoritative resolution result. If no backend was explicitly
   selected and no explicit deterministic selection authority is available,
   it returns a structured :attr:`RuntimeResolutionStatus.NO_BACKEND_SELECTED`
   rather than guessing.

2. Deterministically resolves one runtime *candidate* within that backend
   from the discovered candidates, using explicit identity + health + a
   documented stable tie-break. Candidate input order never changes the
   result. Genuine ambiguity (>=2 healthy candidates on distinct endpoints,
   no explicit profile selector) is returned as
   :attr:`RuntimeResolutionStatus.RUNTIME_AMBIGUOUS`, not silently resolved.

3. Reuses the existing frozen fit machinery -- :func:`topology_budget.
   evaluate_workload_fit` for the placement decision (primary GPU first;
   minimum additional GPUs; STOP by default; GPU + host RAM only with
   explicit ``--allow-ram-spill``) and :func:`ram_spill_preflight.
   resolve_spill_preflight` for the conservative host-RAM check. It does
   **not** implement a second VRAM-fit subsystem, and it does not invent a
   KV/weight formula -- those are inputs; absent, it returns
   :attr:`RuntimeResolutionStatus.FIT_UNKNOWN`.

4. Produces a small typed :class:`ResolvedRuntime` recipe -- backend +
   endpoint + resolved :class:`~llm_modelbench.runtime_identity.RuntimeExecutionSettings`
   (the existing type, not a new one) + selected **physical GPU UUIDs** +
   placement class + the stable
   :class:`~llm_modelbench.identity.RuntimeProfileIdentity` -- only for
   :attr:`RuntimeResolutionStatus.RESOLVED`.

Availability is not suitability: the resolver distinguishes
backend-executable-unavailable / no-usable-endpoint / runtime-unavailable /
runtime-incompatible / identity-insufficient / capability-evidence-
insufficient / capability-incompatible / environment-infeasible /
fit-unknown / resolved, via :class:`RuntimeResolutionStatus`. Downstream
callers switch on the enum, never parse prose.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from .capabilities import MeasuredCapabilityState
from .identity import RuntimeProfileIdentity, resolve_runtime_profile_identity
from .ram_spill_preflight import PLACEMENT_LABELS, resolve_spill_preflight
from .runtime_identity import RuntimeExecutionSettings
from .runtime_profiles import RuntimeCandidate
from .topology_budget import TopologyBudget, WorkloadFit, evaluate_workload_fit

__all__ = [
    "RuntimeResolutionStatus",
    "ResolvedRuntime",
    "RuntimeResolution",
    "RequiredCapability",
    "resolve_runtime",
]

# The two backends ModelBench actually resolves runtimes for.
_KNOWN_BACKENDS = ("ollama", "llama_cpp")


class RuntimeResolutionStatus(str, Enum):
    """Structured outcome of :func:`resolve_runtime`. Availability is not
    suitability -- each value is a materially different preflight state a
    caller must be able to distinguish without parsing prose."""

    RESOLVED = "resolved"

    # --- backend authority -------------------------------------------------
    NO_BACKEND_SELECTED = "no_backend_selected"
    # Explicit backend selected, but it is not a backend ModelBench resolves
    # runtimes for. Materially different from NO_BACKEND_SELECTED (the caller
    # made a choice; the choice is unsupported) -- a downstream caller must
    # distinguish these without parsing `detail` prose. (Stage 3B.3A
    # carryover: the 3B.2 resolver conflated the two.)
    UNSUPPORTED_BACKEND_SELECTED = "unsupported_backend_selected"

    # --- discovery / endpoint --------------------------------------------
    BACKEND_UNAVAILABLE = "backend_unavailable"
    NO_USABLE_ENDPOINT = "no_usable_endpoint"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    RUNTIME_INCOMPATIBLE = "runtime_incompatible"
    RUNTIME_AMBIGUOUS = "runtime_ambiguous"

    # --- identity / provenance -----------------------------------------
    IDENTITY_INSUFFICIENT = "identity_insufficient"

    # --- capability ---------------------------------------------------
    CAPABILITY_EVIDENCE_INSUFFICIENT = "capability_evidence_insufficient"
    CAPABILITY_INCOMPATIBLE = "capability_incompatible"

    # --- fit / environment ---------------------------------------------
    ENVIRONMENT_INFEASIBLE = "environment_infeasible"
    FIT_UNKNOWN = "fit_unknown"


# probe_profile health value -> resolver status for the "cannot resolve"
# branch. Explicit and total over the five known probe_profile health
# strings (runtime_profiles._HEALTH).
_HEALTH_TO_STATUS: Mapping[str, RuntimeResolutionStatus] = {
    "unreachable": RuntimeResolutionStatus.RUNTIME_UNAVAILABLE,
    "unhealthy": RuntimeResolutionStatus.RUNTIME_INCOMPATIBLE,
    "unknown": RuntimeResolutionStatus.RUNTIME_INCOMPATIBLE,
    "unsupported": RuntimeResolutionStatus.NO_USABLE_ENDPOINT,
}


@dataclass(frozen=True)
class RequiredCapability:
    """One capability the benchmark protocol requires, plus the caller's
    already-computed decision about whether the model demonstrates it.

    The resolver never computes a capability decision (that stays
    :mod:`~llm_modelbench.capability_projection`'s job -- capability and
    environment are separate by contract). It only consumes the verdict:

    * ``measured_state is None`` / evidence absent -> the resolver returns
      :attr:`RuntimeResolutionStatus.CAPABILITY_EVIDENCE_INSUFFICIENT`
      (fail closed -- an unproven required capability is not a green light).
    * ``measured_state`` is a non-supporting committing state
      (``MEASURED_UNSUPPORTED`` / ``BACKEND_UNSUPPORTED`` /
      ``NOT_APPLICABLE``) -> :attr:`RuntimeResolutionStatus.CAPABILITY_INCOMPATIBLE`.
    * ``measured_state`` is ``PROBE_INCONCLUSIVE`` -> evidence insufficient.
    * ``measured_state`` is ``MEASURED_SUPPORTED`` -> this capability does
      not block resolution.
    """

    name: str
    measured_state: Optional[MeasuredCapabilityState] = None


@dataclass(frozen=True)
class ResolvedRuntime:
    """The resolved runtime recipe -- the smallest typed structure that
    represents what execution needs. Present only on
    :attr:`RuntimeResolutionStatus.RESOLVED`.

    Persisted/authoritative GPU identity is the **physical GPU UUID**
    (``selected_physical_gpu_uuids``), never a transient ordinal. The
    execution settings are the existing
    :class:`~llm_modelbench.runtime_identity.RuntimeExecutionSettings`
    type -- 3B.2 derives it deterministically, it does not introduce a new
    execution-settings type or an optimization-profile database.
    """

    backend: str
    endpoint: str
    runtime_profile_name: Optional[str]
    execution_settings: RuntimeExecutionSettings
    runtime_profile_identity: RuntimeProfileIdentity
    selected_physical_gpu_uuids: Tuple[str, ...]
    placement_class: str  # one of ram_spill_preflight.PLACEMENT_LABELS
    requested_context: Optional[int]
    allow_ram_spill: bool
    estimated_ram_spill_bytes: Optional[int]
    #: Content-addressed identity of the model bytes this recipe was resolved
    #: for, carried verbatim from ``resolve_runtime(model_primary_sha256=...)``.
    #: ``None`` when the caller supplied none. Stage 3B.3C's *managed*
    #: ``llama-server`` spawn requires this to be present (it must prove it is
    #: loading exactly the resolved artifact and not an arbitrary path); an
    #: *external-reuse* materialisation does not (ModelBench does not choose
    #: what an already-running server has loaded).
    model_primary_sha256: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "endpoint": self.endpoint,
            "runtime_profile_name": self.runtime_profile_name,
            "execution_settings": self.execution_settings.normalized(self.selected_physical_gpu_uuids),
            "runtime_profile_identity_stable_key": self.runtime_profile_identity.stable_key(),
            "runtime_profile_backend": self.runtime_profile_identity.backend,
            "selected_physical_gpu_uuids": list(self.selected_physical_gpu_uuids),
            "placement_class": self.placement_class,
            "requested_context": self.requested_context,
            "allow_ram_spill": self.allow_ram_spill,
            "estimated_ram_spill_bytes": self.estimated_ram_spill_bytes,
            "model_primary_sha256": self.model_primary_sha256,
        }


@dataclass(frozen=True)
class RuntimeResolution:
    """Immutable resolver result. ``resolved`` is populated only when
    ``status is RuntimeResolutionStatus.RESOLVED``."""

    status: RuntimeResolutionStatus
    reason: str
    detail: str
    resolved: Optional[ResolvedRuntime] = None
    selected_candidate: Optional[RuntimeCandidate] = None
    workload_fit: Optional[WorkloadFit] = None
    considered_candidate_endpoints: Tuple[str, ...] = ()

    @property
    def is_resolved(self) -> bool:
        return self.status is RuntimeResolutionStatus.RESOLVED

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "detail": self.detail,
            "resolved": self.resolved.to_dict() if self.resolved is not None else None,
            "selected_candidate": (
                self.selected_candidate.to_dict() if self.selected_candidate is not None else None
            ),
            "workload_fit": (
                {
                    "classification": self.workload_fit.classification,
                    "selected_gpu_uuids": list(self.workload_fit.selected_gpu_uuids),
                    "required_bytes": self.workload_fit.required_bytes,
                    "unknown_components": list(self.workload_fit.unknown_components),
                }
                if self.workload_fit is not None
                else None
            ),
            "considered_candidate_endpoints": list(self.considered_candidate_endpoints),
        }


def _unresolved(
    status: RuntimeResolutionStatus,
    detail: str,
    *,
    selected_candidate: Optional[RuntimeCandidate] = None,
    workload_fit: Optional[WorkloadFit] = None,
    considered_endpoints: Tuple[str, ...] = (),
) -> RuntimeResolution:
    return RuntimeResolution(
        status=status,
        reason=status.value,
        detail=detail,
        selected_candidate=selected_candidate,
        workload_fit=workload_fit,
        considered_candidate_endpoints=considered_endpoints,
    )


def _backend_executable_state(
    backend_executables: Optional[Iterable[Any]], backend: str
) -> Optional[str]:
    """Return the ``BackendExecutable.state`` for ``backend`` from a
    ``discover_backend_executables()`` result, or ``None`` if not present."""
    if backend_executables is None:
        return None
    for entry in backend_executables:
        if getattr(entry, "backend", None) == backend:
            return getattr(entry, "state", None)
    return None


def _select_candidate(
    candidates: Sequence[RuntimeCandidate],
    backend: str,
    *,
    explicit_profile_name: Optional[str],
    backend_executables: Optional[Iterable[Any]],
) -> Tuple[Optional[RuntimeCandidate], Optional[RuntimeResolution], Tuple[str, ...]]:
    """Deterministic within-backend candidate resolution. Returns
    ``(candidate, None, considered_endpoints)`` on success, or
    ``(None, RuntimeResolution, considered_endpoints)`` when it cannot
    resolve. ``_recommended`` / the ``recommended`` flag is never consulted.
    """
    for_backend = [c for c in candidates if c.profile.backend == backend]
    considered_endpoints = tuple(sorted({c.profile.endpoint for c in for_backend}))

    if explicit_profile_name is not None:
        named = [c for c in for_backend if c.profile.name == explicit_profile_name]
        if not named:
            return (
                None,
                _unresolved(
                    RuntimeResolutionStatus.NO_USABLE_ENDPOINT,
                    f"explicit runtime profile {explicit_profile_name!r} was not discovered "
                    f"for backend {backend!r}",
                    considered_endpoints=considered_endpoints,
                ),
                considered_endpoints,
            )
        # An explicit profile name is a total order key (names are unique in
        # the profile store); if discovery somehow yielded more than one,
        # take the lexicographically-first for determinism.
        candidate = sorted(named, key=lambda c: c.profile.name)[0]
        if candidate.health != "healthy":
            status = _HEALTH_TO_STATUS.get(
                candidate.health, RuntimeResolutionStatus.RUNTIME_INCOMPATIBLE
            )
            return (
                None,
                _unresolved(
                    status,
                    f"explicit runtime profile {explicit_profile_name!r} is "
                    f"{candidate.health}: {candidate.detail}",
                    selected_candidate=candidate,
                    considered_endpoints=considered_endpoints,
                ),
                considered_endpoints,
            )
        return candidate, None, considered_endpoints

    healthy = [c for c in for_backend if c.health == "healthy"]
    if len(healthy) == 1:
        return healthy[0], None, considered_endpoints

    if len(healthy) >= 2:
        endpoints = {c.profile.endpoint for c in healthy}
        if len(endpoints) >= 2:
            # Genuinely different runtimes -- never guess (§2/§3).
            return (
                None,
                _unresolved(
                    RuntimeResolutionStatus.RUNTIME_AMBIGUOUS,
                    "multiple healthy runtimes on distinct endpoints for backend "
                    f"{backend!r} ({', '.join(sorted(endpoints))}); an explicit runtime "
                    "profile is required to disambiguate",
                    considered_endpoints=considered_endpoints,
                ),
                considered_endpoints,
            )
        # Same endpoint, >1 candidate. Same (backend, endpoint) is NOT proof
        # of runtime equivalence: RuntimeProfile.physical_gpu_uuids is an
        # identity-bearing field that can genuinely diverge same-endpoint
        # (DEFECT-3B.2-AUDIT-02). Compare it as a frozenset -- __post_init__
        # dedups but does NOT sort physical_gpu_uuids (unlike RuntimeIdentity
        # / runtime_fit which sort), so ("A","B") and ("B","A") are the same
        # physical placement and must not read as divergent.
        gpu_identity_sets = {
            frozenset(c.profile.physical_gpu_uuids) for c in healthy
        }
        if len(gpu_identity_sets) >= 2:
            return (
                None,
                _unresolved(
                    RuntimeResolutionStatus.RUNTIME_AMBIGUOUS,
                    "multiple healthy runtimes on the same endpoint for backend "
                    f"{backend!r} with materially different physical GPU placement "
                    f"({sorted(sorted(s) for s in gpu_identity_sets)}); an explicit "
                    "runtime profile is required to disambiguate",
                    considered_endpoints=considered_endpoints,
                ),
                considered_endpoints,
            )
        # Provably equivalent on (backend, endpoint, physical GPU identity):
        # the remaining candidates differ only in incidental metadata
        # (name / description / provenance / source tuple). Stable tie-break
        # on the profile name; documented and deterministic.
        return sorted(healthy, key=lambda c: c.profile.name)[0], None, considered_endpoints

    # Zero healthy candidates for this backend.
    exec_state = _backend_executable_state(backend_executables, backend)
    if exec_state in {"not_installed", "not_configured"} and not for_backend:
        return (
            None,
            _unresolved(
                RuntimeResolutionStatus.BACKEND_UNAVAILABLE,
                f"backend {backend!r} launch executable is {exec_state} and no runtime "
                "endpoint was discovered",
                considered_endpoints=considered_endpoints,
            ),
            considered_endpoints,
        )
    if not for_backend:
        return (
            None,
            _unresolved(
                RuntimeResolutionStatus.NO_USABLE_ENDPOINT,
                f"no runtime endpoint was discovered for backend {backend!r}",
                considered_endpoints=considered_endpoints,
            ),
            considered_endpoints,
        )
    # There ARE candidates for the backend but none is healthy: pick the
    # least-bad status deterministically from the discovered health values.
    # Priority: unreachable (temporarily unavailable) > unhealthy/unknown
    # (incompatible) > unsupported (no usable endpoint).
    health_priority = ("unreachable", "unhealthy", "unknown", "unsupported")
    worst = min(
        for_backend,
        key=lambda c: (
            health_priority.index(c.health) if c.health in health_priority else len(health_priority),
            c.profile.endpoint,
            c.profile.name,
        ),
    )
    status = _HEALTH_TO_STATUS.get(worst.health, RuntimeResolutionStatus.RUNTIME_INCOMPATIBLE)
    return (
        None,
        _unresolved(
            status,
            f"backend {backend!r} has discovered runtimes but none is healthy "
            f"(best: {worst.profile.endpoint} -> {worst.health}: {worst.detail})",
            selected_candidate=worst,
            considered_endpoints=considered_endpoints,
        ),
        considered_endpoints,
    )


def _capability_block(required_capabilities: Sequence[RequiredCapability]) -> Optional[RuntimeResolution]:
    _supporting = MeasuredCapabilityState.MEASURED_SUPPORTED
    _incompatible = {
        MeasuredCapabilityState.MEASURED_UNSUPPORTED,
        MeasuredCapabilityState.BACKEND_UNSUPPORTED,
        MeasuredCapabilityState.NOT_APPLICABLE,
    }
    for required in required_capabilities:
        state = required.measured_state
        if state is None or state is MeasuredCapabilityState.PROBE_INCONCLUSIVE:
            return _unresolved(
                RuntimeResolutionStatus.CAPABILITY_EVIDENCE_INSUFFICIENT,
                f"required capability {required.name!r} has no committing measured evidence "
                f"({'absent' if state is None else state.value})",
            )
        if state in _incompatible:
            return _unresolved(
                RuntimeResolutionStatus.CAPABILITY_INCOMPATIBLE,
                f"required capability {required.name!r} is {state.value}",
            )
        if state is not _supporting:
            return _unresolved(
                RuntimeResolutionStatus.CAPABILITY_EVIDENCE_INSUFFICIENT,
                f"required capability {required.name!r} state {state.value!r} is not a "
                "supporting measurement",
            )
    return None


def _safe_selected_pool_capacity_bytes(
    topology: TopologyBudget, selected_uuids: Iterable[str]
) -> Optional[int]:
    selected = set(selected_uuids)
    pool = [d for d in topology.devices if d.uuid in selected]
    caps = [d.safe_capacity_bytes for d in pool]
    if not pool or any(c is None for c in caps):
        return None
    return sum(int(c) for c in caps)


def resolve_runtime(
    *,
    selected_backend: Optional[str],
    discovered_candidates: Iterable[RuntimeCandidate],
    topology: TopologyBudget,
    host_meminfo: Mapping[str, Any],
    weight_bytes: Optional[int],
    kv_cache_bytes: Optional[int],
    requested_context: Optional[int] = None,
    allow_ram_spill: bool = False,
    explicit_profile_name: Optional[str] = None,
    backend_executables: Optional[Iterable[Any]] = None,
    required_capabilities: Sequence[RequiredCapability] = (),
    require_content_addressed_model_identity: bool = False,
    model_primary_sha256: Optional[str] = None,
    backend_version: Optional[str] = None,
    runtime_overhead_bytes: Optional[int] = None,
) -> RuntimeResolution:
    """Deterministically resolve the minimum runtime configuration required
    to execute a benchmark protocol fairly, given already-collected inputs.

    Identical authoritative inputs yield an identical result; candidate
    input order never changes the outcome. The resolver launches nothing.

    ``selected_backend`` is the **backend authority** (§2). ``None`` (or an
    unknown backend) yields :attr:`RuntimeResolutionStatus.NO_BACKEND_SELECTED` --
    the resolver never guesses a backend and never consults
    ``runtime_profiles._recommended`` / ``RuntimeCandidate.recommended``.

    ``weight_bytes`` / ``kv_cache_bytes`` are the workload estimate
    **inputs** (§5 -- the resolver invents no formula). Absent, the fit is
    :attr:`RuntimeResolutionStatus.FIT_UNKNOWN`.

    ``allow_ram_spill`` is the explicit operator permission (§6). It never
    self-enables; the resolver passes it straight through to
    :func:`topology_budget.evaluate_workload_fit` and
    :func:`ram_spill_preflight.resolve_spill_preflight`.
    """
    candidates = list(discovered_candidates)

    # --- 1. backend authority -------------------------------------------
    if not selected_backend:
        return _unresolved(
            RuntimeResolutionStatus.NO_BACKEND_SELECTED,
            "no explicit backend was selected; the Stage 3B.2 resolver does not choose a "
            "backend (it never consults _recommended or the recommended flag)",
        )
    if selected_backend not in _KNOWN_BACKENDS:
        return _unresolved(
            RuntimeResolutionStatus.UNSUPPORTED_BACKEND_SELECTED,
            f"selected backend {selected_backend!r} is not a backend ModelBench resolves "
            f"runtimes for (known: {', '.join(_KNOWN_BACKENDS)})",
        )

    # --- 2. deterministic within-backend candidate resolution ----------
    candidate, failure, considered_endpoints = _select_candidate(
        candidates,
        selected_backend,
        explicit_profile_name=explicit_profile_name,
        backend_executables=backend_executables,
    )
    if failure is not None or candidate is None:
        return failure if failure is not None else _unresolved(
            RuntimeResolutionStatus.NO_USABLE_ENDPOINT,
            f"no runtime candidate could be resolved for backend {selected_backend!r}",
        )

    # --- 3. identity / provenance sufficiency --------------------------
    if require_content_addressed_model_identity and not (
        isinstance(model_primary_sha256, str) and model_primary_sha256.strip()
    ):
        return _unresolved(
            RuntimeResolutionStatus.IDENTITY_INSUFFICIENT,
            "content-addressed model identity was required but no primary_sha256 was supplied",
            selected_candidate=candidate,
            considered_endpoints=considered_endpoints,
        )

    # --- 4. required-capability gate (evidence consumed, never computed)
    capability_failure = _capability_block(required_capabilities)
    if capability_failure is not None:
        return RuntimeResolution(
            status=capability_failure.status,
            reason=capability_failure.reason,
            detail=capability_failure.detail,
            selected_candidate=candidate,
            considered_candidate_endpoints=considered_endpoints,
        )

    # --- 5. fit / preflight (reuse the frozen machinery) ---------------
    if weight_bytes is None:
        return _unresolved(
            RuntimeResolutionStatus.FIT_UNKNOWN,
            "model weight estimate was not supplied; cannot resolve GPU placement",
            selected_candidate=candidate,
            considered_endpoints=considered_endpoints,
        )

    fit = evaluate_workload_fit(
        topology,
        weight_bytes=weight_bytes,
        kv_cache_bytes=kv_cache_bytes,
        runtime_overhead_bytes=runtime_overhead_bytes,
        allow_cpu_spill=allow_ram_spill,
    )

    if fit.classification == "confirmed_no_fit":
        return _unresolved(
            RuntimeResolutionStatus.ENVIRONMENT_INFEASIBLE,
            f"the workload cannot fit the available GPU pool: {fit.detail}",
            selected_candidate=candidate,
            workload_fit=fit,
            considered_endpoints=considered_endpoints,
        )
    if fit.classification == "unknown":
        return _unresolved(
            RuntimeResolutionStatus.FIT_UNKNOWN,
            f"GPU fit could not be determined: {fit.detail}",
            selected_candidate=candidate,
            workload_fit=fit,
            considered_endpoints=considered_endpoints,
        )

    known_workload = int(weight_bytes) + int(kv_cache_bytes or 0)
    spill = resolve_spill_preflight(
        fit,
        safe_selected_gpu_capacity_bytes=_safe_selected_pool_capacity_bytes(
            topology, fit.selected_gpu_uuids
        ),
        allow_ram_spill=allow_ram_spill,
        host_meminfo=host_meminfo,
        known_workload_bytes=known_workload,
    )

    if spill.feasible is None:
        return _unresolved(
            RuntimeResolutionStatus.FIT_UNKNOWN,
            f"host-RAM spill preflight is inconclusive: {spill.reason}",
            selected_candidate=candidate,
            workload_fit=fit,
            considered_endpoints=considered_endpoints,
        )
    if spill.feasible is False:
        return _unresolved(
            RuntimeResolutionStatus.ENVIRONMENT_INFEASIBLE,
            f"the environment cannot execute this workload: {spill.reason}",
            selected_candidate=candidate,
            workload_fit=fit,
            considered_endpoints=considered_endpoints,
        )

    placement_class = spill.resolution
    if placement_class not in PLACEMENT_LABELS:  # pragma: no cover -- defensive
        return _unresolved(
            RuntimeResolutionStatus.FIT_UNKNOWN,
            f"spill preflight returned an unexpected resolution {placement_class!r}",
            selected_candidate=candidate,
            workload_fit=fit,
            considered_endpoints=considered_endpoints,
        )

    selected_uuids = tuple(spill.selected_gpu_uuids or fit.selected_gpu_uuids)

    # --- 6. resolved runtime recipe ----------------------------------
    strategy = "single_device" if len(selected_uuids) <= 1 else "layer_split"
    # ``allow_cpu_spill`` carries the resolved RAM-spill *permission*, not the
    # actual resulting placement -- matching the existing convention in
    # ``runtime_identity.collect_runtime_identity`` (Stage 3.2C-2b: "the
    # resolved RAM-spill permission (not the actual resulting placement) is
    # identity-bearing") and ``benchmark_binding._execution_settings_from_config``.
    # An absent/false permission stays ``None`` (identical to the historical
    # default) so an ordinary run keeps a stable identity hash. The actual
    # placement outcome is carried separately in ``placement_class`` /
    # ``estimated_ram_spill_bytes`` on ``ResolvedRuntime``.
    execution_settings = RuntimeExecutionSettings(
        strategy=strategy if len(selected_uuids) >= 1 else None,
        context_size=requested_context,
        allow_cpu_spill=True if allow_ram_spill else None,
    )
    # Validate against the selected UUID set (raises on an inconsistent recipe).
    execution_settings.normalized(selected_uuids)

    profile_identity = resolve_runtime_profile_identity(
        backend=selected_backend,
        backend_version=backend_version,
        execution_settings=execution_settings,
    )

    resolved = ResolvedRuntime(
        backend=selected_backend,
        endpoint=candidate.profile.endpoint,
        runtime_profile_name=candidate.profile.name,
        execution_settings=execution_settings,
        runtime_profile_identity=profile_identity,
        selected_physical_gpu_uuids=selected_uuids,
        placement_class=placement_class,
        requested_context=requested_context,
        allow_ram_spill=allow_ram_spill,
        estimated_ram_spill_bytes=spill.estimated_ram_spill_bytes,
        model_primary_sha256=(
            model_primary_sha256.strip()
            if isinstance(model_primary_sha256, str) and model_primary_sha256.strip()
            else None
        ),
    )
    return RuntimeResolution(
        status=RuntimeResolutionStatus.RESOLVED,
        reason=RuntimeResolutionStatus.RESOLVED.value,
        detail=f"resolved {selected_backend} runtime at {candidate.profile.endpoint} "
        f"({placement_class}, {len(selected_uuids)} GPU(s))",
        resolved=resolved,
        selected_candidate=candidate,
        workload_fit=fit,
        considered_candidate_endpoints=considered_endpoints,
    )
