"""Anvil Stage 3B.3B -- runtime lifecycle domain / ownership contract.

Stage 3B.2 (:mod:`~llm_modelbench.runtime_resolution`) answers *"what
runtime/backend/config is authoritative and suitable?"*. This module answers
the narrow lifecycle question that comes next: *"given an accepted resolver
result, may ModelBench materialise exactly that recipe, use it, and clean up
only what it provably owns?"*

**This slice is pure.** It launches no process, opens no socket, claims no
port, contacts no backend, and integrates with no runner. It defines and
tests the contracts a later slice (3B.3C) will use to materialise a real
``llama-server`` child process. Cleanup is expressed as an *injected*
callback; there is no ``subprocess`` / ``signal`` / ``socket`` import here.

Core separation (owner-accepted "Alternative B", 2026-09-03):

* ``runtime_profiles.RuntimeProfile.mode`` stays ``"external"``-only. No
  ``"managed"`` / ``"owned"`` profile mode is introduced.
* Runtime *identity / discovery* and runtime *lifecycle ownership* are
  separate domains. This module is the second.
* An :class:`OwnedRuntime` is the record of a process ModelBench itself
  created and may therefore terminate. An externally discovered runtime is
  *never* wrapped as an :class:`OwnedRuntime`; it is returned as usable but
  carries no cleanup authority, and the API makes accidental termination of
  it structurally hard.

Consumes the resolver result; never re-resolves backend / GPU placement /
RAM spill / context / fit. The :class:`MaterialisationRequest` references
the already-resolved :class:`~llm_modelbench.runtime_resolution.ResolvedRuntime`
rather than reconstructing it, so "lifecycle code cannot append a GPU" and
"no RAM-spill permission can be minted in materialisation" are structural,
not merely test-enforced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Tuple

from .identity import RuntimeProfileIdentity
from .runtime_resolution import (
    ResolvedRuntime,
    RuntimeResolution,
    RuntimeResolutionStatus,
)

__all__ = [
    "MaterialisationRequestError",
    "LaunchProcessProof",
    "MaterialisationRequest",
    "RuntimeOwnership",
    "LifecycleState",
    "CleanupOutcome",
    "CleanupResult",
    "OwnedRuntime",
    "CleanupAuthority",
    "MaterialisationResult",
    "RuntimeLifecycleController",
    "reuse_external_runtime",
    "materialise_owned_runtime",
]


# Module-private construction token. Identity-bearing lifecycle records
# (:class:`MaterialisationRequest`, :class:`OwnedRuntime`) must not be
# hand-assembled from arbitrary fields by a caller that skipped the resolver
# or the materialiser. Direct construction without this token raises. This is
# the "authoritative construction path is explicit and tested" guarantee the
# stage prompt asks for where Python cannot enforce it perfectly.
class _ConstructionToken:
    __slots__ = ()


_TOKEN = _ConstructionToken()


class MaterialisationRequestError(ValueError):
    """A :class:`MaterialisationRequest` was constructed from something other
    than an accepted (``RESOLVED``) :class:`RuntimeResolution`, or by
    bypassing :meth:`MaterialisationRequest.from_resolution`."""


# ---------------------------------------------------------------------------
# process-identity proof requirement (anti-PID-reuse)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LaunchProcessProof:
    """The process-identity evidence recorded at launch time that 3B.3C must
    be able to *revalidate* before any destructive cleanup is permitted.

    Backend-neutral and I/O-free: this is the *proof requirement*, not a
    procfs collector. 3B.3C supplies the platform adapter that fills these
    fields at spawn and re-reads them before teardown (on Linux: the
    ``/proc/<pid>`` process start-time tick, ``exe`` target, and ``cmdline``;
    ``telemetry.RuntimeProcessIdentity`` already models the same shape and a
    spawn path may legitimately import it -- this domain must not).

    Field -> safety decision it serves:

    * ``pid`` -- names the OS process to signal. Alone it is *insufficient*:
      the kernel reuses PIDs.
    * ``process_start_time_ticks`` -- the anti-reuse discriminator. A reused
      PID has a different start time. **Cleanup is refused unless this is
      present on both the launch record and the revalidation** -- an absent
      value is treated as "identity could not be proven", never as a match.
    * ``executable_path`` -- guards against a same-PID same-start-time
      process that is nonetheless not the backend we launched (defensive;
      cheap).
    * ``command_argv`` -- distinguishes our ``llama-server`` invocation from
      an unrelated one; also the audit record of exactly what was launched.
    * ``parent_pid`` -- lets 3B.3C assert the process is still our child
      where that is meaningful. Optional; not required for the match.
    """

    pid: int
    process_start_time_ticks: Optional[int]
    executable_path: Optional[str]
    command_argv: Tuple[str, ...]
    parent_pid: Optional[int] = None

    def __post_init__(self) -> None:
        if isinstance(self.pid, bool) or not isinstance(self.pid, int) or self.pid <= 0:
            raise ValueError("pid must be a positive integer")
        if self.process_start_time_ticks is not None and (
            isinstance(self.process_start_time_ticks, bool)
            or not isinstance(self.process_start_time_ticks, int)
            or self.process_start_time_ticks <= 0
        ):
            raise ValueError("process_start_time_ticks must be a positive integer or None")
        if self.parent_pid is not None and (
            isinstance(self.parent_pid, bool)
            or not isinstance(self.parent_pid, int)
            or self.parent_pid < 0
        ):
            raise ValueError("parent_pid must be a non-negative integer or None")
        object.__setattr__(self, "command_argv", tuple(self.command_argv))
        if any(not isinstance(arg, str) for arg in self.command_argv):
            raise ValueError("command_argv must be a tuple of strings")

    def revalidation_matches(self, observed: "LaunchProcessProof") -> bool:
        """True only if ``observed`` is provably the same process this record
        was created for.

        Fail-closed rule: an absent ``process_start_time_ticks`` on *either*
        side means the identity could not be revalidated -> not a match. A
        bare ``pid ==`` comparison is never sufficient.
        """
        if self.process_start_time_ticks is None or observed.process_start_time_ticks is None:
            return False
        if self.pid != observed.pid:
            return False
        if self.process_start_time_ticks != observed.process_start_time_ticks:
            return False
        if self.executable_path != observed.executable_path:
            return False
        if self.command_argv != observed.command_argv:
            return False
        return True


# ---------------------------------------------------------------------------
# materialisation request -- derived only from an accepted resolution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MaterialisationRequest:
    """A request to materialise (reuse-or-launch) the runtime described by an
    accepted Stage 3B.2 resolution.

    Holds the :class:`RuntimeResolution` **by reference** -- it does not copy
    ``backend`` / ``selected_physical_gpu_uuids`` / ``allow_ram_spill`` /
    fit into its own fields. Downstream lifecycle code therefore *cannot*
    hand-edit the recipe (add a GPU, flip a spill permission) without going
    back through the resolver.

    Construct via :meth:`from_resolution` only. Direct construction raises
    unless the module-private token is supplied.
    """

    resolution: RuntimeResolution
    #: No default -- an omitted token is a construction error. Only _TOKEN
    #: (held by from_resolution()) is accepted.
    _token: object = field(repr=False)

    def __post_init__(self) -> None:
        if self._token is not _TOKEN:
            raise MaterialisationRequestError(
                "MaterialisationRequest must be built via "
                "MaterialisationRequest.from_resolution(); direct construction "
                "would let a caller bypass the Stage 3B.2 resolver"
            )
        if not isinstance(self.resolution, RuntimeResolution):
            raise MaterialisationRequestError("resolution must be a RuntimeResolution")
        if self.resolution.status is not RuntimeResolutionStatus.RESOLVED:
            raise MaterialisationRequestError(
                f"resolution is not accepted: status={self.resolution.status.value}"
            )
        if self.resolution.resolved is None:
            raise MaterialisationRequestError(
                "RESOLVED resolution has no resolved recipe (malformed)"
            )

    @classmethod
    def from_resolution(cls, resolution: RuntimeResolution) -> "MaterialisationRequest":
        """The authoritative construction path. Returns a request only for a
        ``RESOLVED`` resolution carrying a recipe; every other status
        (unresolved / ambiguous / environment-infeasible / unsupported
        backend / ...) raises :class:`MaterialisationRequestError` naming the
        blocking status -- never a silently-approved launch."""
        if not isinstance(resolution, RuntimeResolution):
            raise MaterialisationRequestError("resolution must be a RuntimeResolution")
        if resolution.status is not RuntimeResolutionStatus.RESOLVED:
            raise MaterialisationRequestError(
                f"cannot materialise a non-accepted resolution: "
                f"status={resolution.status.value} ({resolution.detail})"
            )
        if resolution.resolved is None:
            raise MaterialisationRequestError(
                "RESOLVED resolution carries no ResolvedRuntime recipe"
            )
        return cls(resolution=resolution, _token=_TOKEN)

    @property
    def recipe(self) -> ResolvedRuntime:
        """The already-resolved recipe. Read-only view; this module never
        recomputes any of its fields."""
        assert self.resolution.resolved is not None  # guaranteed by __post_init__
        return self.resolution.resolved

    @property
    def backend(self) -> str:
        return self.recipe.backend

    @property
    def endpoint(self) -> str:
        return self.recipe.endpoint

    @property
    def runtime_profile_identity(self) -> RuntimeProfileIdentity:
        return self.recipe.runtime_profile_identity

    @property
    def selected_physical_gpu_uuids(self) -> Tuple[str, ...]:
        return self.recipe.selected_physical_gpu_uuids

    def identity_key(self) -> str:
        """Structural identity of *what this request asks for*. Derived only
        from resolved-recipe facts -- candidate discovery order, the
        resolver's ``considered_candidate_endpoints`` list and the
        (never-consulted) ``_recommended`` flag do not enter it, so the same
        resolved recipe always yields the same key."""
        r = self.recipe
        return "|".join(
            (
                "materialisation_request_v1",
                r.backend,
                r.endpoint,
                r.runtime_profile_identity.stable_key(),
                ",".join(sorted(r.selected_physical_gpu_uuids)),
                r.placement_class,
                "" if r.requested_context is None else str(r.requested_context),
                "spill" if r.allow_ram_spill else "no_spill",
            )
        )


# ---------------------------------------------------------------------------
# ownership / lifecycle enums
# ---------------------------------------------------------------------------
class RuntimeOwnership(str, Enum):
    """Who owns the runtime process the caller is about to use."""

    #: Discovered, healthy, already-running runtime. ModelBench did not start
    #: it and must never terminate it.
    EXTERNAL_REUSED = "external_reused"
    #: A process ModelBench started from the resolved recipe. ModelBench owns
    #: its lifecycle and is the only party that may clean it up.
    MODELBENCH_OWNED = "modelbench_owned"


class LifecycleState(str, Enum):
    """Lifecycle state of a materialisation. Only members with a real
    producer *in Stage 3B.3B* are defined here. States that only make sense
    once 3B.3C actually spawns (``SPAWN_PENDING`` / ``STARTING`` / a
    materialisation-``FAILED`` state / a ``NOT_REQUIRED`` state for the case
    where the resolver's own endpoint was already usable) are deliberately
    reserved for that slice, not cargo-culted in now -- adding an enum member
    with no producer is the same defect (advertised-but-unused surface) the
    accepted 3B.3A correction removed elsewhere."""

    #: An external runtime is being reused as-is.
    REUSED_EXTERNAL = "reused_external"
    #: An owned runtime has been materialised and is ready for use.
    READY = "ready"
    #: Cleanup of an owned runtime has been requested but not yet completed.
    CLEANUP_PENDING = "cleanup_pending"
    #: An owned runtime has been cleaned up.
    CLEANED = "cleaned"
    #: Cleanup of an owned runtime was attempted and failed.
    CLEANUP_FAILED = "cleanup_failed"


class CleanupOutcome(str, Enum):
    """Structured outcome of a cleanup attempt. Distinct values a caller must
    tell apart without parsing prose."""

    #: The runtime is external / unowned -- cleanup does not apply. No
    #: destructive action was or will be taken.
    NOT_APPLICABLE_EXTERNAL = "not_applicable_external"
    #: The owned runtime was cleaned up by this call.
    SUCCEEDED = "succeeded"
    #: Cleanup already ran for this ownership record; this call was a no-op.
    ALREADY_COMPLETED = "already_completed"
    #: Ownership could not be revalidated (missing/again-unprovable process
    #: identity). Cleanup refused -- fail closed, do not signal a PID we
    #: cannot prove is still ours.
    OWNERSHIP_NOT_REVALIDATED = "ownership_not_revalidated"
    #: Graceful cleanup (3B.3C: SIGTERM within timeout) failed.
    GRACEFUL_FAILED = "graceful_failed"
    #: Forced cleanup (3B.3C: SIGKILL) failed.
    FORCED_FAILED = "forced_failed"


@dataclass(frozen=True)
class CleanupResult:
    """Typed cleanup result -- never a bool / tuple / exception string."""

    outcome: CleanupOutcome
    detail: str
    #: True iff this call actually invoked the destructive cleanup callback.
    #: Exactly one cleanup for one ownership record may have this True.
    destructive_action_performed: bool = False

    @property
    def ok(self) -> bool:
        return self.outcome in (
            CleanupOutcome.SUCCEEDED,
            CleanupOutcome.ALREADY_COMPLETED,
            CleanupOutcome.NOT_APPLICABLE_EXTERNAL,
        )


# ---------------------------------------------------------------------------
# owned runtime -- immutable ownership proof
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OwnedRuntime:
    """Immutable record of a runtime process ModelBench started and therefore
    may terminate. Created only by :func:`materialise_owned_runtime` (token
    gated). An externally discovered runtime is **never** represented as one
    of these.

    Every field must earn its place by serving a lifecycle safety decision:

    * ``backend`` -- selects the 3B.3C teardown adapter; audit.
    * ``endpoint`` -- the local endpoint claimed for this owned process;
      released on cleanup.
    * ``launch_proof`` -- the anti-PID-reuse evidence. 3B.3C must
      re-observe it and call :meth:`LaunchProcessProof.revalidation_matches`
      *before* any signal. Absent proof -> cleanup refused.
    * ``launched_at`` -- ISO-8601 launch timestamp; audit + teardown-timeout
      reasoning. Not part of process-identity matching.
    * ``runtime_profile_identity`` -- proves the launched process corresponds
      to the resolved *recipe* (not some other config); carried through to
      benchmark evidence by 3B.3C.
    * ``recipe_identity_key`` -- the :meth:`MaterialisationRequest.identity_key`
      the launch was authorised against; lets 3B.3C assert it is tearing down
      the runtime it was asked to create.
    * ``selected_physical_gpu_uuids`` -- copied verbatim from the resolved
      recipe. Used only to *verify* placement (3B.3C "verify GPU assignment
      where possible") and to scope GPU-resource release. Lifecycle code has
      no path that appends to this.

    No PID-reuse-unsafe field, no mutable lifecycle state, and no
    llama.cpp-specific launch detail lives here -- 3B.3C's command builder
    owns the backend specifics.
    """

    backend: str
    endpoint: str
    launch_proof: LaunchProcessProof
    launched_at: str
    runtime_profile_identity: RuntimeProfileIdentity
    recipe_identity_key: str
    selected_physical_gpu_uuids: Tuple[str, ...]
    # No default: an omitted token is a construction error, not a pass. The
    # only valid value is the module-private _TOKEN, held by
    # materialise_owned_runtime().
    _token: object = field(repr=False)

    def __post_init__(self) -> None:
        if self._token is not _TOKEN:
            raise MaterialisationRequestError(
                "OwnedRuntime must be created via materialise_owned_runtime(); "
                "direct construction would fake ownership"
            )
        if not isinstance(self.launch_proof, LaunchProcessProof):
            raise MaterialisationRequestError("launch_proof must be a LaunchProcessProof")
        if not (isinstance(self.backend, str) and self.backend):
            raise MaterialisationRequestError("backend is required")
        if not (isinstance(self.endpoint, str) and self.endpoint):
            raise MaterialisationRequestError("endpoint is required")
        object.__setattr__(
            self, "selected_physical_gpu_uuids", tuple(self.selected_physical_gpu_uuids)
        )


@dataclass(frozen=True)
class CleanupAuthority:
    """The right to clean up exactly one owned runtime. Handed out only for
    ``MODELBENCH_OWNED`` materialisations. An external-reuse result yields
    ``None`` here, so a caller cannot write ``result.cleanup_authority.clean()``
    for a runtime ModelBench does not own -- it would be ``None``."""

    owned_runtime: OwnedRuntime


@dataclass(frozen=True)
class MaterialisationResult:
    """Typed result of a materialisation attempt. A downstream caller reads
    ownership and usability directly -- never infers them from ``pid is not
    None`` or ``profile.mode``."""

    state: LifecycleState
    ownership: RuntimeOwnership
    usable: bool
    endpoint: Optional[str] = None
    runtime_profile_identity: Optional[RuntimeProfileIdentity] = None
    owned_runtime: Optional[OwnedRuntime] = None
    failure_reason: Optional[str] = None

    def __post_init__(self) -> None:
        owned = self.ownership is RuntimeOwnership.MODELBENCH_OWNED
        if owned and self.owned_runtime is None:
            raise ValueError("a MODELBENCH_OWNED result must carry an OwnedRuntime")
        if not owned and self.owned_runtime is not None:
            raise ValueError("only a MODELBENCH_OWNED result may carry an OwnedRuntime")

    @property
    def cleanup_authority(self) -> Optional[CleanupAuthority]:
        """A :class:`CleanupAuthority` only for an owned runtime; ``None`` for
        external reuse or a failed materialisation. This is the single place
        cleanup rights are conferred."""
        if self.ownership is RuntimeOwnership.MODELBENCH_OWNED and self.owned_runtime is not None:
            return CleanupAuthority(owned_runtime=self.owned_runtime)
        return None


# ---------------------------------------------------------------------------
# entry points (no spawn) -- 3B.3C replaces the owned path with a real
# process adapter; the reuse path is already complete.
# ---------------------------------------------------------------------------
def reuse_external_runtime(request: MaterialisationRequest) -> MaterialisationResult:
    """Represent reuse of an already-running external runtime described by an
    accepted resolution. No process is started; no ownership is conferred."""
    if not isinstance(request, MaterialisationRequest):
        raise MaterialisationRequestError("request must be a MaterialisationRequest")
    return MaterialisationResult(
        state=LifecycleState.REUSED_EXTERNAL,
        ownership=RuntimeOwnership.EXTERNAL_REUSED,
        usable=True,
        endpoint=request.endpoint,
        runtime_profile_identity=request.runtime_profile_identity,
        owned_runtime=None,
    )


def materialise_owned_runtime(
    request: MaterialisationRequest,
    *,
    launch_proof: LaunchProcessProof,
    launched_at: str,
) -> MaterialisationResult:
    """Record that ModelBench has materialised an owned runtime for
    ``request``.

    **3B.3B does not spawn.** The caller (a test now, 3B.3C's process adapter
    later) is responsible for having actually started the process and for
    passing the :class:`LaunchProcessProof` observed at that spawn. This
    function only builds the immutable :class:`OwnedRuntime` and the typed
    result -- it is the single authoritative construction path for an
    ``OwnedRuntime``.
    """
    if not isinstance(request, MaterialisationRequest):
        raise MaterialisationRequestError("request must be a MaterialisationRequest")
    if not isinstance(launch_proof, LaunchProcessProof):
        raise MaterialisationRequestError("launch_proof must be a LaunchProcessProof")
    if not (isinstance(launched_at, str) and launched_at):
        raise MaterialisationRequestError("launched_at is required")
    recipe = request.recipe
    owned = OwnedRuntime(
        backend=recipe.backend,
        endpoint=recipe.endpoint,
        launch_proof=launch_proof,
        launched_at=launched_at,
        runtime_profile_identity=recipe.runtime_profile_identity,
        recipe_identity_key=request.identity_key(),
        selected_physical_gpu_uuids=recipe.selected_physical_gpu_uuids,
        _token=_TOKEN,
    )
    return MaterialisationResult(
        state=LifecycleState.READY,
        ownership=RuntimeOwnership.MODELBENCH_OWNED,
        usable=True,
        endpoint=recipe.endpoint,
        runtime_profile_identity=recipe.runtime_profile_identity,
        owned_runtime=owned,
    )


# ---------------------------------------------------------------------------
# lifecycle controller -- mutable state + injected cleanup
# ---------------------------------------------------------------------------
#: Given the owned runtime, re-observe its live process identity so ownership
#: can be revalidated before a destructive action. Returns None if the
#: process identity cannot be established (-> cleanup refused, fail closed).
RevalidateFn = Callable[[OwnedRuntime], Optional[LaunchProcessProof]]
#: Perform the actual destructive teardown of a revalidated owned runtime.
#: 3B.3C supplies the SIGTERM/SIGKILL adapter; 3B.3B injects a fake. May
#: raise to signal cleanup failure.
CleanupFn = Callable[[OwnedRuntime], None]


class RuntimeLifecycleController:
    """Mutable lifecycle wrapper around a :class:`MaterialisationResult`.

    Separates the *immutable* ownership proof (:class:`OwnedRuntime`) from
    *mutable* lifecycle state, which lives here. Provides context-manager
    semantics:

    * entering does **not** confer ownership -- an external-reuse result
      wrapped here still has no cleanup authority;
    * ``__exit__`` cleans only an explicitly owned runtime;
    * external reuse exits without any termination;
    * cleanup is idempotent -- exactly one destructive action is possible per
      ownership record;
    * an exception inside the ``with`` block still requests cleanup for an
      owned runtime;
    * a cleanup failure never replaces the original exception.
    """

    def __init__(
        self,
        result: MaterialisationResult,
        *,
        cleanup_fn: Optional[CleanupFn] = None,
        revalidate_fn: Optional[RevalidateFn] = None,
    ) -> None:
        if not isinstance(result, MaterialisationResult):
            raise TypeError("result must be a MaterialisationResult")
        # Invariant: a MODELBENCH_OWNED controller cannot exist without the
        # means to *prove* ownership (revalidate_fn) and *release* it
        # (cleanup_fn). Without both, `cleanup()` would either signal a PID
        # with zero proof or report a destruction that never happened -- the
        # exact fail-open modes this slice exists to prevent. EXTERNAL_REUSED
        # needs neither: that path returns before either is consulted.
        if result.ownership is RuntimeOwnership.MODELBENCH_OWNED:
            if cleanup_fn is None or revalidate_fn is None:
                raise TypeError(
                    "an owned-runtime lifecycle controller requires both "
                    "cleanup_fn and revalidate_fn (ownership must be provable "
                    "and releasable)"
                )
        self._result = result
        self._cleanup_fn = cleanup_fn
        self._revalidate_fn = revalidate_fn
        # The single guard behind "exactly one destructive cleanup per
        # ownership record". Set the instant a destructive attempt begins
        # (whether it then succeeds or fails), so a retry never signals a
        # second time.
        self._cleanup_attempted = False
        self._state = result.state
        self._last_cleanup: Optional[CleanupResult] = None

    @property
    def result(self) -> MaterialisationResult:
        return self._result

    @property
    def state(self) -> LifecycleState:
        return self._state

    @property
    def owns_runtime(self) -> bool:
        return self._result.ownership is RuntimeOwnership.MODELBENCH_OWNED

    @property
    def last_cleanup(self) -> Optional[CleanupResult]:
        return self._last_cleanup

    # -- cleanup ----------------------------------------------------------
    def cleanup(self) -> CleanupResult:
        """Clean up the owned runtime, once. Idempotent and fail-closed.

        * external / unowned -> :attr:`CleanupOutcome.NOT_APPLICABLE_EXTERNAL`,
          no destructive action;
        * already attempted -> :attr:`CleanupOutcome.ALREADY_COMPLETED`,
          no second destructive action;
        * ownership cannot be revalidated -> refused
          (:attr:`CleanupOutcome.OWNERSHIP_NOT_REVALIDATED`);
        * callback raises -> :attr:`CleanupOutcome.GRACEFUL_FAILED`, still
          marked attempted (no automatic retry / no second signal here --
          3B.3C owns the graceful->forced escalation).
        """
        if not self.owns_runtime:
            self._last_cleanup = CleanupResult(
                outcome=CleanupOutcome.NOT_APPLICABLE_EXTERNAL,
                detail="runtime is external / unowned; nothing to clean up",
            )
            return self._last_cleanup

        if self._cleanup_attempted:
            self._last_cleanup = CleanupResult(
                outcome=CleanupOutcome.ALREADY_COMPLETED,
                detail="cleanup already attempted for this ownership record",
            )
            return self._last_cleanup

        owned = self._result.owned_runtime
        assert owned is not None  # owns_runtime implies this

        # Fail closed *before* marking attempted or touching the callback:
        # if we cannot prove the process is still ours, we take no
        # destructive action and leave the door open for a later, proven
        # attempt. revalidate_fn / cleanup_fn are guaranteed present for an
        # owned runtime (constructor invariant) -- re-check as an explicit
        # guard, not an assert, so `python -O` cannot turn a refusal into a
        # `None(owned)` TypeError mid-teardown.
        if self._revalidate_fn is None or self._cleanup_fn is None:
            self._state = LifecycleState.CLEANUP_FAILED
            self._last_cleanup = CleanupResult(
                outcome=CleanupOutcome.OWNERSHIP_NOT_REVALIDATED,
                detail="lifecycle controller has no revalidate/cleanup adapter",
            )
            return self._last_cleanup
        observed = self._revalidate_fn(owned)
        if observed is None or not owned.launch_proof.revalidation_matches(observed):
            self._state = LifecycleState.CLEANUP_FAILED
            self._last_cleanup = CleanupResult(
                outcome=CleanupOutcome.OWNERSHIP_NOT_REVALIDATED,
                detail=(
                    "process identity could not be revalidated; refusing to "
                    "signal a PID that may have been reused"
                ),
            )
            return self._last_cleanup

        # Commit: from here exactly one destructive attempt has occurred.
        self._cleanup_attempted = True
        self._state = LifecycleState.CLEANUP_PENDING

        try:
            self._cleanup_fn(owned)
        except Exception as exc:  # noqa: BLE001 -- surfaced as structured outcome
            self._state = LifecycleState.CLEANUP_FAILED
            self._last_cleanup = CleanupResult(
                outcome=CleanupOutcome.GRACEFUL_FAILED,
                detail=f"cleanup callback failed: {type(exc).__name__}: {exc}",
                destructive_action_performed=True,
            )
            return self._last_cleanup

        self._state = LifecycleState.CLEANED
        self._last_cleanup = CleanupResult(
            outcome=CleanupOutcome.SUCCEEDED,
            detail="owned runtime cleaned up",
            destructive_action_performed=True,
        )
        return self._last_cleanup

    # -- context manager -------------------------------------------------
    def __enter__(self) -> "RuntimeLifecycleController":
        # Entering never confers ownership -- ownership is whatever the
        # MaterialisationResult already established.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Clean up an owned runtime on every exit path. External reuse exits
        without any destructive action. A cleanup failure is recorded but
        never swallows the original exception, and never propagates on a
        clean exit (the outcome is available via :attr:`last_cleanup` /
        structured result -- callers switch on that, not on exceptions)."""
        if self.owns_runtime:
            self.cleanup()
        # Always falsy: do not suppress an in-flight exception.
        return False
