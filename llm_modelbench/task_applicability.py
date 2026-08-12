"""Anvil Stage 2.5 -- TaskApplicability: the single typed composition of
"can this task run, and if not, why?" TaskApplicability = CapabilityDecision
(Stage 2.3) + EnvironmentDecision (Stage 2.4) + operator task policy, per
the master plan's own architecture. This module is deliberately, purely
compositional: it does not rediscover hardware, inspect backend metadata,
rerun capability projection, perform KV/VRAM estimation, or read legacy
profile dictionaries. It consumes three already-typed decisions and
produces one typed :class:`TaskApplicability`.

Scope is deliberately narrow, per this session's Stage 2.5 scoping advice
(``local_only/anvil/codex-advice-stage2-continuing.txt``, itself a
continuation/correction of the earlier Stage 2.4 advice) -- purely
additive, like every Stage 2 module before it. Nothing here is wired into
``planner.py``, ``runner.py``, ``recovery``/``repair.py``, or judge
authority yet; that is Stage 2.6's job, split per that same advice into
several individually-verified consumer migrations rather than one rewrite.

Two deliberate departures from the advice's literal wording, both explained
here so a later reader doesn't assume they're oversights:

1. The advice sketches ``capability_unsupported`` / ``capability_inconclusive``
   / ``capability_reprobe_required`` and
   ``environment_vram_limit`` / ``environment_context_limit`` /
   ``environment_backend_limit`` / ``environment_inconclusive`` as terminal
   reasons. Neither maps 1:1 onto what Stage 2.3's ``CapabilityDecisionReason``
   or Stage 2.4's ``EnvironmentDecisionReason`` actually produce today:
   - ``CapabilityDecisionReason`` has no "reprobe required" value. Mapped
     here from ``NO_CURRENT_PROJECTION`` -- no compatible observation
     exists at all for the current identity, which really does mean
     "nobody has measured this configuration yet," i.e. go probe it.
     ``PROJECTION_AMBIGUOUS`` (contradictory observations exist) maps to
     ``CAPABILITY_INCONCLUSIVE`` instead -- distinct from "never measured":
     evidence exists but doesn't resolve.
   - ``EnvironmentDecisionReason`` has no VRAM/context/backend
     sub-classification -- ``EnvironmentDecision`` (Stage 2.4) does not
     carry that granularity in its typed reason today (only in the free-text
     ``detail``/``source`` fields of the underlying evidence). Inventing
     three typed sub-reasons here that no current code path can actually
     compute would be a false precision -- worse than not having them. This
     module maps to one ``ENVIRONMENT_DOES_NOT_FIT`` instead. If a later
     slice adds VRAM-vs-context-vs-backend typing to ``EnvironmentEvidence``
     itself, this module's reason vocabulary should be revisited to match --
     not before.
2. The advice frames a missing environment evaluation as needing an
   explicit "NOT_EVALUATED/NOT_REQUIRED" state "rather than fabricating
   unknown." Taking that literally: ``ENVIRONMENT_NOT_EVALUATED`` (this
   module, when ``environment_decision`` is ``None``) is kept structurally
   distinct from ``ENVIRONMENT_INCONCLUSIVE`` (an ``EnvironmentDecision``
   *was* supplied, but its own evidence was absent/ambiguous) -- collapsing
   the two would itself be exactly the kind of fabricated-unknown the
   advice warned against.

Terminal precedence (first match wins, deterministic, no other ordering
considered):
1. Operator exclusion -- an explicit administrative decision overrides
   everything else regardless of what capability/environment evidence
   says, and doesn't require either to have been evaluated at all.
2. Capability blocks -- checked before environment, so environment
   measurement is never required for a task already known
   capability-ineligible (this is exactly why ``environment_decision`` is
   optional here, not a design gap).
3. Environment blocks (including "not evaluated").
4. Otherwise applicable.

All three component decisions are preserved intact in the result, even
when only one becomes the primary terminal reason -- this is what makes
:class:`TaskApplicability` useful later for readiness views, reports,
recovery, and explaining *why* something did not run, not just whether it
did.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .capability_projection import CapabilityDecision, CapabilityDecisionReason
from .environment_decision import EnvironmentDecision

__all__ = [
    "OperatorTaskDecision",
    "TaskApplicabilityStatus",
    "TaskApplicabilityReason",
    "TaskApplicability",
    "compose_task_applicability",
]


@dataclass(frozen=True)
class OperatorTaskDecision:
    """Whether operator/administrative policy allows this task to run at
    all, independent of whether it technically could. Deliberately not
    Stage 1's ``DecisionPolicy`` -- that class answers a different
    question (may unattended execution auto-select a backend?), and
    forcing task inclusion/exclusion through it would mean asking
    something conceptually backwards, like
    ``DecisionPolicy.permits(Action.RUN_TASK_X)``. This is a separate,
    minimal typed decision: the actual policy that produces one (e.g.
    from an operator-supplied exclusion list) is not this module's
    concern -- purely additive, callers construct these however they
    already decide today (e.g. the existing task include/exclude regex
    filtering in ``filters.py``/``planner.py``, unmigrated as of this
    slice).
    """

    allowed: bool
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.allowed, bool):
            raise TypeError(
                f"OperatorTaskDecision.allowed must be a bool, not {type(self.allowed).__name__!r}"
            )


class TaskApplicabilityStatus(str, Enum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"


class TaskApplicabilityReason(str, Enum):
    APPLICABLE = "applicable"
    OPERATOR_EXCLUDED = "operator_excluded"
    CAPABILITY_UNSUPPORTED = "capability_unsupported"
    CAPABILITY_INCONCLUSIVE = "capability_inconclusive"
    CAPABILITY_REPROBE_REQUIRED = "capability_reprobe_required"
    ENVIRONMENT_DOES_NOT_FIT = "environment_does_not_fit"
    ENVIRONMENT_INCONCLUSIVE = "environment_inconclusive"
    ENVIRONMENT_NOT_EVALUATED = "environment_not_evaluated"


# CapabilityDecisionReason -> TaskApplicabilityReason for every reason that
# does NOT mean "capability supports this" (MEASURED_SUPPORTED is handled
# separately, as the one reason that lets composition proceed to the next
# tier rather than terminate here).
_CAPABILITY_BLOCK_REASON = {
    CapabilityDecisionReason.MEASURED_UNSUPPORTED: TaskApplicabilityReason.CAPABILITY_UNSUPPORTED,
    CapabilityDecisionReason.BACKEND_UNSUPPORTED: TaskApplicabilityReason.CAPABILITY_UNSUPPORTED,
    CapabilityDecisionReason.NOT_APPLICABLE: TaskApplicabilityReason.CAPABILITY_UNSUPPORTED,
    CapabilityDecisionReason.PROBE_INCONCLUSIVE: TaskApplicabilityReason.CAPABILITY_INCONCLUSIVE,
    CapabilityDecisionReason.PROJECTION_AMBIGUOUS: TaskApplicabilityReason.CAPABILITY_INCONCLUSIVE,
    CapabilityDecisionReason.NO_CURRENT_PROJECTION: TaskApplicabilityReason.CAPABILITY_REPROBE_REQUIRED,
}


@dataclass(frozen=True)
class TaskApplicability:
    status: TaskApplicabilityStatus
    terminal_reason: TaskApplicabilityReason
    capability_decision: CapabilityDecision
    environment_decision: Optional[EnvironmentDecision]
    operator_decision: OperatorTaskDecision
    evidence_refs: Tuple[str, ...] = ()


def compose_task_applicability(
    *,
    capability_decision: CapabilityDecision,
    operator_decision: OperatorTaskDecision,
    environment_decision: Optional[EnvironmentDecision] = None,
) -> TaskApplicability:
    """Compose one :class:`TaskApplicability` from three already-typed
    decisions. Does not compute, measure, or reprobe anything -- see the
    module docstring for the full terminal-reason precedence."""
    if not isinstance(capability_decision, CapabilityDecision):
        raise TypeError(
            "compose_task_applicability() requires a real CapabilityDecision, "
            f"not {type(capability_decision).__name__!r}"
        )
    if not isinstance(operator_decision, OperatorTaskDecision):
        raise TypeError(
            "compose_task_applicability() requires a real OperatorTaskDecision, "
            f"not {type(operator_decision).__name__!r}"
        )
    if environment_decision is not None and not isinstance(environment_decision, EnvironmentDecision):
        raise TypeError(
            "compose_task_applicability() requires environment_decision to be a real "
            f"EnvironmentDecision or None, not {type(environment_decision).__name__!r}"
        )

    evidence_refs = capability_decision.projection.considered_observation_ids

    def _decision(status: TaskApplicabilityStatus, reason: TaskApplicabilityReason) -> TaskApplicability:
        return TaskApplicability(
            status=status,
            terminal_reason=reason,
            capability_decision=capability_decision,
            environment_decision=environment_decision,
            operator_decision=operator_decision,
            evidence_refs=evidence_refs,
        )

    if not operator_decision.allowed:
        return _decision(TaskApplicabilityStatus.NOT_APPLICABLE, TaskApplicabilityReason.OPERATOR_EXCLUDED)

    if not capability_decision.applicable:
        capability_reason = _CAPABILITY_BLOCK_REASON[capability_decision.reason]
        return _decision(TaskApplicabilityStatus.NOT_APPLICABLE, capability_reason)

    if environment_decision is None:
        return _decision(TaskApplicabilityStatus.NOT_APPLICABLE, TaskApplicabilityReason.ENVIRONMENT_NOT_EVALUATED)

    if environment_decision.fits is False:
        return _decision(TaskApplicabilityStatus.NOT_APPLICABLE, TaskApplicabilityReason.ENVIRONMENT_DOES_NOT_FIT)

    if environment_decision.fits is None:
        return _decision(TaskApplicabilityStatus.NOT_APPLICABLE, TaskApplicabilityReason.ENVIRONMENT_INCONCLUSIVE)

    return _decision(TaskApplicabilityStatus.APPLICABLE, TaskApplicabilityReason.APPLICABLE)
