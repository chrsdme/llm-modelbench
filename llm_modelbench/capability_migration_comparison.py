"""Anvil Stage 2.6A (phase 1) -- legacy-vs-new capability decision
comparison harness. Computes today's legacy planner capability-routing
decision and the new typed :class:`~llm_modelbench.capability_projection.CapabilityDecision`
side by side for the same (profile, family) input, and classifies the
result, per this session's Stage 2.6 continuing advice
(``local_only/anvil/codex-advice-stage2-continuing.txt``): "build a
comparison harness that evaluates the legacy planner applicability
decision and the new TaskApplicability result over the regression
corpus and representative real fixtures... classify each comparison as
MATCH / EXPECTED_CORRECTION / UNEXPLAINED_DIFFERENCE."

**Scope note, a deliberate narrowing of the advice's literal ask**: this
compares at the *capability* decision level
(``capability_identity_compatibility()`` + ``measured_supported_families()``
vs. the new ``CapabilityDecision``), not the full
``TaskApplicability`` composition. ``planner.build_plan()``'s actual
family-routing gate (``planner.py:107/120/122``, confirmed by this
session's legacy-authority inventory) is exactly this capability
question -- it does not itself evaluate environment/VRAM fit or operator
policy at that point in the flow. Comparing at the ``TaskApplicability``
level would require sourcing typed ``EnvironmentDecision``/
``OperatorTaskDecision`` inputs that don't have an established legacy
source yet -- a separate open design question, not resolved here. This
harness is scoped to what ``planner.py`` actually gates on today, kept
honest rather than padded with a "full" comparison it can't really
perform.

The legacy side of the comparison calls ``capabilities.capability_identity_compatibility()``
and ``capabilities.measured_supported_families()`` directly -- the exact
same functions ``planner.py`` calls today (confirmed by reading
``planner.py:107,120,122``) -- rather than re-implementing planner logic
from memory, so a discrepancy in this harness reflects a real behavioral
difference, not a harness bug.

The new side uses :func:`~llm_modelbench.capability_evidence_adapter.adapt_legacy_profile_family_to_observation`
(this session, phase 1 of this same slice) to build a typed
``CapabilityObservation`` from the same profile, then runs it through
Stage 2.3's ``project_capability_observation()``/
``decide_capability_from_projection()`` unmodified -- this harness adds
no new capability-authority logic of its own, it only assembles the
already-built pieces so they can be observed side by side.

**Classification is deliberately conservative, matching the advice's
explicit risk asymmetry** ("Any unexplained change from non-applicable
to applicable is a release blocker... Negative-direction differences can
still be bugs, but positive-authority expansion is the dangerous failure
mode Stage 2 was specifically designed to prevent"): every difference
defaults to ``UNEXPLAINED_DIFFERENCE`` regardless of direction. A caller
may only promote a specific comparison to ``EXPECTED_CORRECTION`` by
passing ``known_correction=True`` -- an explicit, deliberate,
case-by-case decision, never inferred automatically. This module does
not maintain its own allowlist of "known" corrections; that judgment
call belongs to whoever runs the comparison, not to this harness.

Still purely additive/observational: not called from ``planner.py``
itself, and does not change what any real run does.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional

from .capabilities import (
    CAPABILITY_SCHEMA_VERSION,
    PROBE_PROTOCOL_VERSION,
    MeasuredCapabilityState,
    capability_identity_compatibility,
    measured_supported_families,
)
from .capability_evidence_adapter import (
    adapt_legacy_profile_family_to_observation,
    typed_identity_from_capability_identity,
)
from .capability_projection import decide_capability_from_projection, project_capability_observation

__all__ = [
    "ComparisonDirection",
    "ComparisonVerdict",
    "CapabilityDecisionComparison",
    "compare_planner_capability_decision",
    "compare_judge_capability_eligibility",
]


class ComparisonDirection(str, Enum):
    NONE = "none"
    EXPANSION = "expansion"  # legacy: not applicable -> new: applicable (dangerous direction)
    CONTRACTION = "contraction"  # legacy: applicable -> new: not applicable


class ComparisonVerdict(str, Enum):
    MATCH = "match"
    EXPECTED_CORRECTION = "expected_correction"
    UNEXPLAINED_DIFFERENCE = "unexplained_difference"


@dataclass(frozen=True)
class CapabilityDecisionComparison:
    family: str
    legacy_applicable: bool
    legacy_reason: str
    new_applicable: bool
    new_reason: str
    direction: ComparisonDirection
    verdict: ComparisonVerdict


def compare_planner_capability_decision(
    profile: Mapping[str, Any],
    family: str,
    *,
    legacy_current_identity: Optional[Mapping[str, Any]],
    current_probe_protocol_version: str = PROBE_PROTOCOL_VERSION,
    current_capability_schema_version: int = CAPABILITY_SCHEMA_VERSION,
    known_correction: bool = False,
) -> CapabilityDecisionComparison:
    """Compute both the legacy and the new capability-applicability
    decision for the same ``(profile, family)`` input against the same
    ``legacy_current_identity`` "what does the live client see right now"
    snapshot, and classify the result. ``known_correction=True`` is the
    only way a difference becomes ``EXPECTED_CORRECTION`` -- explicit
    caller sign-off, never inferred."""
    legacy_compatibility = capability_identity_compatibility(profile, legacy_current_identity)
    legacy_compatible = bool(legacy_compatibility.get("compatible"))
    legacy_families = measured_supported_families(profile) if legacy_compatible else []
    legacy_applicable = family in legacy_families
    legacy_reason = (
        "measured_supported" if legacy_applicable
        else str(legacy_compatibility.get("reason")) if not legacy_compatible
        else "not_measured_supported"
    )

    current_identity = (
        typed_identity_from_capability_identity(
            legacy_current_identity, protocol_version=current_probe_protocol_version
        )
        if legacy_current_identity is not None else None
    )
    observation = adapt_legacy_profile_family_to_observation(profile, family)
    observations = [observation] if observation is not None else []
    projection = project_capability_observation(
        observations,
        capability=family,
        current_model_identity=current_identity.model_identity if current_identity else None,
        current_runtime_profile_identity=current_identity.runtime_profile_identity if current_identity else None,
        current_probe_protocol_version=current_probe_protocol_version,
        current_capability_schema_version=current_capability_schema_version,
        current_template_config_hash=current_identity.template_hash if current_identity else None,
        current_endpoint_identity=current_identity.endpoint_identity if current_identity else None,
    )
    new_decision = decide_capability_from_projection(projection)
    new_applicable = new_decision.applicable
    new_reason = new_decision.reason.value

    if legacy_applicable == new_applicable:
        direction = ComparisonDirection.NONE
        verdict = ComparisonVerdict.MATCH
    elif legacy_applicable and not new_applicable:
        direction = ComparisonDirection.CONTRACTION
        verdict = ComparisonVerdict.EXPECTED_CORRECTION if known_correction else ComparisonVerdict.UNEXPLAINED_DIFFERENCE
    else:
        direction = ComparisonDirection.EXPANSION
        verdict = ComparisonVerdict.EXPECTED_CORRECTION if known_correction else ComparisonVerdict.UNEXPLAINED_DIFFERENCE

    return CapabilityDecisionComparison(
        family=family,
        legacy_applicable=legacy_applicable,
        legacy_reason=legacy_reason,
        new_applicable=new_applicable,
        new_reason=new_reason,
        direction=direction,
        verdict=verdict,
    )


def _legacy_judge_capability_rejection_reference(item: Mapping[str, Any]) -> Optional[str]:
    """Frozen, verbatim reproduction of ``campaign.py``'s pre-Stage-2.6D
    ``_judge_capability_rejection()`` (as it existed at commit ``36be6e6``,
    the parent of the 2.6D migration), reconstructed here from the same
    unchanged helper primitives 2.6D deliberately kept callable as a
    "regression oracle, comparison source" (per
    ``codex-advise_pre2.6D.txt`` part 9) rather than deleted.

    This function is used **only** by :func:`compare_judge_capability_eligibility`
    below (Task #8's comparison gate). It is never called by production
    code and must not be treated as authority -- ``campaign.py``'s current
    ``_judge_capability_rejection()`` (the post-2.6D typed-authority
    version) is the real production decision; this is its frozen
    predecessor, kept only to prove what changed.
    """
    from .campaign import (
        _candidate_capability_identity_compatibility,
        _canonical_candidate_families,
        _judge_measured_text_state,
    )

    if item.get("capability_schema_version") != CAPABILITY_SCHEMA_VERSION:
        return "capability_reprobe_required"
    state = _judge_measured_text_state(item)
    families = _canonical_candidate_families(item)
    if state == MeasuredCapabilityState.MEASURED_SUPPORTED.value:
        if not _candidate_capability_identity_compatibility(item).get("compatible"):
            return "capability_reprobe_required"
        return None
    if state == MeasuredCapabilityState.PROBE_INCONCLUSIVE.value:
        return "capability_reprobe_required"
    if "embedding" in families:
        return "non_generative_embedding_only"
    if "vision" in families:
        return "non_generative_vision_only"
    return "unknown_or_non_generative_capability"


def compare_judge_capability_eligibility(
    item: Dict[str, Any],
    *,
    known_correction: bool = False,
) -> CapabilityDecisionComparison:
    """Anvil Stage 2.6D, Task #8 -- compare the legacy (pre-2.6D, frozen
    above) judge candidate text-capability-eligibility decision against
    ``campaign.py``'s real, current (post-2.6D) ``_judge_capability_rejection()``,
    for the same candidate ``item`` dict.

    Both sides call ``campaign.py``'s actual functions (the frozen
    reference on the legacy side, the live production function on the new
    side) rather than re-implementing judge eligibility logic a third
    time -- a discrepancy here reflects a real behavioral difference, not
    a harness bug. ``known_correction=True`` is the only way a difference
    becomes ``EXPECTED_CORRECTION``, matching
    :func:`compare_planner_capability_decision`'s existing convention.
    """
    from .campaign import _judge_capability_rejection

    legacy_rejection = _legacy_judge_capability_rejection_reference(item)
    legacy_applicable = legacy_rejection is None
    new_rejection = _judge_capability_rejection(item)
    new_applicable = new_rejection is None

    if legacy_applicable == new_applicable:
        direction = ComparisonDirection.NONE
        verdict = ComparisonVerdict.MATCH
    elif legacy_applicable and not new_applicable:
        direction = ComparisonDirection.CONTRACTION
        verdict = ComparisonVerdict.EXPECTED_CORRECTION if known_correction else ComparisonVerdict.UNEXPLAINED_DIFFERENCE
    else:
        direction = ComparisonDirection.EXPANSION
        verdict = ComparisonVerdict.EXPECTED_CORRECTION if known_correction else ComparisonVerdict.UNEXPLAINED_DIFFERENCE

    return CapabilityDecisionComparison(
        family="text",
        legacy_applicable=legacy_applicable,
        legacy_reason=legacy_rejection or "eligible",
        new_applicable=new_applicable,
        new_reason=new_rejection or "eligible",
        direction=direction,
        verdict=verdict,
    )
