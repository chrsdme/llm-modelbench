"""Anvil Stage 2.3 -- CapabilityProjection + CapabilityDecision: the typed
read model over :class:`~llm_modelbench.capability_observation.CapabilityObservation`.

Answers one question: given the current typed model/runtime identity and a
set of typed observations for one capability, which measurement is
currently authoritative (:class:`CapabilityProjection`), and what
capability-only decision follows from that (:class:`CapabilityDecision`)?

Scope was deliberately narrow at introduction, per this session's Codex/GPT
scoping pass (``local_only/anvil/codex-advice_stage2.3.txt``): purely
additive, like Stage 2.1/2.2 before it, with nothing wired into
``capabilities.py``, ``planner.py``, ``runner.py``, ``campaign.py``, or
``repair.py`` yet at that time.

**Status update (Anvil Stage 2.6E, correcting the claim above, which is
stale as originally written)**: this module is now the core of the typed
capability-authority path every migrated consumer sources its positive
authority from --
:func:`~llm_modelbench.capability_evidence_adapter.new_measured_supported_families`
calls ``project_capability_observation()``/``decide_capability_from_projection()``
here directly, and ``planner.py``/``runner.py``/``repair.py``/``campaign.py``'s
judge-selection functions all call that shared helper (Stages 2.6A-D). The
legacy dict-shaped ``capability_identity_compatibility()`` in
``capabilities.py`` is *not* operationally authoritative for any of those
four real command paths any more -- see
``local_only/anvil/stage-2.6E-authority-audit.md`` for the full audit.

Explicit non-goals for this slice (all later Stage 2 work, not this one):
- No ``EnvironmentDecision`` -- "can this specific configuration run here,
  right now" (GPU/runtime-reuse policy, allocation slope, estimator
  precedence, endpoint recovery, live-instance suitability) is Stage 2.4.
- No ``TaskApplicability`` composition or operator policy -- Stage 2.5.
- No planner/runner/campaign/repair/judge call-site migration -- Stage 2.6.
- No legacy profile conversion, no blessing old dict profiles into typed
  evidence, no reprobe/fleet adoption -- Stage 2.7.
- No bypass audit or Stage 2 acceptance audit -- Stage 2.8/2.9.
- No :class:`~llm_modelbench.evidence.ProjectionStore` current-pointer
  writes. Per the master plan, ``CapabilityProjection`` is "computed, not
  stored" -- this module only ever computes a projection on demand, it
  never persists an adoption/current-pointer decision.

Selection policy (conservative, fail-closed on genuine ambiguity):
1. Filter observations to the requested ``capability``.
2. Run :func:`~llm_modelbench.capability_identity.capability_observation_identity_compatibility`
   (Stage 2.2) against every candidate.
3. No observations at all for this capability -> ``NO_OBSERVATIONS``.
4. Candidates exist but none are identity-compatible -> ``NO_COMPATIBLE_OBSERVATION``.
5. Exactly one compatible observation -> select it.
6. Multiple compatible observations that all share the same
   ``evidence_hash`` are the *same* measurement recorded more than once
   (e.g. a repeated probe) -- select deterministically by newest
   ``timestamp``, then ``observation_id`` as a tiebreaker. This is
   duplicate-measurement collapse, never a semantic override.
7. Multiple compatible observations with *differing* ``evidence_hash`` are
   genuinely contradictory measurements with no way to know which is
   authoritative from identity alone -> ``AMBIGUOUS_COMPATIBLE_OBSERVATIONS``,
   fail closed rather than guess (e.g. "newest wins").
8. The ledger-aware entry point additionally resolves
   :class:`~llm_modelbench.evidence.ProvenanceRelation.SUPERSEDES` chains
   via the existing :class:`~llm_modelbench.evidence.EffectiveEvidenceResolver`
   (Stage 0) rather than inventing a second supersession mechanism, before
   applying the same selection policy to the resulting terminal
   observations. A cycle, a fork, or a terminal record that resolves
   outside this capability's observation set is reported as
   ``SUPERSESSION_CONFLICT`` rather than guessed at.

``CapabilityDecision`` adds only capability-level translation on top of a
projection: a measured-supported result is applicable, every other typed
measured result is not -- ``inconclusive`` never silently becomes positive
applicability. It knows nothing about GPUs, memory, task selection, or
operator policy.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Optional, Tuple

from .capabilities import MeasuredCapabilityState
from .capability_identity import (
    CapabilityIdentityCompatibility,
    capability_observation_identity_compatibility,
)
from .capability_observation import CAPABILITY_OBSERVATION_RECORD_TYPE, CapabilityObservation
from .classify import FAMILY_ORDER
from .evidence import EffectiveEvidenceResolutionError, EffectiveEvidenceResolver, EvidenceLedger

if TYPE_CHECKING:
    from .identity import ModelArtifactIdentity, RuntimeProfileIdentity

__all__ = [
    "CapabilityProjectionStatus",
    "CapabilityProjection",
    "CapabilityDecisionReason",
    "CapabilityDecision",
    "project_capability_observation",
    "project_capability_from_ledger",
    "decide_capability_from_projection",
]


class CapabilityProjectionStatus(str, Enum):
    SELECTED = "selected"
    NO_OBSERVATIONS = "no_observations"
    NO_COMPATIBLE_OBSERVATION = "no_compatible_observation"
    AMBIGUOUS_COMPATIBLE_OBSERVATIONS = "ambiguous_compatible_observations"
    SUPERSESSION_CONFLICT = "supersession_conflict"


@dataclass(frozen=True)
class CapabilityProjection:
    """Which ``CapabilityObservation`` (if any) is currently authoritative
    for one capability, computed on demand -- never persisted."""

    capability: str
    status: CapabilityProjectionStatus
    selected_observation: Optional[CapabilityObservation] = None
    selected_record_id: Optional[str] = None
    compatibility: Optional[CapabilityIdentityCompatibility] = None
    considered_observation_ids: Tuple[str, ...] = ()
    incompatible: Tuple[CapabilityIdentityCompatibility, ...] = ()


class CapabilityDecisionReason(str, Enum):
    MEASURED_SUPPORTED = "measured_supported"
    MEASURED_UNSUPPORTED = "measured_unsupported"
    BACKEND_UNSUPPORTED = "backend_unsupported"
    NOT_APPLICABLE = "not_applicable"
    PROBE_INCONCLUSIVE = "probe_inconclusive"
    NO_CURRENT_PROJECTION = "no_current_projection"
    PROJECTION_AMBIGUOUS = "projection_ambiguous"


@dataclass(frozen=True)
class CapabilityDecision:
    capability: str
    applicable: bool
    reason: CapabilityDecisionReason
    projection: CapabilityProjection
    selected_result: Optional[MeasuredCapabilityState] = None


_RESULT_TO_DECISION_REASON = {
    MeasuredCapabilityState.MEASURED_SUPPORTED: CapabilityDecisionReason.MEASURED_SUPPORTED,
    MeasuredCapabilityState.MEASURED_UNSUPPORTED: CapabilityDecisionReason.MEASURED_UNSUPPORTED,
    MeasuredCapabilityState.BACKEND_UNSUPPORTED: CapabilityDecisionReason.BACKEND_UNSUPPORTED,
    MeasuredCapabilityState.NOT_APPLICABLE: CapabilityDecisionReason.NOT_APPLICABLE,
    MeasuredCapabilityState.PROBE_INCONCLUSIVE: CapabilityDecisionReason.PROBE_INCONCLUSIVE,
}

_NON_SELECTED_DECISION_REASON = {
    CapabilityProjectionStatus.NO_OBSERVATIONS: CapabilityDecisionReason.NO_CURRENT_PROJECTION,
    CapabilityProjectionStatus.NO_COMPATIBLE_OBSERVATION: CapabilityDecisionReason.NO_CURRENT_PROJECTION,
    CapabilityProjectionStatus.AMBIGUOUS_COMPATIBLE_OBSERVATIONS: CapabilityDecisionReason.PROJECTION_AMBIGUOUS,
    CapabilityProjectionStatus.SUPERSESSION_CONFLICT: CapabilityDecisionReason.PROJECTION_AMBIGUOUS,
}


def project_capability_observation(
    observations: Iterable[CapabilityObservation],
    *,
    capability: str,
    current_model_identity: Optional["ModelArtifactIdentity"],
    current_runtime_profile_identity: Optional["RuntimeProfileIdentity"],
    current_probe_protocol_version: str,
    current_capability_schema_version: int,
    current_template_config_hash: Optional[str] = None,
    current_endpoint_identity: Optional[str] = None,
) -> CapabilityProjection:
    """Select the currently-authoritative observation for ``capability``
    out of ``observations``, purely in memory -- no ledger, no
    supersession resolution. See the module docstring for the full
    selection policy."""
    if capability not in FAMILY_ORDER:
        raise ValueError(
            f"project_capability_observation() capability {capability!r} is not a known "
            f"family (expected one of {FAMILY_ORDER!r})"
        )

    candidates = []
    for observation in observations:
        if not isinstance(observation, CapabilityObservation):
            raise TypeError(
                "project_capability_observation() requires an iterable of real "
                f"CapabilityObservation instances, not {type(observation).__name__!r}"
            )
        if observation.capability == capability:
            candidates.append(observation)

    considered_ids = tuple(observation.observation_id for observation in candidates)

    if not candidates:
        return CapabilityProjection(
            capability=capability,
            status=CapabilityProjectionStatus.NO_OBSERVATIONS,
            considered_observation_ids=considered_ids,
        )

    compatible: list = []
    incompatible: list = []
    for observation in candidates:
        result = capability_observation_identity_compatibility(
            observation,
            current_model_identity=current_model_identity,
            current_runtime_profile_identity=current_runtime_profile_identity,
            current_probe_protocol_version=current_probe_protocol_version,
            current_capability_schema_version=current_capability_schema_version,
            current_template_config_hash=current_template_config_hash,
            current_endpoint_identity=current_endpoint_identity,
        )
        if result.compatible:
            compatible.append((observation, result))
        else:
            incompatible.append(result)

    if not compatible:
        return CapabilityProjection(
            capability=capability,
            status=CapabilityProjectionStatus.NO_COMPATIBLE_OBSERVATION,
            considered_observation_ids=considered_ids,
            incompatible=tuple(incompatible),
        )

    if len(compatible) == 1:
        observation, result = compatible[0]
        return CapabilityProjection(
            capability=capability,
            status=CapabilityProjectionStatus.SELECTED,
            selected_observation=observation,
            compatibility=result,
            considered_observation_ids=considered_ids,
            incompatible=tuple(incompatible),
        )

    distinct_hashes = {observation.evidence_hash for observation, _ in compatible}
    if len(distinct_hashes) == 1:
        # Same semantic measurement recorded more than once -- duplicate
        # collapse, not a choice between conflicting evidence.
        observation, result = max(
            compatible, key=lambda pair: (pair[0].timestamp or "", pair[0].observation_id)
        )
        return CapabilityProjection(
            capability=capability,
            status=CapabilityProjectionStatus.SELECTED,
            selected_observation=observation,
            compatibility=result,
            considered_observation_ids=considered_ids,
            incompatible=tuple(incompatible),
        )

    # Genuinely contradictory compatible observations -- fail closed rather
    # than guess which one is right.
    return CapabilityProjection(
        capability=capability,
        status=CapabilityProjectionStatus.AMBIGUOUS_COMPATIBLE_OBSERVATIONS,
        considered_observation_ids=considered_ids,
        incompatible=tuple(incompatible),
    )


def project_capability_from_ledger(
    ledger: EvidenceLedger,
    *,
    capability: str,
    current_model_identity: Optional["ModelArtifactIdentity"],
    current_runtime_profile_identity: Optional["RuntimeProfileIdentity"],
    current_probe_protocol_version: str,
    current_capability_schema_version: int,
    current_template_config_hash: Optional[str] = None,
    current_endpoint_identity: Optional[str] = None,
) -> CapabilityProjection:
    """As :func:`project_capability_observation`, but reads
    ``CapabilityObservation`` records from ``ledger`` and first resolves
    any ``SUPERSEDES`` chains via :class:`~llm_modelbench.evidence.EffectiveEvidenceResolver`
    (Stage 0's existing supersession-chain walker, reused rather than
    reinvented) -- the selection policy then runs on the resulting
    terminal observations only."""
    records = {
        record.record_id: record
        for record in ledger.find(record_type=CAPABILITY_OBSERVATION_RECORD_TYPE)
    }
    observations_by_record_id = {
        record_id: CapabilityObservation.from_ledger_payload(record.payload)
        for record_id, record in records.items()
    }
    matching_ids = tuple(
        record_id
        for record_id, observation in observations_by_record_id.items()
        if observation.capability == capability
    )

    if not matching_ids:
        return CapabilityProjection(
            capability=capability,
            status=CapabilityProjectionStatus.NO_OBSERVATIONS,
        )

    resolver = EffectiveEvidenceResolver(ledger)
    terminal_ids = set()
    try:
        for record_id in matching_ids:
            terminal_ids.add(resolver.resolve(record_id).record_id)
    except EffectiveEvidenceResolutionError:
        return CapabilityProjection(
            capability=capability,
            status=CapabilityProjectionStatus.SUPERSESSION_CONFLICT,
            considered_observation_ids=tuple(
                observations_by_record_id[record_id].observation_id for record_id in matching_ids
            ),
        )

    terminal_observations = []
    for terminal_id in sorted(terminal_ids):
        observation = observations_by_record_id.get(terminal_id)
        if observation is None or observation.capability != capability:
            # A supersession chain led outside this capability's own
            # observation set (e.g. a mistaken cross-capability SUPERSEDES
            # link) -- do not guess which side is authoritative.
            return CapabilityProjection(
                capability=capability,
                status=CapabilityProjectionStatus.SUPERSESSION_CONFLICT,
                considered_observation_ids=tuple(
                    observations_by_record_id[record_id].observation_id for record_id in matching_ids
                ),
            )
        terminal_observations.append(observation)

    projection = project_capability_observation(
        terminal_observations,
        capability=capability,
        current_model_identity=current_model_identity,
        current_runtime_profile_identity=current_runtime_profile_identity,
        current_probe_protocol_version=current_probe_protocol_version,
        current_capability_schema_version=current_capability_schema_version,
        current_template_config_hash=current_template_config_hash,
        current_endpoint_identity=current_endpoint_identity,
    )
    if projection.status == CapabilityProjectionStatus.SELECTED:
        projection = dataclasses.replace(
            projection, selected_record_id=projection.selected_observation.observation_id
        )
    return projection


def decide_capability_from_projection(projection: CapabilityProjection) -> CapabilityDecision:
    """Translate a :class:`CapabilityProjection` into a capability-only
    applicability decision. Purely a lookup -- adds no new evidence,
    knows nothing about environment/GPU/task/operator concerns."""
    if not isinstance(projection, CapabilityProjection):
        raise TypeError(
            "decide_capability_from_projection() requires a real CapabilityProjection, "
            f"not {type(projection).__name__!r}"
        )

    if projection.status == CapabilityProjectionStatus.SELECTED:
        result = projection.selected_observation.result
        reason = _RESULT_TO_DECISION_REASON[result]
        return CapabilityDecision(
            capability=projection.capability,
            applicable=result == MeasuredCapabilityState.MEASURED_SUPPORTED,
            reason=reason,
            projection=projection,
            selected_result=result,
        )

    reason = _NON_SELECTED_DECISION_REASON[projection.status]
    return CapabilityDecision(
        capability=projection.capability,
        applicable=False,
        reason=reason,
        projection=projection,
        selected_result=None,
    )
