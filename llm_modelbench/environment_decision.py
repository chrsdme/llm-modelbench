"""Anvil Stage 2.4 -- EnvironmentDecision: a pure typed decision layer
answering exactly one question: "can this exact requested configuration
run here?" It never answers "does this model support this capability?" --
that is :mod:`~llm_modelbench.capability_projection`'s job (Stage 2.3),
kept structurally separate on purpose (see the master plan's "capability
and environment stay separate, by contract").

Scope is deliberately narrow, per this session's Stage 2.4 scoping advice
(``local_only/anvil/Stage 2.4-EnvironmentDecision-codex-advice.txt``):
purely additive, like every Stage 2 module before it. Nothing here is
wired into ``planner.py``, ``runner.py``, ``recovery``/``repair.py``, or
judge authority yet -- that is Stage 2.6's job.

This module does not measure or estimate anything itself. Real, working
precedent for tiers 2 and 4 already exists in ``runner.py`` (confirmed
during Stage 0.0, see ``local_only/anvil/ANVIL_PROGRESS.md``'s Stage 0.0
section): ``_needle_kv_estimate``/``_kv_bytes_per_token`` (metadata-tier
estimate), ``_measured_memory_estimate`` (the "phase 2"/under-load
measurement built from real probe deltas, with its own internal
precedence of Ollama ``/api/ps`` residency over GPU-VRAM slope over
diagnostic-only process-RSS deltas), and ``_needle_environment_skip``
(today's dict-shaped skip verdict). ``topology_budget.evaluate_workload_fit``
similarly already classifies GPU-topology fit from known/unknown
components. None of that measurement logic is reimplemented, duplicated,
or replaced here -- this module only defines the typed evidence/decision
*shape* that a later slice (2.6) can adapt those existing dict-shaped
outputs into. Building the adapter now, before any call site is
migrated, would be scope creep past what this slice was asked to do.

Explicit non-goals for this slice (all later Stage 2 work, not this
one):
- No adapter from ``runner.py``'s existing dict-shaped estimates
  (``_needle_kv_estimate`` et al.) into typed ``EnvironmentEvidence`` --
  2.6.
- No wiring into planner/runner/recovery/judge call sites -- 2.6.
- No ``TaskApplicability`` composition (capability decision + environment
  decision + operator policy) -- 2.5.
- No persistence/ledger integration for environment evidence -- not
  requested by this slice's scoping advice; environment evidence is
  transient, request-scoped input to a pure function, unlike
  ``CapabilityObservation``'s append-only evidence model.

Structural fix for a real, confirmed-by-Stage-0.0 problem (not merely
documented, enforced by the type system): today, when every context-
length probe is skipped for VRAM/KV-budget reasons, the pipeline
hardcodes ``error_kind: "harness_error"`` regardless of cause
(``runner.py``), which ``outcome.py`` maps to ``HarnessError`` --
semantically wrong, since "the measurement machinery failed" and "the
configuration doesn't fit in VRAM" are different facts. ``evidence.py``'s
``EvalStatus`` already anticipated this (``ENVIRONMENT_SKIPPED`` is a
distinct member from ``HARNESS_ERROR`` and ``CAPABILITY_UNSUPPORTED``,
citing ``kv_cache_exceeds_vram_budget`` as the ad hoc precedent) but was
never wired to prevent the conflation. ``EnvironmentDecisionReason`` here
is a separate Python enum from both ``CapabilityDecisionReason`` (Stage
2.3) and ``EvalStatus`` -- a VRAM/capacity failure cannot even
accidentally collide with a capability-unsupported or harness-error
value, because they are different types with disjoint value sets. Wiring
this into the real skip path is still Stage 2.6's job; this slice makes
the conflation impossible to reintroduce in new typed code going
forward.

Evidence precedence (highest first, per the master plan and this
session's scoping advice) -- a request's decision is grounded in the
*highest* tier that has any committing (non-``UNKNOWN``) evidence for
that exact request identity:

1. ``MEASURED_EXECUTION`` -- an exact measured execution of this exact
   requested configuration actually ran (or failed) here.
2. ``OBSERVED_ALLOCATION`` -- a runtime-observed allocation or measured
   slope for this configuration (e.g. ``_measured_memory_estimate``'s
   anchor-and-project machinery), short of a full execution.
3. ``RUNTIME_DECLARATION`` -- an authoritative allocation figure the
   runtime itself declares (e.g. backend-reported usable VRAM limits).
4. ``VALIDATED_ESTIMATOR`` -- a validated estimator with no direct
   measurement backing it (e.g. metadata-only KV-bytes-per-token).
5. ``CONSERVATIVE_UNKNOWN`` -- the fail-closed default when no tier 1-4
   evidence exists for this exact request identity. This is never a tier
   a caller constructs ``EnvironmentEvidence`` for -- it is what
   :func:`decide_environment_fit` falls back to on its own, structurally
   enforced by :meth:`EnvironmentEvidence.__post_init__` rejecting it.

Lower-tier evidence that disagrees with the tier that actually grounds
the decision is never silently discarded -- it is preserved in
:attr:`EnvironmentDecision.contradicting_evidence` ("measured evidence
wins over an estimator conflict, with the contradiction recorded").

Requests are scoped to an exact :class:`EnvironmentRequestIdentity`
(model artifact + runtime profile + GPU topology + context length + KV
cache policy + offload/split policy). Evidence is matched to a decision
by exact identity equality -- a 4k-context measured success does not,
and structurally cannot, satisfy a 64k-context request, because the two
requests' identities are unequal and the 64k evidence lookup simply
never sees the 4k evidence at all.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Optional, Tuple

if TYPE_CHECKING:
    from .identity import ModelArtifactIdentity, RuntimeProfileIdentity

__all__ = [
    "EnvironmentEvidenceTier",
    "EnvironmentFit",
    "EnvironmentRequestIdentity",
    "EnvironmentEvidence",
    "EnvironmentDecisionReason",
    "EnvironmentDecision",
    "decide_environment_fit",
]


def _stable_hash(*parts: Any) -> str:
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class EnvironmentEvidenceTier(str, Enum):
    MEASURED_EXECUTION = "measured_execution"
    OBSERVED_ALLOCATION = "observed_allocation"
    RUNTIME_DECLARATION = "runtime_declaration"
    VALIDATED_ESTIMATOR = "validated_estimator"
    CONSERVATIVE_UNKNOWN = "conservative_unknown"


# Precedence order, highest first. CONSERVATIVE_UNKNOWN is deliberately
# excluded -- it is never walked as an evidence tier, only ever the
# fallback label when no evidence tier below has a committing verdict.
_EVIDENCE_TIER_PRECEDENCE: Tuple[EnvironmentEvidenceTier, ...] = (
    EnvironmentEvidenceTier.MEASURED_EXECUTION,
    EnvironmentEvidenceTier.OBSERVED_ALLOCATION,
    EnvironmentEvidenceTier.RUNTIME_DECLARATION,
    EnvironmentEvidenceTier.VALIDATED_ESTIMATOR,
)


class EnvironmentFit(str, Enum):
    FITS = "fits"
    DOES_NOT_FIT = "does_not_fit"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EnvironmentRequestIdentity:
    """The exact requested configuration an :class:`EnvironmentDecision`
    answers for. Distinct from :class:`~llm_modelbench.identity.RuntimeProfileIdentity`,
    which scopes *capability* evidence compatibility (Stage 2.2) -- this
    identity additionally scopes the memory/capacity-affecting request
    parameters that capability compatibility deliberately does not care
    about. Two requests differing only in ``context_length`` are
    different identities by design: this is the structural mechanism
    behind "a 4k successful run must not prove 64k fit."
    """

    model_identity: "ModelArtifactIdentity"
    runtime_profile_identity: "RuntimeProfileIdentity"
    gpu_topology_key: str
    context_length: int
    kv_cache_policy: Optional[str] = None
    offload_split_policy: Optional[str] = None

    def __post_init__(self) -> None:
        from .identity import ModelArtifactIdentity, RuntimeProfileIdentity

        if not isinstance(self.model_identity, ModelArtifactIdentity):
            raise TypeError(
                "EnvironmentRequestIdentity.model_identity must be a real "
                f"ModelArtifactIdentity, not {type(self.model_identity).__name__!r}"
            )
        if not isinstance(self.runtime_profile_identity, RuntimeProfileIdentity):
            raise TypeError(
                "EnvironmentRequestIdentity.runtime_profile_identity must be a real "
                f"RuntimeProfileIdentity, not {type(self.runtime_profile_identity).__name__!r}"
            )
        if not self.gpu_topology_key or not isinstance(self.gpu_topology_key, str):
            raise ValueError("EnvironmentRequestIdentity.gpu_topology_key must be a non-empty string")
        if isinstance(self.context_length, bool) or not isinstance(self.context_length, int) or self.context_length <= 0:
            raise ValueError("EnvironmentRequestIdentity.context_length must be a positive integer")

    def stable_key(self) -> str:
        return _stable_hash(
            self.model_identity.artifact_set_id,
            self.runtime_profile_identity.stable_key(),
            self.gpu_topology_key,
            self.context_length,
            self.kv_cache_policy,
            self.offload_split_policy,
        )


@dataclass(frozen=True)
class EnvironmentEvidence:
    """One typed piece of evidence about whether ``request_identity``
    fits. Callers construct these from whatever underlying measurement
    or estimator produced them -- this module does not measure anything
    itself (see the module docstring for the existing ``runner.py``
    machinery this is designed to eventually sit on top of)."""

    tier: EnvironmentEvidenceTier
    fit: EnvironmentFit
    request_identity: EnvironmentRequestIdentity
    detail: str = ""
    source: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tier, EnvironmentEvidenceTier):
            raise TypeError(
                f"EnvironmentEvidence.tier must be an EnvironmentEvidenceTier, "
                f"not {type(self.tier).__name__!r}"
            )
        if self.tier == EnvironmentEvidenceTier.CONSERVATIVE_UNKNOWN:
            raise ValueError(
                "EnvironmentEvidenceTier.CONSERVATIVE_UNKNOWN is the fail-closed default "
                "decide_environment_fit() falls back to on its own -- it is never a tier "
                "real evidence is constructed for"
            )
        if not isinstance(self.fit, EnvironmentFit):
            raise TypeError(
                f"EnvironmentEvidence.fit must be an EnvironmentFit, not {type(self.fit).__name__!r}"
            )
        if not isinstance(self.request_identity, EnvironmentRequestIdentity):
            raise TypeError(
                "EnvironmentEvidence.request_identity must be a real EnvironmentRequestIdentity, "
                f"not {type(self.request_identity).__name__!r}"
            )


class EnvironmentDecisionReason(str, Enum):
    FITS = "fits"
    DOES_NOT_FIT = "does_not_fit"
    NO_MATCHING_EVIDENCE = "no_matching_evidence"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class EnvironmentDecision:
    request_identity: EnvironmentRequestIdentity
    fits: Optional[bool]
    reason: EnvironmentDecisionReason
    tier: EnvironmentEvidenceTier
    selected_evidence: Tuple[EnvironmentEvidence, ...] = ()
    contradicting_evidence: Tuple[EnvironmentEvidence, ...] = ()
    considered_evidence: Tuple[EnvironmentEvidence, ...] = ()


def decide_environment_fit(
    evidence: Iterable[EnvironmentEvidence],
    *,
    request_identity: EnvironmentRequestIdentity,
) -> EnvironmentDecision:
    """Decide whether ``request_identity`` fits, from whatever typed
    ``evidence`` is available. See the module docstring for the full
    precedence/contradiction-recording policy."""
    if not isinstance(request_identity, EnvironmentRequestIdentity):
        raise TypeError(
            "decide_environment_fit() requires a real EnvironmentRequestIdentity, "
            f"not {type(request_identity).__name__!r}"
        )

    requested_key = request_identity.stable_key()
    matching = []
    for item in evidence:
        if not isinstance(item, EnvironmentEvidence):
            raise TypeError(
                "decide_environment_fit() requires an iterable of real EnvironmentEvidence "
                f"instances, not {type(item).__name__!r}"
            )
        # stable_key(), not dataclass `==` -- RuntimeProfileIdentity's own
        # contract (identity.py) says compare/key on stable_key(), never
        # field-by-field equality, since e.g. feature_flags order must not
        # matter. Matching on `==` here would let two evidence records for
        # the "same" runtime profile spuriously miss each other.
        if item.request_identity.stable_key() == requested_key:
            matching.append(item)
    considered = tuple(matching)

    if not considered:
        return EnvironmentDecision(
            request_identity=request_identity,
            fits=None,
            reason=EnvironmentDecisionReason.NO_MATCHING_EVIDENCE,
            tier=EnvironmentEvidenceTier.CONSERVATIVE_UNKNOWN,
            considered_evidence=considered,
        )

    authoritative_tier: Optional[EnvironmentEvidenceTier] = None
    committing: list = []
    for tier in _EVIDENCE_TIER_PRECEDENCE:
        tier_committing = [item for item in matching if item.tier == tier and item.fit != EnvironmentFit.UNKNOWN]
        if tier_committing:
            authoritative_tier = tier
            committing = tier_committing
            break

    if authoritative_tier is None:
        return EnvironmentDecision(
            request_identity=request_identity,
            fits=None,
            reason=EnvironmentDecisionReason.INCONCLUSIVE,
            tier=EnvironmentEvidenceTier.CONSERVATIVE_UNKNOWN,
            considered_evidence=considered,
        )

    # Fail closed within the authoritative tier: if it disagrees with
    # itself, a "does not fit" claim outweighs a "fits" claim for the
    # exact same request identity.
    does_not_fit = [item for item in committing if item.fit == EnvironmentFit.DOES_NOT_FIT]
    fits_decision = not does_not_fit
    selected = tuple(committing) if fits_decision else tuple(does_not_fit)
    decided_fit = EnvironmentFit.FITS if fits_decision else EnvironmentFit.DOES_NOT_FIT

    contradicting = tuple(
        item
        for item in matching
        if item.fit != EnvironmentFit.UNKNOWN
        and item not in selected
        and item.fit != decided_fit
    )

    return EnvironmentDecision(
        request_identity=request_identity,
        fits=fits_decision,
        reason=EnvironmentDecisionReason.FITS if fits_decision else EnvironmentDecisionReason.DOES_NOT_FIT,
        tier=authoritative_tier,
        selected_evidence=selected,
        contradicting_evidence=contradicting,
        considered_evidence=considered,
    )
