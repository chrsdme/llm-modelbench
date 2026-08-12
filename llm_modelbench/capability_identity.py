"""Anvil Stage 2.2 -- runtime/model identity compatibility contract for
:class:`~llm_modelbench.capability_observation.CapabilityObservation`.

Scope is deliberately narrow, per this session's Codex/GPT scoping pass
(``local_only/anvil/codex-advice_stage2.2.txt``): this module decides
whether a stored ``CapabilityObservation`` is identity-compatible with the
currently running model/runtime identity, and nothing else. Purely additive
-- like Stage 2.1's ``CapabilityObservation`` before it, nothing here is
wired into ``planner.py``, ``runner.py``, ``campaign.py``, ``repair.py``,
reprobe logic, or judge qualification yet. The existing
``capabilities.capability_identity_compatibility()`` (dict-in, dict-out)
remains the operationally authoritative function for every real command
path until a later slice deliberately migrates call sites (Stage 2.6).

Explicit non-goals for this slice (all later Stage 2 work, not this one):
- No ``CapabilityProjection`` -- selecting which observation is currently
  authoritative among several is Stage 2.3.
- No ``CapabilityDecision`` / task-routing / measured-family applicability
  decisions -- Stage 2.3/2.5.
- No ``EnvironmentDecision``, GPU/runtime reuse policy, endpoint recovery
  policy, or live-instance suitability -- Stage 2.4.
- No ``TaskApplicability`` rewrite or task-gating changes -- Stage 2.5.
- No planner/runner/recovery/judge call-site migration -- Stage 2.6.
- No blessing legacy/unbound profiles into typed evidence, no auto-convert
  of stored dict profiles, no fleet reprobe -- Stage 2.7.

This module does not call the legacy ``capability_identity_compatibility()``
and does not depend on its dict-path shape (``("model", "digest")`` etc.) --
the two are related in vocabulary only, not by implementation.

Known open question, deliberately left unresolved here (flagged by the
Stage 2.2 scoping pass for a later slice to pick up): ``RuntimeProfileIdentity``
is intentionally restart-stable, while ``CapabilityObservation.endpoint_identity``
is a separate, more volatile field. Whether an ``endpoint_identity`` change
alone should invalidate capability evidence is a policy question that
belongs to Stage 2.4's environment/runtime-reuse decision, not to this
identity-comparison contract -- this module simply reports the mismatch as
``ENDPOINT_CHANGED`` and lets a later slice decide what to do with it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from .capability_observation import CapabilityObservation

if TYPE_CHECKING:
    from .identity import ModelArtifactIdentity, RuntimeProfileIdentity


class CapabilityIdentityCompatibilityReason(str, Enum):
    IDENTITY_MATCH = "identity_match"
    CURRENT_IDENTITY_MISSING = "current_identity_missing"
    CAPABILITY_SCHEMA_VERSION_CHANGED = "capability_schema_version_changed"
    PROBE_PROTOCOL_VERSION_CHANGED = "probe_protocol_version_changed"
    MODEL_ARTIFACT_CHANGED = "model_artifact_changed"
    MODEL_PRIMARY_HASH_CHANGED = "model_primary_hash_changed"
    RUNTIME_PROFILE_CHANGED = "runtime_profile_changed"
    TEMPLATE_CONFIG_CHANGED = "template_config_changed"
    ENDPOINT_CHANGED = "endpoint_changed"


@dataclass(frozen=True)
class CapabilityIdentityCompatibility:
    """The typed compatibility verdict for one observation against one
    current identity snapshot. ``stored_evidence_hash`` always identifies
    which observation the verdict is about, whether compatible or not --
    it is not conditioned on the outcome."""

    compatible: bool
    reason: CapabilityIdentityCompatibilityReason
    stored_evidence_hash: Optional[str] = None


def capability_observation_identity_compatibility(
    observation: CapabilityObservation,
    *,
    current_model_identity: Optional["ModelArtifactIdentity"],
    current_runtime_profile_identity: Optional["RuntimeProfileIdentity"],
    current_probe_protocol_version: str,
    current_capability_schema_version: int,
    current_template_config_hash: Optional[str] = None,
    current_endpoint_identity: Optional[str] = None,
) -> CapabilityIdentityCompatibility:
    """Decide whether ``observation`` is identity-compatible with the
    current model/runtime snapshot described by the ``current_*`` keyword
    arguments. Checks run in a fixed precedence order and return on the
    first mismatch found -- see the module-level check list in
    ``local_only/anvil/codex-advice_stage2.2.txt`` for why this order
    matters (e.g. a schema-version bump takes precedence over a model
    change, since the schema change makes the comparison itself suspect)."""
    from .identity import ModelArtifactIdentity, RuntimeProfileIdentity

    if not isinstance(observation, CapabilityObservation):
        raise TypeError(
            "capability_observation_identity_compatibility() requires a real "
            f"CapabilityObservation, not {type(observation).__name__!r}"
        )
    if current_model_identity is not None and not isinstance(current_model_identity, ModelArtifactIdentity):
        raise TypeError(
            "current_model_identity must be a real ModelArtifactIdentity or None, "
            f"not {type(current_model_identity).__name__!r} -- a legacy identity dict "
            "cannot be substituted here by design"
        )
    if current_runtime_profile_identity is not None and not isinstance(
        current_runtime_profile_identity, RuntimeProfileIdentity
    ):
        raise TypeError(
            "current_runtime_profile_identity must be a real RuntimeProfileIdentity or "
            f"None, not {type(current_runtime_profile_identity).__name__!r}"
        )

    evidence_hash = observation.evidence_hash

    def _result(
        compatible: bool, reason: CapabilityIdentityCompatibilityReason
    ) -> CapabilityIdentityCompatibility:
        return CapabilityIdentityCompatibility(
            compatible=compatible, reason=reason, stored_evidence_hash=evidence_hash
        )

    if current_model_identity is None or current_runtime_profile_identity is None:
        return _result(False, CapabilityIdentityCompatibilityReason.CURRENT_IDENTITY_MISSING)

    if observation.capability_schema_version != current_capability_schema_version:
        return _result(False, CapabilityIdentityCompatibilityReason.CAPABILITY_SCHEMA_VERSION_CHANGED)

    if observation.probe_protocol_version != current_probe_protocol_version:
        return _result(False, CapabilityIdentityCompatibilityReason.PROBE_PROTOCOL_VERSION_CHANGED)

    if observation.model_identity.artifact_set_id != current_model_identity.artifact_set_id:
        return _result(False, CapabilityIdentityCompatibilityReason.MODEL_ARTIFACT_CHANGED)

    stored_hash = observation.model_identity.primary_sha256
    current_hash = current_model_identity.primary_sha256
    if stored_hash is not None and current_hash is not None and stored_hash != current_hash:
        return _result(False, CapabilityIdentityCompatibilityReason.MODEL_PRIMARY_HASH_CHANGED)

    if observation.runtime_profile_identity.stable_key() != current_runtime_profile_identity.stable_key():
        return _result(False, CapabilityIdentityCompatibilityReason.RUNTIME_PROFILE_CHANGED)

    if observation.template_config_hash != current_template_config_hash:
        return _result(False, CapabilityIdentityCompatibilityReason.TEMPLATE_CONFIG_CHANGED)

    if observation.endpoint_identity != current_endpoint_identity:
        return _result(False, CapabilityIdentityCompatibilityReason.ENDPOINT_CHANGED)

    return _result(True, CapabilityIdentityCompatibilityReason.IDENTITY_MATCH)
