"""Anvil Stage 3B.2 -- explicit write-time trust classification for fresh
native capability evidence.

Implements the owner's frozen rule (recorded in
``local_only/anvil/stage-3b.1a-source-map.md`` -- "OWNER DECISION RESOLVED"
append, and ``ANVIL_PROGRESS.md`` "STAGE 3B.1 INTEGRATION + OWNER DECISION
RESOLVED"):

    New capability evidence must receive an *explicit* ``EvidenceTrustClass``
    at write time. Trust is **never** inferred from freshness, timestamp,
    schema version, the presence of current-schema fields, or the fact that
    a probe just ran.

``CANONICAL_COMPATIBLE`` is assigned **only** when every condition of the
current supported ModelBench capability-probe contract is explicitly
demonstrated by the observation itself:

* the observation's ``probe_protocol_version`` matches the current
  supported probe protocol version;
* its ``capability_schema_version`` matches the current supported schema;
* it is bound to a real :class:`~llm_modelbench.identity.ModelArtifactIdentity`
  carrying content-addressed provenance the source actually provided --
  ``primary_sha256`` present (a name-hash-only ``artifact_set_id`` fails
  closed);
* it is bound to a real :class:`~llm_modelbench.identity.RuntimeProfileIdentity`
  with a non-empty ``backend``;
* the measured result is a *committing* measurement --
  ``MEASURED_SUPPORTED`` or ``MEASURED_UNSUPPORTED`` (identically: the
  trust class is the trust/comparability of the evidence, not the sign of
  the result). ``PROBE_INCONCLUSIVE`` / ``BACKEND_UNSUPPORTED`` /
  ``NOT_APPLICABLE`` never yield canonical;
* the caller reports no unresolved ambiguity/conflict.

Anything short of that fails **closed** to ``UNKNOWN_LEGACY``.

Scope note (verified against ``capability_evidence_adapter.py`` before
freezing, not assumed): the current probe path builds the observation's
``RuntimeProfileIdentity`` via
``identity.resolve_runtime_profile_identity(execution_settings=None)`` --
so ``backend_version`` / ``runtime_configuration_hash`` / ``gpu_policy``
are ``None`` by design at this stage (the rich resolved-recipe binding is
3B.2+ wiring / 3B.3-3B.4 evidence, per the 3B.1A/3B.1D debt placement).
Requiring those here would make *every* fresh probe ``UNKNOWN_LEGACY`` --
that is the rejected Alternative 2, not the frozen Alternative 1. The
contract therefore checks exactly what the *current* probe contract
demonstrates and no more.

This module performs no I/O. It does not read or write the ledger, run a
probe, or touch historical evidence. It is a pure classifier over one
already-constructed :class:`~llm_modelbench.capability_observation.CapabilityObservation`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .capabilities import (
    CAPABILITY_SCHEMA_VERSION,
    PROBE_PROTOCOL_VERSION,
    MeasuredCapabilityState,
)
from .evidence import EvidenceTrustClass
from .identity import ModelArtifactIdentity, RuntimeProfileIdentity

if TYPE_CHECKING:
    from .capability_observation import CapabilityObservation

__all__ = ["classify_fresh_capability_trust", "COMMITTING_MEASURED_STATES"]

# The two measured states that constitute a committing capability
# measurement. A canonical-compatible trust class describes the trust of
# the evidence, identically for a positive and a negative measured result
# (owner rule). The other three states are non-committing: no measurement
# was actually established, so the evidence cannot be canonical.
COMMITTING_MEASURED_STATES = frozenset(
    {
        MeasuredCapabilityState.MEASURED_SUPPORTED,
        MeasuredCapabilityState.MEASURED_UNSUPPORTED,
    }
)


def classify_fresh_capability_trust(
    observation: "CapabilityObservation",
    *,
    expected_probe_protocol_version: str = PROBE_PROTOCOL_VERSION,
    expected_capability_schema_version: int = CAPABILITY_SCHEMA_VERSION,
    unresolved_ambiguity: bool = False,
) -> EvidenceTrustClass:
    """Return the explicit :class:`EvidenceTrustClass` a freshly-written
    native capability observation must carry.

    Fails **closed** to :attr:`EvidenceTrustClass.UNKNOWN_LEGACY` unless the
    complete current probe contract is explicitly satisfied by
    ``observation`` (see the module docstring). Never infers trust from
    freshness or schema.
    """
    if observation.probe_protocol_version != expected_probe_protocol_version:
        return EvidenceTrustClass.UNKNOWN_LEGACY
    if observation.capability_schema_version != expected_capability_schema_version:
        return EvidenceTrustClass.UNKNOWN_LEGACY

    model_identity = observation.model_identity
    if not isinstance(model_identity, ModelArtifactIdentity):
        return EvidenceTrustClass.UNKNOWN_LEGACY
    primary_sha256 = model_identity.primary_sha256
    if not isinstance(primary_sha256, str) or not primary_sha256.strip():
        # Content-addressed provenance the source actually provided. A
        # name-derived artifact_set_id with no primary_sha256 is not proof
        # of which bytes ran -- fail closed.
        return EvidenceTrustClass.UNKNOWN_LEGACY

    runtime_identity = observation.runtime_profile_identity
    if not isinstance(runtime_identity, RuntimeProfileIdentity):
        return EvidenceTrustClass.UNKNOWN_LEGACY
    if not isinstance(runtime_identity.backend, str) or not runtime_identity.backend.strip():
        return EvidenceTrustClass.UNKNOWN_LEGACY

    if observation.result not in COMMITTING_MEASURED_STATES:
        return EvidenceTrustClass.UNKNOWN_LEGACY

    if unresolved_ambiguity:
        return EvidenceTrustClass.UNKNOWN_LEGACY

    return EvidenceTrustClass.CANONICAL_COMPATIBLE
