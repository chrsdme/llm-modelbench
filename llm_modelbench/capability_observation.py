"""Anvil Stage 2.1 -- CapabilityObservation: an append-only evidence type
for a single functional capability probe, plus its persistence semantics.

Scope was deliberately narrow at introduction, per the second Codex/GPT
pre-Stage-2 review (``local_only/anvil/codex-advice_pre_stage2.txt``, Part
2): this module introduces the evidence type and its persistence, and
nothing else. It does NOT itself define ``CapabilityProjection`` or
``TaskApplicability``, and did not migrate the planner/runner/recovery/
judge call sites on its own -- those stayed on ``capabilities.py``
(``current_capability_identity`` / ``capability_identity_compatibility``)
until later Stage 2 slices deliberately migrated them.

**Status update (Anvil Stage 2.6E, correcting the claim above, which is
stale as originally written)**: ``CapabilityObservation`` (this module) is
now the typed evidence type that flows through the whole migrated
authority path (``capability_evidence_adapter`` ->
``capability_projection`` -> the four real command paths, Stages
2.6A-D) -- not "nothing here is wired into any command path yet." See
``local_only/anvil/stage-2.6E-authority-audit.md`` for the full audit.

Two more explicit non-goals for this slice, both from the same review:

- No function anywhere in this module accepts a legacy/declared-only
  profile dict and produces a ``CapabilityObservation``. The dataclass's
  own ``__post_init__`` enforces this structurally (real
  :class:`~llm_modelbench.identity.ModelArtifactIdentity` /
  :class:`~llm_modelbench.identity.RuntimeProfileIdentity` instances are
  required, not any mapping that merely looks like one) -- the existing
  1,158 legacy/unbound stored profiles are not silently blessed into this
  schema. They remain historical until the later explicit reprobe work
  (Stage 2.4/2.7).
- The known shape inconsistency between the two ``repair.py`` fail-closed
  paths found in Stage 2.0 (a present-but-unbound profile produces the
  labelled reason ``legacy_or_unbound_capability_profile``; a wholly
  missing profile produces an unlabelled ``{}``) is deliberately NOT
  fixed here. Stage 2.0 froze both shapes as regression tests; unifying
  them is left for the later slice that introduces typed capability
  decisions end-to-end, where it can be an intentional, tested semantic
  migration rather than an opportunistic fix bundled into the schema
  introduction.

Result values reuse :class:`~llm_modelbench.capabilities.MeasuredCapabilityState`
rather than a second, near-duplicate enum -- the two already mean the same
five things (measured_supported / measured_unsupported / probe_inconclusive
/ backend_unsupported / not_applicable), and a second enum would only
create ambiguity about which one the later call-site migration (Stage 2.6)
is supposed to target. This is a deliberate API-surface choice, recorded
here so it isn't silently unpicked by a later slice.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple

from .capabilities import MeasuredCapabilityState
from .classify import FAMILY_ORDER
from .evidence import EvidenceLedger, LedgerRecord, ProvenanceLink

if TYPE_CHECKING:
    from .identity import ModelArtifactIdentity, RuntimeProfileIdentity

CAPABILITY_OBSERVATION_RECORD_TYPE = "capability_observation"


def _canonical_hash(value: Any) -> str:
    """Deterministic over semantic content only: ``sort_keys=True`` and
    compact separators mean JSON key order and pretty-printing never
    change the result -- only field values do."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CapabilityObservation:
    """One append-only fact: this capability, under this exact model
    artifact and runtime profile, produced this measured result at this
    time. Immutable from construction -- a reprobe creates a new
    ``CapabilityObservation``, it never edits this one. Which observation
    is currently authoritative for a given identity is a later concern
    (``CapabilityProjection``, Stage 2.3+), deliberately not decided here.

    ``observation_id`` and ``evidence_hash`` are deliberately distinct:
    ``evidence_hash`` covers only the semantic measurement (identity +
    capability + result), excluding ``timestamp`` and ``declared_hint`` --
    two probes of the same capability under identical conditions produce
    the same ``evidence_hash`` even if run minutes apart. ``observation_id``
    additionally folds in ``timestamp``, so two such probes still get
    distinct ledger records and genuinely coexist (see
    :meth:`to_ledger_payload` and :meth:`EvidenceLedger.append`'s own
    content-addressed idempotency, which would otherwise silently collapse
    two identical-payload appends into one).
    """

    model_identity: "ModelArtifactIdentity"
    runtime_profile_identity: "RuntimeProfileIdentity"
    capability: str
    result: MeasuredCapabilityState
    probe_protocol_version: str
    capability_schema_version: int
    template_config_hash: Optional[str] = None
    endpoint_identity: Optional[str] = None
    declared_hint: Tuple[str, ...] = ()
    timestamp: Optional[str] = None
    observation_id: str = field(init=False)
    evidence_hash: str = field(init=False)

    def __post_init__(self) -> None:
        # Import locally to avoid a hard module-level dependency cycle
        # risk (identity.py currently has none on this module, but this
        # keeps the direction explicit rather than assumed).
        from .identity import ModelArtifactIdentity, RuntimeProfileIdentity

        if not isinstance(self.model_identity, ModelArtifactIdentity):
            raise TypeError(
                "CapabilityObservation.model_identity must be a real ModelArtifactIdentity, "
                f"not {type(self.model_identity).__name__!r} -- a legacy/declared-only profile "
                "dict cannot be substituted here by design"
            )
        if not isinstance(self.runtime_profile_identity, RuntimeProfileIdentity):
            raise TypeError(
                "CapabilityObservation.runtime_profile_identity must be a real "
                f"RuntimeProfileIdentity, not {type(self.runtime_profile_identity).__name__!r}"
            )
        if not isinstance(self.result, MeasuredCapabilityState):
            raise TypeError(
                f"CapabilityObservation.result must be a MeasuredCapabilityState, "
                f"not {type(self.result).__name__!r} -- pass the enum member, not a raw string"
            )
        if not self.capability or not isinstance(self.capability, str):
            raise ValueError("CapabilityObservation.capability must be a non-empty string")
        if self.capability not in FAMILY_ORDER:
            raise ValueError(
                f"CapabilityObservation.capability {self.capability!r} is not a known family "
                f"(expected one of {FAMILY_ORDER!r})"
            )
        object.__setattr__(self, "declared_hint", tuple(self.declared_hint))
        object.__setattr__(self, "timestamp", self.timestamp or _now_iso())
        object.__setattr__(self, "evidence_hash", _canonical_hash(self._semantic_content()))
        object.__setattr__(self, "observation_id", _canonical_hash({**self._semantic_content(), "timestamp": self.timestamp}))

    def _semantic_content(self) -> Dict[str, Any]:
        """Everything that affects compatibility or the actual measurement.
        Excludes ``timestamp`` (when it was measured, not what was
        measured) and ``declared_hint`` (a non-authoritative hint field,
        never part of what the measurement itself asserts)."""
        return {
            "model_artifact_set_id": self.model_identity.artifact_set_id,
            "model_primary_sha256": self.model_identity.primary_sha256,
            "runtime_profile_stable_key": self.runtime_profile_identity.stable_key(),
            "template_config_hash": self.template_config_hash,
            "endpoint_identity": self.endpoint_identity,
            "capability": self.capability,
            "result": self.result.value,
            "probe_protocol_version": self.probe_protocol_version,
            "capability_schema_version": self.capability_schema_version,
        }

    def to_ledger_payload(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "evidence_hash": self.evidence_hash,
            "model_identity": {
                "artifact_set_id": self.model_identity.artifact_set_id,
                "primary_sha256": self.model_identity.primary_sha256,
                "auxiliary_artifact_hashes": list(self.model_identity.auxiliary_artifact_hashes),
                "size_bytes": self.model_identity.size_bytes,
                "format": self.model_identity.format,
                "quantization": self.model_identity.quantization,
                "source": self.model_identity.source,
            },
            "runtime_profile_identity": {
                "backend": self.runtime_profile_identity.backend,
                "backend_version": self.runtime_profile_identity.backend_version,
                "protocol_version": self.runtime_profile_identity.protocol_version,
                "template_hash": self.runtime_profile_identity.template_hash,
                "runtime_configuration_hash": self.runtime_profile_identity.runtime_configuration_hash,
                "gpu_policy": self.runtime_profile_identity.gpu_policy,
                "feature_flags": list(self.runtime_profile_identity.feature_flags),
                "stable_key": self.runtime_profile_identity.stable_key(),
            },
            "template_config_hash": self.template_config_hash,
            "endpoint_identity": self.endpoint_identity,
            "probe_protocol_version": self.probe_protocol_version,
            "capability_schema_version": self.capability_schema_version,
            "capability": self.capability,
            "result": self.result.value,
            "declared_hint": list(self.declared_hint),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_ledger_payload(cls, payload: Mapping[str, Any]) -> "CapabilityObservation":
        from .identity import ModelArtifactIdentity, RuntimeProfileIdentity

        model = dict(payload["model_identity"])
        runtime = dict(payload["runtime_profile_identity"])
        observation = cls(
            model_identity=ModelArtifactIdentity(
                artifact_set_id=model["artifact_set_id"],
                primary_sha256=model.get("primary_sha256"),
                auxiliary_artifact_hashes=tuple(model.get("auxiliary_artifact_hashes") or ()),
                size_bytes=model.get("size_bytes"),
                format=model.get("format", "unknown"),
                quantization=model.get("quantization"),
                source=model.get("source"),
            ),
            runtime_profile_identity=RuntimeProfileIdentity(
                backend=runtime["backend"],
                backend_version=runtime.get("backend_version"),
                protocol_version=runtime.get("protocol_version"),
                template_hash=runtime.get("template_hash"),
                runtime_configuration_hash=runtime.get("runtime_configuration_hash"),
                gpu_policy=runtime.get("gpu_policy"),
                feature_flags=tuple(runtime.get("feature_flags") or ()),
            ),
            capability=payload["capability"],
            result=MeasuredCapabilityState(payload["result"]),
            probe_protocol_version=payload["probe_protocol_version"],
            capability_schema_version=payload["capability_schema_version"],
            template_config_hash=payload.get("template_config_hash"),
            endpoint_identity=payload.get("endpoint_identity"),
            declared_hint=tuple(payload.get("declared_hint") or ()),
            timestamp=payload.get("timestamp"),
        )
        if observation.observation_id != payload.get("observation_id"):
            raise ValueError(
                "reconstructed observation_id does not match stored payload -- "
                "the ledger record has been tampered with or is corrupt"
            )
        if observation.evidence_hash != payload.get("evidence_hash"):
            raise ValueError(
                "reconstructed evidence_hash does not match stored payload -- "
                "the ledger record has been tampered with or is corrupt"
            )
        return observation


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append_capability_observation(
    ledger: EvidenceLedger,
    observation: CapabilityObservation,
    *,
    provenance: Tuple[ProvenanceLink, ...] = (),
) -> LedgerRecord:
    """Thin wrapper over ``EvidenceLedger.append`` fixing the record type
    and payload shape. Kept as a function rather than a method on
    ``CapabilityObservation`` -- the observation itself has no knowledge of
    ledgers, matching Stage 0's evidence-model separation of typed facts
    from their storage."""
    return ledger.append(
        CAPABILITY_OBSERVATION_RECORD_TYPE,
        observation.to_ledger_payload(),
        provenance=provenance,
        record_id=observation.observation_id,
    )
