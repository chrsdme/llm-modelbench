"""Anvil Stage 3.0 -- BenchmarkProtocol and BenchmarkRuntimeBinding.

BenchmarkProtocol and RuntimeProfileIdentity are deliberately independent
identity axes (``codex-stage3-advice.txt`` part 2, "hard rule for 3.0"): the
same runtime configuration under two different benchmark-protocol versions
must not become two different runtime identities, and changing a runtime
configuration must not silently invalidate an unrelated protocol's identity.

``BenchmarkRuntimeBinding`` is the one-directional join between the two: it
may reference a ``RuntimeProfileIdentity`` (by its ``stable_key()``), but
``RuntimeProfileIdentity`` (``identity.py``) must never reference a
``BenchmarkProtocol`` -- that direction is what would let benchmark policy
identity leak into generic runtime-configuration identity.

Schema/type-contract freeze only. No migration, persistence wiring, or
consumer code is introduced in this stage -- see
``local_only/anvil/stage-3.0-schema-freeze.md``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .identity import ModelArtifactIdentity


def _stable_hash(*parts: Any) -> str:
    """Deterministic short hash over JSON-serializable parts. Mirrors
    identity.py's own helper -- never pass wall-clock/timing/random fields."""
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class BenchmarkProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkProtocol:
    """What must stay fixed for two benchmark runs to be comparable.

    Immutable once published: a comparability-relevant policy change (e.g.
    reweighting sampling policy, bumping a scorer version) is a new
    ``version``, never an in-place edit -- mirrors ``RuntimeProfile``'s own
    revision rule (identity-bearing edits create a new identity, not a
    silent mutation of the old one).
    """

    protocol_id: str
    version: str
    task_ids: Tuple[str, ...]
    prompt_semantics_hash: str
    sampling_policy_hash: str
    output_budget_policy_hash: str
    scorer_versions: Tuple[Tuple[str, str], ...]
    allowed_adaptations: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.protocol_id:
            raise BenchmarkProtocolError("protocol_id must be a non-empty string")
        if not self.version:
            raise BenchmarkProtocolError("version must be a non-empty string")
        if not self.task_ids:
            raise BenchmarkProtocolError("task_ids must be non-empty")
        object.__setattr__(self, "task_ids", tuple(dict.fromkeys(self.task_ids)))
        object.__setattr__(self, "scorer_versions", tuple(sorted(self.scorer_versions)))
        object.__setattr__(self, "allowed_adaptations", tuple(dict.fromkeys(self.allowed_adaptations)))
        scored_tasks = {task_id for task_id, _ in self.scorer_versions}
        missing = set(self.task_ids) - scored_tasks
        if missing:
            raise BenchmarkProtocolError(f"tasks missing a scorer version: {sorted(missing)}")

    def identity_key(self) -> str:
        """Stable identity for this exact protocol version. Two
        ``BenchmarkProtocol`` values sharing a ``protocol_id`` but differing
        in ``version`` are, by design, different identities: comparability
        policy changes are new versions, not edits to an existing identity.
        """
        return _stable_hash(
            self.protocol_id, self.version, self.task_ids,
            self.prompt_semantics_hash, self.sampling_policy_hash,
            self.output_budget_policy_hash, self.scorer_versions,
            self.allowed_adaptations,
        )


@dataclass(frozen=True)
class BenchmarkRuntimeBinding:
    """Answers "how does this particular runtime profile implement this
    particular benchmark protocol, for this model?"

    References other identities by their stable key strings rather than
    embedding the objects themselves -- consistent with this codebase's
    existing provenance-by-reference pattern (``evidence.ProvenanceLink``
    references records by ID, not by embedding them). One-directional: this
    type may point at a ``RuntimeProfileIdentity`` key; nothing about a
    ``RuntimeProfileIdentity`` ever points back here, and this type's own
    ``binding_key()`` never feeds into ``RuntimeProfileIdentity.stable_key()``.
    """

    model_artifact_identity: ModelArtifactIdentity
    benchmark_protocol_identity_key: str
    runtime_profile_identity_key: str
    allowed_adaptations_used: Tuple[str, ...] = ()
    provenance: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_artifact_identity, ModelArtifactIdentity):
            raise BenchmarkProtocolError("model_artifact_identity must be a ModelArtifactIdentity")
        if not self.benchmark_protocol_identity_key:
            raise BenchmarkProtocolError("benchmark_protocol_identity_key is required")
        if not self.runtime_profile_identity_key:
            raise BenchmarkProtocolError("runtime_profile_identity_key is required")
        object.__setattr__(
            self, "allowed_adaptations_used", tuple(dict.fromkeys(self.allowed_adaptations_used))
        )

    def binding_key(self) -> str:
        return _stable_hash(
            self.model_artifact_identity.artifact_set_id,
            self.benchmark_protocol_identity_key,
            self.runtime_profile_identity_key,
            self.allowed_adaptations_used,
        )
