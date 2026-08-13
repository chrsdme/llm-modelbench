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

**Identity-material canonicalization (``codex-stage3-advice.txt`` part 3's
fixup)**: ``allowed_adaptations`` and ``allowed_adaptations_used`` are
*sets* of permitted/used adaptations -- insertion order carries no meaning,
so both are sorted before hashing; two protocol/binding values built from
the same adaptations in different insertion order must produce identical
identities. ``scorer_versions`` must name exactly one scorer version per
task in ``task_ids`` -- no missing tasks, no unknown extra tasks, no
duplicate entries for the same task. ``task_ids`` itself is treated as an
**ordered execution sequence, not a set** -- a deliberate decision, not an
oversight: Anvil's own run model treats benchmark execution order as
potentially thermal/runtime-state-bearing (models are "kept warm between
tests," per ``ANVIL_MASTER_PLAN.md`` Stage 4), so two protocols that run the
same tasks in a different order are not guaranteed to be comparable and
therefore must not collapse to the same identity.

Schema/type-contract freeze only. No migration, persistence wiring, or
consumer code is introduced in this stage -- see
``local_only/anvil/stage-3.0-schema-freeze.md``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
        # task_ids is an ordered execution sequence (identity-bearing order)
        # -- deduplicated, but never reordered.
        object.__setattr__(self, "task_ids", tuple(dict.fromkeys(self.task_ids)))
        object.__setattr__(self, "scorer_versions", _canonical_scorer_versions(
            self.task_ids, self.scorer_versions
        ))
        # allowed_adaptations is a set -- canonicalize away insertion order.
        object.__setattr__(self, "allowed_adaptations", tuple(sorted(set(self.allowed_adaptations))))

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


def _canonical_scorer_versions(
    task_ids: Tuple[str, ...], scorer_versions: Tuple[Tuple[str, str], ...]
) -> Tuple[Tuple[str, str], ...]:
    """Exactly one scorer-version entry per task in ``task_ids`` -- no
    missing tasks, no unknown extra tasks, no duplicate entries for the same
    task. Duplicate detection happens on the raw input, before any
    conversion, so a silently-overwritten duplicate can never slip through."""
    scorer_map: Dict[str, str] = {}
    for task_id, scorer_version in scorer_versions:
        if task_id in scorer_map:
            raise BenchmarkProtocolError(f"duplicate scorer_versions entry for task_id: {task_id}")
        scorer_map[task_id] = scorer_version
    task_id_set = set(task_ids)
    missing = task_id_set - set(scorer_map)
    if missing:
        raise BenchmarkProtocolError(f"tasks missing a scorer version: {sorted(missing)}")
    extra = set(scorer_map) - task_id_set
    if extra:
        raise BenchmarkProtocolError(f"scorer_versions has entries for unknown task_ids: {sorted(extra)}")
    return tuple((task_id, scorer_map[task_id]) for task_id in task_ids)


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

    Prefer constructing this via :func:`bind_runtime_to_protocol`, which
    validates ``allowed_adaptations_used`` against the actual protocol
    before producing the binding. Direct construction (e.g. when
    reconstructing a previously-validated binding from persisted data) does
    not re-verify that subset relationship -- callers resolving a binding
    from storage must re-run the equivalent check against the resolved
    protocol themselves.
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
        # allowed_adaptations_used is a set -- canonicalize away insertion order.
        object.__setattr__(self, "allowed_adaptations_used", tuple(sorted(set(self.allowed_adaptations_used))))

    def binding_key(self) -> str:
        return _stable_hash(
            self.model_artifact_identity.artifact_set_id,
            self.benchmark_protocol_identity_key,
            self.runtime_profile_identity_key,
            self.allowed_adaptations_used,
        )


def bind_runtime_to_protocol(
    protocol: BenchmarkProtocol,
    *,
    model_artifact_identity: ModelArtifactIdentity,
    runtime_profile_identity_key: str,
    allowed_adaptations_used: Tuple[str, ...] = (),
    provenance: Optional[str] = None,
) -> BenchmarkRuntimeBinding:
    """The sanctioned way to construct a ``BenchmarkRuntimeBinding`` that
    claims specific adaptations were used: validates that
    ``allowed_adaptations_used`` is a subset of ``protocol.allowed_adaptations``
    before constructing the binding, so a binding can never claim an
    adaptation its own referenced protocol does not permit. Deliberately
    does not store the whole ``protocol`` object on the binding -- the
    binding stays reference-oriented (``benchmark_protocol_identity_key``
    only); this function is where the two are actually checked together.
    """
    used = set(allowed_adaptations_used)
    permitted = set(protocol.allowed_adaptations)
    if not used <= permitted:
        raise BenchmarkProtocolError(
            f"adaptations not permitted by protocol {protocol.protocol_id}@{protocol.version}: "
            f"{sorted(used - permitted)}"
        )
    return BenchmarkRuntimeBinding(
        model_artifact_identity=model_artifact_identity,
        benchmark_protocol_identity_key=protocol.identity_key(),
        runtime_profile_identity_key=runtime_profile_identity_key,
        allowed_adaptations_used=allowed_adaptations_used,
        provenance=provenance,
    )
