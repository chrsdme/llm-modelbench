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

**Verified vs. unresolved binding material (``codex-stage3-advice.txt``
part 3's micro-fix)**: ``BenchmarkRuntimeBinding`` means "this binding's
``allowed_adaptations_used`` has been checked against its referenced
protocol's ``allowed_adaptations``" -- never "possibly validated." It is
constructible only via :func:`bind_runtime_to_protocol` or
:func:`resolve_binding`; direct construction raises. Raw, not-yet-checked
binding material (e.g. deserialized from persisted data before its
protocol has been resolved) is a structurally distinct type,
:class:`UnresolvedBenchmarkRuntimeBinding`, freely constructible and never
treated by any code path as if it were validated. Promotion from unresolved
to validated happens only through :func:`resolve_binding`, which re-runs
the same subset check against the actually-resolved protocol.

**Identity-material canonicalization (part 3's earlier fixup, unchanged
here)**: ``allowed_adaptations`` and ``allowed_adaptations_used`` are
*sets* of permitted/used adaptations -- insertion order carries no meaning,
so both are sorted before hashing. ``scorer_versions`` must name exactly
one scorer version per task in ``task_ids`` -- no missing tasks, no unknown
extra tasks, no duplicate entries for the same task. ``task_ids`` itself is
treated as an **ordered execution sequence, not a set** -- a deliberate
decision: Anvil's own run model treats benchmark execution order as
potentially thermal/runtime-state-bearing (models are "kept warm between
tests," per ``ANVIL_MASTER_PLAN.md`` Stage 4), so two protocols that run
the same tasks in a different order are not guaranteed to be comparable and
therefore must not collapse to the same identity.

**Duplicate task_ids fail closed (this micro-fix)**: since order is
identity-bearing, a duplicate task ID is not silently collapsed the way an
unordered set's duplicates would be -- ``("t1", "t1", "t2")`` is rejected
outright rather than silently rewritten to ``("t1", "t2")``. If repeated
execution of the same task is ever deliberately supported as protocol
semantics, that will need its own explicit representation; today's
architecture does not require it, so rejection is the safer default.

Schema/type-contract freeze only. No migration, persistence wiring, or
consumer code is introduced in this stage -- see
``local_only/anvil/stage-3.0-schema-freeze.md``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
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
        # -- duplicates are rejected outright, never silently collapsed.
        task_ids = tuple(self.task_ids)
        seen: set = set()
        duplicates: list = []
        for task_id in task_ids:
            if task_id in seen:
                duplicates.append(task_id)
            else:
                seen.add(task_id)
        if duplicates:
            raise BenchmarkProtocolError(f"duplicate task_ids: {sorted(set(duplicates))}")
        object.__setattr__(self, "task_ids", task_ids)
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


def _validate_binding_fields(
    model_artifact_identity: Any, benchmark_protocol_identity_key: str, runtime_profile_identity_key: str,
) -> None:
    if not isinstance(model_artifact_identity, ModelArtifactIdentity):
        raise BenchmarkProtocolError("model_artifact_identity must be a ModelArtifactIdentity")
    if not benchmark_protocol_identity_key:
        raise BenchmarkProtocolError("benchmark_protocol_identity_key is required")
    if not runtime_profile_identity_key:
        raise BenchmarkProtocolError("runtime_profile_identity_key is required")


@dataclass(frozen=True)
class UnresolvedBenchmarkRuntimeBinding:
    """Raw, not-yet-checked binding material -- e.g. deserialized from
    persisted data before its referenced :class:`BenchmarkProtocol` has been
    resolved and re-checked. Freely constructible; structurally distinct
    from :class:`BenchmarkRuntimeBinding`, so no code path can mistake this
    for a validated binding. Promote via :func:`resolve_binding`, which
    re-runs the ``allowed_adaptations_used`` subset check against the
    actually-resolved protocol before producing a validated
    ``BenchmarkRuntimeBinding``.
    """

    model_artifact_identity: ModelArtifactIdentity
    benchmark_protocol_identity_key: str
    runtime_profile_identity_key: str
    allowed_adaptations_used: Tuple[str, ...] = ()
    provenance: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_binding_fields(
            self.model_artifact_identity, self.benchmark_protocol_identity_key,
            self.runtime_profile_identity_key,
        )
        object.__setattr__(self, "allowed_adaptations_used", tuple(sorted(set(self.allowed_adaptations_used))))


class _VerifiedBindingMarker:
    """Private sentinel gating :class:`BenchmarkRuntimeBinding` construction
    -- only :func:`bind_runtime_to_protocol` and :func:`resolve_binding` may
    pass it. Not a defense against a deliberately adversarial caller within
    the same trust boundary (Python has no true private constructors); it
    exists to prevent *accidental* direct construction of a type whose
    entire meaning is "this was checked against its protocol"."""


_VERIFIED = _VerifiedBindingMarker()


@dataclass(frozen=True)
class BenchmarkRuntimeBinding:
    """Answers "how does this particular runtime profile implement this
    particular benchmark protocol, for this model?" -- and, unlike
    :class:`UnresolvedBenchmarkRuntimeBinding`, does so *verifiably*:
    ``allowed_adaptations_used`` has been checked as a subset of the
    referenced protocol's ``allowed_adaptations`` at construction time.

    References other identities by their stable key strings rather than
    embedding the objects themselves -- consistent with this codebase's
    existing provenance-by-reference pattern (``evidence.ProvenanceLink``
    references records by ID, not by embedding them). One-directional: this
    type may point at a ``RuntimeProfileIdentity`` key; nothing about a
    ``RuntimeProfileIdentity`` ever points back here, and this type's own
    ``binding_key()`` never feeds into ``RuntimeProfileIdentity.stable_key()``.

    **Constructible only via** :func:`bind_runtime_to_protocol` **or**
    :func:`resolve_binding` **-- direct construction raises.** This is the
    structural guarantee that a consumer can never mistake unresolved
    binding material for one actually verified against its protocol.
    """

    model_artifact_identity: ModelArtifactIdentity
    benchmark_protocol_identity_key: str
    runtime_profile_identity_key: str
    allowed_adaptations_used: Tuple[str, ...] = ()
    provenance: Optional[str] = None
    _verified_marker: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._verified_marker is not _VERIFIED:
            raise BenchmarkProtocolError(
                "BenchmarkRuntimeBinding must be constructed via bind_runtime_to_protocol() "
                "or resolve_binding() -- direct construction is not permitted"
            )
        _validate_binding_fields(
            self.model_artifact_identity, self.benchmark_protocol_identity_key,
            self.runtime_profile_identity_key,
        )
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
    """Constructs a validated ``BenchmarkRuntimeBinding`` fresh, checking
    that ``allowed_adaptations_used`` is a subset of ``protocol.
    allowed_adaptations`` before construction. Deliberately does not store
    the whole ``protocol`` object on the binding -- the binding stays
    reference-oriented (``benchmark_protocol_identity_key`` only); this
    function is where the two are actually checked together.
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
        _verified_marker=_VERIFIED,
    )


def resolve_binding(
    material: UnresolvedBenchmarkRuntimeBinding, protocol: BenchmarkProtocol,
) -> BenchmarkRuntimeBinding:
    """Promotes previously-persisted/deserialized raw binding material to a
    validated ``BenchmarkRuntimeBinding``, after checking it against the
    actually-resolved protocol it claims to reference. This is the required
    path from unresolved to validated -- there is no other way to obtain a
    ``BenchmarkRuntimeBinding`` from an ``UnresolvedBenchmarkRuntimeBinding``.

    Raises if ``material``'s ``benchmark_protocol_identity_key`` does not
    match ``protocol.identity_key()`` (the material claims to reference a
    different protocol than the one supplied), or if its
    ``allowed_adaptations_used`` is not a subset of what ``protocol``
    actually permits.
    """
    if material.benchmark_protocol_identity_key != protocol.identity_key():
        raise BenchmarkProtocolError(
            "unresolved binding's benchmark_protocol_identity_key does not match the given protocol"
        )
    used = set(material.allowed_adaptations_used)
    permitted = set(protocol.allowed_adaptations)
    if not used <= permitted:
        raise BenchmarkProtocolError(
            f"adaptations not permitted by protocol {protocol.protocol_id}@{protocol.version}: "
            f"{sorted(used - permitted)}"
        )
    return BenchmarkRuntimeBinding(
        model_artifact_identity=material.model_artifact_identity,
        benchmark_protocol_identity_key=material.benchmark_protocol_identity_key,
        runtime_profile_identity_key=material.runtime_profile_identity_key,
        allowed_adaptations_used=material.allowed_adaptations_used,
        provenance=material.provenance,
        _verified_marker=_VERIFIED,
    )
