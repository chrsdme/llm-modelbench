"""Anvil Stage 3.0 -- scoped runtime-profile validation and the four
master-plan-required profile roles per model: ``validated_runtime_profiles``,
``benchmark_canonical_profile``, ``best_observed_profile``,
``recommended_production_profile`` (``ANVIL_MASTER_PLAN.md``, Stage 3
section).

Validation is deliberately never a bare boolean (``codex-stage3-advice.txt``
part 2, "define exactly what 'validated runtime profile' means"): a
successful tool-capability probe proves the profile ran that probe
successfully, not that it is a valid 64k benchmark profile, or validated for
vision, or suitable as the canonical benchmark profile. ``ValidationKind``
records what was actually checked -- "validated somewhere" is never treated
as "canonical for everything". Both ``validation_kind`` and ``status`` are
runtime-checked against their enums in ``__post_init__`` (part 3's fixup: a
plain type annotation does not stop a raw string from bypassing the scope
checks below, which compare with ``is``, not ``==``/``in``).

``benchmark_canonical_profile``, ``best_observed_profile``, and
``recommended_production_profile`` are each a genuinely distinct *type*
(:class:`BenchmarkCanonicalSelection`, :class:`BestObservedSelection`,
:class:`RuntimeProfileRef` respectively), not interchangeable string keys
(part 3's fixup: a bare ``Optional[str]`` on ``benchmark_canonical_profile``
could not structurally distinguish a protocol-bound
``BenchmarkRuntimeBinding`` selection from an arbitrary string). A caller
cannot substitute one role for another without an explicit, visible type
conversion, and :meth:`RuntimeProfileRoles.canonical_for_comparison` only
ever reads the canonical field, with no fallback path to the best-observed
one.

Schema/type-contract freeze only. No migration, persistence wiring, or
consumer code is introduced in this stage -- see
``local_only/anvil/stage-3.0-schema-freeze.md``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple


def _stable_hash(*parts: Any) -> str:
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class RuntimeProfileRoleError(ValueError):
    pass


class ValidationKind(str, Enum):
    """Scope of what a :class:`RuntimeProfileValidation` record actually
    proved. Never conflate "validated somewhere" with "canonical for
    everything"."""

    RUNTIME_STARTUP = "runtime_startup"
    CAPABILITY_PROBE = "capability_probe"
    BENCHMARK_PROTOCOL = "benchmark_protocol"
    PRODUCTION_OBSERVATION = "production_observation"


class ValidationStatus(str, Enum):
    """Typed outcome vocabulary -- matches Stage 2's discipline that typed
    authority vocabulary should not silently become stringly-typed at the
    next layer."""

    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class RuntimeProfileValidation:
    """One scoped validation fact.

    ``benchmark_canonical_profile`` selection must require a
    ``BENCHMARK_PROTOCOL``-scoped validation bound to the specific protocol
    in question -- not mere membership in a model's
    ``validated_runtime_profiles`` set, which may only record e.g. a
    ``CAPABILITY_PROBE``-scoped pass.
    """

    runtime_profile_identity_key: str
    model_artifact_identity_key: str
    validation_kind: ValidationKind
    validated_at: str
    status: ValidationStatus
    evidence_refs: Tuple[str, ...] = ()
    benchmark_protocol_identity_key: Optional[str] = None
    capability_family: Optional[str] = None
    environment_scope: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.runtime_profile_identity_key:
            raise RuntimeProfileRoleError("runtime_profile_identity_key is required")
        if not self.model_artifact_identity_key:
            raise RuntimeProfileRoleError("model_artifact_identity_key is required")
        if not self.validated_at:
            raise RuntimeProfileRoleError("validated_at is required")
        if not isinstance(self.validation_kind, ValidationKind):
            raise RuntimeProfileRoleError("validation_kind must be a ValidationKind")
        if not isinstance(self.status, ValidationStatus):
            raise RuntimeProfileRoleError("status must be a ValidationStatus")
        if self.validation_kind is ValidationKind.BENCHMARK_PROTOCOL and not self.benchmark_protocol_identity_key:
            raise RuntimeProfileRoleError(
                "benchmark_protocol validation requires benchmark_protocol_identity_key"
            )
        if self.validation_kind is ValidationKind.CAPABILITY_PROBE and not self.capability_family:
            raise RuntimeProfileRoleError("capability_probe validation requires capability_family")
        object.__setattr__(self, "evidence_refs", tuple(dict.fromkeys(self.evidence_refs)))

    def validation_key(self) -> str:
        return _stable_hash(
            self.runtime_profile_identity_key, self.model_artifact_identity_key,
            self.validation_kind.value, self.validated_at, self.status.value,
            self.evidence_refs, self.benchmark_protocol_identity_key,
            self.capability_family, self.environment_scope,
        )


@dataclass(frozen=True)
class BestObservedSelection:
    """"Best observed" never means an unexplained universal best -- there
    may be no single dominant configuration (one profile wins on quality,
    another on generation speed, another on VRAM). Provenance records what
    it was best *at*, not just that it was chosen."""

    runtime_profile_identity_key: str
    selection_objective: str
    measurement_ref: str
    policy_version: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.runtime_profile_identity_key:
            raise RuntimeProfileRoleError("runtime_profile_identity_key is required")
        if not self.selection_objective:
            raise RuntimeProfileRoleError("selection_objective is required")
        if not self.measurement_ref:
            raise RuntimeProfileRoleError("measurement_ref is required")


@dataclass(frozen=True)
class RuntimeProfileRef:
    """A structural reference to a ``RuntimeProfileIdentity``. Distinguished
    by type from :class:`BenchmarkCanonicalSelection` and
    :class:`BestObservedSelection` so a generic profile reference can never
    be silently accepted where a protocol-bound canonical selection or a
    best-observed selection is required."""

    runtime_profile_identity_key: str

    def __post_init__(self) -> None:
        if not self.runtime_profile_identity_key:
            raise RuntimeProfileRoleError("runtime_profile_identity_key is required")


@dataclass(frozen=True)
class BenchmarkCanonicalSelection:
    """The only type :meth:`RuntimeProfileRoles.canonical_for_comparison`
    may return. Structurally represents a protocol-bound
    ``BenchmarkRuntimeBinding`` selection, carrying the relevant
    benchmark-protocol validation provenance -- not an arbitrary string, and
    not interchangeable with a bare :class:`RuntimeProfileRef` or a
    :class:`BestObservedSelection`."""

    binding_key: str
    benchmark_protocol_identity_key: str
    runtime_profile_identity_key: str
    validation_ref: str
    selection_provenance: str

    def __post_init__(self) -> None:
        for field_name in (
            "binding_key", "benchmark_protocol_identity_key", "runtime_profile_identity_key",
            "validation_ref", "selection_provenance",
        ):
            if not getattr(self, field_name):
                raise RuntimeProfileRoleError(f"{field_name} is required")


@dataclass(frozen=True)
class RuntimeProfileRoles:
    """The four master-plan-required roles for one model. References only
    -- never copies of the underlying ``RuntimeProfile``/
    ``BenchmarkRuntimeBinding``. Each role field is its own type, not a
    shared ``str``, so roles cannot collapse into interchangeable string
    keys."""

    model_artifact_identity_key: str
    validated_runtime_profiles: Tuple[RuntimeProfileRef, ...] = ()
    benchmark_canonical_profile: Optional[BenchmarkCanonicalSelection] = None
    best_observed_profile: Optional[BestObservedSelection] = None
    recommended_production_profile: Optional[RuntimeProfileRef] = None

    def __post_init__(self) -> None:
        if not self.model_artifact_identity_key:
            raise RuntimeProfileRoleError("model_artifact_identity_key is required")
        for ref in self.validated_runtime_profiles:
            if not isinstance(ref, RuntimeProfileRef):
                raise RuntimeProfileRoleError("validated_runtime_profiles must contain RuntimeProfileRef values")
        object.__setattr__(
            self, "validated_runtime_profiles", tuple(dict.fromkeys(self.validated_runtime_profiles))
        )
        if self.benchmark_canonical_profile is not None and not isinstance(
            self.benchmark_canonical_profile, BenchmarkCanonicalSelection
        ):
            raise RuntimeProfileRoleError("benchmark_canonical_profile must be a BenchmarkCanonicalSelection")
        if self.best_observed_profile is not None and not isinstance(
            self.best_observed_profile, BestObservedSelection
        ):
            raise RuntimeProfileRoleError("best_observed_profile must be a BestObservedSelection")
        if self.recommended_production_profile is not None and not isinstance(
            self.recommended_production_profile, RuntimeProfileRef
        ):
            raise RuntimeProfileRoleError("recommended_production_profile must be a RuntimeProfileRef")

    def canonical_for_comparison(self) -> Optional[BenchmarkCanonicalSelection]:
        """The only role usable for a comparability benchmark run.
        Deliberately never falls back to ``best_observed_profile``: absence
        means "no canonical profile yet", not "use the best observed one
        instead". Return type is the typed selection itself, not an
        unqualified string -- a caller cannot mistake a plain
        ``RuntimeProfileRef`` or a ``BestObservedSelection`` for it, because
        neither can be stored in this field in the first place."""
        return self.benchmark_canonical_profile
