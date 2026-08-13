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
as "canonical for everything".

``best_observed_profile`` is structurally kept distinct from
``benchmark_canonical_profile`` on ``RuntimeProfileRoles``: they are
different fields of different types, and :meth:`RuntimeProfileRoles.
canonical_for_comparison` only ever reads the canonical field, with no
fallback path to the best-observed one. A caller cannot substitute one for
the other without a visible type change at the call site.

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
    status: str
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
        if self.status not in {"passed", "failed"}:
            raise RuntimeProfileRoleError("status must be 'passed' or 'failed'")
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
            self.validation_kind.value, self.validated_at, self.status,
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
class RuntimeProfileRoles:
    """The four master-plan-required roles for one model. References only
    -- never copies of the underlying ``RuntimeProfile``/
    ``BenchmarkRuntimeBinding``."""

    model_artifact_identity_key: str
    validated_runtime_profiles: Tuple[str, ...] = ()
    benchmark_canonical_profile: Optional[str] = None
    best_observed_profile: Optional[BestObservedSelection] = None
    recommended_production_profile: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.model_artifact_identity_key:
            raise RuntimeProfileRoleError("model_artifact_identity_key is required")
        object.__setattr__(
            self, "validated_runtime_profiles", tuple(dict.fromkeys(self.validated_runtime_profiles))
        )

    def canonical_for_comparison(self) -> Optional[str]:
        """The only role usable for a comparability benchmark run.
        Deliberately never falls back to ``best_observed_profile``: absence
        means "no canonical profile yet", not "use the best observed one
        instead"."""
        return self.benchmark_canonical_profile
