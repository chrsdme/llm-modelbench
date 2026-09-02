"""Anvil Stage 0.4 — model and runtime identity primitives.

Two identity concerns that RC21's backend abstraction already anticipated
but deliberately deferred: ``backend.BackendIdentity``'s own docstring says
"runtime-profile identity is a later-stage concern." This module is that
later stage.

``ModelArtifactIdentity`` is backend-neutral, content-addressed identity for
the bytes actually executed -- needed because ``fingerprint.model_identity``
(fingerprint.py) derives identity from an Ollama tag row's digest, which has
no equivalent for a raw GGUF file (possibly with a separate ``mmproj`` file
for vision). Without this, GGUF fleet switching (planned Stage 3B.5) risks
falling back to string identity ("Qwen 14B Q4") instead of proving which
exact artifact ran.

``RuntimeProfileIdentity`` (stable: backend/version/protocol/template/config)
is kept distinct from ``RuntimeInstanceIdentity`` (concrete: endpoint/PID/
started_at/live GPU assignment) so that restarting the same validated
profile -- PID 4127 becoming PID 8192 -- does not invalidate capability
evidence bound to the profile, while benchmark execution evidence can still
record exactly which process instance produced it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple


def _stable_hash(*parts: Any) -> str:
    """Deterministic short hash over JSON-serializable parts. Never includes
    wall-clock/timing/random fields -- callers must not pass any."""
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ModelArtifactIdentity:
    """Backend-neutral, content-addressed identity for the bytes executed."""

    artifact_set_id: str
    primary_sha256: Optional[str]
    auxiliary_artifact_hashes: Tuple[str, ...] = ()
    size_bytes: Optional[int] = None
    format: str = "unknown"
    quantization: Optional[str] = None
    source: Optional[str] = None

    @classmethod
    def from_ollama_tag_row(cls, row: Mapping[str, Any]) -> "ModelArtifactIdentity":
        """Build from an Ollama ``/api/tags`` row (matches fingerprint.py's
        existing digest/id/model_id lookup order -- extends that pattern,
        does not replace it)."""
        digest = None
        for key in ("digest", "id", "model_id"):
            value = row.get(key)
            if value:
                digest = str(value)
                break
        details = row.get("details") or {}
        if digest is None:
            for key in ("digest", "id", "model_id"):
                value = details.get(key)
                if value:
                    digest = str(value)
                    break
        artifact_set_id = digest or _stable_hash("ollama-tag", row.get("name"))
        size = row.get("size")
        return cls(
            artifact_set_id=artifact_set_id,
            primary_sha256=digest,
            size_bytes=int(size) if isinstance(size, (int, float)) else None,
            format="ollama-blob",
            quantization=details.get("quantization_level") if isinstance(details, Mapping) else None,
            source=row.get("name"),
        )

    @classmethod
    def from_gguf_path(
        cls,
        primary_sha256: str,
        *,
        size_bytes: Optional[int] = None,
        quantization: Optional[str] = None,
        source: Optional[str] = None,
        mmproj_sha256: Optional[str] = None,
    ) -> "ModelArtifactIdentity":
        """Build from a GGUF file's own content hash, optionally paired with
        a vision ``mmproj`` companion file's hash."""
        auxiliary = (mmproj_sha256,) if mmproj_sha256 else ()
        artifact_set_id = _stable_hash("gguf", primary_sha256, auxiliary)
        return cls(
            artifact_set_id=artifact_set_id,
            primary_sha256=primary_sha256,
            auxiliary_artifact_hashes=auxiliary,
            size_bytes=size_bytes,
            format="gguf",
            quantization=quantization,
            source=source,
        )


@dataclass(frozen=True)
class RuntimeProfileIdentity:
    """Stable enough for evidence compatibility across restarts of the same
    configuration. Compare/key on :meth:`stable_key`, not on object identity
    or field-by-field equality of unrelated metadata."""

    backend: str
    backend_version: Optional[str] = None
    protocol_version: Optional[str] = None
    template_hash: Optional[str] = None
    runtime_configuration_hash: Optional[str] = None
    gpu_policy: Optional[str] = None
    feature_flags: Tuple[str, ...] = ()

    def stable_key(self) -> str:
        return _stable_hash(
            self.backend,
            self.backend_version,
            self.protocol_version,
            self.template_hash,
            self.runtime_configuration_hash,
            self.gpu_policy,
            sorted(self.feature_flags),
        )


# Anvil Stage 3.2D-1: the frozen 3.2B GPU-placement policy descriptor.
# It names the *policy* (primary GPU / GPU0 first, minimum-prefix multi-GPU
# fallback -- topology_budget.placement_order), never concrete GPU UUIDs or a
# device count, which are per-environment execution facts (§8) and belong to
# RuntimeInstanceIdentity.gpu_uuid_assignment / execution evidence.
GPU_PLACEMENT_POLICY = "primary_gpu_first_minimum_multi_gpu"

# Settings that materially distinguish the reusable runtime *recipe*. A value
# that is not set becomes an explicit sentinel rather than being dropped, so
# "unset" and "set to X" never collide.
_RUNTIME_RECIPE_SETTINGS: Tuple[str, ...] = (
    "strategy",
    "context_size",
    "batch_size",
    "micro_batch_size",
    "kv_cache_type",
    "parallel_sequences",
    "offload_layers",
)
_RUNTIME_SETTING_UNSET = "unset"


def resolve_runtime_profile_identity(
    *,
    backend: str,
    backend_version: Optional[str] = None,
    execution_settings: Any = None,
    protocol_version: Optional[str] = None,
    template_hash: Optional[str] = None,
    feature_flags: Tuple[str, ...] = (),
) -> "RuntimeProfileIdentity":
    """Deterministic stable :class:`RuntimeProfileIdentity` from normalized
    resolved runtime configuration (Anvil Stage 3.2D-1).

    This is the identity used for :class:`BenchmarkRuntimeBinding`'s
    ``runtime_profile_identity_key`` -- a *reusable configuration* identity,
    not the concrete-instance hash from ``runtime_identity.RuntimeIdentity``.

    ``execution_settings`` is any object exposing the
    :class:`runtime_identity.RuntimeExecutionSettings` attribute surface. Its
    ``allocation_weights`` are deliberately **excluded** -- they are keyed by
    physical GPU UUID (an environment fact, §8); ``strategy`` carries the
    recipe-level split choice. ``allow_cpu_spill`` is also **excluded**: it is
    an execution-time operator permission, already identity-bearing in
    ``RuntimeIdentity.identity_hash`` (Stage 3.2C-2b), and replicating a
    per-invocation CLI flag into reusable-recipe identity would be wrong (§9).

    ``protocol_version``/``template_hash``/``feature_flags`` stay ``None``/
    empty unless a real source is supplied -- ``backend.py`` already declares
    template-hash derivation as later-stage work; nothing is invented here.

    ``execution_settings=None`` (Anvil Stage 3.4B): the caller has **no
    resolved runtime recipe** -- e.g. the capability-evidence path, where
    ``interrogate_model`` never receives one. In that case
    ``runtime_configuration_hash`` and ``gpu_policy`` are left ``None`` rather
    than fabricated. Hashing seven ``"unset"`` sentinels would mint a
    *concrete-looking* recipe hash that no run measured, and -- worse --
    every recipe-less caller would share that one hash, so a minimal identity
    could collide with (or be mistaken for) a genuinely rich one whose recipe
    fields all happened to be unset. A recipe-less identity is honestly
    distinguished by ``runtime_configuration_hash is None`` /
    ``gpu_policy is None`` in the canonical serialization (§7, §14).
    """
    if execution_settings is None:
        runtime_configuration_hash = None
        gpu_policy = None
    else:
        recipe: Dict[str, Any] = {}
        for name in _RUNTIME_RECIPE_SETTINGS:
            value = getattr(execution_settings, name, None)
            recipe[name] = _RUNTIME_SETTING_UNSET if value is None else value
        runtime_configuration_hash = _stable_hash("runtime_recipe_v1", recipe)
        gpu_policy = GPU_PLACEMENT_POLICY
    return RuntimeProfileIdentity(
        backend=backend,
        backend_version=backend_version,
        protocol_version=protocol_version,
        template_hash=template_hash,
        runtime_configuration_hash=runtime_configuration_hash,
        gpu_policy=gpu_policy,
        feature_flags=tuple(feature_flags),
    )


@dataclass(frozen=True)
class RuntimeInstanceIdentity:
    """The concrete running process. Never used as a capability-observation
    key (that's :class:`RuntimeProfileIdentity`'s job) -- used to record
    exactly which process instance produced a piece of benchmark evidence."""

    profile: RuntimeProfileIdentity
    endpoint: str
    process_id: Optional[int] = None
    started_at: Optional[str] = None
    gpu_uuid_assignment: Tuple[str, ...] = ()
    loaded_artifact: Optional[ModelArtifactIdentity] = None
    live_configuration_hash: Optional[str] = None

    def instance_key(self) -> str:
        """Distinct per concrete process/load, unlike stable_key()."""
        return _stable_hash(
            self.profile.stable_key(),
            self.endpoint,
            self.process_id,
            self.started_at,
            sorted(self.gpu_uuid_assignment),
            self.loaded_artifact.artifact_set_id if self.loaded_artifact else None,
            self.live_configuration_hash,
        )
