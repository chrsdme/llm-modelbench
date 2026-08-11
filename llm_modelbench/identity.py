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
from typing import Any, Mapping, Optional, Tuple


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
