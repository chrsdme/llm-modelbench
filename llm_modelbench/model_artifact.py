"""Anvil Stage 3.0 -- ``ModelArtifact``: stable, content-addressed identity
kept strictly separate from mutable local install/inventory state.

``codex-stage3-advice.txt`` part 1: "ModelArtifact currently mixes stable
artifact facts with host-local availability. ModelArtifactIdentity is
stable/content-addressed. local presence and discovery timestamp are not."
This split matters starting Stage 3B, when GGUFs are switched, removed,
rediscovered, or served from another location: an artifact's identity must
not change, and must not need to be re-derived, just because it was
temporarily absent from (or moved on) the host.

Schema/type-contract freeze only. No migration, persistence wiring, or
consumer code is introduced in this stage -- see
``local_only/anvil/stage-3.0-schema-freeze.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .identity import ModelArtifactIdentity

_DISCOVERY_SOURCES = {"ollama_tag", "gguf_path"}


class ModelArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class ModelArtifact:
    """Authoritative fields only: stable, content-addressed identity.
    Deliberately carries nothing else -- no capability claim, no benchmark
    result, no ranking data, and no install/inventory state (see
    :class:`ArtifactInventoryObservation`)."""

    identity: ModelArtifactIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ModelArtifactIdentity):
            raise ModelArtifactError("ModelArtifact requires a ModelArtifactIdentity")


@dataclass(frozen=True)
class ArtifactInventoryObservation:
    """Mutable, host-local, current-state-only. Never authoritative for
    artifact identity or durable model truth: an artifact that is
    temporarily absent (removed, not yet rediscovered, served from a
    different path) does not thereby lose or change its identity -- only
    this observation record changes."""

    artifact_set_id: str
    discovery_source: str
    present: bool
    observed_at: str
    local_path: Optional[str] = None
    size_bytes_on_disk: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.artifact_set_id:
            raise ModelArtifactError("artifact_set_id is required")
        if self.discovery_source not in _DISCOVERY_SOURCES:
            raise ModelArtifactError(f"discovery_source must be one of {sorted(_DISCOVERY_SOURCES)}")
        if not self.observed_at:
            raise ModelArtifactError("observed_at is required")
        if not isinstance(self.present, bool):
            raise ModelArtifactError("present must be a bool")
