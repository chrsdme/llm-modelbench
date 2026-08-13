"""Anvil Stage 3.0 schema freeze: ModelArtifact identity/install-state split."""
import pytest

from llm_modelbench.identity import ModelArtifactIdentity
from llm_modelbench.model_artifact import (
    ArtifactInventoryObservation,
    ModelArtifact,
    ModelArtifactError,
)


def _identity(**overrides):
    kwargs = dict(
        artifact_set_id="artifact-1", primary_sha256="sha-1", size_bytes=1_000,
        format="gguf", quantization="Q4_K_M", source="test-model.gguf",
    )
    kwargs.update(overrides)
    return ModelArtifactIdentity(**kwargs)


def test_model_artifact_wraps_identity():
    artifact = ModelArtifact(identity=_identity())
    assert artifact.identity.artifact_set_id == "artifact-1"


def test_model_artifact_rejects_non_identity_value():
    with pytest.raises(ModelArtifactError):
        ModelArtifact(identity="not-an-identity")  # type: ignore[arg-type]


def test_inventory_observation_requires_known_discovery_source():
    with pytest.raises(ModelArtifactError):
        ArtifactInventoryObservation(
            artifact_set_id="artifact-1", discovery_source="unknown_source",
            present=True, observed_at="2026-08-13T00:00:00Z",
        )


def test_inventory_observation_requires_artifact_set_id_and_observed_at():
    with pytest.raises(ModelArtifactError):
        ArtifactInventoryObservation(
            artifact_set_id="", discovery_source="gguf_path",
            present=True, observed_at="2026-08-13T00:00:00Z",
        )
    with pytest.raises(ModelArtifactError):
        ArtifactInventoryObservation(
            artifact_set_id="artifact-1", discovery_source="gguf_path",
            present=True, observed_at="",
        )


def test_inventory_observation_present_must_be_bool():
    with pytest.raises(ModelArtifactError):
        ArtifactInventoryObservation(
            artifact_set_id="artifact-1", discovery_source="gguf_path",
            present="yes",  # type: ignore[arg-type]
            observed_at="2026-08-13T00:00:00Z",
        )


def test_absence_does_not_change_artifact_identity():
    """The core point of the split: an artifact temporarily absent from the
    host is a fact about ArtifactInventoryObservation only -- the identity
    object itself carries no notion of presence and is unaffected."""
    identity = _identity()
    artifact = ModelArtifact(identity=identity)
    present = ArtifactInventoryObservation(
        artifact_set_id=identity.artifact_set_id, discovery_source="gguf_path",
        present=True, observed_at="2026-08-13T00:00:00Z", local_path="/models/test.gguf",
    )
    absent = ArtifactInventoryObservation(
        artifact_set_id=identity.artifact_set_id, discovery_source="gguf_path",
        present=False, observed_at="2026-08-13T01:00:00Z", local_path=None,
    )
    assert present.artifact_set_id == absent.artifact_set_id == artifact.identity.artifact_set_id
    assert artifact.identity == identity  # unaffected by either observation
