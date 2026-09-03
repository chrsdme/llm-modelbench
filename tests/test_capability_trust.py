"""Anvil Stage 3B.2 slice B -- explicit write-time trust classification for
fresh native capability evidence (owner's frozen rule).

Key negative test (prompt §8): a fresh / current-schema observation lacking
the complete required identity/provenance/measurement contract must NOT
become ``CANONICAL_COMPATIBLE``. Also proves the complete contract CAN
explicitly produce ``CANONICAL_COMPATIBLE``, and that historical/legacy
absent-trust stays fail-closed.
"""
import pytest

from llm_modelbench.capabilities import (
    CAPABILITY_SCHEMA_VERSION,
    PROBE_PROTOCOL_VERSION,
    MeasuredCapabilityState,
)
from llm_modelbench.capability_observation import CapabilityObservation
from llm_modelbench.capability_trust import classify_fresh_capability_trust
from llm_modelbench.evidence import EvidenceTrustClass
from llm_modelbench.identity import ModelArtifactIdentity


def _content_addressed_model(digest="sha256:abc123"):
    return ModelArtifactIdentity(
        artifact_set_id=digest, primary_sha256=digest, size_bytes=9_000_000_000,
        format="ollama-blob", quantization="Q4_K_M", source="qwen2.5-coder:14b",
    )


def _name_only_model():
    # Exactly what ModelArtifactIdentity.from_ollama_tag_row produces when
    # the tag row carried no digest: a name-hash artifact_set_id, no sha256.
    return ModelArtifactIdentity.from_ollama_tag_row({"name": "mystery-model:latest"})


def _probe_path_runtime_identity(backend="ollama"):
    # The shape the real probe path builds: resolve_runtime_profile_identity
    # with execution_settings=None -> backend_version / config hash / gpu
    # policy all None. The contract must accept this.
    from llm_modelbench.identity import resolve_runtime_profile_identity

    return resolve_runtime_profile_identity(
        backend=backend, execution_settings=None,
        protocol_version=PROBE_PROTOCOL_VERSION, template_hash="tmpl-1",
    )


def _observation(**overrides):
    kwargs = dict(
        model_identity=_content_addressed_model(),
        runtime_profile_identity=_probe_path_runtime_identity(),
        capability="text",
        result=MeasuredCapabilityState.MEASURED_SUPPORTED,
        probe_protocol_version=PROBE_PROTOCOL_VERSION,
        capability_schema_version=CAPABILITY_SCHEMA_VERSION,
    )
    kwargs.update(overrides)
    return CapabilityObservation(**kwargs)


def test_complete_current_contract_yields_canonical_compatible():
    """The positive case: a probe under the current contract, bound to
    content-addressed identity, with a committing measurement -> canonical.
    Proven with the *realistic* probe-path identity shape (no
    backend_version), not a hand-built ideal one."""
    assert classify_fresh_capability_trust(_observation()) is EvidenceTrustClass.CANONICAL_COMPATIBLE


def test_canonical_applies_identically_to_measured_unsupported():
    """Trust class is the trust of the evidence, not the sign of the result
    (owner rule)."""
    obs = _observation(result=MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    assert classify_fresh_capability_trust(obs) is EvidenceTrustClass.CANONICAL_COMPATIBLE


def test_fresh_schema_alone_cannot_create_canonical_trust():
    """KEY NEGATIVE TEST (prompt §8). Current schema + current protocol
    version, but the artifact identity has no content-addressed provenance
    (name-hash only). Must fail closed -- freshness/schema is not trust."""
    obs = _observation(model_identity=_name_only_model())
    assert obs.capability_schema_version == CAPABILITY_SCHEMA_VERSION  # it IS current-schema
    assert obs.probe_protocol_version == PROBE_PROTOCOL_VERSION
    assert classify_fresh_capability_trust(obs) is EvidenceTrustClass.UNKNOWN_LEGACY


def test_incomplete_provenance_cannot_create_canonical_trust():
    """primary_sha256 explicitly blank -> fail closed."""
    obs = _observation(
        model_identity=ModelArtifactIdentity(
            artifact_set_id="x", primary_sha256=None, format="gguf",
        )
    )
    assert classify_fresh_capability_trust(obs) is EvidenceTrustClass.UNKNOWN_LEGACY


def test_stale_probe_protocol_version_fails_closed():
    obs = _observation(probe_protocol_version="capability-smoke-v1")
    assert classify_fresh_capability_trust(obs) is EvidenceTrustClass.UNKNOWN_LEGACY


def test_wrong_capability_schema_version_fails_closed():
    obs = _observation(capability_schema_version=CAPABILITY_SCHEMA_VERSION - 1)
    assert classify_fresh_capability_trust(obs) is EvidenceTrustClass.UNKNOWN_LEGACY


@pytest.mark.parametrize(
    "state",
    [
        MeasuredCapabilityState.PROBE_INCONCLUSIVE,
        MeasuredCapabilityState.BACKEND_UNSUPPORTED,
        MeasuredCapabilityState.NOT_APPLICABLE,
    ],
)
def test_non_committing_measured_state_is_not_canonical(state):
    obs = _observation(result=state)
    assert classify_fresh_capability_trust(obs) is EvidenceTrustClass.UNKNOWN_LEGACY


def test_unresolved_ambiguity_fails_closed():
    assert (
        classify_fresh_capability_trust(_observation(), unresolved_ambiguity=True)
        is EvidenceTrustClass.UNKNOWN_LEGACY
    )


def test_no_trust_inference_from_timestamp_or_recency():
    """Two observations, same everything except one has a much newer
    timestamp. The classifier must not consult it."""
    old = _observation(timestamp="2020-01-01T00:00:00Z")
    new = _observation(timestamp="2099-01-01T00:00:00Z")
    assert classify_fresh_capability_trust(old) is classify_fresh_capability_trust(new)
