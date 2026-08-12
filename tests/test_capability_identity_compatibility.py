"""Anvil Stage 2.2: runtime/model identity compatibility contract.

Covers the exact test matrix from this session's Codex/GPT Stage 2.2
scoping pass (`local_only/anvil/codex-advice_stage2.2.txt`). Scope is
deliberately the typed compatibility contract only -- no CapabilityProjection,
no CapabilityDecision/EnvironmentDecision, no TaskApplicability, no call-site
migration. See capability_identity.py's module docstring for the full
non-goal list.
"""
import pytest

from llm_modelbench.capabilities import MeasuredCapabilityState
from llm_modelbench.capability_identity import (
    CapabilityIdentityCompatibility,
    CapabilityIdentityCompatibilityReason,
    capability_observation_identity_compatibility,
)
from llm_modelbench.capability_observation import CapabilityObservation
from llm_modelbench.identity import ModelArtifactIdentity, RuntimeProfileIdentity

REASON = CapabilityIdentityCompatibilityReason


def _model(*, digest="digest-1", primary_sha256="digest-1"):
    return ModelArtifactIdentity(
        artifact_set_id=digest, primary_sha256=primary_sha256, size_bytes=9_000_000_000,
        format="ollama-blob", quantization="Q4_K_M", source="qwen2.5-coder:14b",
    )


def _profile(*, backend="ollama", template_hash="template-hash-1"):
    return RuntimeProfileIdentity(
        backend=backend, backend_version="0.5.1", protocol_version="capability-smoke-v2",
        template_hash=template_hash, runtime_configuration_hash="cfg-1",
    )


def _observation(**overrides):
    kwargs = dict(
        model_identity=_model(), runtime_profile_identity=_profile(),
        capability="text", result=MeasuredCapabilityState.MEASURED_SUPPORTED,
        probe_protocol_version="capability-smoke-v2", capability_schema_version=2,
        template_config_hash="tmpl-abc", endpoint_identity="http://127.0.0.1:11434",
    )
    kwargs.update(overrides)
    return CapabilityObservation(**kwargs)


def _current(observation, **overrides):
    """The current-identity kwargs that, unmodified, exactly match `observation`."""
    kwargs = dict(
        current_model_identity=observation.model_identity,
        current_runtime_profile_identity=observation.runtime_profile_identity,
        current_probe_protocol_version=observation.probe_protocol_version,
        current_capability_schema_version=observation.capability_schema_version,
        current_template_config_hash=observation.template_config_hash,
        current_endpoint_identity=observation.endpoint_identity,
    )
    kwargs.update(overrides)
    return kwargs


def test_exact_match_is_compatible():
    observation = _observation()
    result = capability_observation_identity_compatibility(observation, **_current(observation))
    assert result == CapabilityIdentityCompatibility(
        compatible=True, reason=REASON.IDENTITY_MATCH, stored_evidence_hash=observation.evidence_hash,
    )


def test_missing_current_model_identity():
    observation = _observation()
    result = capability_observation_identity_compatibility(
        observation, **_current(observation, current_model_identity=None)
    )
    assert result == CapabilityIdentityCompatibility(
        compatible=False, reason=REASON.CURRENT_IDENTITY_MISSING, stored_evidence_hash=observation.evidence_hash,
    )


def test_missing_current_runtime_profile_identity():
    observation = _observation()
    result = capability_observation_identity_compatibility(
        observation, **_current(observation, current_runtime_profile_identity=None)
    )
    assert result == CapabilityIdentityCompatibility(
        compatible=False, reason=REASON.CURRENT_IDENTITY_MISSING, stored_evidence_hash=observation.evidence_hash,
    )


def test_capability_schema_version_changed():
    observation = _observation()
    result = capability_observation_identity_compatibility(
        observation, **_current(observation, current_capability_schema_version=99)
    )
    assert result == CapabilityIdentityCompatibility(
        compatible=False, reason=REASON.CAPABILITY_SCHEMA_VERSION_CHANGED,
        stored_evidence_hash=observation.evidence_hash,
    )


def test_probe_protocol_version_changed():
    observation = _observation()
    result = capability_observation_identity_compatibility(
        observation, **_current(observation, current_probe_protocol_version="capability-smoke-v3")
    )
    assert result == CapabilityIdentityCompatibility(
        compatible=False, reason=REASON.PROBE_PROTOCOL_VERSION_CHANGED,
        stored_evidence_hash=observation.evidence_hash,
    )


def test_model_artifact_changed():
    observation = _observation()
    result = capability_observation_identity_compatibility(
        observation, **_current(observation, current_model_identity=_model(digest="digest-2", primary_sha256="digest-2"))
    )
    assert result == CapabilityIdentityCompatibility(
        compatible=False, reason=REASON.MODEL_ARTIFACT_CHANGED, stored_evidence_hash=observation.evidence_hash,
    )


def test_model_primary_hash_changed_when_artifact_set_id_matches():
    # Same artifact_set_id (the content-addressed key) but a differing
    # primary_sha256 -- an inconsistent/tampered identity, distinct from a
    # genuinely different artifact.
    observation = _observation(model_identity=_model(digest="shared-id", primary_sha256="hash-a"))
    current = _model(digest="shared-id", primary_sha256="hash-b")
    result = capability_observation_identity_compatibility(
        observation, **_current(observation, current_model_identity=current)
    )
    assert result == CapabilityIdentityCompatibility(
        compatible=False, reason=REASON.MODEL_PRIMARY_HASH_CHANGED, stored_evidence_hash=observation.evidence_hash,
    )


def test_null_primary_hash_does_not_falsely_prove_change_or_equality():
    # One side missing primary_sha256 (e.g. an artifact identified purely by
    # artifact_set_id) must not trip MODEL_PRIMARY_HASH_CHANGED -- absence
    # is not proof of mismatch. artifact_set_id already matches, so this is
    # IDENTITY_MATCH.
    observation = _observation(model_identity=_model(digest="shared-id", primary_sha256=None))
    current = _model(digest="shared-id", primary_sha256="hash-b")
    result = capability_observation_identity_compatibility(
        observation, **_current(observation, current_model_identity=current)
    )
    assert result == CapabilityIdentityCompatibility(
        compatible=True, reason=REASON.IDENTITY_MATCH, stored_evidence_hash=observation.evidence_hash,
    )


def test_runtime_profile_changed():
    observation = _observation()
    result = capability_observation_identity_compatibility(
        observation, **_current(observation, current_runtime_profile_identity=_profile(backend="llama_cpp"))
    )
    assert result == CapabilityIdentityCompatibility(
        compatible=False, reason=REASON.RUNTIME_PROFILE_CHANGED, stored_evidence_hash=observation.evidence_hash,
    )


def test_runtime_profile_restart_is_stable_across_instance_details():
    # RuntimeProfileIdentity (stable_key) deliberately carries no PID/endpoint/
    # started_at -- restarting the same validated profile must NOT trip
    # RUNTIME_PROFILE_CHANGED. Two independently-constructed but field-identical
    # RuntimeProfileIdentity instances (standing in for "same profile, process
    # restarted") compare equal.
    observation = _observation()
    restarted_profile = RuntimeProfileIdentity(
        backend="ollama", backend_version="0.5.1", protocol_version="capability-smoke-v2",
        template_hash="template-hash-1", runtime_configuration_hash="cfg-1",
    )
    assert restarted_profile is not observation.runtime_profile_identity
    result = capability_observation_identity_compatibility(
        observation, **_current(observation, current_runtime_profile_identity=restarted_profile)
    )
    assert result == CapabilityIdentityCompatibility(
        compatible=True, reason=REASON.IDENTITY_MATCH, stored_evidence_hash=observation.evidence_hash,
    )


def test_template_config_hash_changed():
    observation = _observation()
    result = capability_observation_identity_compatibility(
        observation, **_current(observation, current_template_config_hash="different-template")
    )
    assert result == CapabilityIdentityCompatibility(
        compatible=False, reason=REASON.TEMPLATE_CONFIG_CHANGED, stored_evidence_hash=observation.evidence_hash,
    )


def test_endpoint_identity_changed():
    observation = _observation()
    result = capability_observation_identity_compatibility(
        observation, **_current(observation, current_endpoint_identity="http://127.0.0.1:11500")
    )
    assert result == CapabilityIdentityCompatibility(
        compatible=False, reason=REASON.ENDPOINT_CHANGED, stored_evidence_hash=observation.evidence_hash,
    )


def test_precedence_schema_version_wins_over_model_artifact_change():
    observation = _observation()
    result = capability_observation_identity_compatibility(
        observation,
        **_current(
            observation,
            current_capability_schema_version=99,
            current_model_identity=_model(digest="digest-2", primary_sha256="digest-2"),
        ),
    )
    assert result == CapabilityIdentityCompatibility(
        compatible=False, reason=REASON.CAPABILITY_SCHEMA_VERSION_CHANGED,
        stored_evidence_hash=observation.evidence_hash,
    )


def test_precedence_probe_protocol_wins_over_runtime_profile_change():
    observation = _observation()
    result = capability_observation_identity_compatibility(
        observation,
        **_current(
            observation,
            current_probe_protocol_version="capability-smoke-v3",
            current_runtime_profile_identity=_profile(backend="llama_cpp"),
        ),
    )
    assert result == CapabilityIdentityCompatibility(
        compatible=False, reason=REASON.PROBE_PROTOCOL_VERSION_CHANGED,
        stored_evidence_hash=observation.evidence_hash,
    )


def test_rejects_non_observation_argument():
    with pytest.raises(TypeError, match="CapabilityObservation"):
        capability_observation_identity_compatibility(
            {"capability": "text"},  # a dict, not a CapabilityObservation
            current_model_identity=_model(),
            current_runtime_profile_identity=_profile(),
            current_probe_protocol_version="capability-smoke-v2",
            current_capability_schema_version=2,
        )


def test_rejects_legacy_dict_as_current_model_identity():
    observation = _observation()
    with pytest.raises(TypeError, match="ModelArtifactIdentity"):
        capability_observation_identity_compatibility(
            observation,
            **_current(observation, current_model_identity={"digest": "digest-1"}),
        )


def test_rejects_legacy_dict_as_current_runtime_profile_identity():
    observation = _observation()
    with pytest.raises(TypeError, match="RuntimeProfileIdentity"):
        capability_observation_identity_compatibility(
            observation,
            **_current(observation, current_runtime_profile_identity={"backend": "ollama"}),
        )
