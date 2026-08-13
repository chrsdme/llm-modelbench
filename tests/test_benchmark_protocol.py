"""Anvil Stage 3.0 schema freeze: BenchmarkProtocol and BenchmarkRuntimeBinding."""
import pytest

from llm_modelbench.benchmark_protocol import (
    BenchmarkProtocol,
    BenchmarkProtocolError,
    BenchmarkRuntimeBinding,
)
from llm_modelbench.identity import ModelArtifactIdentity


def _artifact(*, artifact_set_id="artifact-1"):
    return ModelArtifactIdentity(
        artifact_set_id=artifact_set_id, primary_sha256="sha-1", size_bytes=1_000,
        format="gguf", quantization="Q4_K_M", source="test-model.gguf",
    )


def _protocol(*, protocol_id="core", version="1.0.0", task_ids=("t1", "t2")):
    return BenchmarkProtocol(
        protocol_id=protocol_id, version=version, task_ids=task_ids,
        prompt_semantics_hash="prompt-hash", sampling_policy_hash="sampling-hash",
        output_budget_policy_hash="budget-hash",
        scorer_versions=tuple((task_id, "scorer-v1") for task_id in task_ids),
    )


def test_protocol_identity_key_stable_for_identical_material():
    a = _protocol()
    b = _protocol()
    assert a.identity_key() == b.identity_key()


def test_protocol_identity_key_changes_with_version():
    v1 = _protocol(version="1.0.0")
    v2 = _protocol(version="2.0.0")
    assert v1.identity_key() != v2.identity_key()


def test_protocol_identity_key_changes_with_protocol_id():
    a = _protocol(protocol_id="core")
    b = _protocol(protocol_id="extended")
    assert a.identity_key() != b.identity_key()


def test_protocol_requires_non_empty_task_ids():
    with pytest.raises(BenchmarkProtocolError):
        BenchmarkProtocol(
            protocol_id="core", version="1.0.0", task_ids=(),
            prompt_semantics_hash="p", sampling_policy_hash="s",
            output_budget_policy_hash="b", scorer_versions=(),
        )


def test_protocol_requires_scorer_version_for_every_task():
    with pytest.raises(BenchmarkProtocolError):
        BenchmarkProtocol(
            protocol_id="core", version="1.0.0", task_ids=("t1", "t2"),
            prompt_semantics_hash="p", sampling_policy_hash="s",
            output_budget_policy_hash="b", scorer_versions=(("t1", "v1"),),
        )


def test_protocol_deduplicates_task_ids_and_adaptations():
    protocol = BenchmarkProtocol(
        protocol_id="core", version="1.0.0", task_ids=("t1", "t1", "t2"),
        prompt_semantics_hash="p", sampling_policy_hash="s",
        output_budget_policy_hash="b",
        scorer_versions=(("t1", "v1"), ("t2", "v1")),
        allowed_adaptations=("template_substitution", "template_substitution"),
    )
    assert protocol.task_ids == ("t1", "t2")
    assert protocol.allowed_adaptations == ("template_substitution",)


def test_binding_key_stable_for_identical_material():
    artifact = _artifact()
    a = BenchmarkRuntimeBinding(
        model_artifact_identity=artifact, benchmark_protocol_identity_key="proto-key-1",
        runtime_profile_identity_key="profile-key-1",
    )
    b = BenchmarkRuntimeBinding(
        model_artifact_identity=artifact, benchmark_protocol_identity_key="proto-key-1",
        runtime_profile_identity_key="profile-key-1",
    )
    assert a.binding_key() == b.binding_key()


def test_binding_key_changes_with_runtime_profile():
    artifact = _artifact()
    a = BenchmarkRuntimeBinding(
        model_artifact_identity=artifact, benchmark_protocol_identity_key="proto-key-1",
        runtime_profile_identity_key="profile-key-1",
    )
    b = BenchmarkRuntimeBinding(
        model_artifact_identity=artifact, benchmark_protocol_identity_key="proto-key-1",
        runtime_profile_identity_key="profile-key-2",
    )
    assert a.binding_key() != b.binding_key()


def test_binding_requires_protocol_and_profile_keys():
    artifact = _artifact()
    with pytest.raises(BenchmarkProtocolError):
        BenchmarkRuntimeBinding(
            model_artifact_identity=artifact, benchmark_protocol_identity_key="",
            runtime_profile_identity_key="profile-key-1",
        )
    with pytest.raises(BenchmarkProtocolError):
        BenchmarkRuntimeBinding(
            model_artifact_identity=artifact, benchmark_protocol_identity_key="proto-key-1",
            runtime_profile_identity_key="",
        )


def test_binding_requires_model_artifact_identity_type():
    with pytest.raises(BenchmarkProtocolError):
        BenchmarkRuntimeBinding(
            model_artifact_identity="not-an-identity",  # type: ignore[arg-type]
            benchmark_protocol_identity_key="proto-key-1",
            runtime_profile_identity_key="profile-key-1",
        )


def test_binding_two_different_protocol_versions_same_runtime_are_distinct_bindings():
    """The hard rule from part 2: same runtime configuration under two
    different protocol versions must produce two different bindings, not
    two different RuntimeProfileIdentity values -- this test only checks
    the binding side, since RuntimeProfileIdentity itself is untouched."""
    artifact = _artifact()
    protocol_v1 = _protocol(version="1.0.0").identity_key()
    protocol_v2 = _protocol(version="2.0.0").identity_key()
    binding_v1 = BenchmarkRuntimeBinding(
        model_artifact_identity=artifact, benchmark_protocol_identity_key=protocol_v1,
        runtime_profile_identity_key="profile-key-1",
    )
    binding_v2 = BenchmarkRuntimeBinding(
        model_artifact_identity=artifact, benchmark_protocol_identity_key=protocol_v2,
        runtime_profile_identity_key="profile-key-1",
    )
    assert binding_v1.binding_key() != binding_v2.binding_key()
