"""Anvil Stage 3.0 schema freeze: BenchmarkProtocol and BenchmarkRuntimeBinding.

Includes the part-3 fixup regression tests: canonicalization of set-like
identity material (allowed_adaptations, allowed_adaptations_used), exact
one-scorer-per-task validation for scorer_versions, an explicit test that
task_ids order IS identity-bearing (a deliberate decision, not an
oversight), and the bind_runtime_to_protocol() adaptation-validation rule.
"""
import pytest

from llm_modelbench.benchmark_protocol import (
    BenchmarkProtocol,
    BenchmarkProtocolError,
    BenchmarkRuntimeBinding,
    bind_runtime_to_protocol,
)
from llm_modelbench.identity import ModelArtifactIdentity


def _artifact(*, artifact_set_id="artifact-1"):
    return ModelArtifactIdentity(
        artifact_set_id=artifact_set_id, primary_sha256="sha-1", size_bytes=1_000,
        format="gguf", quantization="Q4_K_M", source="test-model.gguf",
    )


def _protocol(*, protocol_id="core", version="1.0.0", task_ids=("t1", "t2"), allowed_adaptations=()):
    return BenchmarkProtocol(
        protocol_id=protocol_id, version=version, task_ids=task_ids,
        prompt_semantics_hash="prompt-hash", sampling_policy_hash="sampling-hash",
        output_budget_policy_hash="budget-hash",
        scorer_versions=tuple((task_id, "scorer-v1") for task_id in task_ids),
        allowed_adaptations=allowed_adaptations,
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


def test_protocol_rejects_scorer_versions_for_unknown_task():
    with pytest.raises(BenchmarkProtocolError):
        BenchmarkProtocol(
            protocol_id="core", version="1.0.0", task_ids=("t1",),
            prompt_semantics_hash="p", sampling_policy_hash="s",
            output_budget_policy_hash="b",
            scorer_versions=(("t1", "v1"), ("not-in-protocol", "v1")),
        )


def test_protocol_rejects_duplicate_scorer_versions_for_same_task():
    with pytest.raises(BenchmarkProtocolError):
        BenchmarkProtocol(
            protocol_id="core", version="1.0.0", task_ids=("t1",),
            prompt_semantics_hash="p", sampling_policy_hash="s",
            output_budget_policy_hash="b",
            scorer_versions=(("t1", "scorer-v1"), ("t1", "scorer-v2")),
        )


def test_protocol_deduplicates_task_ids_but_preserves_order():
    protocol = BenchmarkProtocol(
        protocol_id="core", version="1.0.0", task_ids=("t1", "t1", "t2"),
        prompt_semantics_hash="p", sampling_policy_hash="s",
        output_budget_policy_hash="b",
        scorer_versions=(("t1", "v1"), ("t2", "v1")),
    )
    assert protocol.task_ids == ("t1", "t2")


def test_task_ids_order_is_identity_bearing():
    """Deliberate decision (not an oversight): task_ids is an ordered
    execution sequence, so reordering the same task set must produce a
    different protocol identity -- unlike allowed_adaptations, which is a
    set and must NOT be sensitive to order (see the adaptations tests
    below)."""
    forward = _protocol(task_ids=("t1", "t2"))
    reversed_order = _protocol(task_ids=("t2", "t1"))
    assert forward.identity_key() != reversed_order.identity_key()


def test_allowed_adaptations_canonicalized_regardless_of_insertion_order():
    a = _protocol(allowed_adaptations=("template_substitution", "stop_adjustment"))
    b = _protocol(allowed_adaptations=("stop_adjustment", "template_substitution"))
    assert a.allowed_adaptations == b.allowed_adaptations
    assert a.identity_key() == b.identity_key()


def test_allowed_adaptations_deduplicated():
    protocol = _protocol(allowed_adaptations=("template_substitution", "template_substitution"))
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


def test_binding_allowed_adaptations_used_canonicalized_regardless_of_order():
    artifact = _artifact()
    a = BenchmarkRuntimeBinding(
        model_artifact_identity=artifact, benchmark_protocol_identity_key="proto-key-1",
        runtime_profile_identity_key="profile-key-1",
        allowed_adaptations_used=("template_substitution", "stop_adjustment"),
    )
    b = BenchmarkRuntimeBinding(
        model_artifact_identity=artifact, benchmark_protocol_identity_key="proto-key-1",
        runtime_profile_identity_key="profile-key-1",
        allowed_adaptations_used=("stop_adjustment", "template_substitution"),
    )
    assert a.allowed_adaptations_used == b.allowed_adaptations_used
    assert a.binding_key() == b.binding_key()


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


def test_bind_runtime_to_protocol_accepts_permitted_adaptations():
    protocol = _protocol(allowed_adaptations=("template_substitution",))
    binding = bind_runtime_to_protocol(
        protocol, model_artifact_identity=_artifact(), runtime_profile_identity_key="profile-key-1",
        allowed_adaptations_used=("template_substitution",),
    )
    assert binding.benchmark_protocol_identity_key == protocol.identity_key()
    assert binding.allowed_adaptations_used == ("template_substitution",)


def test_bind_runtime_to_protocol_rejects_adaptation_not_permitted_by_protocol():
    protocol = _protocol(allowed_adaptations=("template_substitution",))
    with pytest.raises(BenchmarkProtocolError):
        bind_runtime_to_protocol(
            protocol, model_artifact_identity=_artifact(), runtime_profile_identity_key="profile-key-1",
            allowed_adaptations_used=("anything-i-want",),
        )


def test_bind_runtime_to_protocol_with_no_adaptations_used_always_succeeds():
    protocol = _protocol(allowed_adaptations=())
    binding = bind_runtime_to_protocol(
        protocol, model_artifact_identity=_artifact(), runtime_profile_identity_key="profile-key-1",
    )
    assert binding.allowed_adaptations_used == ()


def test_direct_binding_construction_does_not_validate_adaptations_against_a_protocol():
    """Documents the boundary explicitly: raw BenchmarkRuntimeBinding
    construction has no protocol object to validate against (reference-
    oriented by design) -- only bind_runtime_to_protocol() enforces the
    subset rule. This is intentional, not a gap: callers resolving a
    binding from persisted data must re-run the equivalent check
    themselves against the resolved protocol."""
    artifact = _artifact()
    binding = BenchmarkRuntimeBinding(
        model_artifact_identity=artifact, benchmark_protocol_identity_key="proto-key-1",
        runtime_profile_identity_key="profile-key-1",
        allowed_adaptations_used=("anything-i-want",),
    )
    assert binding.allowed_adaptations_used == ("anything-i-want",)
