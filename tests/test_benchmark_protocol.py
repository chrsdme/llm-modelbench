"""Anvil Stage 3.0 schema freeze: BenchmarkProtocol and BenchmarkRuntimeBinding.

Includes the part-3 fixup regression tests: canonicalization of set-like
identity material (allowed_adaptations, allowed_adaptations_used), exact
one-scorer-per-task validation for scorer_versions, an explicit test that
task_ids order IS identity-bearing (a deliberate decision, not an
oversight), and the bind_runtime_to_protocol() adaptation-validation rule.

Plus the Stage 3.0 micro-fix regression tests: BenchmarkRuntimeBinding is
now constructible only via bind_runtime_to_protocol()/resolve_binding() --
direct construction raises -- and duplicate task_ids are rejected outright
rather than silently collapsed.
"""
import pytest

from llm_modelbench.benchmark_protocol import (
    BenchmarkProtocol,
    BenchmarkProtocolError,
    BenchmarkRuntimeBinding,
    UnresolvedBenchmarkRuntimeBinding,
    bind_runtime_to_protocol,
    resolve_binding,
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


def test_protocol_rejects_duplicate_task_ids():
    """task_ids order is identity-bearing, so duplicates are rejected
    outright rather than silently collapsed the way an unordered set's
    duplicates would be (contrast with allowed_adaptations, which IS
    deduplicated -- see the adaptations tests below)."""
    with pytest.raises(BenchmarkProtocolError):
        BenchmarkProtocol(
            protocol_id="core", version="1.0.0", task_ids=("t1", "t1", "t2"),
            prompt_semantics_hash="p", sampling_policy_hash="s",
            output_budget_policy_hash="b",
            scorer_versions=(("t1", "v1"), ("t2", "v1")),
        )


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
    a = bind_runtime_to_protocol(
        _protocol(), model_artifact_identity=artifact, runtime_profile_identity_key="profile-key-1",
    )
    b = bind_runtime_to_protocol(
        _protocol(), model_artifact_identity=artifact, runtime_profile_identity_key="profile-key-1",
    )
    assert a.binding_key() == b.binding_key()


def test_binding_key_changes_with_runtime_profile():
    artifact = _artifact()
    protocol = _protocol()
    a = bind_runtime_to_protocol(
        protocol, model_artifact_identity=artifact, runtime_profile_identity_key="profile-key-1",
    )
    b = bind_runtime_to_protocol(
        protocol, model_artifact_identity=artifact, runtime_profile_identity_key="profile-key-2",
    )
    assert a.binding_key() != b.binding_key()


def test_binding_allowed_adaptations_used_canonicalized_regardless_of_order():
    artifact = _artifact()
    protocol = _protocol(allowed_adaptations=("template_substitution", "stop_adjustment"))
    a = bind_runtime_to_protocol(
        protocol, model_artifact_identity=artifact, runtime_profile_identity_key="profile-key-1",
        allowed_adaptations_used=("template_substitution", "stop_adjustment"),
    )
    b = bind_runtime_to_protocol(
        protocol, model_artifact_identity=artifact, runtime_profile_identity_key="profile-key-1",
        allowed_adaptations_used=("stop_adjustment", "template_substitution"),
    )
    assert a.allowed_adaptations_used == b.allowed_adaptations_used
    assert a.binding_key() == b.binding_key()


def test_binding_two_different_protocol_versions_same_runtime_are_distinct_bindings():
    """The hard rule from part 2: same runtime configuration under two
    different protocol versions must produce two different bindings, not
    two different RuntimeProfileIdentity values -- this test only checks
    the binding side, since RuntimeProfileIdentity itself is untouched."""
    artifact = _artifact()
    protocol_v1 = _protocol(version="1.0.0")
    protocol_v2 = _protocol(version="2.0.0")
    binding_v1 = bind_runtime_to_protocol(
        protocol_v1, model_artifact_identity=artifact, runtime_profile_identity_key="profile-key-1",
    )
    binding_v2 = bind_runtime_to_protocol(
        protocol_v2, model_artifact_identity=artifact, runtime_profile_identity_key="profile-key-1",
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


# -- Stage 3.0 micro-fix: verified vs. unresolved binding material --------


def test_direct_benchmark_runtime_binding_construction_is_rejected():
    """The central fix: BenchmarkRuntimeBinding means 'checked against its
    protocol', never 'possibly validated'. Direct construction -- with no
    way to prove that check happened -- must raise."""
    with pytest.raises(BenchmarkProtocolError):
        BenchmarkRuntimeBinding(
            model_artifact_identity=_artifact(), benchmark_protocol_identity_key="proto-key-1",
            runtime_profile_identity_key="profile-key-1",
        )


def test_unresolved_binding_is_freely_constructible():
    """Raw, not-yet-checked material -- e.g. what you'd get deserializing
    persisted data before resolving its protocol -- has no such gate."""
    material = UnresolvedBenchmarkRuntimeBinding(
        model_artifact_identity=_artifact(), benchmark_protocol_identity_key="proto-key-1",
        runtime_profile_identity_key="profile-key-1", allowed_adaptations_used=("anything-i-want",),
    )
    assert material.allowed_adaptations_used == ("anything-i-want",)


def test_unresolved_binding_requires_the_same_basic_fields():
    artifact = _artifact()
    with pytest.raises(BenchmarkProtocolError):
        UnresolvedBenchmarkRuntimeBinding(
            model_artifact_identity=artifact, benchmark_protocol_identity_key="",
            runtime_profile_identity_key="profile-key-1",
        )
    with pytest.raises(BenchmarkProtocolError):
        UnresolvedBenchmarkRuntimeBinding(
            model_artifact_identity="not-an-identity",  # type: ignore[arg-type]
            benchmark_protocol_identity_key="proto-key-1", runtime_profile_identity_key="profile-key-1",
        )


def test_unverified_record_is_not_equal_to_and_not_a_validated_binding():
    """'unverified record != validated binding' -- both structurally (an
    UnresolvedBenchmarkRuntimeBinding is never an instance of
    BenchmarkRuntimeBinding) and by value equality."""
    artifact = _artifact()
    protocol = _protocol(allowed_adaptations=("template_substitution",))
    material = UnresolvedBenchmarkRuntimeBinding(
        model_artifact_identity=artifact, benchmark_protocol_identity_key=protocol.identity_key(),
        runtime_profile_identity_key="profile-key-1", allowed_adaptations_used=("template_substitution",),
    )
    validated = resolve_binding(material, protocol)
    assert not isinstance(material, BenchmarkRuntimeBinding)
    assert not isinstance(validated, UnresolvedBenchmarkRuntimeBinding)
    assert material != validated


def test_resolve_binding_rejects_illegal_adaptation():
    """'illegal adaptation cannot produce validated binding' -- via the
    resolve_binding() promotion path."""
    artifact = _artifact()
    protocol = _protocol(allowed_adaptations=("template_substitution",))
    material = UnresolvedBenchmarkRuntimeBinding(
        model_artifact_identity=artifact, benchmark_protocol_identity_key=protocol.identity_key(),
        runtime_profile_identity_key="profile-key-1", allowed_adaptations_used=("anything-i-want",),
    )
    with pytest.raises(BenchmarkProtocolError):
        resolve_binding(material, protocol)


def test_bind_runtime_to_protocol_rejects_illegal_adaptation_too():
    """'illegal adaptation cannot produce validated binding' -- via the
    fresh-construction path too (already covered above, restated here
    alongside the resolve_binding() case for a direct side-by-side)."""
    protocol = _protocol(allowed_adaptations=("template_substitution",))
    with pytest.raises(BenchmarkProtocolError):
        bind_runtime_to_protocol(
            protocol, model_artifact_identity=_artifact(), runtime_profile_identity_key="profile-key-1",
            allowed_adaptations_used=("anything-i-want",),
        )


def test_resolve_binding_accepts_legal_adaptation():
    """'legal adaptation can produce validated binding' -- via
    resolve_binding()."""
    artifact = _artifact()
    protocol = _protocol(allowed_adaptations=("template_substitution",))
    material = UnresolvedBenchmarkRuntimeBinding(
        model_artifact_identity=artifact, benchmark_protocol_identity_key=protocol.identity_key(),
        runtime_profile_identity_key="profile-key-1", allowed_adaptations_used=("template_substitution",),
    )
    validated = resolve_binding(material, protocol)
    assert isinstance(validated, BenchmarkRuntimeBinding)
    assert validated.allowed_adaptations_used == ("template_substitution",)


def test_resolve_binding_requires_material_to_reference_the_given_protocol():
    """'deserialized/unresolved reference requires protocol resolution
    before promotion' -- and resolution must actually match: material
    claiming a different protocol than the one supplied is rejected, not
    silently repointed."""
    artifact = _artifact()
    protocol = _protocol(protocol_id="core", allowed_adaptations=("template_substitution",))
    other_protocol = _protocol(protocol_id="other", allowed_adaptations=("template_substitution",))
    material = UnresolvedBenchmarkRuntimeBinding(
        model_artifact_identity=artifact, benchmark_protocol_identity_key=protocol.identity_key(),
        runtime_profile_identity_key="profile-key-1", allowed_adaptations_used=("template_substitution",),
    )
    with pytest.raises(BenchmarkProtocolError):
        resolve_binding(material, other_protocol)


def test_verified_marker_is_not_part_of_binding_equality_or_repr():
    """The private gating field must not leak into a validated binding's
    observable equality/repr -- two validated bindings with identical real
    fields must compare equal regardless of internal marker plumbing, and
    the marker itself must not appear in repr() output."""
    artifact = _artifact()
    protocol = _protocol()
    a = bind_runtime_to_protocol(
        protocol, model_artifact_identity=artifact, runtime_profile_identity_key="profile-key-1",
    )
    b = bind_runtime_to_protocol(
        protocol, model_artifact_identity=artifact, runtime_profile_identity_key="profile-key-1",
    )
    assert a == b
    assert "_verified_marker" not in repr(a)
