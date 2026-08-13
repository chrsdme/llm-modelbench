"""Anvil Stage 3.0 schema freeze: scoped runtime-profile validation and the
four master-plan-required profile roles.

Includes the part-3 fixup regression tests: RuntimeProfileValidation's
runtime type bypass (a raw string masquerading as a ValidationKind/
ValidationStatus must fail construction, not silently skip scope checks),
and RuntimeProfileRoles' typed canonical-profile references (a plain
RuntimeProfileRef or BestObservedSelection must never be accepted where a
BenchmarkCanonicalSelection is required).
"""
import pytest

from llm_modelbench.runtime_profile_roles import (
    BenchmarkCanonicalSelection,
    BestObservedSelection,
    RuntimeProfileRef,
    RuntimeProfileRoleError,
    RuntimeProfileRoles,
    RuntimeProfileValidation,
    ValidationKind,
    ValidationStatus,
)


def _validation(**overrides):
    kwargs = dict(
        runtime_profile_identity_key="profile-1", model_artifact_identity_key="artifact-1",
        validation_kind=ValidationKind.RUNTIME_STARTUP, validated_at="2026-08-13T00:00:00Z",
        status=ValidationStatus.PASSED,
    )
    kwargs.update(overrides)
    return RuntimeProfileValidation(**kwargs)


def _canonical_selection(**overrides):
    kwargs = dict(
        binding_key="binding-1", benchmark_protocol_identity_key="protocol-1",
        runtime_profile_identity_key="profile-1", validation_ref="validation-1",
        selection_provenance="manual",
    )
    kwargs.update(overrides)
    return BenchmarkCanonicalSelection(**kwargs)


def test_validation_key_stable_for_identical_material():
    a = _validation()
    b = _validation()
    assert a.validation_key() == b.validation_key()


def test_validation_requires_validation_status_enum():
    with pytest.raises(RuntimeProfileRoleError):
        _validation(status="passed")  # raw string, not ValidationStatus
    with pytest.raises(RuntimeProfileRoleError):
        _validation(status="PASS")
    with pytest.raises(RuntimeProfileRoleError):
        _validation(status=True)
    with pytest.raises(RuntimeProfileRoleError):
        _validation(status=None)


def test_validation_requires_validation_kind_enum():
    with pytest.raises(RuntimeProfileRoleError):
        _validation(validation_kind="benchmark_protocol")  # raw string bypass
    with pytest.raises(RuntimeProfileRoleError):
        _validation(validation_kind="capability_probe")
    with pytest.raises(RuntimeProfileRoleError):
        _validation(validation_kind=None)


def test_raw_string_validation_kind_cannot_bypass_benchmark_protocol_scope_check():
    """The exact bug from part 3: 'benchmark_protocol' == ValidationKind.BENCHMARK_PROTOCOL.value
    but 'benchmark_protocol' is not ValidationKind.BENCHMARK_PROTOCOL. Without the
    isinstance guard, this raw string would silently skip the
    benchmark_protocol_identity_key requirement instead of being rejected outright."""
    with pytest.raises(RuntimeProfileRoleError):
        RuntimeProfileValidation(
            runtime_profile_identity_key="profile-1", model_artifact_identity_key="artifact-1",
            validation_kind="benchmark_protocol", validated_at="2026-08-13T00:00:00Z",
            status=ValidationStatus.PASSED, benchmark_protocol_identity_key=None,
        )


def test_capability_probe_validation_requires_capability_family():
    with pytest.raises(RuntimeProfileRoleError):
        _validation(validation_kind=ValidationKind.CAPABILITY_PROBE)
    # Passes once the required scope field is supplied.
    _validation(validation_kind=ValidationKind.CAPABILITY_PROBE, capability_family="tools")


def test_benchmark_protocol_validation_requires_protocol_key():
    with pytest.raises(RuntimeProfileRoleError):
        _validation(validation_kind=ValidationKind.BENCHMARK_PROTOCOL)
    # Passes once the required scope field is supplied.
    _validation(
        validation_kind=ValidationKind.BENCHMARK_PROTOCOL,
        benchmark_protocol_identity_key="protocol-key-1",
    )


def test_a_capability_probe_pass_does_not_imply_benchmark_protocol_validation():
    """Direct check of the advice's core point: 'validated somewhere' is a
    distinct fact from 'canonical for everything' -- a CAPABILITY_PROBE
    validation record carries no benchmark_protocol_identity_key at all,
    so it cannot be mistaken for (or read as) a BENCHMARK_PROTOCOL-scoped
    validation."""
    probe = _validation(validation_kind=ValidationKind.CAPABILITY_PROBE, capability_family="tools")
    assert probe.validation_kind is not ValidationKind.BENCHMARK_PROTOCOL
    assert probe.benchmark_protocol_identity_key is None


def test_best_observed_selection_requires_objective_and_measurement_ref():
    with pytest.raises(RuntimeProfileRoleError):
        BestObservedSelection(
            runtime_profile_identity_key="profile-1", selection_objective="",
            measurement_ref="measurement-1",
        )
    with pytest.raises(RuntimeProfileRoleError):
        BestObservedSelection(
            runtime_profile_identity_key="profile-1", selection_objective="quality",
            measurement_ref="",
        )


def test_runtime_profile_ref_requires_identity_key():
    with pytest.raises(RuntimeProfileRoleError):
        RuntimeProfileRef(runtime_profile_identity_key="")


def test_benchmark_canonical_selection_requires_all_fields():
    for missing_field in (
        "binding_key", "benchmark_protocol_identity_key", "runtime_profile_identity_key",
        "validation_ref", "selection_provenance",
    ):
        with pytest.raises(RuntimeProfileRoleError):
            _canonical_selection(**{missing_field: ""})


def test_roles_requires_model_artifact_identity_key():
    with pytest.raises(RuntimeProfileRoleError):
        RuntimeProfileRoles(model_artifact_identity_key="")


def test_roles_deduplicates_validated_profiles():
    ref1 = RuntimeProfileRef(runtime_profile_identity_key="profile-1")
    ref1_dup = RuntimeProfileRef(runtime_profile_identity_key="profile-1")
    ref2 = RuntimeProfileRef(runtime_profile_identity_key="profile-2")
    roles = RuntimeProfileRoles(
        model_artifact_identity_key="artifact-1",
        validated_runtime_profiles=(ref1, ref1_dup, ref2),
    )
    assert roles.validated_runtime_profiles == (ref1, ref2)


def test_roles_rejects_non_ref_validated_profiles():
    with pytest.raises(RuntimeProfileRoleError):
        RuntimeProfileRoles(
            model_artifact_identity_key="artifact-1",
            validated_runtime_profiles=("profile-1",),  # raw string, not RuntimeProfileRef
        )


def test_canonical_for_comparison_returns_only_the_canonical_field():
    selection = _canonical_selection()
    roles = RuntimeProfileRoles(
        model_artifact_identity_key="artifact-1",
        benchmark_canonical_profile=selection,
        best_observed_profile=BestObservedSelection(
            runtime_profile_identity_key="profile-best", selection_objective="quality",
            measurement_ref="measurement-1",
        ),
    )
    assert roles.canonical_for_comparison() is selection
    assert isinstance(roles.canonical_for_comparison(), BenchmarkCanonicalSelection)


def test_canonical_for_comparison_never_falls_back_to_best_observed():
    """The structural no-substitution guarantee: when no canonical profile
    is set, canonical_for_comparison() must return None, never silently
    substitute best_observed_profile."""
    roles = RuntimeProfileRoles(
        model_artifact_identity_key="artifact-1",
        benchmark_canonical_profile=None,
        best_observed_profile=BestObservedSelection(
            runtime_profile_identity_key="profile-best", selection_objective="quality",
            measurement_ref="measurement-1",
        ),
    )
    assert roles.canonical_for_comparison() is None


def test_a_plain_runtime_profile_ref_cannot_be_accepted_as_canonical_selection():
    """Part 3's central fix, tested directly: benchmark_canonical_profile's
    type is BenchmarkCanonicalSelection, not str/RuntimeProfileRef -- a
    plain profile reference cannot be substituted for a protocol-bound
    canonical selection."""
    with pytest.raises(RuntimeProfileRoleError):
        RuntimeProfileRoles(
            model_artifact_identity_key="artifact-1",
            benchmark_canonical_profile=RuntimeProfileRef(runtime_profile_identity_key="profile-1"),
        )


def test_a_best_observed_selection_cannot_be_accepted_as_canonical_selection():
    with pytest.raises(RuntimeProfileRoleError):
        RuntimeProfileRoles(
            model_artifact_identity_key="artifact-1",
            benchmark_canonical_profile=BestObservedSelection(
                runtime_profile_identity_key="profile-best", selection_objective="quality",
                measurement_ref="measurement-1",
            ),
        )


def test_recommended_production_profile_must_be_a_runtime_profile_ref():
    with pytest.raises(RuntimeProfileRoleError):
        RuntimeProfileRoles(
            model_artifact_identity_key="artifact-1",
            recommended_production_profile="profile-1",  # raw string
        )
    # Passes once wrapped in the correct type.
    RuntimeProfileRoles(
        model_artifact_identity_key="artifact-1",
        recommended_production_profile=RuntimeProfileRef(runtime_profile_identity_key="profile-1"),
    )


def test_best_observed_and_canonical_are_structurally_distinct_types():
    """best_observed_profile is a BestObservedSelection (carries an
    objective); benchmark_canonical_profile is a BenchmarkCanonicalSelection
    (carries protocol/validation provenance). A caller cannot pass one
    where the other is expected without a type mismatch that is visible at
    construction time."""
    roles = RuntimeProfileRoles(
        model_artifact_identity_key="artifact-1",
        benchmark_canonical_profile=_canonical_selection(),
        best_observed_profile=BestObservedSelection(
            runtime_profile_identity_key="profile-best", selection_objective="generation_speed",
            measurement_ref="measurement-2",
        ),
    )
    assert isinstance(roles.benchmark_canonical_profile, BenchmarkCanonicalSelection)
    assert isinstance(roles.best_observed_profile, BestObservedSelection)
    assert roles.best_observed_profile.selection_objective == "generation_speed"
