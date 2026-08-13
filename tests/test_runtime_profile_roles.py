"""Anvil Stage 3.0 schema freeze: scoped runtime-profile validation and the
four master-plan-required profile roles."""
import pytest

from llm_modelbench.runtime_profile_roles import (
    BestObservedSelection,
    RuntimeProfileRoleError,
    RuntimeProfileRoles,
    RuntimeProfileValidation,
    ValidationKind,
)


def _validation(**overrides):
    kwargs = dict(
        runtime_profile_identity_key="profile-1", model_artifact_identity_key="artifact-1",
        validation_kind=ValidationKind.RUNTIME_STARTUP, validated_at="2026-08-13T00:00:00Z",
        status="passed",
    )
    kwargs.update(overrides)
    return RuntimeProfileValidation(**kwargs)


def test_validation_key_stable_for_identical_material():
    a = _validation()
    b = _validation()
    assert a.validation_key() == b.validation_key()


def test_validation_requires_status_passed_or_failed():
    with pytest.raises(RuntimeProfileRoleError):
        _validation(status="unknown")


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


def test_roles_requires_model_artifact_identity_key():
    with pytest.raises(RuntimeProfileRoleError):
        RuntimeProfileRoles(model_artifact_identity_key="")


def test_roles_deduplicates_validated_profiles():
    roles = RuntimeProfileRoles(
        model_artifact_identity_key="artifact-1",
        validated_runtime_profiles=("profile-1", "profile-1", "profile-2"),
    )
    assert roles.validated_runtime_profiles == ("profile-1", "profile-2")


def test_canonical_for_comparison_returns_only_the_canonical_field():
    roles = RuntimeProfileRoles(
        model_artifact_identity_key="artifact-1",
        benchmark_canonical_profile="binding-key-1",
        best_observed_profile=BestObservedSelection(
            runtime_profile_identity_key="profile-best", selection_objective="quality",
            measurement_ref="measurement-1",
        ),
    )
    assert roles.canonical_for_comparison() == "binding-key-1"


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


def test_best_observed_and_canonical_are_structurally_distinct_types():
    """best_observed_profile is a BestObservedSelection (carries an
    objective); benchmark_canonical_profile is a plain reference string.
    A caller cannot pass one where the other is expected without a type
    mismatch that is visible at construction time."""
    roles = RuntimeProfileRoles(
        model_artifact_identity_key="artifact-1",
        benchmark_canonical_profile="binding-key-1",
        best_observed_profile=BestObservedSelection(
            runtime_profile_identity_key="profile-best", selection_objective="generation_speed",
            measurement_ref="measurement-2",
        ),
    )
    assert isinstance(roles.benchmark_canonical_profile, str)
    assert isinstance(roles.best_observed_profile, BestObservedSelection)
    assert roles.best_observed_profile.selection_objective == "generation_speed"
