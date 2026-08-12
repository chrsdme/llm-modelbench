"""Anvil Stage 2.5: TaskApplicability composition.

Covers the test matrix from this session's Stage 2.5 continuing advice
(`local_only/anvil/codex-advice-stage2-continuing.txt`), adjusted where
that advice's suggested reason vocabulary didn't match what Stage 2.3/2.4
actually produce -- see task_applicability.py's module docstring for the
two deliberate departures and their rationale. Purely additive, not wired
into any command path.
"""
import pytest

from llm_modelbench.capabilities import MeasuredCapabilityState
from llm_modelbench.capability_observation import CapabilityObservation
from llm_modelbench.capability_projection import (
    CapabilityDecisionReason,
    decide_capability_from_projection,
    project_capability_observation,
)
from llm_modelbench.environment_decision import (
    EnvironmentEvidence,
    EnvironmentEvidenceTier,
    EnvironmentFit,
    EnvironmentRequestIdentity,
    decide_environment_fit,
)
from llm_modelbench.identity import ModelArtifactIdentity, RuntimeProfileIdentity
from llm_modelbench.task_applicability import (
    OperatorTaskDecision,
    TaskApplicability,
    TaskApplicabilityReason,
    TaskApplicabilityStatus,
    compose_task_applicability,
)

STATUS = TaskApplicabilityStatus
REASON = TaskApplicabilityReason


def _model():
    return ModelArtifactIdentity(
        artifact_set_id="digest-1", primary_sha256="digest-1", size_bytes=9_000_000_000,
        format="ollama-blob", quantization="Q4_K_M", source="qwen2.5-coder:14b",
    )


def _profile():
    return RuntimeProfileIdentity(
        backend="ollama", backend_version="0.5.1", protocol_version="capability-smoke-v2",
        template_hash="template-hash-1", runtime_configuration_hash="cfg-1",
    )


def _capability_current(**overrides):
    kwargs = dict(
        current_model_identity=_model(),
        current_runtime_profile_identity=_profile(),
        current_probe_protocol_version="capability-smoke-v2",
        current_capability_schema_version=2,
        current_template_config_hash="tmpl-abc",
        current_endpoint_identity="http://127.0.0.1:11434",
    )
    kwargs.update(overrides)
    return kwargs


def _observation(**overrides):
    kwargs = dict(
        model_identity=_model(), runtime_profile_identity=_profile(),
        capability="text", result=MeasuredCapabilityState.MEASURED_SUPPORTED,
        probe_protocol_version="capability-smoke-v2", capability_schema_version=2,
        template_config_hash="tmpl-abc", endpoint_identity="http://127.0.0.1:11434",
    )
    kwargs.update(overrides)
    return CapabilityObservation(**kwargs)


def _capability_decision(**observation_overrides):
    observation = _observation(**observation_overrides)
    projection = project_capability_observation([observation], capability="text", **_capability_current())
    return decide_capability_from_projection(projection)


def _capability_decision_no_projection():
    projection = project_capability_observation([], capability="text", **_capability_current())
    return decide_capability_from_projection(projection)


def _capability_decision_ambiguous():
    a = _observation(timestamp="2026-08-10T00:00:00Z", result=MeasuredCapabilityState.MEASURED_SUPPORTED)
    b = _observation(timestamp="2026-08-11T00:00:00Z", result=MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    projection = project_capability_observation([a, b], capability="text", **_capability_current())
    return decide_capability_from_projection(projection)


def _environment_request():
    return EnvironmentRequestIdentity(
        model_identity=_model(), runtime_profile_identity=_profile(),
        gpu_topology_key="gpu-0", context_length=4096,
    )


def _environment_decision(fit):
    request = _environment_request()
    evidence = EnvironmentEvidence(tier=EnvironmentEvidenceTier.MEASURED_EXECUTION, fit=fit, request_identity=request)
    return decide_environment_fit([evidence], request_identity=request)


def _environment_decision_inconclusive():
    return decide_environment_fit([], request_identity=_environment_request())


ALLOWED = OperatorTaskDecision(allowed=True)
EXCLUDED = OperatorTaskDecision(allowed=False, reason="cost budget")


# --- the advice's core composition matrix ---


def test_capability_supported_environment_fits_operator_allows_is_applicable():
    capability = _capability_decision(result=MeasuredCapabilityState.MEASURED_SUPPORTED)
    environment = _environment_decision(EnvironmentFit.FITS)
    result = compose_task_applicability(
        capability_decision=capability, environment_decision=environment, operator_decision=ALLOWED,
    )
    assert result == TaskApplicability(
        status=STATUS.APPLICABLE, terminal_reason=REASON.APPLICABLE,
        capability_decision=capability, environment_decision=environment, operator_decision=ALLOWED,
        evidence_refs=capability.projection.considered_observation_ids,
    )


def test_capability_unsupported_short_circuits_regardless_of_environment():
    capability = _capability_decision(result=MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    environment = _environment_decision(EnvironmentFit.FITS)  # would otherwise be fine
    result = compose_task_applicability(
        capability_decision=capability, environment_decision=environment, operator_decision=ALLOWED,
    )
    assert result.status == STATUS.NOT_APPLICABLE
    assert result.terminal_reason == REASON.CAPABILITY_UNSUPPORTED
    assert result.environment_decision == environment  # preserved even though not the terminal reason


def test_capability_inconclusive_fails_closed():
    capability = _capability_decision(result=MeasuredCapabilityState.PROBE_INCONCLUSIVE)
    result = compose_task_applicability(capability_decision=capability, operator_decision=ALLOWED)
    assert result.terminal_reason == REASON.CAPABILITY_INCONCLUSIVE
    assert result.status == STATUS.NOT_APPLICABLE


def test_capability_ambiguous_projection_is_inconclusive_not_reprobe_required():
    # Distinct from "never measured": contradictory evidence exists.
    capability = _capability_decision_ambiguous()
    assert capability.reason == CapabilityDecisionReason.PROJECTION_AMBIGUOUS
    result = compose_task_applicability(capability_decision=capability, operator_decision=ALLOWED)
    assert result.terminal_reason == REASON.CAPABILITY_INCONCLUSIVE


def test_capability_no_projection_requires_reprobe():
    # No compatible observation exists at all -- distinct from ambiguous.
    capability = _capability_decision_no_projection()
    assert capability.reason == CapabilityDecisionReason.NO_CURRENT_PROJECTION
    result = compose_task_applicability(capability_decision=capability, operator_decision=ALLOWED)
    assert result.terminal_reason == REASON.CAPABILITY_REPROBE_REQUIRED


@pytest.mark.parametrize(
    "measured_result,expected_reason",
    [
        (MeasuredCapabilityState.MEASURED_UNSUPPORTED, REASON.CAPABILITY_UNSUPPORTED),
        (MeasuredCapabilityState.BACKEND_UNSUPPORTED, REASON.CAPABILITY_UNSUPPORTED),
        (MeasuredCapabilityState.NOT_APPLICABLE, REASON.CAPABILITY_UNSUPPORTED),
        (MeasuredCapabilityState.PROBE_INCONCLUSIVE, REASON.CAPABILITY_INCONCLUSIVE),
    ],
)
def test_every_capability_block_reason_maps_to_a_real_task_applicability_reason(measured_result, expected_reason):
    # _CAPABILITY_BLOCK_REASON has six keys; the four MeasuredCapabilityState-
    # derived ones are exercised here, the other two (NO_CURRENT_PROJECTION,
    # PROJECTION_AMBIGUOUS -- reachable only via projection status, not a
    # measured result) are covered by test_capability_no_projection_requires_reprobe
    # and test_capability_ambiguous_projection_is_inconclusive_not_reprobe_required
    # above. Together these exercise every key in the dict lookup, so a future
    # CapabilityDecisionReason addition without a matching dict entry fails
    # loudly here (KeyError) rather than only in production.
    capability = _capability_decision(result=measured_result)
    result = compose_task_applicability(capability_decision=capability, operator_decision=ALLOWED)
    assert result.terminal_reason == expected_reason
    assert result.status == STATUS.NOT_APPLICABLE


def test_capability_supported_environment_does_not_fit():
    capability = _capability_decision(result=MeasuredCapabilityState.MEASURED_SUPPORTED)
    environment = _environment_decision(EnvironmentFit.DOES_NOT_FIT)
    result = compose_task_applicability(
        capability_decision=capability, environment_decision=environment, operator_decision=ALLOWED,
    )
    assert result.terminal_reason == REASON.ENVIRONMENT_DOES_NOT_FIT
    assert result.status == STATUS.NOT_APPLICABLE


def test_capability_supported_environment_not_evaluated():
    # Distinct from "evaluated but inconclusive" -- environment_decision
    # simply was not supplied (e.g. capability already blocked earlier in
    # a real pipeline, or measurement genuinely hasn't run yet).
    capability = _capability_decision(result=MeasuredCapabilityState.MEASURED_SUPPORTED)
    result = compose_task_applicability(capability_decision=capability, operator_decision=ALLOWED)
    assert result.terminal_reason == REASON.ENVIRONMENT_NOT_EVALUATED
    assert result.environment_decision is None


def test_capability_supported_environment_evaluated_but_inconclusive():
    capability = _capability_decision(result=MeasuredCapabilityState.MEASURED_SUPPORTED)
    environment = _environment_decision_inconclusive()
    result = compose_task_applicability(
        capability_decision=capability, environment_decision=environment, operator_decision=ALLOWED,
    )
    assert result.terminal_reason == REASON.ENVIRONMENT_INCONCLUSIVE
    assert result.environment_decision is environment


def test_everything_valid_but_operator_excludes():
    capability = _capability_decision(result=MeasuredCapabilityState.MEASURED_SUPPORTED)
    environment = _environment_decision(EnvironmentFit.FITS)
    result = compose_task_applicability(
        capability_decision=capability, environment_decision=environment, operator_decision=EXCLUDED,
    )
    assert result == TaskApplicability(
        status=STATUS.NOT_APPLICABLE, terminal_reason=REASON.OPERATOR_EXCLUDED,
        capability_decision=capability, environment_decision=environment, operator_decision=EXCLUDED,
        evidence_refs=capability.projection.considered_observation_ids,
    )


def test_operator_exclusion_wins_even_over_capability_failure():
    # Precedence proof: operator exclusion is checked first, regardless of
    # what capability/environment would otherwise have said.
    capability = _capability_decision(result=MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    result = compose_task_applicability(capability_decision=capability, operator_decision=EXCLUDED)
    assert result.terminal_reason == REASON.OPERATOR_EXCLUDED


def test_environment_failure_never_becomes_capability_unsupported_or_harness_error():
    # TaskApplicabilityReason deliberately REUSES "capability_unsupported"/
    # "capability_inconclusive"/"operator_excluded" as vocabulary shared
    # with evidence.EvalStatus -- that's correct, not a leak (a
    # CAPABILITY_UNSUPPORTED TaskApplicability should map onto an
    # EvalStatus.CAPABILITY_UNSUPPORTED row in a later slice). What must
    # never happen is an *environment*-tier outcome being mis-tagged as
    # one of those capability/harness values. Check only the three
    # environment-outcome reasons, which have no legitimate reason to
    # collide with EvalStatus at all.
    from llm_modelbench.evidence import EvalStatus

    environment_only_values = {
        REASON.ENVIRONMENT_DOES_NOT_FIT.value,
        REASON.ENVIRONMENT_INCONCLUSIVE.value,
        REASON.ENVIRONMENT_NOT_EVALUATED.value,
    }
    eval_status_values = {member.value for member in EvalStatus}
    assert environment_only_values.isdisjoint(eval_status_values)
    assert "harness_error" not in {member.value for member in TaskApplicabilityReason}

    capability = _capability_decision(result=MeasuredCapabilityState.MEASURED_SUPPORTED)
    environment = _environment_decision(EnvironmentFit.DOES_NOT_FIT)
    result = compose_task_applicability(
        capability_decision=capability, environment_decision=environment, operator_decision=ALLOWED,
    )
    assert result.terminal_reason == REASON.ENVIRONMENT_DOES_NOT_FIT
    assert result.terminal_reason != REASON.CAPABILITY_UNSUPPORTED


# --- component decisions retained intact ---


def test_component_decisions_retained_intact_even_when_not_the_terminal_reason():
    capability = _capability_decision(result=MeasuredCapabilityState.MEASURED_SUPPORTED)
    environment = _environment_decision(EnvironmentFit.DOES_NOT_FIT)
    result = compose_task_applicability(
        capability_decision=capability, environment_decision=environment, operator_decision=ALLOWED,
    )
    assert result.capability_decision is capability
    assert result.environment_decision is environment
    assert result.operator_decision is ALLOWED


# --- structural type guards ---


def test_rejects_legacy_dict_as_capability_decision():
    with pytest.raises(TypeError, match="CapabilityDecision"):
        compose_task_applicability(
            capability_decision={"applicable": True}, operator_decision=ALLOWED,
        )


def test_rejects_legacy_dict_as_operator_decision():
    capability = _capability_decision(result=MeasuredCapabilityState.MEASURED_SUPPORTED)
    with pytest.raises(TypeError, match="OperatorTaskDecision"):
        compose_task_applicability(capability_decision=capability, operator_decision={"allowed": True})


def test_rejects_legacy_dict_as_environment_decision():
    capability = _capability_decision(result=MeasuredCapabilityState.MEASURED_SUPPORTED)
    with pytest.raises(TypeError, match="EnvironmentDecision"):
        compose_task_applicability(
            capability_decision=capability, operator_decision=ALLOWED,
            environment_decision={"fits": True},
        )


def test_operator_task_decision_rejects_non_bool_allowed():
    with pytest.raises(TypeError, match="bool"):
        OperatorTaskDecision(allowed="yes")
