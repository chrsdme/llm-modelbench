"""Anvil Stage 2.4: EnvironmentDecision.

Covers the exact boundary from this session's Stage 2.4 scoping advice
(`local_only/anvil/Stage 2.4-EnvironmentDecision-codex-advice.txt`):
EnvironmentDecision answers only "can this exact requested configuration
run here?", never "does this model support this capability?" -- that
stays capability_projection.py's job. Purely additive, not wired into
any command path. See environment_decision.py's module docstring for
the full precedence/non-goal list.
"""
import pytest

from llm_modelbench.environment_decision import (
    EnvironmentDecision,
    EnvironmentDecisionReason,
    EnvironmentEvidence,
    EnvironmentEvidenceTier,
    EnvironmentFit,
    EnvironmentRequestIdentity,
    decide_environment_fit,
)
from llm_modelbench.identity import ModelArtifactIdentity, RuntimeProfileIdentity

TIER = EnvironmentEvidenceTier
FIT = EnvironmentFit
REASON = EnvironmentDecisionReason


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


def _request(**overrides):
    kwargs = dict(
        model_identity=_model(), runtime_profile_identity=_profile(),
        gpu_topology_key="gpu-0", context_length=4096,
        kv_cache_policy="f16", offload_split_policy="single_gpu",
    )
    kwargs.update(overrides)
    return EnvironmentRequestIdentity(**kwargs)


def _evidence(request_identity, *, tier, fit, detail="", source=None):
    return EnvironmentEvidence(
        tier=tier, fit=fit, request_identity=request_identity, detail=detail, source=source,
    )


# --- EnvironmentRequestIdentity validation ---


def test_request_identity_rejects_legacy_model_dict():
    with pytest.raises(TypeError, match="ModelArtifactIdentity"):
        EnvironmentRequestIdentity(
            model_identity={"digest": "x"}, runtime_profile_identity=_profile(),
            gpu_topology_key="gpu-0", context_length=4096,
        )


def test_request_identity_rejects_legacy_runtime_dict():
    with pytest.raises(TypeError, match="RuntimeProfileIdentity"):
        EnvironmentRequestIdentity(
            model_identity=_model(), runtime_profile_identity={"backend": "ollama"},
            gpu_topology_key="gpu-0", context_length=4096,
        )


def test_request_identity_rejects_empty_gpu_topology_key():
    with pytest.raises(ValueError, match="gpu_topology_key"):
        _request(gpu_topology_key="")


@pytest.mark.parametrize("bad_context_length", [0, -1, True])
def test_request_identity_rejects_invalid_context_length(bad_context_length):
    with pytest.raises(ValueError, match="context_length"):
        _request(context_length=bad_context_length)


# --- EnvironmentEvidence validation ---


def test_evidence_rejects_conservative_unknown_as_input_tier():
    with pytest.raises(ValueError, match="CONSERVATIVE_UNKNOWN"):
        EnvironmentEvidence(
            tier=TIER.CONSERVATIVE_UNKNOWN, fit=FIT.UNKNOWN, request_identity=_request(),
        )


def test_evidence_rejects_non_enum_tier():
    with pytest.raises(TypeError, match="EnvironmentEvidenceTier"):
        EnvironmentEvidence(tier="measured_execution", fit=FIT.FITS, request_identity=_request())


def test_evidence_rejects_non_enum_fit():
    with pytest.raises(TypeError, match="EnvironmentFit"):
        EnvironmentEvidence(tier=TIER.MEASURED_EXECUTION, fit="fits", request_identity=_request())


def test_evidence_rejects_legacy_dict_as_request_identity():
    with pytest.raises(TypeError, match="EnvironmentRequestIdentity"):
        EnvironmentEvidence(
            tier=TIER.MEASURED_EXECUTION, fit=FIT.FITS, request_identity={"context_length": 4096},
        )


# --- decide_environment_fit: no evidence / identity scoping ---


def test_no_matching_evidence_is_conservative_unknown():
    request = _request()
    decision = decide_environment_fit([], request_identity=request)
    assert decision == EnvironmentDecision(
        request_identity=request, fits=None, reason=REASON.NO_MATCHING_EVIDENCE,
        tier=TIER.CONSERVATIVE_UNKNOWN, considered_evidence=(),
    )


def test_4k_success_does_not_prove_64k_fit():
    request_4k = _request(context_length=4096)
    request_64k = _request(context_length=65536)
    evidence_4k = _evidence(request_4k, tier=TIER.MEASURED_EXECUTION, fit=FIT.FITS)
    decision = decide_environment_fit([evidence_4k], request_identity=request_64k)
    assert decision == EnvironmentDecision(
        request_identity=request_64k, fits=None, reason=REASON.NO_MATCHING_EVIDENCE,
        tier=TIER.CONSERVATIVE_UNKNOWN, considered_evidence=(),
    )


def test_matching_uses_stable_key_not_field_order_sensitive_equality():
    # RuntimeProfileIdentity's own contract (identity.py) says compare/key
    # on stable_key(), never field-by-field equality -- feature_flags order
    # must not matter. Evidence recorded under one flag order must still be
    # found when queried under a different order for the "same" profile.
    profile_recorded = RuntimeProfileIdentity(
        backend="ollama", backend_version="0.5.1", protocol_version="capability-smoke-v2",
        template_hash="template-hash-1", runtime_configuration_hash="cfg-1",
        feature_flags=("vision", "tools"),
    )
    profile_queried = RuntimeProfileIdentity(
        backend="ollama", backend_version="0.5.1", protocol_version="capability-smoke-v2",
        template_hash="template-hash-1", runtime_configuration_hash="cfg-1",
        feature_flags=("tools", "vision"),
    )
    assert profile_recorded != profile_queried  # different tuple order -> not dataclass-equal
    assert profile_recorded.stable_key() == profile_queried.stable_key()  # but same stable identity

    request_recorded = _request(runtime_profile_identity=profile_recorded)
    request_queried = _request(runtime_profile_identity=profile_queried)
    evidence = _evidence(request_recorded, tier=TIER.MEASURED_EXECUTION, fit=FIT.FITS)
    decision = decide_environment_fit([evidence], request_identity=request_queried)
    assert decision.fits is True
    assert decision.tier == TIER.MEASURED_EXECUTION
    assert decision.selected_evidence == (evidence,)


def test_only_unknown_evidence_is_inconclusive_not_no_evidence():
    request = _request()
    unknown = _evidence(request, tier=TIER.VALIDATED_ESTIMATOR, fit=FIT.UNKNOWN, detail="kv_estimate_unavailable")
    decision = decide_environment_fit([unknown], request_identity=request)
    assert decision == EnvironmentDecision(
        request_identity=request, fits=None, reason=REASON.INCONCLUSIVE,
        tier=TIER.CONSERVATIVE_UNKNOWN, considered_evidence=(unknown,),
    )


# --- decide_environment_fit: single-tier decisions ---


def test_single_measured_execution_fits():
    request = _request()
    evidence = _evidence(request, tier=TIER.MEASURED_EXECUTION, fit=FIT.FITS)
    decision = decide_environment_fit([evidence], request_identity=request)
    assert decision == EnvironmentDecision(
        request_identity=request, fits=True, reason=REASON.FITS, tier=TIER.MEASURED_EXECUTION,
        selected_evidence=(evidence,), considered_evidence=(evidence,),
    )


def test_single_measured_execution_does_not_fit():
    request = _request()
    evidence = _evidence(request, tier=TIER.MEASURED_EXECUTION, fit=FIT.DOES_NOT_FIT, detail="kv_cache_exceeds_vram_budget")
    decision = decide_environment_fit([evidence], request_identity=request)
    assert decision == EnvironmentDecision(
        request_identity=request, fits=False, reason=REASON.DOES_NOT_FIT, tier=TIER.MEASURED_EXECUTION,
        selected_evidence=(evidence,), considered_evidence=(evidence,),
    )


@pytest.mark.parametrize(
    "tier",
    [TIER.MEASURED_EXECUTION, TIER.OBSERVED_ALLOCATION, TIER.RUNTIME_DECLARATION, TIER.VALIDATED_ESTIMATOR],
)
def test_each_tier_alone_can_ground_a_fits_decision(tier):
    request = _request()
    evidence = _evidence(request, tier=tier, fit=FIT.FITS)
    decision = decide_environment_fit([evidence], request_identity=request)
    assert decision.fits is True
    assert decision.tier == tier
    assert decision.reason == REASON.FITS


# --- decide_environment_fit: precedence across tiers ---


def test_higher_tier_wins_regardless_of_lower_tier_agreement():
    request = _request()
    measured = _evidence(request, tier=TIER.MEASURED_EXECUTION, fit=FIT.FITS)
    estimator = _evidence(request, tier=TIER.VALIDATED_ESTIMATOR, fit=FIT.FITS)
    decision = decide_environment_fit([estimator, measured], request_identity=request)
    assert decision.tier == TIER.MEASURED_EXECUTION
    assert decision.selected_evidence == (measured,)
    assert decision.contradicting_evidence == ()  # agreement is not a contradiction


def test_measured_evidence_wins_over_estimator_conflict_and_contradiction_is_recorded():
    request = _request()
    measured_fails = _evidence(request, tier=TIER.MEASURED_EXECUTION, fit=FIT.DOES_NOT_FIT, detail="oom")
    estimator_says_fits = _evidence(request, tier=TIER.VALIDATED_ESTIMATOR, fit=FIT.FITS, detail="metadata_estimate")
    decision = decide_environment_fit([estimator_says_fits, measured_fails], request_identity=request)
    assert decision == EnvironmentDecision(
        request_identity=request, fits=False, reason=REASON.DOES_NOT_FIT, tier=TIER.MEASURED_EXECUTION,
        selected_evidence=(measured_fails,),
        contradicting_evidence=(estimator_says_fits,),
        considered_evidence=(estimator_says_fits, measured_fails),
    )


def test_unknown_at_higher_tier_falls_through_to_next_tier():
    request = _request()
    measured_unknown = _evidence(request, tier=TIER.MEASURED_EXECUTION, fit=FIT.UNKNOWN)
    observed_fits = _evidence(request, tier=TIER.OBSERVED_ALLOCATION, fit=FIT.FITS)
    decision = decide_environment_fit([measured_unknown, observed_fits], request_identity=request)
    assert decision == EnvironmentDecision(
        request_identity=request, fits=True, reason=REASON.FITS, tier=TIER.OBSERVED_ALLOCATION,
        selected_evidence=(observed_fits,),
        considered_evidence=(measured_unknown, observed_fits),
    )


def test_within_tier_disagreement_fails_closed_to_does_not_fit():
    request = _request()
    fits_item = _evidence(request, tier=TIER.MEASURED_EXECUTION, fit=FIT.FITS, source="run-a")
    does_not_fit_item = _evidence(request, tier=TIER.MEASURED_EXECUTION, fit=FIT.DOES_NOT_FIT, source="run-b")
    decision = decide_environment_fit([fits_item, does_not_fit_item], request_identity=request)
    assert decision == EnvironmentDecision(
        request_identity=request, fits=False, reason=REASON.DOES_NOT_FIT, tier=TIER.MEASURED_EXECUTION,
        selected_evidence=(does_not_fit_item,),
        contradicting_evidence=(fits_item,),
        considered_evidence=(fits_item, does_not_fit_item),
    )


# --- decide_environment_fit: capability/environment separation ---


def test_decision_has_no_capability_concept_leaking_in():
    # Structural proof that EnvironmentDecisionReason cannot collide with
    # CapabilityDecisionReason or EvalStatus -- disjoint enum value sets,
    # not just documentation.
    from llm_modelbench.capability_projection import CapabilityDecisionReason
    from llm_modelbench.evidence import EvalStatus

    environment_values = {member.value for member in EnvironmentDecisionReason}
    capability_values = {member.value for member in CapabilityDecisionReason}
    eval_status_values = {member.value for member in EvalStatus}
    assert environment_values.isdisjoint(capability_values)
    assert environment_values.isdisjoint(eval_status_values)


# --- type guards on decide_environment_fit itself ---


def test_rejects_non_environment_evidence_item():
    with pytest.raises(TypeError, match="EnvironmentEvidence"):
        decide_environment_fit([{"tier": "measured_execution"}], request_identity=_request())


def test_rejects_non_request_identity():
    with pytest.raises(TypeError, match="EnvironmentRequestIdentity"):
        decide_environment_fit([], request_identity={"context_length": 4096})
