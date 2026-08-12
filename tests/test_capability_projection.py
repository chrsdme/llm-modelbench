"""Anvil Stage 2.3: CapabilityProjection + CapabilityDecision.

Covers the exact test matrix from this session's Codex/GPT Stage 2.3
scoping pass (`local_only/anvil/codex-advice_stage2.3.txt`). Scope is
deliberately the typed projection/decision read model only -- no
EnvironmentDecision, no TaskApplicability, no call-site migration, no
ProjectionStore current-pointer adoption. See capability_projection.py's
module docstring for the full non-goal list.
"""
import pytest

from llm_modelbench.capabilities import MeasuredCapabilityState
from llm_modelbench.capability_identity import (
    CapabilityIdentityCompatibility,
    CapabilityIdentityCompatibilityReason,
)
from llm_modelbench.capability_observation import CapabilityObservation, append_capability_observation
from llm_modelbench.capability_projection import (
    CapabilityDecision,
    CapabilityDecisionReason,
    CapabilityProjection,
    CapabilityProjectionStatus,
    decide_capability_from_projection,
    project_capability_from_ledger,
    project_capability_observation,
)
from llm_modelbench.evidence import EvidenceLedger, ProvenanceLink, ProvenanceRelation
from llm_modelbench.identity import ModelArtifactIdentity, RuntimeProfileIdentity

STATUS = CapabilityProjectionStatus
REASON = CapabilityIdentityCompatibilityReason
DREASON = CapabilityDecisionReason


def _model(*, digest="digest-1"):
    return ModelArtifactIdentity(
        artifact_set_id=digest, primary_sha256=digest, size_bytes=9_000_000_000,
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


def _current(**overrides):
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


def test_no_observations_for_capability():
    projection = project_capability_observation([], capability="text", **_current())
    assert projection == CapabilityProjection(
        capability="text", status=STATUS.NO_OBSERVATIONS, considered_observation_ids=(),
    )


def test_observations_for_other_capability_do_not_count():
    vision_only = _observation(capability="vision", result=MeasuredCapabilityState.MEASURED_SUPPORTED)
    projection = project_capability_observation([vision_only], capability="text", **_current())
    assert projection == CapabilityProjection(
        capability="text", status=STATUS.NO_OBSERVATIONS, considered_observation_ids=(),
    )


def test_only_incompatible_observations():
    observation = _observation(model_identity=_model(digest="digest-2"))
    projection = project_capability_observation([observation], capability="text", **_current())
    expected_incompat = CapabilityIdentityCompatibility(
        compatible=False, reason=REASON.MODEL_ARTIFACT_CHANGED, stored_evidence_hash=observation.evidence_hash,
    )
    assert projection == CapabilityProjection(
        capability="text", status=STATUS.NO_COMPATIBLE_OBSERVATION,
        considered_observation_ids=(observation.observation_id,),
        incompatible=(expected_incompat,),
    )


def test_single_compatible_observation_is_selected():
    observation = _observation()
    projection = project_capability_observation([observation], capability="text", **_current())
    assert projection == CapabilityProjection(
        capability="text", status=STATUS.SELECTED,
        selected_observation=observation,
        compatibility=CapabilityIdentityCompatibility(
            compatible=True, reason=REASON.IDENTITY_MATCH, stored_evidence_hash=observation.evidence_hash,
        ),
        considered_observation_ids=(observation.observation_id,),
    )


def test_duplicate_evidence_hash_selects_newest_timestamp():
    older = _observation(timestamp="2026-08-10T00:00:00Z")
    newer = _observation(timestamp="2026-08-11T00:00:00Z")
    assert older.evidence_hash == newer.evidence_hash  # same semantic measurement
    projection = project_capability_observation([older, newer], capability="text", **_current())
    assert projection.status == STATUS.SELECTED
    assert projection == CapabilityProjection(
        capability="text", status=STATUS.SELECTED,
        selected_observation=newer,
        compatibility=CapabilityIdentityCompatibility(
            compatible=True, reason=REASON.IDENTITY_MATCH, stored_evidence_hash=newer.evidence_hash,
        ),
        considered_observation_ids=(older.observation_id, newer.observation_id),
    )


def test_contradictory_compatible_observations_are_ambiguous():
    supported = _observation(
        timestamp="2026-08-10T00:00:00Z", result=MeasuredCapabilityState.MEASURED_SUPPORTED,
    )
    unsupported = _observation(
        timestamp="2026-08-11T00:00:00Z", result=MeasuredCapabilityState.MEASURED_UNSUPPORTED,
    )
    assert supported.evidence_hash != unsupported.evidence_hash
    projection = project_capability_observation([supported, unsupported], capability="text", **_current())
    assert projection == CapabilityProjection(
        capability="text", status=STATUS.AMBIGUOUS_COMPATIBLE_OBSERVATIONS,
        considered_observation_ids=(supported.observation_id, unsupported.observation_id),
    )


def test_probe_inconclusive_never_becomes_applicable():
    observation = _observation(result=MeasuredCapabilityState.PROBE_INCONCLUSIVE)
    projection = project_capability_observation([observation], capability="text", **_current())
    decision = decide_capability_from_projection(projection)
    assert decision == CapabilityDecision(
        capability="text", applicable=False, reason=DREASON.PROBE_INCONCLUSIVE,
        projection=projection, selected_result=MeasuredCapabilityState.PROBE_INCONCLUSIVE,
    )


@pytest.mark.parametrize(
    "result,expected_reason",
    [
        (MeasuredCapabilityState.MEASURED_UNSUPPORTED, DREASON.MEASURED_UNSUPPORTED),
        (MeasuredCapabilityState.BACKEND_UNSUPPORTED, DREASON.BACKEND_UNSUPPORTED),
        (MeasuredCapabilityState.NOT_APPLICABLE, DREASON.NOT_APPLICABLE),
    ],
)
def test_negative_results_never_become_applicable(result, expected_reason):
    observation = _observation(result=result)
    projection = project_capability_observation([observation], capability="text", **_current())
    decision = decide_capability_from_projection(projection)
    assert decision == CapabilityDecision(
        capability="text", applicable=False, reason=expected_reason,
        projection=projection, selected_result=result,
    )


def test_measured_supported_is_applicable():
    observation = _observation(result=MeasuredCapabilityState.MEASURED_SUPPORTED)
    projection = project_capability_observation([observation], capability="text", **_current())
    decision = decide_capability_from_projection(projection)
    assert decision == CapabilityDecision(
        capability="text", applicable=True, reason=DREASON.MEASURED_SUPPORTED,
        projection=projection, selected_result=MeasuredCapabilityState.MEASURED_SUPPORTED,
    )


def test_schema_version_mismatch_cannot_fall_through_to_positive_decision():
    observation = _observation(result=MeasuredCapabilityState.MEASURED_SUPPORTED)
    projection = project_capability_observation(
        [observation], capability="text", **_current(current_capability_schema_version=99)
    )
    decision = decide_capability_from_projection(projection)
    assert projection.status == STATUS.NO_COMPATIBLE_OBSERVATION
    assert decision == CapabilityDecision(
        capability="text", applicable=False, reason=DREASON.NO_CURRENT_PROJECTION,
        projection=projection, selected_result=None,
    )


def test_no_observations_decision_is_not_applicable():
    projection = project_capability_observation([], capability="text", **_current())
    decision = decide_capability_from_projection(projection)
    assert decision == CapabilityDecision(
        capability="text", applicable=False, reason=DREASON.NO_CURRENT_PROJECTION,
        projection=projection, selected_result=None,
    )


def test_ambiguous_decision_is_not_applicable():
    supported = _observation(timestamp="2026-08-10T00:00:00Z", result=MeasuredCapabilityState.MEASURED_SUPPORTED)
    unsupported = _observation(timestamp="2026-08-11T00:00:00Z", result=MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    projection = project_capability_observation([supported, unsupported], capability="text", **_current())
    decision = decide_capability_from_projection(projection)
    assert decision == CapabilityDecision(
        capability="text", applicable=False, reason=DREASON.PROJECTION_AMBIGUOUS,
        projection=projection, selected_result=None,
    )


def test_rejects_non_observation_in_iterable():
    with pytest.raises(TypeError, match="CapabilityObservation"):
        project_capability_observation(
            [{"capability": "text"}], capability="text", **_current()
        )


def test_rejects_unknown_capability_family():
    with pytest.raises(ValueError, match="not a known family"):
        project_capability_observation([], capability="not_a_real_family", **_current())


def test_decide_rejects_non_projection():
    with pytest.raises(TypeError, match="CapabilityProjection"):
        decide_capability_from_projection({"status": "selected"})


# --- Ledger-aware projection (SUPERSEDES resolution via EffectiveEvidenceResolver) ---


def test_ledger_projection_with_no_records(tmp_path):
    ledger = EvidenceLedger(tmp_path / "capability.jsonl")
    projection = project_capability_from_ledger(ledger, capability="text", **_current())
    assert projection == CapabilityProjection(capability="text", status=STATUS.NO_OBSERVATIONS)


def test_ledger_projection_selects_single_observation(tmp_path):
    ledger = EvidenceLedger(tmp_path / "capability.jsonl")
    observation = _observation()
    append_capability_observation(ledger, observation)
    projection = project_capability_from_ledger(ledger, capability="text", **_current())
    assert projection == CapabilityProjection(
        capability="text", status=STATUS.SELECTED,
        selected_observation=observation,
        selected_record_id=observation.observation_id,
        compatibility=CapabilityIdentityCompatibility(
            compatible=True, reason=REASON.IDENTITY_MATCH, stored_evidence_hash=observation.evidence_hash,
        ),
        considered_observation_ids=(observation.observation_id,),
    )


def test_ledger_supersession_chain_selects_terminal_observation(tmp_path):
    ledger = EvidenceLedger(tmp_path / "capability.jsonl")
    original = _observation(
        timestamp="2026-08-10T00:00:00Z", result=MeasuredCapabilityState.MEASURED_UNSUPPORTED,
    )
    append_capability_observation(ledger, original)
    replacement = _observation(
        timestamp="2026-08-11T00:00:00Z", result=MeasuredCapabilityState.MEASURED_SUPPORTED,
    )
    append_capability_observation(
        ledger, replacement,
        provenance=(ProvenanceLink(ProvenanceRelation.SUPERSEDES, original.observation_id),),
    )
    projection = project_capability_from_ledger(ledger, capability="text", **_current())
    assert projection.status == STATUS.SELECTED
    assert projection.selected_observation == replacement
    assert projection.selected_record_id == replacement.observation_id


def test_ledger_supersession_fork_is_conflict(tmp_path):
    ledger = EvidenceLedger(tmp_path / "capability.jsonl")
    original = _observation(timestamp="2026-08-10T00:00:00Z")
    append_capability_observation(ledger, original)
    fork_a = _observation(timestamp="2026-08-11T00:00:00Z", template_config_hash="fork-a")
    fork_b = _observation(timestamp="2026-08-11T00:00:00Z", template_config_hash="fork-b")
    append_capability_observation(
        ledger, fork_a, provenance=(ProvenanceLink(ProvenanceRelation.SUPERSEDES, original.observation_id),)
    )
    append_capability_observation(
        ledger, fork_b, provenance=(ProvenanceLink(ProvenanceRelation.SUPERSEDES, original.observation_id),)
    )
    projection = project_capability_from_ledger(ledger, capability="text", **_current())
    assert projection.status == STATUS.SUPERSESSION_CONFLICT
    assert projection.selected_observation is None


def test_ledger_projection_terminal_order_is_deterministic(tmp_path):
    # Two independent, non-superseding compatible observations both reach
    # project_capability_observation()'s delegation path as separate
    # terminals -- project_capability_from_ledger() collects candidate
    # terminals via a plain `set()` internally (record_ids as hash keys),
    # so considered_observation_ids must be explicitly sorted before use,
    # or this field would be flaky across interpreter runs under
    # PYTHONHASHSEED rather than a stable, assertable value.
    ledger = EvidenceLedger(tmp_path / "capability.jsonl")
    left = _observation(timestamp="2026-08-10T00:00:00Z", result=MeasuredCapabilityState.MEASURED_SUPPORTED)
    right = _observation(timestamp="2026-08-11T00:00:00Z", result=MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    append_capability_observation(ledger, left)
    append_capability_observation(ledger, right)
    expected_considered = tuple(sorted((left.observation_id, right.observation_id)))
    projection = project_capability_from_ledger(ledger, capability="text", **_current())
    assert projection.status == STATUS.AMBIGUOUS_COMPATIBLE_OBSERVATIONS
    assert projection.considered_observation_ids == expected_considered


def test_ledger_supersession_cycle_is_conflict(tmp_path):
    ledger = EvidenceLedger(tmp_path / "capability.jsonl")
    # observation_id is computed purely from an observation's own semantic
    # content (see capability_observation.py), independent of ledger append
    # order -- so both sides of a genuine two-node SUPERSEDES cycle can be
    # constructed before either is appended.
    obs_a = _observation(timestamp="2026-08-10T00:00:00Z", template_config_hash="cycle-a")
    obs_b = _observation(timestamp="2026-08-11T00:00:00Z", template_config_hash="cycle-b")
    append_capability_observation(
        ledger, obs_a, provenance=(ProvenanceLink(ProvenanceRelation.SUPERSEDES, obs_b.observation_id),)
    )
    append_capability_observation(
        ledger, obs_b, provenance=(ProvenanceLink(ProvenanceRelation.SUPERSEDES, obs_a.observation_id),)
    )
    projection = project_capability_from_ledger(ledger, capability="text", **_current())
    assert projection.status == STATUS.SUPERSESSION_CONFLICT
    assert projection.selected_observation is None
