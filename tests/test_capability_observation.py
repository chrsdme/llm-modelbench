"""Anvil Stage 2.1: CapabilityObservation schema + persistence semantics.

Covers the exact minimum test list from the second Codex/GPT pre-Stage-2
review (`local_only/anvil/codex-advice_pre_stage2.txt`, Part 2). Scope is
deliberately the evidence type and its persistence only -- no planner/
runner/recovery/judge migration, no CapabilityProjection/TaskApplicability,
nothing wired into any command path. See capability_observation.py's module
docstring for the full non-goal list.
"""
import pytest

from llm_modelbench.capabilities import MeasuredCapabilityState
from llm_modelbench.capability_observation import (
    CAPABILITY_OBSERVATION_RECORD_TYPE,
    CapabilityObservation,
)
from llm_modelbench.evidence import EvidenceLedger
from llm_modelbench.identity import ModelArtifactIdentity, RuntimeProfileIdentity



# Anvil Stage 3B.2: append_capability_observation now requires an explicit
# EvidenceTrustClass (owner's frozen rule). These tests exercise ledger /
# projection behaviour, not trust classification, so they pass an explicit
# CANONICAL_COMPATIBLE via this thin shim rather than at every call site.
from llm_modelbench.capability_observation import append_capability_observation as _acobs_real
from llm_modelbench.evidence import EvidenceTrustClass as _ETC
# These files exercise ledger / projection / adapter behaviour, not the
# trust decision itself -- so they pin an explicit CANONICAL_COMPATIBLE
# rather than thread a trust class through every call site. The write-time
# trust decision is proven in tests/test_capability_trust.py and the three
# writer tests in tests/test_capability_reprobe_execute.py. The helper is
# deliberately NOT named like the real function so it cannot be mistaken
# for the production writer (which has no default and never assumes canonical).
def _append_with_explicit_canonical_trust(ledger, observation, *, trust_class=_ETC.CANONICAL_COMPATIBLE, provenance=()):
    return _acobs_real(ledger, observation, trust_class=trust_class, provenance=provenance)
append_capability_observation = _append_with_explicit_canonical_trust

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
    )
    kwargs.update(overrides)
    return CapabilityObservation(**kwargs)


def test_round_trip_observation_without_loss(tmp_path):
    ledger = EvidenceLedger(tmp_path / "capability.jsonl")
    observation = _observation(declared_hint=("completion",), template_config_hash="tmpl-abc")
    record = append_capability_observation(ledger, observation)
    assert record.record_type == CAPABILITY_OBSERVATION_RECORD_TYPE

    reloaded_ledger = EvidenceLedger(tmp_path / "capability.jsonl")
    stored = reloaded_ledger.get(observation.observation_id)
    assert stored is not None
    restored = CapabilityObservation.from_ledger_payload(stored.payload)
    assert restored == observation


def test_two_observations_for_same_capability_coexist(tmp_path):
    ledger = EvidenceLedger(tmp_path / "capability.jsonl")
    first = _observation(timestamp="2026-08-10T00:00:00Z")
    second = _observation(timestamp="2026-08-11T00:00:00Z")
    assert first.observation_id != second.observation_id  # timestamp folded into observation_id
    assert first.evidence_hash == second.evidence_hash  # identical semantic content
    append_capability_observation(ledger, first)
    append_capability_observation(ledger, second)
    stored_ids = {record.record_id for record in ledger.all()}
    assert {first.observation_id, second.observation_id} <= stored_ids


def test_second_observation_does_not_mutate_first(tmp_path):
    ledger = EvidenceLedger(tmp_path / "capability.jsonl")
    first = _observation(timestamp="2026-08-10T00:00:00Z", result=MeasuredCapabilityState.MEASURED_SUPPORTED)
    append_capability_observation(ledger, first)
    second = _observation(timestamp="2026-08-11T00:00:00Z", result=MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    append_capability_observation(ledger, second)
    reloaded = ledger.get(first.observation_id)
    assert reloaded.payload["result"] == MeasuredCapabilityState.MEASURED_SUPPORTED.value
    assert reloaded.payload["timestamp"] == "2026-08-10T00:00:00Z"


def test_result_enum_rejects_unknown_values():
    with pytest.raises(TypeError):
        CapabilityObservation(
            model_identity=_model(), runtime_profile_identity=_profile(),
            capability="text", result="not_a_real_state",
            probe_protocol_version="capability-smoke-v2", capability_schema_version=2,
        )
    with pytest.raises(ValueError):
        MeasuredCapabilityState("not_a_real_state")


def test_inconclusive_stays_distinct_from_unsupported():
    inconclusive = _observation(result=MeasuredCapabilityState.PROBE_INCONCLUSIVE)
    unsupported = _observation(result=MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    assert inconclusive.result != unsupported.result
    assert inconclusive.evidence_hash != unsupported.evidence_hash


def test_declared_hint_can_coexist_with_measured_unsupported():
    observation = _observation(
        declared_hint=("vision",),
        result=MeasuredCapabilityState.MEASURED_UNSUPPORTED,
    )
    assert observation.declared_hint == ("vision",)
    assert observation.result == MeasuredCapabilityState.MEASURED_UNSUPPORTED
    # declared_hint is metadata only -- never authoritative -- and must not
    # be folded into the measurement's own semantic identity.
    without_hint = _observation(declared_hint=(), result=MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    assert observation.evidence_hash == without_hint.evidence_hash


def test_observation_identity_and_hash_are_deterministic():
    one = _observation(timestamp="2026-08-10T00:00:00Z")
    two = _observation(timestamp="2026-08-10T00:00:00Z")
    assert one.evidence_hash == two.evidence_hash
    assert one.observation_id == two.observation_id


def test_material_semantic_change_changes_evidence_hash():
    baseline = _observation()
    assert baseline.evidence_hash != _observation(capability="vision").evidence_hash
    assert baseline.evidence_hash != _observation(result=MeasuredCapabilityState.MEASURED_UNSUPPORTED).evidence_hash
    assert baseline.evidence_hash != _observation(model_identity=_model(digest="digest-2")).evidence_hash
    assert baseline.evidence_hash != _observation(runtime_profile_identity=_profile(backend="llama_cpp")).evidence_hash
    assert baseline.evidence_hash != _observation(template_config_hash="different-template").evidence_hash
    assert baseline.evidence_hash != _observation(capability_schema_version=99).evidence_hash
    assert baseline.evidence_hash != _observation(probe_protocol_version="capability-smoke-v3").evidence_hash


def test_json_field_order_does_not_change_evidence_hash():
    # RuntimeProfileIdentity/ModelArtifactIdentity are positional dataclasses,
    # so this constructs the same logical identity via differently-ordered
    # keyword calls to prove the underlying canonical-JSON hash (sort_keys)
    # is what actually determines equality, not construction order.
    model_a = ModelArtifactIdentity(artifact_set_id="d1", primary_sha256="d1", format="ollama-blob", size_bytes=100)
    model_b = ModelArtifactIdentity(size_bytes=100, format="ollama-blob", primary_sha256="d1", artifact_set_id="d1")
    profile_a = RuntimeProfileIdentity(backend="ollama", backend_version="0.5.1", protocol_version="v2")
    profile_b = RuntimeProfileIdentity(protocol_version="v2", backend_version="0.5.1", backend="ollama")
    left = _observation(model_identity=model_a, runtime_profile_identity=profile_a)
    right = _observation(model_identity=model_b, runtime_profile_identity=profile_b)
    assert left.evidence_hash == right.evidence_hash


def test_legacy_capability_profile_is_not_silently_convertible():
    """Structural guarantee, not an absent-function claim: CapabilityObservation's
    __post_init__ rejects anything that isn't a real ModelArtifactIdentity/
    RuntimeProfileIdentity, so today's legacy profile dict shape
    (declared_capabilities/supported_families, no bound identity objects)
    cannot be passed through -- by type, not by convention."""
    legacy_profile = {
        "model": "legacy:latest",
        "declared_capabilities": ["completion"],
        "supported_families": ["text"],
    }
    with pytest.raises(TypeError, match="ModelArtifactIdentity"):
        CapabilityObservation(
            model_identity=legacy_profile,  # a dict, not a ModelArtifactIdentity
            runtime_profile_identity=_profile(),
            capability="text", result=MeasuredCapabilityState.MEASURED_SUPPORTED,
            probe_protocol_version="capability-smoke-v2", capability_schema_version=2,
        )
    with pytest.raises(TypeError, match="RuntimeProfileIdentity"):
        CapabilityObservation(
            model_identity=_model(),
            runtime_profile_identity={"backend": "ollama"},  # a dict, not a RuntimeProfileIdentity
            capability="text", result=MeasuredCapabilityState.MEASURED_SUPPORTED,
            probe_protocol_version="capability-smoke-v2", capability_schema_version=2,
        )


def test_unknown_capability_family_is_rejected():
    with pytest.raises(ValueError, match="not a known family"):
        _observation(capability="not_a_real_family")


def test_ledger_payload_tamper_detection_on_reload():
    # Tampering `result` changes both the reconstructed observation_id and
    # evidence_hash (both derive from semantic content) -- either mismatch
    # is an acceptable proof of tamper detection, so match their shared
    # trailing message rather than one specific field name.
    payload = _observation().to_ledger_payload()
    tampered = {**payload, "result": MeasuredCapabilityState.MEASURED_UNSUPPORTED.value}
    with pytest.raises(ValueError, match="does not match stored payload"):
        CapabilityObservation.from_ledger_payload(tampered)


# --- Anvil Stage 3.4C: legacy-minimal ledger payloads stay historical ----------


def _minimal_profile():
    """Legacy capability-path identity: no resolved runtime recipe -- the
    shape every capability observation has today (runtime_configuration_hash
    and gpu_policy both None)."""
    return RuntimeProfileIdentity(
        backend="ollama", protocol_version="capability-smoke-v2",
        template_hash="template-hash-1",
    )


def test_legacy_minimal_payload_round_trips_without_enrichment(tmp_path):
    ledger = EvidenceLedger(tmp_path / "capability.jsonl")
    observation = _observation(runtime_profile_identity=_minimal_profile())
    append_capability_observation(ledger, observation)

    stored = EvidenceLedger(tmp_path / "capability.jsonl").get(observation.observation_id)
    restored = CapabilityObservation.from_ledger_payload(stored.payload)
    assert restored == observation
    assert restored.runtime_profile_identity.runtime_configuration_hash is None
    assert restored.runtime_profile_identity.gpu_policy is None
    assert restored.runtime_profile_identity.stable_key() == observation.runtime_profile_identity.stable_key()
    assert stored.payload["runtime_profile_identity"]["runtime_configuration_hash"] is None
    assert stored.payload["runtime_profile_identity"]["gpu_policy"] is None


def test_reconstruction_cannot_enrich_a_minimal_identity_into_a_rich_one():
    # "Modernizing" a stored minimal identity on reload by supplying a
    # runtime_configuration_hash / gpu_policy would recompute a divergent
    # stable_key -> evidence_hash, and the tamper check must reject it.
    # Enrichment on read is structurally impossible, not just discouraged.
    payload = _observation(runtime_profile_identity=_minimal_profile()).to_ledger_payload()
    enriched = {
        **payload,
        "runtime_profile_identity": {
            **payload["runtime_profile_identity"],
            "runtime_configuration_hash": "cfg-1",
            "gpu_policy": "primary_gpu_first_minimum_multi_gpu",
        },
    }
    with pytest.raises(ValueError, match="does not match stored payload"):
        CapabilityObservation.from_ledger_payload(enriched)
