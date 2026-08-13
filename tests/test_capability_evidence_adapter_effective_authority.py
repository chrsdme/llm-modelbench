"""Anvil Stage 2.9: native-evidence-preferred capability authority.

Exercises `capability_evidence_adapter.effective_measured_supported_families()`
-- the sibling of `new_measured_supported_families()` that prefers a current,
identity-compatible native `EvidenceLedger` observation over the legacy
adapter path, records (never silently resolves) disagreement between the
two, and fails closed when native evidence is itself ambiguous rather than
falling back to a convenient legacy answer.

Profile/identity fixtures mirror `test_capability_evidence_adapter.py`'s own
(the real `capabilities.interrogate_model()` shape), and native observations
are appended directly to a real `EvidenceLedger`, mirroring
`test_capability_reprobe_execute.py`'s pattern of deriving "current identity"
straight from the observation that was just appended so the two cannot
silently drift apart (the exact bug class Stage 2.7C caught).
"""
from pathlib import Path

from llm_modelbench.capabilities import CAPABILITY_SCHEMA_VERSION, PROBE_PROTOCOL_VERSION, MeasuredCapabilityState
from llm_modelbench.capabilities import _canonical_hash as legacy_canonical_hash
from llm_modelbench.capability_evidence_adapter import (
    effective_measured_supported_families,
    new_measured_supported_families,
    typed_identity_from_capability_identity,
)
from llm_modelbench.capability_observation import CapabilityObservation, append_capability_observation
from llm_modelbench.evidence import EvidenceLedger, ProvenanceLink, ProvenanceRelation


def _template_config(*, num_ctx=8192):
    material = {
        "template": "{{ .System }}\n{{ .Prompt }}",
        "parameters": f"num_ctx {num_ctx}",
        "modelfile": None,
        "system": None,
        "model_info": {"llama.context_length": num_ctx},
    }
    return {"available": True, "hash": legacy_canonical_hash(material), "material": material}


def _capability_identity(*, digest="sha256:abc123", canonical_name="qwen2.5-coder:14b", backend="ollama", endpoint="http://127.0.0.1:11434", num_ctx=8192):
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "model": {
            "canonical_name": canonical_name,
            "backend_model_id": canonical_name,
            "digest": digest,
            "size": 9_000_000_000,
            "details": {"quantization_level": "Q4_K_M"},
        },
        "backend": {"backend": backend, "implementation": "OllamaClient", "endpoint": endpoint},
        "runtime": {"endpoint": endpoint, "implementation": "OllamaClient"},
        "template_config": _template_config(num_ctx=num_ctx),
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "identity_hash": "irrelevant-composite-hash",
    }


def _profile(*, family="text", state=MeasuredCapabilityState.MEASURED_SUPPORTED, digest="sha256:abc123"):
    identity = _capability_identity(digest=digest)
    return {
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "model": "qwen2.5-coder:14b",
        "capability_identity": identity,
        "declared_capabilities": ["completion", "tools"],
        "measured_capabilities": {
            family: {"state": state.value, "legacy_probe_state": "responded_ok", "route_scored_tasks": True},
        },
        "measured_supported_families": [family] if state == MeasuredCapabilityState.MEASURED_SUPPORTED else [],
        "functional_probes_enabled": True,
        "routing_policy": "functional_probe_required",
        "warnings": [],
    }


def _native_observation(*, family="text", state=MeasuredCapabilityState.MEASURED_SUPPORTED, digest="sha256:abc123"):
    typed = typed_identity_from_capability_identity(_capability_identity(digest=digest), protocol_version=PROBE_PROTOCOL_VERSION)
    return CapabilityObservation(
        model_identity=typed.model_identity,
        runtime_profile_identity=typed.runtime_profile_identity,
        capability=family,
        result=state,
        probe_protocol_version=PROBE_PROTOCOL_VERSION,
        capability_schema_version=CAPABILITY_SCHEMA_VERSION,
        template_config_hash=typed.template_hash,
        endpoint_identity=typed.endpoint_identity,
    )


def test_no_ledger_falls_back_to_legacy_path_identically(tmp_path: Path):
    profile = _profile()
    identity = profile["capability_identity"]
    legacy = new_measured_supported_families(profile, identity)
    effective = effective_measured_supported_families(profile, identity, ledger=None)
    assert effective.families == tuple(legacy)
    assert effective.disagreements == ()


def test_ledger_given_but_no_native_evidence_falls_back_to_legacy(tmp_path: Path):
    profile = _profile()
    identity = profile["capability_identity"]
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    legacy = new_measured_supported_families(profile, identity)
    effective = effective_measured_supported_families(profile, identity, ledger=ledger)
    assert effective.families == tuple(legacy)
    assert effective.disagreements == ()


def test_native_and_legacy_agree_no_disagreement_recorded(tmp_path: Path):
    profile = _profile(family="text", state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    identity = profile["capability_identity"]
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    append_capability_observation(ledger, _native_observation(family="text", state=MeasuredCapabilityState.MEASURED_SUPPORTED))
    effective = effective_measured_supported_families(profile, identity, ledger=ledger)
    assert "text" in effective.families
    assert effective.disagreements == ()


def test_native_positive_wins_over_legacy_negative_and_disagreement_recorded(tmp_path: Path):
    # Legacy profile says unsupported; native ledger says supported. Native
    # must win (family included), and the disagreement must be visible, not
    # silently dropped.
    profile = _profile(family="text", state=MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    identity = profile["capability_identity"]
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    append_capability_observation(ledger, _native_observation(family="text", state=MeasuredCapabilityState.MEASURED_SUPPORTED))
    effective = effective_measured_supported_families(profile, identity, ledger=ledger)
    assert "text" in effective.families
    assert len(effective.disagreements) == 1
    d = effective.disagreements[0]
    assert d.family == "text"
    assert d.native_applicable is True
    assert d.legacy_applicable is False


def test_native_negative_wins_over_legacy_positive_and_disagreement_recorded(tmp_path: Path):
    # Legacy profile says supported; native ledger says unsupported. Native
    # must win (family excluded) -- never let a stale/legacy positive answer
    # override a genuine measured negative.
    profile = _profile(family="text", state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    identity = profile["capability_identity"]
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    append_capability_observation(ledger, _native_observation(family="text", state=MeasuredCapabilityState.MEASURED_UNSUPPORTED))
    effective = effective_measured_supported_families(profile, identity, ledger=ledger)
    assert "text" not in effective.families
    assert len(effective.disagreements) == 1
    d = effective.disagreements[0]
    assert d.native_applicable is False
    assert d.legacy_applicable is True


def test_ambiguous_native_evidence_fails_closed_even_if_legacy_is_positive(tmp_path: Path):
    profile = _profile(family="text", state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    identity = profile["capability_identity"]
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    # Two genuinely contradictory compatible native observations for the
    # same identity -- must be recorded with distinct evidence_hash, so use
    # different results with no supersession link between them.
    append_capability_observation(ledger, _native_observation(family="text", state=MeasuredCapabilityState.MEASURED_SUPPORTED))
    append_capability_observation(ledger, _native_observation(family="text", state=MeasuredCapabilityState.MEASURED_UNSUPPORTED))
    effective = effective_measured_supported_families(profile, identity, ledger=ledger)
    assert "text" not in effective.families
    assert len(effective.disagreements) == 1
    d = effective.disagreements[0]
    assert d.native_applicable is None
    assert d.native_reason == "ambiguous_compatible_observations"
    assert d.legacy_applicable is True


def test_ambiguous_native_evidence_no_disagreement_when_legacy_also_negative(tmp_path: Path):
    profile = _profile(family="text", state=MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    identity = profile["capability_identity"]
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    append_capability_observation(ledger, _native_observation(family="text", state=MeasuredCapabilityState.MEASURED_SUPPORTED))
    append_capability_observation(ledger, _native_observation(family="text", state=MeasuredCapabilityState.MEASURED_UNSUPPORTED))
    effective = effective_measured_supported_families(profile, identity, ledger=ledger)
    assert "text" not in effective.families
    # Both sides land on "not applicable" -- nothing to disagree about.
    assert effective.disagreements == ()


def test_supersession_conflict_fails_closed_regardless_of_legacy(tmp_path: Path):
    # A genuine cycle in the ledger's own provenance graph (structural
    # defect, not "multiple valid but disagreeing measurements") -- built
    # the same way test_resolver_detects_a_cycle does: explicit record_ids
    # that supersede one another.
    from llm_modelbench.capability_observation import CAPABILITY_OBSERVATION_RECORD_TYPE

    profile = _profile(family="text", state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    identity = profile["capability_identity"]
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    obs_a = _native_observation(family="text", state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    obs_b = _native_observation(family="text", state=MeasuredCapabilityState.MEASURED_UNSUPPORTED, digest="sha256:def456")
    ledger.append(
        CAPABILITY_OBSERVATION_RECORD_TYPE, obs_a.to_ledger_payload(), record_id="oa",
        provenance=[ProvenanceLink(ProvenanceRelation.SUPERSEDES, "ob")],
    )
    ledger.append(
        CAPABILITY_OBSERVATION_RECORD_TYPE, obs_b.to_ledger_payload(), record_id="ob",
        provenance=[ProvenanceLink(ProvenanceRelation.SUPERSEDES, "oa")],
    )
    effective = effective_measured_supported_families(profile, identity, ledger=ledger)
    assert "text" not in effective.families
    assert len(effective.disagreements) == 1
    d = effective.disagreements[0]
    assert d.native_applicable is None
    assert d.native_reason == "supersession_conflict"
    assert d.legacy_applicable is True


def test_family_order_preserved(tmp_path: Path):
    from llm_modelbench.classify import FAMILY_ORDER
    profile = {
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "model": "qwen2.5-coder:14b",
        "capability_identity": _capability_identity(),
        "declared_capabilities": [],
        "measured_capabilities": {
            fam: {"state": MeasuredCapabilityState.MEASURED_SUPPORTED.value, "legacy_probe_state": "responded_ok", "route_scored_tasks": True}
            for fam in FAMILY_ORDER
        },
        "measured_supported_families": list(FAMILY_ORDER),
        "functional_probes_enabled": True,
        "routing_policy": "functional_probe_required",
        "warnings": [],
    }
    identity = profile["capability_identity"]
    effective = effective_measured_supported_families(profile, identity, ledger=None)
    assert list(effective.families) == [f for f in FAMILY_ORDER]
