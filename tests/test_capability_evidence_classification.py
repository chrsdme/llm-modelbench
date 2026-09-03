"""Anvil Stage 2.7A: read-only fleet capability-evidence classification.

Fixture profiles mirror the real shape ``capabilities.interrogate_model()``
produces (same construction as ``tests/test_capability_evidence_adapter.py``,
verified against the real function, not guessed), so these tests exercise
the classifier against realistic legacy data.
"""
import json
from pathlib import Path

import pytest

from llm_modelbench.capabilities import CAPABILITY_SCHEMA_VERSION, PROBE_PROTOCOL_VERSION, MeasuredCapabilityState
from llm_modelbench.capabilities import _canonical_hash as legacy_canonical_hash
from llm_modelbench.capability_evidence_adapter import typed_identity_from_capability_identity
from llm_modelbench.capability_evidence_classification import (
    REPROBE_NOT_REQUIRED,
    EvidenceCellStatus,
    classify_fleet,
    classify_model_capability,
    discover_capability_report_files,
    load_fleet_evidence,
)
from llm_modelbench.capability_observation import CAPABILITY_OBSERVATION_RECORD_TYPE, CapabilityObservation
from llm_modelbench.evidence import EvidenceLedger, ProvenanceLink, ProvenanceRelation



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

def _template_config(*, num_ctx=8192):
    material = {
        "template": "{{ .System }}\n{{ .Prompt }}",
        "parameters": f"num_ctx {num_ctx}",
        "modelfile": None,
        "system": None,
        "model_info": {"llama.context_length": num_ctx},
    }
    return {"available": True, "hash": legacy_canonical_hash(material), "material": material}


def _capability_identity(*, digest="sha256:abc123", canonical_name="qwen2.5-coder:14b",
                          backend="ollama", endpoint="http://127.0.0.1:11434", num_ctx=8192):
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


def _profile(*, family="text", state=MeasuredCapabilityState.MEASURED_SUPPORTED,
             digest="sha256:abc123", backend="ollama", num_ctx=8192, families=None):
    measured = {}
    for fam in (families or [family]):
        measured[fam] = {
            "state": state.value, "legacy_probe_state": "responded_ok",
            "route_scored_tasks": state == MeasuredCapabilityState.MEASURED_SUPPORTED,
        }
    return {
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "model": "qwen2.5-coder:14b",
        "capability_identity": _capability_identity(digest=digest, backend=backend, num_ctx=num_ctx),
        "declared_capabilities": ["completion", "tools"],
        "measured_capabilities": measured,
        "measured_supported_families": [f for f in (families or [family]) if state == MeasuredCapabilityState.MEASURED_SUPPORTED],
        "functional_probes_enabled": True,
        "warnings": [],
    }


def _current_identity(*, digest="sha256:abc123", backend="ollama", num_ctx=8192):
    ci = _capability_identity(digest=digest, backend=backend, num_ctx=num_ctx)
    del ci["identity_hash"]
    from llm_modelbench.capabilities import _canonical_hash
    ci["identity_hash"] = _canonical_hash(ci)
    return ci


class _FakeClient:
    def __init__(self, models, show_by_model=None):
        self._models = models
        self._show_by_model = show_by_model or {}

    def tags(self):
        return [{"name": m, "digest": d} for m, d in self._models.items()]


def test_current_valid_when_measured_supported_and_identity_matches():
    stored = [(Path("runs/r1/capability_report.json"), _profile(state=MeasuredCapabilityState.MEASURED_SUPPORTED))]
    cell = classify_model_capability("qwen2.5-coder:14b", "text", stored, _current_identity())
    assert cell.status == EvidenceCellStatus.CURRENT_VALID
    assert cell.typed_decision_reason == "measured_supported"
    assert cell.stored_profile_count == 1
    assert cell.structurally_adaptable_profile_count == 1
    # Anvil Stage 2.7B needs these surfaced directly, not re-derived.
    assert cell.current_backend == "ollama"
    assert cell.current_model_primary_sha256 == "sha256:abc123"
    assert cell.current_model_artifact_set_id
    assert cell.current_runtime_profile_stable_key
    assert cell.selected_evidence_hash
    assert cell.considered_evidence_hashes == (cell.selected_evidence_hash,)


@pytest.mark.parametrize("state,expected", [
    (MeasuredCapabilityState.MEASURED_UNSUPPORTED, EvidenceCellStatus.MEASURED_UNSUPPORTED),
    (MeasuredCapabilityState.BACKEND_UNSUPPORTED, EvidenceCellStatus.BACKEND_UNSUPPORTED),
    (MeasuredCapabilityState.NOT_APPLICABLE, EvidenceCellStatus.NOT_APPLICABLE),
    (MeasuredCapabilityState.PROBE_INCONCLUSIVE, EvidenceCellStatus.PROBE_INCONCLUSIVE),
])
def test_negative_and_inconclusive_states_classify_faithfully(state, expected):
    stored = [(Path("runs/r1/capability_report.json"), _profile(state=state))]
    cell = classify_model_capability("qwen2.5-coder:14b", "text", stored, _current_identity())
    assert cell.status == expected
    # These are legitimate measured negatives / genuine unknowns, not
    # missing evidence -- the reprobe_required flag must reflect that.
    if expected == EvidenceCellStatus.PROBE_INCONCLUSIVE:
        assert expected not in REPROBE_NOT_REQUIRED
    else:
        assert expected in REPROBE_NOT_REQUIRED


def test_missing_when_no_evidence_found_anywhere():
    cell = classify_model_capability("never-probed:7b", "text", [], _current_identity())
    assert cell.status == EvidenceCellStatus.MISSING
    assert cell.stored_profile_count == 0
    assert cell.to_dict()["reprobe_required"] is True


def test_missing_when_family_never_assessed_in_any_stored_profile():
    # Stored evidence exists and is structurally adaptable, but only ever
    # measured "text" -- "vision" was never assessed for this model.
    stored = [(Path("runs/r1/capability_report.json"),
               _profile(families=["text"], state=MeasuredCapabilityState.MEASURED_SUPPORTED))]
    cell = classify_model_capability("qwen2.5-coder:14b", "vision", stored, _current_identity())
    assert cell.status == EvidenceCellStatus.MISSING
    assert cell.structurally_adaptable_profile_count == 1


def test_legacy_schema_when_schema_version_stale():
    profile = _profile()
    profile["capability_schema_version"] = CAPABILITY_SCHEMA_VERSION - 1
    stored = [(Path("runs/r1/capability_report.json"), profile)]
    cell = classify_model_capability("qwen2.5-coder:14b", "text", stored, _current_identity())
    assert cell.status == EvidenceCellStatus.LEGACY_SCHEMA
    assert cell.structurally_adaptable_profile_count == 0


def test_legacy_schema_when_protocol_version_stale():
    profile = _profile()
    profile["probe_protocol_version"] = "capability-smoke-v1"
    profile["capability_identity"]["probe_protocol_version"] = "capability-smoke-v1"
    stored = [(Path("runs/r1/capability_report.json"), profile)]
    cell = classify_model_capability("qwen2.5-coder:14b", "text", stored, _current_identity())
    assert cell.status == EvidenceCellStatus.LEGACY_SCHEMA


def test_unbound_identity_when_capability_identity_missing():
    profile = _profile()
    del profile["capability_identity"]
    stored = [(Path("runs/r1/capability_report.json"), profile)]
    cell = classify_model_capability("qwen2.5-coder:14b", "text", stored, _current_identity())
    assert cell.status == EvidenceCellStatus.UNBOUND_IDENTITY


def test_model_identity_changed_on_digest_drift():
    stored = [(Path("runs/r1/capability_report.json"), _profile(digest="sha256:old000"))]
    cell = classify_model_capability("qwen2.5-coder:14b", "text", stored, _current_identity(digest="sha256:new111"))
    assert cell.status == EvidenceCellStatus.MODEL_IDENTITY_CHANGED
    assert cell.legacy_compatibility_reason == "model_digest_changed"


def test_backend_changed_when_backend_family_differs():
    stored = [(Path("runs/r1/capability_report.json"), _profile(backend="ollama"))]
    cell = classify_model_capability("qwen2.5-coder:14b", "text", stored, _current_identity(backend="llama_cpp"))
    assert cell.status == EvidenceCellStatus.BACKEND_CHANGED
    assert cell.legacy_compatibility_reason == "backend_changed"


def test_runtime_profile_changed_when_template_hash_differs():
    stored = [(Path("runs/r1/capability_report.json"), _profile(num_ctx=8192))]
    cell = classify_model_capability("qwen2.5-coder:14b", "text", stored, _current_identity(num_ctx=32768))
    assert cell.status == EvidenceCellStatus.RUNTIME_PROFILE_CHANGED
    assert cell.legacy_compatibility_reason == "template_config_changed"


def test_missing_when_model_not_currently_reachable():
    # No live current_identity available (model absent from client.tags()).
    stored = [(Path("runs/r1/capability_report.json"), _profile())]
    cell = classify_model_capability("qwen2.5-coder:14b", "text", stored, None)
    assert cell.status == EvidenceCellStatus.MISSING
    assert cell.current_identity_available is False
    assert cell.legacy_compatibility_reason == "current_capability_identity_missing"


def test_ambiguous_when_two_compatible_observations_disagree():
    # Same identity both times (both compatible with current), but one run
    # measured supported and another measured unsupported -- genuinely
    # contradictory evidence with no supersession relation to resolve it.
    stored = [
        (Path("runs/r1/capability_report.json"), _profile(state=MeasuredCapabilityState.MEASURED_SUPPORTED)),
        (Path("runs/r2/capability_report.json"), _profile(state=MeasuredCapabilityState.MEASURED_UNSUPPORTED)),
    ]
    cell = classify_model_capability("qwen2.5-coder:14b", "text", stored, _current_identity())
    assert cell.status == EvidenceCellStatus.AMBIGUOUS_COMPATIBLE_OBSERVATIONS
    assert cell.structurally_adaptable_profile_count == 2


def test_duplicate_identical_observations_are_not_ambiguous():
    # Two runs, same identity, same measured result -- a repeated probe,
    # not conflicting evidence. Must select cleanly, never flag ambiguous.
    stored = [
        (Path("runs/r1/capability_report.json"), _profile(state=MeasuredCapabilityState.MEASURED_SUPPORTED)),
        (Path("runs/r2/capability_report.json"), _profile(state=MeasuredCapabilityState.MEASURED_SUPPORTED)),
    ]
    cell = classify_model_capability("qwen2.5-coder:14b", "text", stored, _current_identity())
    assert cell.status == EvidenceCellStatus.CURRENT_VALID


def test_monkeypatch_lie_does_not_flip_classification(monkeypatch):
    # Decisive proof: patch the legacy (messaging-only) compatibility
    # function to lie in the safe direction (claim compatible when it is
    # not) and confirm the typed pipeline -- not the legacy function --
    # still decides the outcome.
    import llm_modelbench.capability_evidence_classification as mod

    def _lying_compatible(profile, current_identity):
        return {"compatible": True, "reason": "identity_match"}

    monkeypatch.setattr(mod, "legacy_capability_identity_compatibility", _lying_compatible)
    stored = [(Path("runs/r1/capability_report.json"), _profile(digest="sha256:old000"))]
    cell = classify_model_capability("qwen2.5-coder:14b", "text", stored, _current_identity(digest="sha256:new111"))
    # The typed pipeline (capability_observation_identity_compatibility)
    # still correctly rejects the digest-mismatched observation regardless
    # of what the patched legacy function claims -- only the *label*
    # chosen from the (now-lying) legacy reason could be wrong, never the
    # underlying MODEL_IDENTITY_CHANGED-vs-CURRENT_VALID applicability call.
    assert cell.status != EvidenceCellStatus.CURRENT_VALID
    assert cell.typed_decision_reason == "no_current_projection"


# ---------------------------------------------------------------------------
# Anvil Stage 2.9: native EvidenceLedger evidence, preferred over the legacy
# axis, failing closed when itself ambiguous/conflicted -- SUPERSESSION_
# CONFLICT and AMBIGUOUS_COMPATIBLE_OBSERVATIONS are reachable through this
# module's own classify_model_capability()/classify_fleet(), not just
# through the adapter's effective_measured_supported_families() (covered
# separately in test_capability_evidence_adapter_effective_authority.py).
# ---------------------------------------------------------------------------

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


def test_native_selected_overrides_legacy_missing(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")

    append_capability_observation(ledger, _native_observation(state=MeasuredCapabilityState.MEASURED_SUPPORTED))
    cell = classify_model_capability("qwen2.5-coder:14b", "text", [], _current_identity(), ledger=ledger)
    assert cell.status == EvidenceCellStatus.CURRENT_VALID
    assert cell.reason.startswith("selected native EvidenceLedger observation")


def test_native_ambiguous_fails_closed_through_classify_model_capability(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")

    append_capability_observation(ledger, _native_observation(state=MeasuredCapabilityState.MEASURED_SUPPORTED))
    append_capability_observation(ledger, _native_observation(state=MeasuredCapabilityState.MEASURED_UNSUPPORTED))
    # A positive legacy profile exists too -- must not rescue the cell from
    # the ambiguous native evidence.
    stored = [(Path("runs/r1/capability_report.json"), _profile(state=MeasuredCapabilityState.MEASURED_SUPPORTED))]
    cell = classify_model_capability("qwen2.5-coder:14b", "text", stored, _current_identity(), ledger=ledger)
    assert cell.status == EvidenceCellStatus.AMBIGUOUS_COMPATIBLE_OBSERVATIONS
    assert cell.status not in REPROBE_NOT_REQUIRED


def test_native_supersession_conflict_reachable_through_classify_fleet(tmp_path):
    # A genuine cycle in the ledger's own provenance graph, same
    # construction as test_resolver_detects_a_cycle in the Stage 0 evidence
    # model suite.
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    obs_a = _native_observation(state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    obs_b = _native_observation(state=MeasuredCapabilityState.MEASURED_UNSUPPORTED, digest="sha256:def456")
    ledger.append(
        CAPABILITY_OBSERVATION_RECORD_TYPE, obs_a.to_ledger_payload(), record_id="oa",
        provenance=[ProvenanceLink(ProvenanceRelation.SUPERSEDES, "ob")],
    )
    ledger.append(
        CAPABILITY_OBSERVATION_RECORD_TYPE, obs_b.to_ledger_payload(), record_id="ob",
        provenance=[ProvenanceLink(ProvenanceRelation.SUPERSEDES, "oa")],
    )
    client = _FakeClient({"qwen2.5-coder:14b": "sha256:abc123"})
    report = classify_fleet(client, runs_dir=tmp_path / "runs", campaigns_root=tmp_path / "campaigns", ledger=ledger)
    cell = next(c for c in report.cells if c.model == "qwen2.5-coder:14b" and c.capability == "text")
    assert cell.status == EvidenceCellStatus.SUPERSESSION_CONFLICT
    assert cell.status not in REPROBE_NOT_REQUIRED


# ---------------------------------------------------------------------------
# Fleet scan
# ---------------------------------------------------------------------------

def test_discover_finds_top_level_run_and_nested_campaign_evidence(tmp_path):
    runs_dir = tmp_path / "runs"
    campaigns_dir = tmp_path / "campaigns"
    (runs_dir / "run1").mkdir(parents=True)
    (runs_dir / "run1" / "capability_report.json").write_text("{}")
    (campaigns_dir / "camp1" / "evidence" / "primary").mkdir(parents=True)
    (campaigns_dir / "camp1" / "evidence" / "primary" / "capability_report.json").write_text("{}")
    (campaigns_dir / "camp1" / "evidence" / "recovery" / "children" / "recovery-0001").mkdir(parents=True)
    (campaigns_dir / "camp1" / "evidence" / "recovery" / "children" / "recovery-0001" / "capability_report.json").write_text("{}")

    found = discover_capability_report_files(runs_dir, campaigns_dir)
    assert len(found) == 3
    assert all(path.name == "capability_report.json" for path in found)


def test_discover_tolerates_missing_roots(tmp_path):
    assert discover_capability_report_files(tmp_path / "does-not-exist") == []


def test_load_fleet_evidence_groups_by_model_across_files(tmp_path):
    run1 = tmp_path / "run1"
    run1.mkdir()
    (run1 / "capability_report.json").write_text(json.dumps({"model-a": _profile()}))
    run2 = tmp_path / "run2"
    run2.mkdir()
    (run2 / "capability_report.json").write_text(json.dumps({"model-a": _profile(), "model-b": _profile()}))

    paths = discover_capability_report_files(tmp_path)
    by_model = load_fleet_evidence(paths)
    assert set(by_model.keys()) == {"model-a", "model-b"}
    assert len(by_model["model-a"]) == 2
    assert len(by_model["model-b"]) == 1


def test_classify_fleet_end_to_end(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    campaigns_dir = tmp_path / "campaigns"
    run1 = runs_dir / "run1"
    run1.mkdir(parents=True)
    (run1 / "capability_report.json").write_text(json.dumps({
        "qwen2.5-coder:14b": _profile(state=MeasuredCapabilityState.MEASURED_SUPPORTED),
    }))

    client = _FakeClient({"qwen2.5-coder:14b": "sha256:abc123"})
    monkeypatch.setattr(
        "llm_modelbench.capability_evidence_classification.current_capability_identity",
        lambda client, model: _current_identity(),
    )

    report = classify_fleet(client, runs_dir=runs_dir, campaigns_root=campaigns_dir)
    assert "qwen2.5-coder:14b" in report.models_considered
    assert len(report.source_files_scanned) == 1
    text_cell = next(c for c in report.cells if c.model == "qwen2.5-coder:14b" and c.capability == "text")
    assert text_cell.status == EvidenceCellStatus.CURRENT_VALID
    # Every FAMILY_ORDER capability got classified, not just the one with evidence.
    from llm_modelbench.classify import FAMILY_ORDER
    covered = {c.capability for c in report.cells if c.model == "qwen2.5-coder:14b"}
    assert covered == set(FAMILY_ORDER)
    counts = report.by_status()
    assert sum(counts.values()) == len(report.cells)
    assert report.to_dict()["cell_count"] == len(report.cells)


def test_classify_fleet_includes_historical_models_no_longer_installed(tmp_path):
    runs_dir = tmp_path / "runs"
    run1 = runs_dir / "run1"
    run1.mkdir(parents=True)
    (run1 / "capability_report.json").write_text(json.dumps({"retired-model:7b": _profile()}))

    client = _FakeClient({})  # nothing currently installed
    report = classify_fleet(client, runs_dir=runs_dir, campaigns_root=tmp_path / "campaigns")
    assert "retired-model:7b" in report.models_considered
    cell = next(c for c in report.cells if c.model == "retired-model:7b" and c.capability == "text")
    assert cell.status == EvidenceCellStatus.MISSING
    assert cell.current_identity_available is False


def test_classify_fleet_is_read_only(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    run1 = runs_dir / "run1"
    run1.mkdir(parents=True)
    report_path = run1 / "capability_report.json"
    payload = json.dumps({"qwen2.5-coder:14b": _profile()})
    report_path.write_text(payload)
    before_mtime = report_path.stat().st_mtime_ns
    before_content = report_path.read_text()

    client = _FakeClient({"qwen2.5-coder:14b": "sha256:abc123"})
    monkeypatch.setattr(
        "llm_modelbench.capability_evidence_classification.current_capability_identity",
        lambda client, model: _current_identity(),
    )
    classify_fleet(client, runs_dir=runs_dir, campaigns_root=tmp_path / "campaigns")
    classify_fleet(client, runs_dir=runs_dir, campaigns_root=tmp_path / "campaigns")

    assert report_path.stat().st_mtime_ns == before_mtime
    assert report_path.read_text() == before_content
