"""Anvil Stage 2.7B: deterministic, read-only reprobe planning.

Fixture profiles mirror the real shape ``capabilities.interrogate_model()``
produces, same construction as ``tests/test_capability_evidence_classification.py``.
"""
import json
import random
from pathlib import Path

import pytest

from llm_modelbench.capabilities import CAPABILITY_SCHEMA_VERSION, PROBE_PROTOCOL_VERSION, MeasuredCapabilityState
from llm_modelbench.capabilities import _canonical_hash as legacy_canonical_hash
from llm_modelbench.capability_evidence_classification import (
    EvidenceCell,
    EvidenceCellStatus,
    FleetEvidenceReport,
    classify_model_capability,
)
from llm_modelbench.capability_reprobe_plan import (
    ReprobeActionKind,
    build_reprobe_plan,
    plan_fleet_reprobes,
)
from llm_modelbench.classify import FAMILY_ORDER


# ---------------------------------------------------------------------------
# Fixtures shared in spirit with test_capability_evidence_classification.py
# ---------------------------------------------------------------------------

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
             digest="sha256:abc123", backend="ollama", num_ctx=8192,
             declared_capabilities=None):
    return {
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "model": "qwen2.5-coder:14b",
        "capability_identity": _capability_identity(digest=digest, backend=backend, num_ctx=num_ctx),
        "declared_capabilities": declared_capabilities if declared_capabilities is not None else ["completion", "tools"],
        "measured_capabilities": {
            family: {
                "state": state.value, "legacy_probe_state": "responded_ok",
                "route_scored_tasks": state == MeasuredCapabilityState.MEASURED_SUPPORTED,
            },
        },
        "measured_supported_families": [family] if state == MeasuredCapabilityState.MEASURED_SUPPORTED else [],
        "functional_probes_enabled": True,
        "warnings": [],
    }


def _current_identity(*, digest="sha256:abc123", backend="ollama", num_ctx=8192):
    from llm_modelbench.capabilities import _canonical_hash
    ci = _capability_identity(digest=digest, backend=backend, num_ctx=num_ctx)
    del ci["identity_hash"]
    ci["identity_hash"] = _canonical_hash(ci)
    return ci


def _bare_cell(model: str, capability: str, status: EvidenceCellStatus, **kwargs) -> EvidenceCell:
    return EvidenceCell(
        model=model, capability=capability, status=status,
        reason=kwargs.pop("reason", f"fixture reason for {status.value}"),
        **kwargs,
    )


class _FakeClient:
    def __init__(self, models):
        self._models = models

    def tags(self):
        return [{"name": m, "digest": d} for m, d in self._models.items()]


# ---------------------------------------------------------------------------
# Bucket -> action mapping (13 buckets, per the advice's explicit list)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", [
    EvidenceCellStatus.MISSING,
    EvidenceCellStatus.LEGACY_SCHEMA,
    EvidenceCellStatus.UNBOUND_IDENTITY,
    EvidenceCellStatus.MODEL_IDENTITY_CHANGED,
    EvidenceCellStatus.RUNTIME_PROFILE_CHANGED,
    EvidenceCellStatus.BACKEND_CHANGED,
    EvidenceCellStatus.AMBIGUOUS_COMPATIBLE_OBSERVATIONS,
    EvidenceCellStatus.SUPERSESSION_CONFLICT,
    EvidenceCellStatus.PROBE_INCONCLUSIVE,
])
def test_reprobe_required_buckets_produce_reprobe_action(status):
    report = FleetEvidenceReport(cells=(_bare_cell("m", "text", status),), models_considered=("m",), source_files_scanned=())
    plan = build_reprobe_plan(report)
    assert plan.actions[0].action == ReprobeActionKind.REPROBE
    assert plan.actions[0].classification == status.value


@pytest.mark.parametrize("status", [
    EvidenceCellStatus.CURRENT_VALID,
    EvidenceCellStatus.MEASURED_UNSUPPORTED,
    EvidenceCellStatus.BACKEND_UNSUPPORTED,
    EvidenceCellStatus.NOT_APPLICABLE,
])
def test_valid_terminal_buckets_produce_no_action(status):
    report = FleetEvidenceReport(cells=(_bare_cell("m", "text", status),), models_considered=("m",), source_files_scanned=())
    plan = build_reprobe_plan(report)
    assert plan.actions[0].action == ReprobeActionKind.NO_ACTION
    assert plan.actions[0].classification == status.value


def test_classification_and_action_are_separate_fields_never_merged():
    report = FleetEvidenceReport(
        cells=(_bare_cell("m", "text", EvidenceCellStatus.MODEL_IDENTITY_CHANGED),),
        models_considered=("m",), source_files_scanned=(),
    )
    action = build_reprobe_plan(report).actions[0]
    d = action.to_dict()
    assert d["classification"] == "model_identity_changed"
    assert d["action"] == "reprobe"
    assert "REPROBE_MODEL_IDENTITY_CHANGED" not in json.dumps(d)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_identical_inputs_produce_canonically_identical_plan():
    report = FleetEvidenceReport(
        cells=(
            _bare_cell("m1", "text", EvidenceCellStatus.MISSING),
            _bare_cell("m2", "vision", EvidenceCellStatus.CURRENT_VALID),
        ),
        models_considered=("m1", "m2"), source_files_scanned=(),
    )
    plan_a = build_reprobe_plan(report)
    plan_b = build_reprobe_plan(report)
    assert plan_a.canonical_plan_hash() == plan_b.canonical_plan_hash()
    assert plan_a.to_dict()["actions"] == plan_b.to_dict()["actions"]


def test_shuffled_input_order_produces_same_canonical_plan():
    cells = []
    models = ["zeta", "alpha", "mu", "beta"]
    for model in models:
        for family in FAMILY_ORDER:
            cells.append(_bare_cell(model, family, EvidenceCellStatus.MISSING))

    ordered_report = FleetEvidenceReport(cells=tuple(cells), models_considered=tuple(models), source_files_scanned=())
    shuffled = list(cells)
    random.Random(1234).shuffle(shuffled)
    shuffled_report = FleetEvidenceReport(cells=tuple(shuffled), models_considered=tuple(reversed(models)), source_files_scanned=())

    plan_ordered = build_reprobe_plan(ordered_report)
    plan_shuffled = build_reprobe_plan(shuffled_report)
    assert plan_ordered.canonical_plan_hash() == plan_shuffled.canonical_plan_hash()
    assert [a.to_dict() for a in plan_ordered.actions] == [a.to_dict() for a in plan_shuffled.actions]
    # models_considered is also canonicalized (sorted), independent of report order.
    assert plan_ordered.models_considered == plan_shuffled.models_considered == tuple(sorted(models))


def test_plan_hash_has_no_wall_clock_or_random_dependency():
    report = FleetEvidenceReport(cells=(_bare_cell("m", "text", EvidenceCellStatus.MISSING),), models_considered=("m",), source_files_scanned=())
    hashes = {build_reprobe_plan(report).canonical_plan_hash() for _ in range(5)}
    assert len(hashes) == 1
    payload = build_reprobe_plan(report).to_dict()
    serialized = json.dumps(payload)
    assert "timestamp" not in serialized and "generated_at" not in serialized


# ---------------------------------------------------------------------------
# Filtering: subset-only, never reclassifies
# ---------------------------------------------------------------------------

def test_filters_produce_subset_and_never_change_classification():
    report = FleetEvidenceReport(
        cells=(
            _bare_cell("m1", "text", EvidenceCellStatus.MISSING, current_backend="ollama"),
            _bare_cell("m1", "vision", EvidenceCellStatus.CURRENT_VALID, current_backend="ollama"),
            _bare_cell("m2", "text", EvidenceCellStatus.BACKEND_CHANGED, current_backend="llama_cpp"),
        ),
        models_considered=("m1", "m2"), source_files_scanned=(),
    )
    plan = build_reprobe_plan(report)
    full = set(a.to_dict()["model"] + a.to_dict()["capability"] for a in plan.actions)

    by_model = plan.filtered(model="m1")
    assert {a.model for a in by_model} == {"m1"}
    assert set(a.to_dict()["model"] + a.to_dict()["capability"] for a in by_model) <= full

    by_capability = plan.filtered(capability="text")
    assert {a.capability for a in by_capability} == {"text"}

    by_backend = plan.filtered(backend="llama_cpp")
    assert {a.model for a in by_backend} == {"m2"}

    by_reason = plan.filtered(reason="backend_changed")
    assert {a.classification for a in by_reason} == {"backend_changed"}

    only_required = plan.filtered(only_required=True)
    assert all(a.action == ReprobeActionKind.REPROBE for a in only_required)
    assert "vision" not in {a.capability for a in only_required if a.model == "m1"}

    # Filtering never mutates the underlying plan.
    assert len(plan.actions) == 3


def test_valid_negative_cell_stays_out_of_required_plan_even_with_misleading_declared_metadata():
    # Declared metadata claims embedding is supported, but the measured
    # state is a genuine negative -- the plan must not reprobe it, and it
    # must be excluded from the only-required view.
    profile = _profile(family="embedding", state=MeasuredCapabilityState.MEASURED_UNSUPPORTED,
                        declared_capabilities=["embedding", "completion"])
    cell = classify_model_capability("qwen2.5-coder:14b", "embedding", [(Path("runs/r1/capability_report.json"), profile)], _current_identity())
    assert cell.status == EvidenceCellStatus.MEASURED_UNSUPPORTED
    report = FleetEvidenceReport(cells=(cell,), models_considered=("qwen2.5-coder:14b",), source_files_scanned=())
    plan = build_reprobe_plan(report)
    assert plan.actions[0].action == ReprobeActionKind.NO_ACTION
    assert plan.filtered(only_required=True) == ()


def test_misleading_declared_metadata_cannot_suppress_a_required_reprobe():
    # Declared metadata claims the model is fine (text/tools), but the
    # stored identity has drifted -- misleading declared data must not
    # suppress the reprobe this cell genuinely needs.
    profile = _profile(digest="sha256:old000", declared_capabilities=["completion", "tools", "vision", "embedding"])
    cell = classify_model_capability("qwen2.5-coder:14b", "text", [(Path("runs/r1/capability_report.json"), profile)], _current_identity(digest="sha256:new111"))
    assert cell.status == EvidenceCellStatus.MODEL_IDENTITY_CHANGED
    report = FleetEvidenceReport(cells=(cell,), models_considered=("qwen2.5-coder:14b",), source_files_scanned=())
    plan = build_reprobe_plan(report)
    assert plan.actions[0].action == ReprobeActionKind.REPROBE


# ---------------------------------------------------------------------------
# Ambiguity/conflict: never resolved by selection
# ---------------------------------------------------------------------------

def test_ambiguous_observations_reprobe_and_retain_every_evidence_hash():
    stored = [
        (Path("runs/r1/capability_report.json"), _profile(state=MeasuredCapabilityState.MEASURED_SUPPORTED)),
        (Path("runs/r2/capability_report.json"), _profile(state=MeasuredCapabilityState.MEASURED_UNSUPPORTED)),
    ]
    cell = classify_model_capability("qwen2.5-coder:14b", "text", stored, _current_identity())
    assert cell.status == EvidenceCellStatus.AMBIGUOUS_COMPATIBLE_OBSERVATIONS
    assert len(cell.considered_evidence_hashes) == 2

    report = FleetEvidenceReport(cells=(cell,), models_considered=("qwen2.5-coder:14b",), source_files_scanned=())
    action = build_reprobe_plan(report).actions[0]
    assert action.action == ReprobeActionKind.REPROBE
    # No "newest wins" selection: both distinct hashes are preserved, not
    # narrowed to one, and no single hash is promoted as "the" evidence.
    assert len(action.considered_evidence_hashes) == 2
    assert action.previous_evidence_hash is None


def test_supersession_conflict_reprobes_without_selection():
    cell = _bare_cell(
        "m", "text", EvidenceCellStatus.SUPERSESSION_CONFLICT,
        considered_evidence_hashes=("hash-a", "hash-b"),
    )
    report = FleetEvidenceReport(cells=(cell,), models_considered=("m",), source_files_scanned=())
    action = build_reprobe_plan(report).actions[0]
    assert action.action == ReprobeActionKind.REPROBE
    assert action.considered_evidence_hashes == ("hash-a", "hash-b")
    assert action.previous_evidence_hash is None


# ---------------------------------------------------------------------------
# Read-only / no mutation, end-to-end
# ---------------------------------------------------------------------------

def test_plan_fleet_reprobes_is_read_only(tmp_path, monkeypatch):
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
    plan_fleet_reprobes(client, runs_dir=runs_dir, campaigns_root=tmp_path / "campaigns")
    plan_fleet_reprobes(client, runs_dir=runs_dir, campaigns_root=tmp_path / "campaigns")

    assert report_path.stat().st_mtime_ns == before_mtime
    assert report_path.read_text() == before_content


def test_plan_fleet_reprobes_end_to_end_summary(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
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
    plan = plan_fleet_reprobes(client, runs_dir=runs_dir, campaigns_root=tmp_path / "campaigns")
    summary = plan.summary()
    assert summary["total_cells_examined"] == len(FAMILY_ORDER)
    assert summary["current_valid"] == 1  # "text"
    assert summary["missing"] == len(FAMILY_ORDER) - 1  # every other family never assessed
    assert summary["native_capability_observation_records"] == 0
    assert summary["total_actions"] == summary["total_cells_examined"] - summary["current_valid"]
