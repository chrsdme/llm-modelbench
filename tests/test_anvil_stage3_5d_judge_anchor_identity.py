"""Anvil Stage 3.5D — judge anchor contract identity.

OWNER DECISION RESOLVED: material judge-anchor semantics are part of
judge-result comparability and MUST be identity-bearing for newly produced
judge evidence. Historical evidence remains immutable.

The identity is ``judge.JUDGE_ANCHOR_POLICY_HASH`` — a deterministic canonical
hash over ``{JUDGE_REQUEST_CONTRACT_VERSION, ANCHORS, JUDGE_PANEL_PERSONAS}`` —
recorded as an additive top-level ``judge_anchor_policy_hash`` on every newly
written ``judge_results.jsonl`` entry. It is NOT folded into
``_compatibility_fingerprint`` / ``_execution_fingerprint`` (that would
reclassify legacy sidecars as drift via the structural-failure /
exhausted-execution reuse matchers).
"""
import json
from pathlib import Path

from llm_modelbench import judge as judge_mod
from llm_modelbench import judge_dumps
from llm_modelbench.runner import _task_hash
from llm_modelbench.tasks import TASKS


# --- fixtures -----------------------------------------------------------------


def _subjective_task():
    return next(task for task in TASKS if task.scorer == "subjective")


def _write_subjective_run(root: Path, rows):
    task = _subjective_task()
    run = root / "run"
    raw_rows = []
    for row in rows:
        model = row["model"]
        digest = row["digest"]
        dump = run / "subjective" / task.id / f"{model}.md"
        dump.parent.mkdir(parents=True, exist_ok=True)
        dump.write_text(f"# TASK {task.id}\n\n## OUTPUT\nanswer from {model}\n")
        raw_rows.append({
            "model": model,
            "model_digest": digest,
            "model_digest_resolved": digest,
            "task": task.id,
            "category": task.category,
            "family": task.family,
            "task_hash": _task_hash(task),
            "score": None,
            "error_kind": None,
            "subjective_path": str(dump.relative_to(run)),
            "timestamp": "2026-01-01T00:00:00Z",
        })
    (run / "raw_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw_rows))
    return run


def _qualified(name: str, digest: str, *, runtime_backend: str = "mock-runtime"):
    return {
        "name": name,
        "digest": digest,
        "roles": ["judge"],
        "runtime_identity": {"backend": runtime_backend, "model": name},
        "qualified": True,
        "qualification": {
            "protocol_version": "judge-qualification-v1",
            "aggregate_disposition": "qualified",
            "runtime_identity": {"backend": runtime_backend, "model": name},
            "controls": [{"control_id": "synthetic", "passed": True}],
        },
    }


class RecordingJudgeClient:
    def __init__(self, failures=None):
        self.failures = dict(failures or {})
        self.calls = []

    def chat(self, model, prompt, **kwargs):
        self.calls.append(model)
        failure = self.failures.get(model)
        if failure == "timeout_exception":
            raise TimeoutError("deadline exceeded")
        if failure:
            return dict(failure)
        return {"ok": True, "text": '{"score": 88, "confidence": 1, "verdict": "synthetic"}'}


def _judge_a_run(tmp_path: Path, *, judge_mode: str = "single", failures=None):
    run = _write_subjective_run(tmp_path, [{"model": "source", "digest": "digest-source"}])
    pool = [_qualified("preferred", "digest-preferred")]
    client = RecordingJudgeClient(failures=failures)
    result = judge_dumps.judge_run(
        client, run, judge_model="preferred", qualified_judges=pool, judge_mode=judge_mode,
    )
    return run, result


# --- 1. stability ------------------------------------------------------------


def test_the_anchor_policy_hash_is_deterministic_and_stable():
    # Two independent computations agree; the exposed constant matches.
    first = judge_mod._judge_anchor_policy_hash()
    second = judge_mod._judge_anchor_policy_hash()
    assert first == second == judge_mod.JUDGE_ANCHOR_POLICY_HASH
    assert len(judge_mod.JUDGE_ANCHOR_POLICY_HASH) == 64  # sha256 hex


def test_same_anchors_same_contract_same_identity(monkeypatch):
    # Re-binding the constants to equal values reproduces the same hash —
    # identity is a pure function of the semantic content, nothing else
    # (no mtime, no source hash, no repr ordering).
    monkeypatch.setattr(judge_mod, "ANCHORS", str(judge_mod.ANCHORS))
    monkeypatch.setattr(judge_mod, "JUDGE_PANEL_PERSONAS", tuple(judge_mod.JUDGE_PANEL_PERSONAS))
    assert judge_mod._judge_anchor_policy_hash() == judge_mod.JUDGE_ANCHOR_POLICY_HASH


# --- 2. drift (discriminating) ---------------------------------------------


def test_a_scoring_relevant_anchor_change_moves_the_identity(monkeypatch):
    baseline = judge_mod._judge_anchor_policy_hash()
    monkeypatch.setattr(
        judge_mod,
        "ANCHORS",
        judge_mod.ANCHORS.replace("90 = accurate", "80 = accurate"),
    )
    assert judge_mod._judge_anchor_policy_hash() != baseline


def test_a_panel_persona_change_moves_the_identity(monkeypatch):
    baseline = judge_mod._judge_anchor_policy_hash()
    mutated = ("a wholly different persona",) + tuple(judge_mod.JUDGE_PANEL_PERSONAS[1:])
    monkeypatch.setattr(judge_mod, "JUDGE_PANEL_PERSONAS", mutated)
    assert judge_mod._judge_anchor_policy_hash() != baseline


def test_a_contract_version_change_moves_the_identity(monkeypatch):
    # The explicit algorithm/interpretation version is mixed in, so an
    # interpretation change with unchanged numeric constants still moves it.
    baseline = judge_mod._judge_anchor_policy_hash()
    monkeypatch.setattr(judge_mod, "JUDGE_REQUEST_CONTRACT_VERSION", "posthoc-judge-request-v2")
    assert judge_mod._judge_anchor_policy_hash() != baseline


# --- 3. serialization: identity travels with judge provenance --------------


def test_written_judge_entries_carry_the_anchor_policy_hash(tmp_path: Path):
    _run, result = _judge_a_run(tmp_path)
    assert result["judged"] == 1
    entry = result["entries"][0]
    assert entry["status"] == "judged"
    assert entry["judge_anchor_policy_hash"] == judge_mod.JUDGE_ANCHOR_POLICY_HASH


def test_persisted_sidecar_entries_carry_the_anchor_policy_hash(tmp_path: Path):
    run, result = _judge_a_run(tmp_path)
    lines = (run / "judge_results.jsonl").read_text().splitlines()
    assert lines
    for line in lines:
        entry = json.loads(line)
        assert entry["judge_anchor_policy_hash"] == judge_mod.JUDGE_ANCHOR_POLICY_HASH


def test_panel_mode_entries_also_carry_the_anchor_policy_hash(tmp_path: Path):
    run, result = _judge_a_run(tmp_path, judge_mode="panel")
    entry = result["entries"][0]
    assert entry["judge_anchor_policy_hash"] == judge_mod.JUDGE_ANCHOR_POLICY_HASH


def test_awaiting_and_exhausted_entries_also_carry_the_anchor_policy_hash(tmp_path: Path):
    # A structural failure with no independent fallback drives the
    # awaiting/exhausted entry dicts (the other two schema_version:1 sites).
    run, result = _judge_a_run(
        tmp_path,
        failures={"preferred": {"ok": False, "error": "unsupported request schema", "http_status": 415}},
    )
    assert result["entries"]
    for entry in result["entries"]:
        assert entry["judge_anchor_policy_hash"] == judge_mod.JUDGE_ANCHOR_POLICY_HASH


# --- 4. drift is detectable on written evidence ----------------------------


def test_a_material_anchor_change_produces_non_comparable_recorded_identity(tmp_path: Path, monkeypatch):
    _run_a, result_a = _judge_a_run(tmp_path / "before")
    hash_before = result_a["entries"][0]["judge_anchor_policy_hash"]

    monkeypatch.setattr(judge_mod, "ANCHORS", judge_mod.ANCHORS + "\n- extra scoring guidance.")
    monkeypatch.setattr(judge_mod, "JUDGE_ANCHOR_POLICY_HASH", judge_mod._judge_anchor_policy_hash())

    _run_b, result_b = _judge_a_run(tmp_path / "after")
    hash_after = result_b["entries"][0]["judge_anchor_policy_hash"]

    assert hash_before != hash_after  # proven drift is visible in the evidence


# --- 5. legacy compatibility ---------------------------------------------


def test_legacy_judge_entry_without_the_field_still_reads(tmp_path: Path):
    # A pre-3.5D sidecar entry: readable, original fields intact, no current
    # hash invented for it.
    run = tmp_path / "legacy_run"
    run.mkdir()
    legacy_entry = {
        "schema_version": 1,
        "run_id": "legacy_run",
        "source_row_hash": "abc123",
        "task": "some_task",
        "judge_model": "old-judge",
        "status": "judged",
        "score": 71.0,
        "reason": "posthoc single judge",
        "request_contract_version": "posthoc-judge-request-v1",
    }
    sidecar = run / "judge_results.jsonl"
    sidecar.write_text(json.dumps(legacy_entry, sort_keys=True) + "\n")

    reloaded = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert reloaded == [legacy_entry]
    assert "judge_anchor_policy_hash" not in reloaded[0]
    # It is not silently enriched on read.


def test_the_anchor_hash_is_not_folded_into_the_judge_fingerprints():
    # Regression guard for the advisor finding: the compatibility /
    # execution fingerprints (which drive prior-structural-failure and
    # exhausted-execution reuse matching) must NOT depend on the anchor
    # policy hash — otherwise every legacy sidecar entry stops matching and
    # "legacy lacks anchor identity" is silently reclassified as drift.
    judge = _qualified("preferred", "digest-preferred")
    config = {"judge_mode": "single", "num_ctx": None, "think": "auto"}
    before = judge_dumps._compatibility_fingerprint(judge, config)
    before_exec = judge_dumps._execution_fingerprint([judge], config)

    original = judge_mod.JUDGE_ANCHOR_POLICY_HASH
    try:
        judge_mod.JUDGE_ANCHOR_POLICY_HASH = "deadbeef" * 8
        assert judge_dumps._compatibility_fingerprint(judge, config) == before
        assert judge_dumps._execution_fingerprint([judge], config) == before_exec
    finally:
        judge_mod.JUDGE_ANCHOR_POLICY_HASH = original
