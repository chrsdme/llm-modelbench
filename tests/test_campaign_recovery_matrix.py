"""Orchestrator-boundary matrix for bounded campaign recovery."""
import json
from pathlib import Path
import pytest
from llm_modelbench import campaign


class Plan:
    def to_dict(self): return {"actions": [{"kind": "retry_generation"}]}


def _run(tmp_path, actions):
    paths, manifest = campaign.create_campaign("matrix", models=["m"], campaigns_root=tmp_path / "campaigns")
    paths.primary_raw_results.write_text('{"model":"m","task":"exact","error_kind":"thinking_only"}\n')
    manifest = campaign.transition(paths, manifest, "planned")
    campaign.transition(paths, manifest, "generating")
    before = paths.primary_raw_results.read_bytes()
    result = campaign.execute_recovery_phase(
        paths, object(), object(), build_plan_fn=lambda *a, **k: Plan(),
        apply_plan_fn=lambda *a, **k: {"actions": actions, "completed": len(actions)},
    )
    records = [json.loads(line) for line in paths.recovery_attempts.read_text().splitlines()]
    return paths, result, records, before


@pytest.mark.parametrize("case,status,score,visible", [
    ("thinking_correct", "recovered_correct", 100, True),
    ("thinking_wrong", "recovered_visible_wrong", 0, True),
    ("thinking_exhausted", "terminal_thinking_only", None, False),
    ("empty_recovered", "recovered_correct", 100, True),
    ("empty_exhausted", "terminal_empty_output", None, False),
    ("transient_recovered", "recovered_correct", 100, True),
    ("transient_exhausted", "terminal_transient_failure", None, False),
    ("think_off_ineffective", "terminal_think_control_ineffective", None, False),
])
def test_recovery_terminal_matrix(tmp_path, case, status, score, visible):
    action = {"status": status, "score": score, "visible_answer": visible, "reason": case,
              "model": "m", "model_digest": "d", "task": "exact", "task_hash": "h",
              "output_budget": 2048, "context": 4096, "think_mode": "off",
              "started_at": "2026-01-01T00:00:00Z", "ended_at": "2026-01-01T00:00:01Z"}
    paths, _, records, before = _run(tmp_path, [action])
    assert records[0]["output_classification"] == status
    assert records[0]["score"] == score and records[0]["visible_answer"] is visible
    assert paths.primary_raw_results.read_bytes() == before
    assert (paths.recovery_children_dir / "recovery-0001" / "attempt.json").exists()
    assert "recovering" in [e["state"] for e in campaign.load_manifest(paths).state_history]


@pytest.mark.parametrize("score,reason", [(0, "wrong"), (50, "partial"), (0, "refusal")])
def test_visible_scored_primary_never_retries(score, reason):
    assert campaign.classify_recovery_row({"score": score, "reason": reason}) == {
        "disposition": "scored", "retry": False, "reason": "visible scorable answer"}


@pytest.mark.parametrize("state", ["confirmed_capability_unavailable", "environment_limited"])
def test_terminal_non_generation_states_are_not_retryable(state):
    assert state in campaign.TERMINAL_DISPOSITIONS


@pytest.mark.parametrize("breaker", ["max_attempts_per_cell", "max_campaign_actions", "max_extra_per_model", "max_wall_time"])
def test_breaker_stop_reasons_are_persisted(tmp_path, breaker):
    action = {"status": "terminal_model_failure", "stop_reason": breaker, "model": "m", "task": "exact"}
    _, _, records, _ = _run(tmp_path, [action])
    assert records[0]["stop_reason"] == breaker


def test_interruption_records_exact_recovery_resume(tmp_path):
    paths, manifest = campaign.create_campaign("interrupt", models=["m"], campaigns_root=tmp_path / "campaigns")
    manifest = campaign.transition(paths, manifest, "planned")
    manifest = campaign.transition(paths, manifest, "generating")
    manifest = campaign.transition(paths, manifest, "recovering")
    manifest = campaign.transition(paths, manifest, "interrupted")
    assert manifest.resume_state == "recovering"
    assert campaign.transition(paths, manifest, "recovering").state == "recovering"


def test_attempt_provenance_fields_and_idempotent_persisted_child(tmp_path):
    action = {"status": "recovered", "child_run_id": "child", "parent_row_hash": "p", "model": "m",
              "model_digest": "d", "task": "exact", "task_hash": "h", "attempt_number": 1,
              "output_budget": 2048, "context": 4096, "configuration": {"timeout": 30},
              "started_at": "s", "ended_at": "e", "wall_time_seconds": 1, "visible_answer": True, "score": 0}
    paths, _, records, before = _run(tmp_path, [action])
    required = {"campaign_id","parent_row_hash","parent_run_id","child_run_id","model","model_digest","task","task_hash","attempt_number","output_budget","context","think_mode","configuration","started_at","ended_at","wall_time_seconds","raw_response_reference","output_classification","visible_answer","score","reason","error_classification","stop_reason","policy_version"}
    assert required <= records[0].keys()
    assert paths.primary_raw_results.read_bytes() == before
