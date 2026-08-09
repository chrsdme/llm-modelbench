import json
from pathlib import Path

import pytest

from llm_modelbench import campaign, repair


OMITTED_TASKS = ("git_conflict", "txt_emails", "agent_plan", "txt_sort", "json_extract", "git_commit")


def _row(task="exact", **overrides):
    row = {"model": "m", "model_digest_resolved": "d", "task": task}
    row.update(overrides)
    return row


def _hash(row):
    return campaign._primary_row_hash(row)


def _plan_for(*rows, duplicate=False, include_unexpected=None):
    source_map = {str(row["task"]): _hash(row) for row in rows}
    if duplicate and rows:
        source_map[str(rows[0]["task"]) + "_duplicate"] = _hash(rows[0])
    if include_unexpected:
        source_map[str(include_unexpected["task"])] = _hash(include_unexpected)
    return {"actions": [{"action_id": "a1", "kind": "retry_generation", "model": "m",
                         "model_digest": "d", "source_row_hashes": source_map}]}


@pytest.mark.parametrize("task_id", OMITTED_TASKS)
def test_stage2a_historically_omitted_task_ids_are_recovery_eligible(task_id):
    row = _row(task_id, error_kind="empty_output")
    state = campaign.classify_recovery_row(row)
    assert state["retry"] is True
    reconciliation = campaign.assert_recovery_reconciled([row], _plan_for(row))
    assert reconciliation["exact"] is True
    assert reconciliation["eligible_source_row_hashes"] == [_hash(row)]


@pytest.mark.parametrize("score", [100, 50, 0])
def test_stage2a_visible_scorable_answers_are_not_retried(score):
    state = campaign.classify_recovery_row(_row(score=score, reason="visible answer"))
    assert state == {"disposition": "scored", "retry": False, "reason": "visible scorable answer"}


@pytest.mark.parametrize("row", [
    _row(error_kind="thinking_only"),
    _row(error_kind="empty_output"),
    _row(error_kind="timeout"),
    _row(error_kind="transient_backend_failure"),
])
def test_stage2a_policy_approved_generation_failures_are_eligible(row):
    assert campaign.classify_recovery_row(row)["retry"] is True


@pytest.mark.parametrize("row", [
    _row(reason="raw only, judge off: visible subjective output"),
    _row(error_kind="awaiting_external_judge"),
    _row(error_kind="awaiting_independent_judge"),
    _row(error_kind="judge_exhausted_unavailable"),
    _row(error_kind="judge_output_failure"),
    _row(error_kind="timeout", evidence_lane="judge"),
    _row(error_kind="transient_backend_failure", evidence_lane="judge"),
])
def test_stage2a_judge_side_states_are_not_generation_recovery(row):
    assert campaign.classify_recovery_row(row)["retry"] is False


def test_stage2a_exact_eligible_equals_planned_plus_explicit_exclusions_succeeds():
    planned = _row("json_extract", error_kind="empty_output")
    excluded = _row("git_commit", error_kind="thinking_only")
    result = campaign.assert_recovery_reconciled(
        [planned, excluded],
        _plan_for(planned),
        excluded={_hash(excluded): "operator-approved policy exclusion"},
    )
    assert result["exact"] is True
    assert result["planned_source_row_hashes"] == [_hash(planned)]
    assert result["explicit_exclusions"][0]["source_row_hash"] == _hash(excluded)


def test_stage2a_exact_reconciliation_rejects_omitted_eligible_row():
    row = _row("json_extract", error_kind="empty_output")
    evidence = campaign.recovery_reconciliation_evidence([row], {"actions": []})
    assert evidence["missing_source_row_hashes"] == [_hash(row)]
    assert evidence["exact"] is False
    with pytest.raises(campaign.CampaignError, match="recovery_plan_incomplete"):
        campaign.assert_recovery_reconciled([row], {"actions": []})


def test_stage2a_exact_reconciliation_reports_unexpected_planned_source():
    row = _row("json_extract", error_kind="empty_output")
    other = _row("other", error_kind="empty_output")
    plan = _plan_for(row, include_unexpected=other)
    evidence = campaign.recovery_reconciliation_evidence([row], plan)
    assert evidence["unexpected_planned_source_row_hashes"] == [_hash(other)]
    assert evidence["exact"] is False
    with pytest.raises(campaign.CampaignError, match="recovery_plan_incomplete"):
        campaign.assert_recovery_reconciled([row], plan)


def test_stage2a_exact_reconciliation_reports_duplicate_planned_source():
    row = _row("json_extract", error_kind="empty_output")
    evidence = campaign.recovery_reconciliation_evidence([row], _plan_for(row, duplicate=True))
    assert evidence["duplicate_planned_source_row_hashes"] == [_hash(row)]
    assert evidence["exact"] is False
    with pytest.raises(campaign.CampaignError, match="recovery_plan_incomplete"):
        campaign.assert_recovery_reconciled([row], _plan_for(row, duplicate=True))


def test_stage2a_unexpected_exclusion_and_empty_reason_fail():
    row = _row("json_extract", error_kind="empty_output")
    other = _row("git_commit", error_kind="empty_output")
    with pytest.raises(campaign.CampaignError, match="recovery_plan_incomplete"):
        campaign.assert_recovery_reconciled([row], _plan_for(row), excluded={_hash(other): "not eligible here"})
    with pytest.raises(campaign.CampaignError, match="recovery_plan_incomplete"):
        campaign.assert_recovery_reconciled([row], {"actions": []}, excluded={_hash(row): "   "})


def test_stage2a_planned_excluded_overlap_fails():
    row = _row("json_extract", error_kind="empty_output")
    with pytest.raises(campaign.CampaignError, match="recovery_plan_incomplete"):
        campaign.assert_recovery_reconciled([row], _plan_for(row), excluded={_hash(row): "do not run"})


def test_stage2a_unattributed_native_action_fails():
    row = _row("json_extract", error_kind="empty_output")
    with pytest.raises(campaign.CampaignError, match="recovery_plan_incomplete"):
        campaign.assert_recovery_reconciled([row], {"actions": [{"action_id": "a1", "kind": "retry_generation"}]})


class _Plan:
    def __init__(self, actions):
        self._actions = actions

    def to_dict(self):
        return {"actions": self._actions}


def _execute_fixture(tmp_path, primary):
    paths, manifest = campaign.create_campaign("stage2a", models=["m"], campaigns_root=tmp_path / "campaigns")
    paths.primary_raw_results.write_text(json.dumps(primary) + "\n")
    manifest = campaign.transition(paths, manifest, "planned")
    campaign.transition(paths, manifest, "generating")
    return paths, paths.primary_raw_results.read_bytes()


@pytest.mark.parametrize("actions", [
    [],
    [{"action_id": "a1", "kind": "retry_generation"}],
    [{"action_id": "a1", "kind": "retry_generation", "source_row_hashes": {"other": "unexpected"}}],
])
def test_stage2a_failed_reconciliation_prevents_execution_callback(tmp_path, actions):
    primary = _row("json_extract", error_kind="empty_output")
    paths, before = _execute_fixture(tmp_path, primary)
    calls = []
    with pytest.raises(campaign.CampaignError, match="recovery_plan_incomplete"):
        campaign.execute_recovery_phase(
            paths, object(), object(),
            build_plan_fn=lambda *a, **k: _Plan(actions),
            apply_plan_fn=lambda *a, **k: calls.append("called") or {"actions": []},
        )
    assert calls == []
    persisted = json.loads(paths.recovery_plan.read_text())
    assert persisted["reconciliation"]["exact"] is False
    assert paths.primary_raw_results.read_bytes() == before


def test_stage2a_successful_reconciliation_permits_mocked_execution(tmp_path):
    primary = _row("json_extract", error_kind="empty_output")
    paths, before = _execute_fixture(tmp_path, primary)
    actions = [{"action_id": "a1", "kind": "retry_generation", "model": "m", "task": "json_extract",
                "source_row_hashes": {"json_extract": _hash(primary)}}]
    calls = []
    result = campaign.execute_recovery_phase(
        paths, object(), object(),
        build_plan_fn=lambda *a, **k: _Plan(actions),
        apply_plan_fn=lambda *a, **k: calls.append("called") or {"actions": [], "completed": 0},
    )
    assert calls == ["called"]
    assert result["reconciliation"]["exact"] is True
    assert paths.primary_raw_results.read_bytes() == before


def test_stage2a_reconciliation_ordering_is_semantically_deterministic():
    rows = [_row("json_extract", error_kind="empty_output"), _row("git_commit", error_kind="thinking_only")]
    forward = campaign.assert_recovery_reconciled(rows, _plan_for(*rows))
    reverse = campaign.assert_recovery_reconciled(list(reversed(rows)), _plan_for(*reversed(rows)))
    assert set(forward["eligible_source_row_hashes"]) == set(reverse["eligible_source_row_hashes"])
    assert forward["planned_source_row_hashes"] == reverse["planned_source_row_hashes"]
    assert forward["exact"] == reverse["exact"] is True


def test_stage2a_classification_and_reconciliation_do_not_mutate_primary_file(tmp_path):
    primary = _row("json_extract", error_kind="empty_output")
    path = tmp_path / "raw_results.jsonl"
    path.write_text(json.dumps(primary) + "\n")
    before = path.read_bytes()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    campaign.classify_recovery_row(rows[0])
    campaign.assert_recovery_reconciled(rows, _plan_for(rows[0]))
    with pytest.raises(campaign.CampaignError, match="recovery_plan_incomplete"):
        campaign.assert_recovery_reconciled(rows, {"actions": []})
    assert path.read_bytes() == before


def test_stage2a_repair_plan_includes_difficulty_zero_recovery_rows(tmp_path, monkeypatch):
    run = tmp_path / "primary"
    run.mkdir()
    rows = [_row(task, error_kind="empty_output") for task in OMITTED_TASKS]
    run.joinpath("raw_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    run.joinpath("model_identities.json").write_text(json.dumps({"m": {"digest": "d"}}))
    run.joinpath("status.json").write_text(json.dumps({"finished_at": "2026-01-01T00:00:00Z"}))
    monkeypatch.setattr(repair, "detect_gpu", lambda: type("GPU", (), {"total_vram_gb": 0.0})())
    monkeypatch.setattr(repair, "inspect_ollama_kv_environment", lambda: {"effective_kv_type": None, "verified": False})

    plan = repair.build_plan(Path(tmp_path), run_id="primary", include_missing=False)

    planned_tasks = {task for action in plan.actions for task in action.tasks}
    assert set(OMITTED_TASKS) <= planned_tasks
    planned_sources = {source for action in plan.actions for source in (action.source_row_hashes or {}).values()}
    assert planned_sources == {_hash(row) for row in rows}
