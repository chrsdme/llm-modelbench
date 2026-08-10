import json

import pytest

from llm_modelbench import campaign


def _paths(tmp_path, name):
    return campaign.create_campaign(name, models=["m"], campaigns_root=tmp_path / "campaigns")[0]


def _write_primary(paths, rows):
    paths.primary_raw_results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(task, *, score=None, reason="ok", error_kind=None):
    row = {
        "model": "m",
        "model_digest_resolved": "digest-m",
        "task": task,
        "task_hash": f"{task}-hash",
        "reason": reason,
    }
    if score is not None:
        row["score"] = score
    if error_kind:
        row["error_kind"] = error_kind
    return row


def test_stage5_primary_success_and_genuine_zero_are_ready(tmp_path):
    paths = _paths(tmp_path, "stage5_zero")
    rows = [_row("exact", score=100), _row("visible_wrong", score=0, reason="wrong but scorable")]
    _write_primary(paths, rows)

    summary = campaign.write_readiness(paths, rows)
    effective = [json.loads(line) for line in paths.effective_rows.read_text().splitlines()]
    assert summary["readiness"] == "ready_for_adoption"
    assert [row["effective_score"] for row in effective] == [100, 0]
    assert all(row["terminal_disposition"] == "scored" for row in effective)


def test_stage5_recoverable_error_without_recovery_blocks_readiness(tmp_path):
    paths = _paths(tmp_path, "stage5_recovery_gap")
    rows = [_row("needle", error_kind="empty_output", reason="empty")]
    _write_primary(paths, rows)

    summary = campaign.write_readiness(paths, rows)
    effective = json.loads(paths.effective_rows.read_text().splitlines()[0])
    assert summary["readiness"] == "not_ready_manual_items"
    assert "empty_output_pending_retry" in summary["blockers"]
    assert effective["terminal_disposition"] == "empty_output_pending_retry"


def test_stage5_complete_recovery_to_zero_is_ready(tmp_path):
    paths = _paths(tmp_path, "stage5_recovered_zero")
    primary = _row("needle", error_kind="empty_output", reason="empty")
    _write_primary(paths, [primary])
    source = campaign._primary_row_hash(primary)
    recovery = _row("needle", score=0, reason="recovered but wrong")
    recovery.update({
        "repair_source_row_hash": source,
        "repair_action_id": "a1",
        "repair_attempt_number": 1,
        "repair_policy_version": campaign.RECOVERY_POLICY_VERSION,
        "_stage2b_final_outcome": {"evidence_source": "child_raw"},
    })
    child = paths.recovery_children_dir / "child-1"
    child.mkdir(parents=True)
    (child / "raw_results.jsonl").write_text(json.dumps(recovery, sort_keys=True) + "\n", encoding="utf-8")

    summary = campaign.write_readiness(paths, [primary])
    effective = json.loads(paths.effective_rows.read_text().splitlines()[0])
    assert summary["readiness"] == "ready_for_adoption"
    assert effective["result_origin"] == "recovered"
    assert effective["effective_score"] == 0


def test_stage5_pending_and_failed_judging_block_readiness(tmp_path):
    pending_paths = _paths(tmp_path, "stage5_pending_judge")
    pending = _row("subjective", reason="raw only, judge off")
    _write_primary(pending_paths, [pending])
    pending_summary = campaign.write_readiness(pending_paths, [pending])
    assert pending_summary["readiness"] == "not_ready_external_judge"
    assert "awaiting_external_judge" in pending_summary["blockers"]

    failed_paths = _paths(tmp_path, "stage5_failed_judge")
    failed = _row("subjective", reason="requires judge")
    failed["judge_source_row_hash"] = "judge-source"
    _write_primary(failed_paths, [failed])
    failed_paths.judge_results.write_text(
        json.dumps({"source_row_hash": "judge-source", "status": "judge_exhausted_unavailable"}) + "\n",
        encoding="utf-8",
    )
    failed_summary = campaign.write_readiness(failed_paths, [failed])
    assert failed_summary["readiness"] == "not_ready_external_judge"
    assert "judge_exhausted_unavailable" in failed_summary["blockers"]


def test_stage5_superseded_evidence_and_invalid_fork_readiness(tmp_path):
    paths = _paths(tmp_path, "stage5_superseded")
    primary = _row("needle", error_kind="harness_error", reason="bad ceiling")
    replacement = _row("needle", score=100, reason="corrected")
    _write_primary(paths, [primary])
    campaign.record_supersession(
        paths,
        source_campaign_id=paths.campaign_id,
        source_row=primary,
        replacement_run_id="catchup",
        replacement_row=replacement,
        reason="corrected synthetic evidence",
        operator="test",
    )

    summary = campaign.write_readiness(paths, [primary])
    effective = json.loads(paths.effective_rows.read_text().splitlines()[0])
    assert summary["readiness"] == "ready_for_adoption"
    assert effective["result_origin"] == "superseded"
    assert effective["effective_score"] == 100
    assert effective["supersession"]["terminal_replacement_row_hash"] == campaign._primary_row_hash(replacement)

    conflict = _row("needle", score=0, reason="conflict")
    with pytest.raises(campaign.CampaignError, match="ambiguous_supersession_fork"):
        campaign.record_supersession(
            paths,
            source_campaign_id=paths.campaign_id,
            source_row=primary,
            replacement_run_id="conflict",
            replacement_row=conflict,
            reason="conflict",
            operator="test",
        )
