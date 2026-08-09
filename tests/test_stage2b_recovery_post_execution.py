import json

import pytest

from llm_modelbench import campaign


def _primary(task="json_extract", **overrides):
    row = {"model": "m", "model_digest_resolved": "d", "task": task}
    row.update(overrides)
    return row


def _hash(row):
    return campaign._primary_row_hash(row)


def _action(rows, *, action_id="a1", status="recovered"):
    return {
        "action_id": action_id,
        "kind": "retry_generation",
        "model": "m",
        "model_digest": "d",
        "tasks": [row["task"] for row in rows],
        "source_row_hashes": {row["task"]: _hash(row) for row in rows},
        "status": status,
    }


def _child(source, *, task=None, score=100, error_kind=None, reason="ok", model="m", digest="d", action_id="a1",
           child_id="child", attempt=1, policy_version=campaign.RECOVERY_POLICY_VERSION):
    row = {
        "model": model,
        "model_digest_resolved": digest,
        "task": task or source["task"],
        "score": score,
        "reason": reason,
        "repair_source_row_hash": _hash(source),
        "repair_action_id": action_id,
        "repair_attempt_number": attempt,
        "_recovery_child_id": child_id,
    }
    if policy_version is not None:
        row["repair_policy_version"] = policy_version
    if error_kind is not None:
        row["error_kind"] = error_kind
        row["score"] = None
    return row


def _stage2b(rows, plan_actions, children):
    plan = {"actions": plan_actions, "reconciliation": {"exact": True}}
    result = {"actions": [{"action_id": action["action_id"], "tasks": action["tasks"], "status": action.get("status", "recovered")}
                          for action in plan_actions]}
    return campaign.reconcile_recovery_post_execution(rows, plan, result, children)


@pytest.mark.parametrize("score", [100, 50, 0])
def test_stage2b_visible_recovered_scores_are_terminal_scored(score):
    source = _primary(error_kind="empty_output")
    evidence = _stage2b([source], [_action([source])], [_child(source, score=score)])
    assert evidence["exact"] is True
    assert evidence["per_row_outcomes"][0]["disposition"] == "scored"
    assert evidence["per_row_outcomes"][0]["category"] == "recovered"


@pytest.mark.parametrize("error_kind,expected", [
    ("thinking_only", "terminal_thinking_only"),
    ("empty_output", "terminal_empty"),
    ("timeout", "terminal_transient"),
    ("transient_backend_failure", "terminal_transient"),
])
def test_stage2b_terminal_recovery_dispositions(error_kind, expected):
    source = _primary(error_kind=error_kind)
    child = _child(source, score=None, error_kind=error_kind, reason=error_kind)
    evidence = _stage2b([source], [_action([source], status=error_kind)], [child])
    assert evidence["exact"] is True
    assert evidence["per_row_outcomes"][0]["disposition"] == expected


@pytest.mark.parametrize("text", ["HTTP 500 Internal Server Error", "HTTP/1.1 503 Service Unavailable"])
def test_stage2b_narrow_legacy_transport_text_terminal_transient(text):
    source = _primary(error_kind="timeout", reason="request failed")
    child = _child(source, score=None, error_kind="harness_error", reason=text)
    evidence = _stage2b([source], [_action([source], status="timeout")], [child])
    assert evidence["per_row_outcomes"][0]["disposition"] == "terminal_transient"


@pytest.mark.parametrize("text", [
    "expected 5 keys but received 4",
    "validation failed after 5 attempts",
    "model returned 50 tokens",
    "parser rejected 5 fields",
])
def test_stage2b_digit_five_prose_is_not_terminal_transient(text):
    source = _primary(error_kind="empty_output", reason=text)
    assert campaign._terminal_after_recovery(source, {"status": "terminal_model_failure"}, None) != "terminal_transient"


def test_stage2b_planned_row_without_outcome_is_missing():
    source = _primary(error_kind="empty_output")
    evidence = _stage2b([source], [_action([source])], [])
    assert evidence["exact"] is False
    assert evidence["missing_source_identities"][0]["source_row_hash"] == _hash(source)


def test_stage2b_unexpected_child_source_fails():
    planned = _primary("json_extract", error_kind="empty_output")
    unexpected = _primary("git_commit", error_kind="empty_output")
    evidence = _stage2b([planned], [_action([planned])], [_child(unexpected)])
    assert evidence["exact"] is False
    assert evidence["unexpected_recovery_identities"][0]["source_row_hash"] == _hash(unexpected)


def test_stage2b_wrong_child_task_for_source_fails():
    source = _primary("json_extract", error_kind="empty_output")
    evidence = _stage2b([source], [_action([source])], [_child(source, task="git_commit")])
    assert evidence["exact"] is False
    assert evidence["invalid_child_attributions"][0]["reason"] == "task_source_mismatch"


def test_stage2b_contradictory_child_digest_fails():
    source = _primary("json_extract", model_digest_resolved="digest-A", error_kind="empty_output")
    action = _action([source])
    action["model_digest"] = "digest-A"
    child = _child(source, digest="digest-B")
    evidence = _stage2b([source], [action], [child])
    assert evidence["exact"] is False
    assert evidence["invalid_child_attributions"][0]["reason"] == "model_digest_source_mismatch"


def test_stage2b_wrong_action_id_for_correct_source_task_digest_fails():
    source = _primary("json_extract", error_kind="empty_output")
    evidence = _stage2b([source], [_action([source], action_id="a1")], [_child(source, action_id="a2")])
    assert evidence["exact"] is False
    assert evidence["invalid_child_attributions"][0]["reason"] == "action_id_mismatch"


def test_stage2b_native_child_missing_action_id_fails():
    source = _primary("json_extract", error_kind="empty_output")
    child = _child(source, action_id=None)
    evidence = _stage2b([source], [_action([source])], [child])
    assert evidence["exact"] is False
    assert evidence["invalid_child_attributions"][0]["reason"] == "missing_action_id"


@pytest.mark.parametrize("attempt", [None, 0, -1, "not-a-number"])
def test_stage2b_invalid_native_attempt_number_fails(attempt):
    source = _primary("json_extract", error_kind="empty_output")
    evidence = _stage2b([source], [_action([source])], [_child(source, attempt=attempt)])
    assert evidence["exact"] is False
    assert evidence["invalid_child_attributions"][0]["reason"] == "invalid_attempt_number"


def test_stage2b_contradictory_native_policy_version_fails():
    source = _primary("json_extract", error_kind="empty_output")
    evidence = _stage2b([source], [_action([source])], [_child(source, policy_version="old-policy")])
    assert evidence["exact"] is False
    assert evidence["invalid_child_attributions"][0]["reason"] == "policy_version_mismatch"


def test_stage2b_valid_full_child_provenance_succeeds():
    source = _primary("json_extract", error_kind="empty_output")
    evidence = _stage2b([source], [_action([source])], [_child(source, child_id="child-a", attempt=1)])
    assert evidence["exact"] is True
    outcome = evidence["final_per_row_outcomes"][0]
    assert outcome["action_id"] == "a1"
    assert outcome["child_run_id"] == "child-a"
    assert outcome["attempt_number"] == 1
    assert outcome["policy_version"] == campaign.RECOVERY_POLICY_VERSION


def test_stage2b_duplicate_child_attribution_fails():
    source = _primary(error_kind="empty_output")
    evidence = _stage2b([source], [_action([source])], [
        _child(source, child_id="c1"),
        _child(source, child_id="c2"),
    ])
    assert evidence["exact"] is False
    assert evidence["invalid_child_attributions"][0]["reason"] == "duplicate_attempt_number"


def test_stage2b_legitimate_two_attempt_chain_is_not_duplicate():
    source = _primary(error_kind="thinking_only")
    evidence = _stage2b([source], [_action([source])], [
        _child(source, score=None, error_kind="thinking_only", child_id="c1", attempt=1),
        _child(source, score=100, child_id="c2", attempt=2),
    ])
    assert evidence["exact"] is True
    assert [item["attempt_number"] for item in evidence["attempt_history"]] == [1, 2]
    assert evidence["final_per_row_outcomes"][0]["disposition"] == "scored"


def test_stage2b_three_attempt_chain_ending_visible_zero_final_scored():
    source = _primary(error_kind="thinking_only")
    evidence = _stage2b([source], [_action([source])], [
        _child(source, score=None, error_kind="thinking_only", child_id="c1", attempt=1),
        _child(source, score=None, error_kind="empty_output", child_id="c2", attempt=2),
        _child(source, score=0, reason="visible wrong", child_id="c3", attempt=3),
    ])
    assert evidence["exact"] is True
    assert evidence["final_per_row_outcomes"][0]["disposition"] == "scored"
    assert evidence["final_per_row_outcomes"][0]["score"] == 0


def test_stage2b_bounded_repeated_transient_attempts_final_terminal_transient():
    source = _primary(error_kind="timeout")
    evidence = _stage2b([source], [_action([source])], [
        _child(source, score=None, error_kind="timeout", child_id="c1", attempt=1),
        _child(source, score=None, error_kind="timeout", child_id="c2", attempt=2),
    ])
    assert evidence["exact"] is True
    assert evidence["final_per_row_outcomes"][0]["disposition"] == "terminal_transient"


def test_stage2b_same_action_same_attempt_incompatible_rows_fail():
    source = _primary(error_kind="empty_output")
    evidence = _stage2b([source], [_action([source])], [
        _child(source, score=100, reason="ok", child_id="c1", attempt=1),
        _child(source, score=0, reason="wrong", child_id="c2", attempt=1),
    ])
    assert evidence["exact"] is False
    assert evidence["invalid_child_attributions"][0]["reason"] == "duplicate_attempt_number"


def test_stage2b_incompatible_child_and_action_result_for_same_source_fails():
    source = _primary(error_kind="empty_output")
    plan = {"actions": [_action([source])], "reconciliation": {"exact": True}}
    result = {"actions": [{"action_id": "a1", "tasks": ["json_extract"], "status": "terminal_empty_output",
                           "error_kind": "empty_output", "score": None}]}
    evidence = campaign.reconcile_recovery_post_execution([source], plan, result, [_child(source, score=100)])
    assert evidence["exact"] is False
    assert any(item["reason"] == "conflicting_attempt_evidence" for item in evidence["invalid_child_attributions"])


def test_stage2b_grouped_action_all_rows_resolved_succeeds():
    rows = [_primary("json_extract", error_kind="empty_output"), _primary("git_commit", error_kind="thinking_only")]
    evidence = _stage2b(rows, [_action(rows)], [_child(row, score=index) for index, row in enumerate(rows)])
    assert evidence["exact"] is True
    assert {item["source_row_hash"] for item in evidence["recovered_source_identities"]} == {_hash(row) for row in rows}


def test_stage2b_grouped_action_with_one_missing_member_is_not_complete():
    rows = [_primary("json_extract", error_kind="empty_output"), _primary("git_commit", error_kind="thinking_only")]
    evidence = _stage2b(rows, [_action(rows)], [_child(rows[0])])
    assert evidence["exact"] is False
    assert evidence["missing_source_identities"][0]["source_row_hash"] == _hash(rows[1])


def test_stage2b_grouped_action_with_invalid_member_fails():
    rows = [_primary("json_extract", error_kind="empty_output"), _primary("git_commit", error_kind="thinking_only")]
    evidence = _stage2b(rows, [_action(rows)], [_child(rows[0]), _child(rows[1], task="json_extract")])
    assert evidence["exact"] is False
    assert any(item["reason"] == "task_source_mismatch" for item in evidence["invalid_child_attributions"])


@pytest.mark.parametrize("task,error_kind", [("git_commit", "empty_output"), ("git_conflict", "thinking_only")])
def test_stage2b_executable_lane_recovery_is_per_row(task, error_kind):
    source = _primary(task, error_kind=error_kind)
    child = _child(source, score=None, error_kind=error_kind)
    evidence = _stage2b([source], [_action([source])], [child])
    assert evidence["exact"] is True
    assert evidence["per_row_outcomes"][0]["source_row_hash"] == _hash(source)


def test_stage2b_visible_executable_zero_is_terminal_evidence():
    source = _primary("git_commit", error_kind="empty_output")
    evidence = _stage2b([source], [_action([source])], [_child(source, score=0, reason="wrong")])
    assert evidence["exact"] is True
    assert evidence["per_row_outcomes"][0]["disposition"] == "scored"


def test_stage2b_primary_evidence_immutable_and_reconciliation_idempotent(tmp_path):
    primary = _primary(error_kind="empty_output")
    path = tmp_path / "raw_results.jsonl"
    path.write_text(json.dumps(primary) + "\n")
    before = path.read_bytes()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    action = _action(rows)
    child = _child(rows[0])
    first = _stage2b(rows, [action], [child])
    second = _stage2b(rows, [action], [child])
    assert first == second
    assert path.read_bytes() == before


def test_stage2b_attempt_history_and_final_outcomes_are_deterministic():
    source = _primary(error_kind="thinking_only")
    children = [
        _child(source, score=0, child_id="c3", attempt=3),
        _child(source, score=None, error_kind="thinking_only", child_id="c1", attempt=1),
        _child(source, score=None, error_kind="empty_output", child_id="c2", attempt=2),
    ]
    first = _stage2b([source], [_action([source])], children)
    second = _stage2b([source], [_action([source])], list(reversed(children)))
    assert first["attempt_history"] == second["attempt_history"]
    assert len(first["final_per_row_outcomes"]) == 1
    assert first["final_per_row_outcomes"][0]["attempt_number"] == 3


def test_stage2b_mixed_child_and_action_result_evidence_resolves_per_source():
    h1 = _primary("json_extract", error_kind="empty_output")
    h2 = _primary("git_commit", error_kind="thinking_only")
    plan = {"actions": [_action([h1, h2])], "reconciliation": {"exact": True}}
    result = {"actions": [{"action_id": "a1", "tasks": ["git_commit"], "status": "recovered",
                           "score": 0, "reason": "visible wrong"}]}
    evidence = campaign.reconcile_recovery_post_execution([h1, h2], plan, result, [_child(h1, score=100)])
    assert evidence["exact"] is True
    outcomes = {item["source_row_hash"]: item for item in evidence["final_per_row_outcomes"]}
    assert outcomes[_hash(h1)]["evidence_source"] == "child_raw"
    assert outcomes[_hash(h2)]["evidence_source"] == "action_result"


def test_stage2b_child_for_one_source_does_not_suppress_action_result_for_another():
    h1 = _primary("json_extract", error_kind="empty_output")
    h2 = _primary("git_commit", error_kind="thinking_only")
    plan = {"actions": [_action([h1, h2])], "reconciliation": {"exact": True}}
    result = {"actions": [{"action_id": "a1", "tasks": ["git_commit"], "status": "timeout",
                           "error_kind": "timeout", "score": None}]}
    evidence = campaign.reconcile_recovery_post_execution([h1, h2], plan, result, [_child(h1, score=100)])
    assert evidence["exact"] is True
    assert {item["source_row_hash"] for item in evidence["final_per_row_outcomes"]} == {_hash(h1), _hash(h2)}


def test_stage2b_execute_persists_post_reconciliation(tmp_path):
    paths, manifest = campaign.create_campaign("stage2b", models=["m"], campaigns_root=tmp_path / "campaigns")
    primary = _primary(error_kind="empty_output")
    paths.primary_raw_results.write_text(json.dumps(primary) + "\n")
    manifest = campaign.transition(paths, manifest, "planned")
    campaign.transition(paths, manifest, "generating")
    before = paths.primary_raw_results.read_bytes()
    action = _action([primary])

    class Plan:
        def to_dict(self):
            return {"actions": [action]}

    result = campaign.execute_recovery_phase(
        paths, object(), object(),
        build_plan_fn=lambda *a, **k: Plan(),
        apply_plan_fn=lambda *a, **k: {"actions": [{"action_id": "a1", "tasks": ["json_extract"], "status": "recovered",
                                                    "score": 0, "visible_answer": True, "reason": "wrong"}]},
    )
    assert result["post_execution_reconciliation"]["exact"] is True
    assert result["post_execution_reconciliation"]["per_row_outcomes"][0]["disposition"] == "scored"
    assert paths.primary_raw_results.read_bytes() == before


def test_stage2b_execute_invalid_post_reconciliation_persists_then_raises(tmp_path):
    paths, manifest = campaign.create_campaign("stage2b_bad", models=["m"], campaigns_root=tmp_path / "campaigns")
    primary = _primary(model_digest_resolved="digest-A", error_kind="empty_output")
    paths.primary_raw_results.write_text(json.dumps(primary) + "\n")
    manifest = campaign.transition(paths, manifest, "planned")
    campaign.transition(paths, manifest, "generating")
    before = paths.primary_raw_results.read_bytes()
    action = _action([primary])
    action["model_digest"] = "digest-A"

    class Plan:
        def to_dict(self):
            return {"actions": [action]}

    with pytest.raises(campaign.CampaignError, match="recovery_post_execution_incomplete"):
        campaign.execute_recovery_phase(
            paths, object(), object(),
            build_plan_fn=lambda *a, **k: Plan(),
            apply_plan_fn=lambda *a, **k: {"actions": [{"action_id": "a1", "tasks": ["json_extract"],
                                                        "model_digest": "digest-B", "status": "recovered",
                                                        "score": 100, "visible_answer": True}]},
        )
    persisted = json.loads(paths.recovery_result.read_text())
    assert persisted["post_execution_reconciliation"]["invalid_child_attributions"][0]["reason"] == "model_digest_source_mismatch"
    assert paths.primary_raw_results.read_bytes() == before


def test_stage2b_invalid_post_reconciliation_is_not_effective_recovery(tmp_path):
    paths, _ = campaign.create_campaign("stage2b_readiness", models=["m"], campaigns_root=tmp_path / "campaigns")
    primary = _primary(error_kind="empty_output")
    source = _hash(primary)
    paths.primary_raw_results.write_text(json.dumps(primary) + "\n")
    child_dir = paths.recovery_children_dir / "child"
    child_dir.mkdir(parents=True)
    child_dir.joinpath("raw_results.jsonl").write_text(json.dumps(_child(primary, score=100)) + "\n")
    paths.recovery_result.write_text(json.dumps({
        "post_execution_reconciliation": {
            "exact": False,
            "recovered_source_identities": [{"source_row_hash": source}],
            "terminal_source_identities": [],
        }
    }))
    paths.recovery_plan.write_text(json.dumps({"actions": [{"action_id": "a1", "tasks": ["json_extract"],
                                                            "source_row_hashes": {"json_extract": source}}]}))

    summary = campaign.write_readiness(paths, [primary])
    row = json.loads(paths.effective_rows.read_text().splitlines()[0])
    assert row["result_origin"] == "primary"
    assert row["terminal_disposition"] == "empty_output_pending_retry"
    assert summary["readiness"] == "not_ready_manual_items"
