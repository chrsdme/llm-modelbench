import json

import pytest

from llm_modelbench import campaign


def _primary(task="json_extract", **overrides):
    row = {"model": "m", "model_digest_resolved": "d", "task": task}
    row.update(overrides)
    return row


def _hash(row):
    return campaign._primary_row_hash(row)


def _action(rows, *, action_id="a1", status="recovered", attempt_limit=1):
    action = {
        "action_id": action_id,
        "kind": "retry_generation",
        "model": "m",
        "model_digest": "d",
        "tasks": [row["task"] for row in rows],
        "source_row_hashes": {row["task"]: _hash(row) for row in rows},
        "status": status,
    }
    if attempt_limit != 1:
        action["overrides"] = {"retry_profiles": [{} for _ in range(attempt_limit)]}
    return action


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
    evidence = _stage2b([source], [_action([source], attempt_limit=2)], [
        _child(source, score=None, error_kind="thinking_only", child_id="c1", attempt=1),
        _child(source, score=100, child_id="c2", attempt=2),
    ])
    assert evidence["exact"] is True
    assert [item["attempt_number"] for item in evidence["attempt_history"]] == [1, 2]
    assert evidence["final_per_row_outcomes"][0]["disposition"] == "scored"


def test_stage2b_three_attempt_chain_ending_visible_zero_final_scored():
    source = _primary(error_kind="thinking_only")
    evidence = _stage2b([source], [_action([source], attempt_limit=3)], [
        _child(source, score=None, error_kind="thinking_only", child_id="c1", attempt=1),
        _child(source, score=None, error_kind="empty_output", child_id="c2", attempt=2),
        _child(source, score=0, reason="visible wrong", child_id="c3", attempt=3),
    ])
    assert evidence["exact"] is True
    assert evidence["final_per_row_outcomes"][0]["disposition"] == "scored"
    assert evidence["final_per_row_outcomes"][0]["score"] == 0


def test_stage2b_bounded_repeated_transient_attempts_final_terminal_transient():
    source = _primary(error_kind="timeout")
    evidence = _stage2b([source], [_action([source], attempt_limit=2)], [
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
    first = _stage2b([source], [_action([source], attempt_limit=3)], children)
    second = _stage2b([source], [_action([source], attempt_limit=3)], list(reversed(children)))
    assert first["attempt_history"] == second["attempt_history"]
    assert len(first["final_per_row_outcomes"]) == 1
    assert first["final_per_row_outcomes"][0]["attempt_number"] == 3


@pytest.mark.parametrize("first_score", [0, 50])
def test_stage2b_attempt_after_first_visible_result_fails(first_score):
    source = _primary(error_kind="empty_output")
    evidence = _stage2b([source], [_action([source], attempt_limit=2)], [
        _child(source, score=first_score, child_id="z-first", attempt=1),
        _child(source, score=100, child_id="a-second", attempt=2),
    ])
    assert evidence["exact"] is False
    assert any(item["reason"] == "attempt_after_visible_result" for item in evidence["invalid_child_attributions"])
    assert [item["attempt_number"] for item in evidence["attempt_history"]] == [1, 2]


def test_stage2b_valid_bounded_retry_sequence_from_policy_succeeds():
    source = _primary(error_kind="thinking_only")
    evidence = _stage2b([source], [_action([source], attempt_limit=3)], [
        _child(source, score=None, error_kind="thinking_only", child_id="c1", attempt=1),
        _child(source, score=None, error_kind="empty_output", child_id="c2", attempt=2),
        _child(source, score=100, child_id="c3", attempt=3),
    ])
    assert evidence["exact"] is True
    assert [item["attempt_number"] for item in evidence["attempt_history"]] == [1, 2, 3]


@pytest.mark.parametrize("attempt", [3, 999])
def test_stage2b_attempt_beyond_policy_bound_fails(attempt):
    source = _primary(error_kind="timeout")
    evidence = _stage2b([source], [_action([source], attempt_limit=2)], [
        _child(source, score=None, error_kind="timeout", child_id="c1", attempt=1),
        _child(source, score=None, error_kind="timeout", child_id="cX", attempt=attempt),
    ])
    assert evidence["exact"] is False
    assert any(item["reason"] == "attempt_out_of_policy" for item in evidence["invalid_child_attributions"])


def test_stage2b_malformed_bounded_attempt_gap_fails():
    source = _primary(error_kind="timeout")
    evidence = _stage2b([source], [_action([source], attempt_limit=3)], [
        _child(source, score=None, error_kind="timeout", child_id="c1", attempt=1),
        _child(source, score=None, error_kind="timeout", child_id="c3", attempt=3),
    ])
    assert evidence["exact"] is False
    assert any(item["reason"] == "invalid_attempt_sequence" for item in evidence["invalid_child_attributions"])


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


def test_stage2b_readiness_uses_final_outcome_not_child_directory_order(tmp_path):
    paths, _ = campaign.create_campaign("stage2b_authoritative_child", models=["m"], campaigns_root=tmp_path / "campaigns")
    primary = _primary(error_kind="thinking_only")
    before_row = json.dumps(primary) + "\n"
    paths.primary_raw_results.write_text(before_row)
    action = _action([primary], attempt_limit=2)
    plan = {"actions": [action], "reconciliation": {"exact": True}}
    paths.recovery_plan.write_text(json.dumps(plan, sort_keys=True))

    later_sorting_first = paths.recovery_children_dir / "a_attempt2"
    earlier_sorting_later = paths.recovery_children_dir / "z_attempt1"
    later_sorting_first.mkdir(parents=True)
    earlier_sorting_later.mkdir(parents=True)
    attempt1 = _child(primary, score=None, error_kind="empty_output", child_id="z_attempt1", attempt=1)
    attempt2 = _child(primary, score=0, reason="visible wrong", child_id="a_attempt2", attempt=2)
    earlier_sorting_later.joinpath("raw_results.jsonl").write_text(json.dumps(attempt1) + "\n")
    later_sorting_first.joinpath("raw_results.jsonl").write_text(json.dumps(attempt2) + "\n")
    post = campaign.reconcile_recovery_post_execution([primary], plan, {"actions": []}, [attempt1, attempt2])
    assert post["exact"] is True
    paths.recovery_result.write_text(json.dumps({
        "actions": [{"action_id": "a1", "tasks": ["json_extract"], "status": "recovered"}],
        "post_execution_reconciliation": post,
    }, sort_keys=True))

    campaign.write_readiness(paths, [primary])
    row = json.loads(paths.effective_rows.read_text().splitlines()[0])
    assert row["effective_score"] == 0
    assert row["result_origin"] == "recovered"
    assert row["terminal_disposition"] == "scored"
    assert row["recovery_attempt_number"] == 2
    assert row["recovery_child_id"] == "a_attempt2"
    assert row["provenance"]["recovery_evidence_source"] == "child_raw"
    assert paths.primary_raw_results.read_text() == before_row


def test_stage2b_readiness_materializes_mixed_child_and_action_result_finals(tmp_path):
    paths, _ = campaign.create_campaign("stage2b_authoritative_mixed", models=["m"], campaigns_root=tmp_path / "campaigns")
    h1 = _primary("json_extract", error_kind="empty_output")
    h2 = _primary("git_commit", error_kind="thinking_only")
    primary_text = "".join(json.dumps(row) + "\n" for row in [h1, h2])
    paths.primary_raw_results.write_text(primary_text)
    action = _action([h1, h2])
    plan = {"actions": [action], "reconciliation": {"exact": True}}
    result = {"actions": [{"action_id": "a1", "tasks": ["git_commit"], "status": "recovered",
                           "score": 0, "reason": "visible wrong"}]}
    child_dir = paths.recovery_children_dir / "child-h1"
    child_dir.mkdir(parents=True)
    h1_child = _child(h1, score=100, child_id="child-h1")
    child_dir.joinpath("raw_results.jsonl").write_text(json.dumps(h1_child) + "\n")
    post = campaign.reconcile_recovery_post_execution([h1, h2], plan, result, [h1_child])
    assert post["exact"] is True
    paths.recovery_plan.write_text(json.dumps(plan, sort_keys=True))
    paths.recovery_result.write_text(json.dumps(result | {"post_execution_reconciliation": post}, sort_keys=True))

    campaign.write_readiness(paths, [h1, h2])
    rows = [json.loads(line) for line in paths.effective_rows.read_text().splitlines()]
    by_task = {row["task"]: row for row in rows}
    assert by_task["json_extract"]["effective_score"] == 100
    assert by_task["json_extract"]["recovery_child_id"] == "child-h1"
    assert by_task["json_extract"]["provenance"]["recovery_evidence_source"] == "child_raw"
    assert by_task["git_commit"]["effective_score"] == 0
    assert by_task["git_commit"]["result_origin"] == "recovered"
    assert by_task["git_commit"]["terminal_disposition"] == "scored"
    assert by_task["git_commit"]["recovery_child_id"] is None
    assert by_task["git_commit"]["provenance"]["recovery_evidence_source"] == "action_result"
    assert paths.primary_raw_results.read_text() == primary_text


def test_stage2b_readiness_materializes_action_result_only_scored_final(tmp_path):
    paths, _ = campaign.create_campaign("stage2b_action_result_scored", models=["m"], campaigns_root=tmp_path / "campaigns")
    primary = _primary(error_kind="empty_output")
    paths.primary_raw_results.write_text(json.dumps(primary) + "\n")
    action = _action([primary])
    plan = {"actions": [action], "reconciliation": {"exact": True}}
    result = {"actions": [{"action_id": "a1", "tasks": ["json_extract"], "status": "recovered",
                           "score": 0, "reason": "visible wrong"}]}
    post = campaign.reconcile_recovery_post_execution([primary], plan, result, [])
    assert post["exact"] is True
    paths.recovery_plan.write_text(json.dumps(plan, sort_keys=True))
    paths.recovery_result.write_text(json.dumps(result | {"post_execution_reconciliation": post}, sort_keys=True))

    campaign.write_readiness(paths, [primary])
    row = json.loads(paths.effective_rows.read_text().splitlines()[0])
    assert row["effective_score"] == 0
    assert row["result_origin"] == "recovered"
    assert row["terminal_disposition"] == "scored"
    assert row["recovery_child_id"] is None
    assert row["provenance"]["recovery_evidence_source"] == "action_result"


def test_stage2b_readiness_materializes_action_result_terminal_final(tmp_path):
    paths, _ = campaign.create_campaign("stage2b_action_result_terminal", models=["m"], campaigns_root=tmp_path / "campaigns")
    primary = _primary(error_kind="empty_output")
    paths.primary_raw_results.write_text(json.dumps(primary) + "\n")
    action = _action([primary])
    plan = {"actions": [action], "reconciliation": {"exact": True}}
    result = {"actions": [{"action_id": "a1", "tasks": ["json_extract"], "status": "terminal_empty_output",
                           "error_kind": "empty_output", "score": None, "reason": "still empty"}]}
    post = campaign.reconcile_recovery_post_execution([primary], plan, result, [])
    assert post["exact"] is True
    paths.recovery_plan.write_text(json.dumps(plan, sort_keys=True))
    paths.recovery_result.write_text(json.dumps(result | {"post_execution_reconciliation": post}, sort_keys=True))

    campaign.write_readiness(paths, [primary])
    row = json.loads(paths.effective_rows.read_text().splitlines()[0])
    assert row["effective_score"] is None
    assert row["result_origin"] == "recovery_terminal"
    assert row["terminal_disposition"] == "terminal_empty"
    assert row["provenance"]["recovery_evidence_source"] == "action_result"


def test_stage2b_repeated_write_readiness_is_deterministic_with_post_reconciliation(tmp_path):
    paths, _ = campaign.create_campaign("stage2b_readiness_idempotent", models=["m"], campaigns_root=tmp_path / "campaigns")
    primary = _primary(error_kind="empty_output")
    paths.primary_raw_results.write_text(json.dumps(primary) + "\n")
    action = _action([primary])
    plan = {"actions": [action], "reconciliation": {"exact": True}}
    result = {"actions": [{"action_id": "a1", "tasks": ["json_extract"], "status": "recovered",
                           "score": 0, "reason": "visible wrong"}]}
    post = campaign.reconcile_recovery_post_execution([primary], plan, result, [])
    paths.recovery_plan.write_text(json.dumps(plan, sort_keys=True))
    paths.recovery_result.write_text(json.dumps(result | {"post_execution_reconciliation": post}, sort_keys=True))

    first = campaign.write_readiness(paths, [primary])
    first_rows = paths.effective_rows.read_text()
    second = campaign.write_readiness(paths, [primary])
    second_rows = paths.effective_rows.read_text()
    assert first == second
    assert first_rows == second_rows


def _write_exact_post_with_child_final(paths, primary, *, child, persisted_child=None):
    action = _action([primary])
    plan = {"actions": [action], "reconciliation": {"exact": True}}
    post = campaign.reconcile_recovery_post_execution([primary], plan, {"actions": []}, [persisted_child or child])
    assert post["exact"] is True
    paths.recovery_plan.write_text(json.dumps(plan, sort_keys=True))
    paths.recovery_result.write_text(json.dumps({
        "actions": [{"action_id": "a1", "tasks": [primary["task"]], "status": "recovered", "score": 100}],
        "post_execution_reconciliation": post,
    }, sort_keys=True))
    return post


def test_stage2b_exact_post_missing_child_raw_does_not_fallback_to_action(tmp_path):
    paths, _ = campaign.create_campaign("stage2b_missing_final_child", models=["m"], campaigns_root=tmp_path / "campaigns")
    primary = _primary(error_kind="empty_output")
    paths.primary_raw_results.write_text(json.dumps(primary) + "\n")
    _write_exact_post_with_child_final(paths, primary, child=_child(primary, score=0, child_id="child-X"))

    campaign.write_readiness(paths, [primary])
    row = json.loads(paths.effective_rows.read_text().splitlines()[0])
    assert row["result_origin"] == "primary"
    assert row["effective_score"] is None
    assert row["terminal_disposition"] == "empty_output_pending_retry"
    assert row["provenance"]["recovery_action_id"] is None


def test_stage2b_exact_post_changed_child_score_is_non_effective(tmp_path):
    paths, _ = campaign.create_campaign("stage2b_changed_child_score", models=["m"], campaigns_root=tmp_path / "campaigns")
    primary = _primary(error_kind="empty_output")
    paths.primary_raw_results.write_text(json.dumps(primary) + "\n")
    persisted = _child(primary, score=0, reason="visible wrong", child_id="child-X")
    _write_exact_post_with_child_final(paths, primary, child=persisted)
    child_dir = paths.recovery_children_dir / "child-X"
    child_dir.mkdir(parents=True)
    child_dir.joinpath("raw_results.jsonl").write_text(json.dumps(_child(primary, score=100, child_id="child-X")) + "\n")

    campaign.write_readiness(paths, [primary])
    row = json.loads(paths.effective_rows.read_text().splitlines()[0])
    assert row["result_origin"] == "primary"
    assert row["effective_score"] != 100
    assert row["terminal_disposition"] == "empty_output_pending_retry"


@pytest.mark.parametrize("field,value", [
    ("task", "git_commit"),
    ("repair_action_id", "wrong-action"),
    ("repair_attempt_number", 2),
])
def test_stage2b_exact_post_child_identity_mismatch_is_non_effective(tmp_path, field, value):
    paths, _ = campaign.create_campaign(f"stage2b_child_identity_{field}", models=["m"], campaigns_root=tmp_path / "campaigns")
    primary = _primary(error_kind="empty_output")
    paths.primary_raw_results.write_text(json.dumps(primary) + "\n")
    persisted = _child(primary, score=0, child_id="child-X")
    _write_exact_post_with_child_final(paths, primary, child=persisted)
    modified = dict(persisted)
    modified[field] = value
    child_dir = paths.recovery_children_dir / "child-X"
    child_dir.mkdir(parents=True)
    child_dir.joinpath("raw_results.jsonl").write_text(json.dumps(modified) + "\n")

    campaign.write_readiness(paths, [primary])
    row = json.loads(paths.effective_rows.read_text().splitlines()[0])
    assert row["result_origin"] == "primary"
    assert row["terminal_disposition"] == "empty_output_pending_retry"


@pytest.mark.parametrize("field,value", [
    ("model_digest_resolved", "changed-digest"),
    ("repair_policy_version", "changed-policy"),
])
def test_stage2b_exact_post_child_digest_or_policy_mismatch_is_non_effective(tmp_path, field, value):
    paths, _ = campaign.create_campaign(f"stage2b_child_semantic_{field}", models=["m"], campaigns_root=tmp_path / "campaigns")
    primary = _primary(error_kind="empty_output")
    paths.primary_raw_results.write_text(json.dumps(primary) + "\n")
    persisted = _child(primary, score=0, child_id="child-X")
    _write_exact_post_with_child_final(paths, primary, child=persisted)
    modified = dict(persisted)
    modified[field] = value
    child_dir = paths.recovery_children_dir / "child-X"
    child_dir.mkdir(parents=True)
    child_dir.joinpath("raw_results.jsonl").write_text(json.dumps(modified) + "\n")

    campaign.write_readiness(paths, [primary])
    row = json.loads(paths.effective_rows.read_text().splitlines()[0])
    assert row["result_origin"] == "primary"
    assert row["terminal_disposition"] == "empty_output_pending_retry"


def test_stage2b_exact_post_valid_child_uses_persisted_final_semantics(tmp_path):
    paths, _ = campaign.create_campaign("stage2b_valid_child_final_semantics", models=["m"], campaigns_root=tmp_path / "campaigns")
    primary = _primary(error_kind="empty_output")
    raw_text = json.dumps(primary) + "\n"
    paths.primary_raw_results.write_text(raw_text)
    child = _child(primary, score=0, reason="persisted visible wrong", child_id="child-X")
    post = _write_exact_post_with_child_final(paths, primary, child=child)
    child_dir = paths.recovery_children_dir / "child-X"
    child_dir.mkdir(parents=True)
    child_dir.joinpath("raw_results.jsonl").write_text(json.dumps(child) + "\n")
    final = post["final_per_row_outcomes"][0]

    campaign.write_readiness(paths, [primary])
    row = json.loads(paths.effective_rows.read_text().splitlines()[0])
    assert row["effective_score"] == final["score"] == 0
    assert row["effective_reason"] == final["reason_text"]
    assert row["terminal_disposition"] == final["disposition"] == "scored"
    assert row["recovery_attempt_number"] == final["attempt_number"]
    assert row["provenance"]["recovery_action_id"] == final["action_id"]
    assert row["recovery_child_id"] == final["child_run_id"]
    assert row["provenance"]["recovery_evidence_source"] == "child_raw"
    assert paths.primary_raw_results.read_text() == raw_text


def test_stage2b_legacy_no_post_reconciliation_keeps_action_compatibility(tmp_path):
    paths, _ = campaign.create_campaign("stage2b_legacy_no_post", models=["m"], campaigns_root=tmp_path / "campaigns")
    primary = _primary(error_kind="empty_output")
    source = _hash(primary)
    paths.primary_raw_results.write_text(json.dumps(primary) + "\n")
    paths.recovery_plan.write_text(json.dumps({"actions": [{"action_id": "a1", "tasks": ["json_extract"],
                                                            "source_row_hashes": {"json_extract": source}}]}))
    paths.recovery_result.write_text(json.dumps({"actions": [{"action_id": "a1", "tasks": ["json_extract"],
                                                              "status": "terminal_empty_output"}]}))

    campaign.write_readiness(paths, [primary])
    row = json.loads(paths.effective_rows.read_text().splitlines()[0])
    assert row["result_origin"] == "recovery_terminal"
    assert row["terminal_disposition"] == "terminal_empty"
    assert row["provenance"]["recovery_action_id"] == "a1"
