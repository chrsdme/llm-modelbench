import pytest

from llm_modelbench import campaign


def test_embedding_only_judge_is_never_selected_and_qwen_policy_is_respected():
    chosen = campaign.select_campaign_judge([
        {"name": "bge-m3:latest", "digest": "bge", "capabilities": ["embedding"]},
        {"name": "qwen2.5:14b", "digest": "q", "capabilities": ["completion"]},
        {"name": "selene", "digest": "s", "capabilities": ["completion"]},
    ], [])
    assert chosen["name"] == "selene"


def test_structured_judge_qualification_rejects_first_http_failure_without_retrying():
    class Broken:
        calls = 0
        def chat(self, *args, **kwargs):
            self.calls += 1
            return {"ok": False, "error": "HTTP 400 unsupported endpoint"}

    client = Broken()
    result = campaign.qualify_judge(client, {"name": "judge", "capabilities": ["completion"]})
    assert result["qualified"] is False
    assert client.calls == 1  # immediate candidate stop; no source rows are touched
    assert result["checks"]["errors"]


def test_judge_qualification_uses_deterministic_fallback_chain():
    class Fallback:
        def chat(self, model, *args, **kwargs):
            if model == "first":
                return {"ok": False, "error": "HTTP 400 unsupported endpoint"}
            return {"ok": True, "text": '{"score": 90, "confidence": 1, "verdict": "correct"}'}

    selected, chain = campaign.select_qualified_campaign_judge(Fallback(), [
        {"name": "first", "digest": "1", "capabilities": ["completion"]},
        {"name": "second", "digest": "2", "capabilities": ["completion"]},
    ], [], configured=["first", "second"])
    assert selected["name"] == "second"
    assert [item["model"] for item in chain] == ["first", "second"]


def test_recovery_reconciliation_fails_for_omitted_eligible_task_rows():
    rows = [{"model": "m", "task": task, "error_kind": "empty_output"}
            for task in ("git_conflict", "txt_emails", "agent_plan", "txt_sort", "json_extract", "git_commit")]
    with pytest.raises(campaign.CampaignError, match="recovery_plan_incomplete"):
        campaign.assert_recovery_reconciled(rows, {"actions": []})


def test_supersession_is_immutable_traceable_and_unambiguous(tmp_path):
    paths, _ = campaign.create_campaign("c", models=["m"], campaigns_root=tmp_path / "campaigns")
    source = {"model": "m", "task": "needle", "error_kind": "harness_error"}
    replacement = {"model": "m", "task": "needle", "score": 100, "reason": "corrected 66560 ceiling"}
    item = campaign.record_supersession(paths, source_campaign_id="old", source_row=source,
                                        replacement_run_id="catchup", replacement_row=replacement,
                                        reason="corrected needle ceiling")
    assert item["source_row_hash"] != item["replacement_row_hash"]
    assert campaign.supersession_map(paths)[item["source_row_hash"]]["replacement_run_id"] == "catchup"
    with pytest.raises(campaign.CampaignError, match="ambiguous"):
        campaign.record_supersession(paths, source_campaign_id="old", source_row=source,
                                     replacement_run_id="other", replacement_row={"score": 0}, reason="conflict")
