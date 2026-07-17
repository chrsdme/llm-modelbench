import json

from llm_modelbench import scoring, sensitivity
from llm_modelbench.config import Config
from llm_modelbench.finals.agentic import manifest
from llm_modelbench.ollama import MockClient
from llm_modelbench.runner import _run_once
from llm_modelbench.tasks import TASKS, tasks_for


def _task(task_id):
    return next(t for t in TASKS if t.id == task_id)


def test_agentic_tool_tasks_registered_in_short_suite():
    ids = {t.id for t in tasks_for("short", ["agentic_tool"], ["text"])}
    assert {
        "agent_tool_select",
        "agent_tool_refuse",
        "agent_tool_repair",
        "agent_tool_state",
        "agent_schema_strict",
    }.issubset(ids)
    mf = manifest()
    assert mf["status"] in {"active_seed", "active_hardened"}
    assert mf["registered_in_tasks"] is True
    assert mf["task_count"] >= 5


def test_agentic_action_scorer_accepts_expected_call():
    task = _task("agent_tool_repair")
    score, reason = scoring.score_agentic_action('{"tool":"calculator.add","args":{"a":12,"b":7}}', task.meta)
    assert score == 100.0, reason


def test_agentic_action_scorer_penalizes_prose_contamination():
    task = _task("agent_tool_select")
    score, reason = scoring.score_agentic_action('Sure. {"tool":"read_file","args":{"path":"README.md"}}', task.meta)
    assert score == 75.0
    assert "format=prose_plus_json" in reason


def test_agentic_refusal_requires_no_call_and_reason():
    task = _task("agent_tool_refuse")
    good = scoring.score_agentic_action('{"tool":null,"args":{},"reason":"No safe offered tool can do that."}', task.meta)[0]
    bad = scoring.score_agentic_action('{"tool":"read_file","args":{"path":"/etc/passwd"},"reason":"try anyway"}', task.meta)[0]
    assert good == 100.0
    assert bad < 100.0


def test_mock_agentic_tool_run_scores_all_seed_tasks():
    cfg = Config()
    client = MockClient()
    for task_id in ["agent_tool_select", "agent_tool_refuse", "agent_tool_repair", "agent_tool_state", "agent_schema_strict"]:
        res = _run_once(client, cfg, "qwen2.5-coder:14b", _task(task_id))
        assert res["score"] == 100.0, (task_id, res)
        assert json.loads(res["output"])["args"] is not None


def test_needle_plan_uses_probe_aligned_ctx_values():
    script = sensitivity.plan_commands(
        run_prefix="probe",
        include_regex="m1",
        tasks="needle",
        level="short",
        ctx_values="default,4096,8192,16384,50000",
        num_predict_values="512",
    )
    assert "--level full" in script
    assert "--ctx 20000" in script
    assert "--ctx 40960" in script
    assert "--ctx 4096 \\" not in script
    assert "--ctx 8192 \\" not in script
    assert "--ctx 16384 \\" not in script
    assert "probe_ctx20000_np512" in script
    assert "probe_ctx40960_np512" in script
