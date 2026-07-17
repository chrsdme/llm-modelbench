import json
from pathlib import Path

from llm_modelbench import scoring, compare
from llm_modelbench.config import Config
from llm_modelbench.ollama import MockClient
from llm_modelbench.runner import _run_once
from llm_modelbench.tasks import TASKS, tasks_for


def _task(task_id):
    return next(t for t in TASKS if t.id == task_id)


def test_v9515_hardening_tasks_registered():
    ids = {t.id for t in tasks_for("short", ["agentic_tool"], ["text"])}
    assert {
        "agent_unknown_tool_reject",
        "agent_schema_collision",
        "agent_state_delta",
        "agent_malformed_repair",
        "agent_nested_args",
    }.issubset(ids)


def test_agentic_schema_collision_rejects_action_kwargs_shape():
    task = _task("agent_schema_collision")
    bad = '{"action":"send_email","kwargs":{"to":"ops@example.com","subject":"Incident acknowledged"}}'
    score, reason = scoring.score_agentic_action(bad, task.meta)
    assert score < 100.0
    assert "forbidden_key=action" in reason or "tool" in reason


def test_agentic_state_delta_trap_scores_final_quantity_only():
    task = _task("agent_state_delta")
    good = '{"tool":"cart.update","args":{"sku":"B2","quantity":3}}'
    bad = '{"tool":"cart.update","args":{"sku":"B2","quantity":2}}'
    assert scoring.score_agentic_action(good, task.meta)[0] == 100.0
    assert scoring.score_agentic_action(bad, task.meta)[0] < 100.0


def test_agentic_nested_args_requires_nested_ticket_object():
    task = _task("agent_nested_args")
    good = '{"tool":"ticket.create","args":{"ticket":{"title":"Disk alert","priority":"high","labels":["infra","disk"]}}}'
    flat = '{"tool":"ticket.create","args":{"title":"Disk alert","priority":"high","labels":["infra","disk"]}}'
    assert scoring.score_agentic_action(good, task.meta)[0] == 100.0
    assert scoring.score_agentic_action(flat, task.meta)[0] < 100.0


def test_mock_scores_all_v9515_agentic_tasks():
    cfg = Config()
    client = MockClient()
    for task in tasks_for("short", ["agentic_tool"], ["text"]):
        res = _run_once(client, cfg, "qwen2.5-coder:14b", task)
        assert res["score"] == 100.0, (task.id, res)
        assert json.loads(res["output"])["args"] is not None


def test_repeatability_report_flags_moving_score(tmp_path: Path):
    def write_run(name, score):
        d = tmp_path / name
        d.mkdir()
        (d / "summary_meta.json").write_text('{"llm_modelbench_version":"9.5.15"}')
        row = {"model":"m1", "task":"agent_tool_select", "score":score, "reason":"r"}
        (d / "raw_results.jsonl").write_text(json.dumps(row) + "\n")
        return d
    a = write_run("a", 100.0)
    b = write_run("b", 85.0)
    text = compare.repeatability_report([a, b])
    assert "moving" in text
    assert "15.0" in text
