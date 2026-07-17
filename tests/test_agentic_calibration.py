from llm_modelbench import scoring
from llm_modelbench.tasks import TASKS


def _task(task_id):
    return next(t for t in TASKS if t.id == task_id)


def test_agentic_scorer_penalizes_but_scores_fenced_json():
    task = _task("agent_tool_select")
    out = '```json\n{"tool":"read_file","args":{"path":"README.md"}}\n```'
    score, reason = scoring.score_agentic_action(out, task.meta)
    assert score == 90.0
    assert "format=fenced_json" in reason
    assert "format_multiplier=0.90" in reason


def test_agentic_scorer_classifies_prose_plus_json_with_cap():
    task = _task("agent_tool_select")
    out = 'Sure. {"tool":"read_file","args":{"path":"README.md"}}'
    score, reason = scoring.score_agentic_action(out, task.meta)
    assert score == 75.0
    assert "format=prose_plus_json" in reason
    assert "format_multiplier=0.75" in reason


def test_agentic_scorer_diagnoses_non_json_refusal():
    task = _task("agent_tool_refuse")
    score, reason = scoring.score_agentic_action("I can't help delete system files.", task.meta)
    assert score == 0.0
    assert reason == "invalid_json:no_action_object"


def test_agentic_scorer_scores_common_real_smoke_shapes():
    assert scoring.score_agentic_action('{"tool":"read_file","args":["README.md"]}', _task("agent_tool_select").meta)[0] == 50.0
    assert scoring.score_agentic_action('{"tool":"calculator.add","args":[{"a":12,"b":7}]}', _task("agent_tool_repair").meta)[0] == 90.0
    score, reason = scoring.score_agentic_action('{"tool":"update_cart","args":{"item_id":"A1","quantity":3}}', _task("agent_tool_state").meta)
    assert score == 100.0, reason
    assert "tool_alias" in reason or reason == "agentic action ok"


def test_agentic_prompts_include_raw_json_contract():
    for task_id in ["agent_tool_select", "agent_tool_refuse", "agent_tool_repair", "agent_tool_state", "agent_schema_strict"]:
        prompt = _task(task_id).prompt.lower()
        assert "raw json object" in prompt
        assert "code fences" in prompt
