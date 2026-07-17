
from llm_modelbench import scoring
from llm_modelbench.aggregate import aggregate
from llm_modelbench.tasks import tasks_for


def _agentic_task(task_id):
    return next(t for t in tasks_for("short", ["agentic_tool"], ["text"]) if t.id == task_id)


def test_agentic_score_decomposes_exactly_for_representative_cells():
    cases = [
        ("agent_tool_select", '{"tool":"read_file","args":{"path":"README.md"}}'),
        ("agent_tool_select", '```json\n{"tool":"read_file","args":{"path":"README.md"}}\n```'),
        ("agent_tool_refuse", '{"tool":null,"args":{},"reason":"No safe offered tool can do that."}'),
        ("agent_tool_repair", '{"tool":"add","args":{"a":12,"b":7}}'),
        ("agent_nested_args", '{"tool":"ticket.create","args":{"ticket":{"title":"Disk alert","priority":"high","labels":["infra","disk"]}}}'),
    ]
    for task_id, out in cases:
        task = _agentic_task(task_id)
        detail = scoring.score_agentic_action_details(out, task.meta)
        assert abs(detail["decision_score"] * detail["format_multiplier"] - detail["score"]) < 1e-9


def test_caps_fired_uses_scorer_notes_not_comma_split_reason_text():
    task = _agentic_task("agent_nested_args")
    out = '{"tool":"ticket.create","args":{"ticket":{"title":"Disk alert","priority":"low","labels":["infra"]}}}'
    detail = scoring.score_agentic_action_details(out, task.meta)
    caps = detail["caps_fired"]
    assert any(c.startswith("args.ticket=") for c in caps)
    assert "'priority': 'high'" not in caps
    assert "'labels': ['infra'" not in caps
    assert "'disk']}" not in caps


def test_no_flawless_agentic_decision_model_ranks_below_flawed_model():
    rows = []
    for i in range(10):
        rows.append({
            "model": "always_right_fenced", "task": f"a{i}", "category": "agentic_tool",
            "score": 90.0, "decision_score": 100.0, "format_multiplier": 0.9,
            "reason": "agentic action ok (format=fenced_json, format_multiplier=0.90)",
            "size_gb": 4.68, "tps": 70.0, "output_chars": 10,
        })
    for i in range(10):
        decision = 40.0 if i == 0 else 100.0
        rows.append({
            "model": "one_wrong_strict", "task": f"a{i}", "category": "agentic_tool",
            "score": decision, "decision_score": decision, "format_multiplier": 1.0,
            "reason": "agentic action ok" if decision == 100.0 else "agentic action 40.0/100, missing: tool=x, decision_cap=40",
            "size_gb": 4.68, "tps": 70.0, "output_chars": 10,
        })
    difficulty = {f"a{i}": 1.0 for i in range(10)}
    lb, _ = aggregate(rows, {"agentic_tool": 1.0}, difficulty)
    by_model = {r["model"]: r for r in lb}
    assert by_model["always_right_fenced"]["quality"] == 100.0
    assert by_model["always_right_fenced"]["score_blended"] == 90.0
    assert by_model["one_wrong_strict"]["quality"] == 94.0
    assert by_model["always_right_fenced"]["quality"] > by_model["one_wrong_strict"]["quality"]
    assert by_model["always_right_fenced"]["value_per_gb"] > by_model["one_wrong_strict"]["value_per_gb"]


def test_format_compliance_reports_strict_rate_not_only_mean_multiplier():
    rows = []
    for i in range(10):
        rows.append({
            "model": "always_fenced", "task": f"a{i}", "category": "agentic_tool",
            "score": 90.0, "decision_score": 100.0, "format_multiplier": 0.9,
            "format_deviation": "fenced_json", "reason": "agentic action ok",
            "size_gb": 1.0, "tps": 1.0, "output_chars": 10,
        })
        mult = 0.75 if i == 0 else 1.0
        rows.append({
            "model": "strict_then_prose", "task": f"a{i}", "category": "agentic_tool",
            "score": 100.0 * mult, "decision_score": 100.0, "format_multiplier": mult,
            "format_deviation": "prose_plus_json" if mult < 1.0 else "strict_json", "reason": "agentic action ok",
            "size_gb": 1.0, "tps": 1.0, "output_chars": 10,
        })
    difficulty = {f"a{i}": 1.0 for i in range(10)}
    lb, _ = aggregate(rows, {"agentic_tool": 1.0}, difficulty)
    by_model = {r["model"]: r for r in lb}
    assert by_model["always_fenced"]["agentic_format_strict_rate"] == 0.0
    assert by_model["always_fenced"]["agentic_format_modal_deviation"] == "fenced_json"
    assert by_model["strict_then_prose"]["agentic_format_strict_rate"] == 0.9
    assert by_model["strict_then_prose"]["agentic_format_modal_deviation"] == "prose_plus_json"
