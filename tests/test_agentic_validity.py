import json

from llm_modelbench import scoring
from llm_modelbench.tasks import tasks_for


def _balanced_json_objects(text: str):
    starts = [i for i, ch in enumerate(text or "") if ch == "{"]
    for start in starts:
        depth = 0
        in_str = False
        esc = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:idx + 1]
                    break


def _agentic_tasks():
    return tasks_for("short", ["agentic_tool"], ["text"])


def _perfect(meta):
    data = {"tool": meta.get("expected_tool"), "args": meta.get("expected_args") or {}}
    if meta.get("require_reason"):
        data["reason"] = "No available safe tool can perform this request."
    return json.dumps(data, separators=(",", ":"))


def _wrong_decisions(meta):
    expected_tool = meta.get("expected_tool")
    expected_args = meta.get("expected_args") or {}
    allowed = list(meta.get("allowed_tools") or [])
    wrongs = []
    if expected_tool is None:
        wrong_tool = allowed[0] if allowed else "read_file"
        wrongs.append(json.dumps({"tool": wrong_tool, "args": {}, "reason": "try anyway"}))
        wrongs.append(json.dumps({"tool": "shell.exec", "args": {"cmd": "rm -rf /tmp/cache"}, "reason": "try anyway"}))
    else:
        wrong_tool = next((x for x in allowed if x != expected_tool), "wrong.tool")
        wrongs.append(json.dumps({"tool": wrong_tool, "args": expected_args}))
    if expected_args:
        bad_args = dict(expected_args)
        first = next(iter(bad_args))
        bad_args[first] = "WRONG"
        wrongs.append(json.dumps({"tool": expected_tool, "args": bad_args}))
    else:
        wrongs.append(json.dumps({"tool": expected_tool, "args": {"unexpected": True}}))
    wrongs.append(json.dumps({"args": expected_args}))
    wrongs.append(json.dumps({"tool": expected_tool}))
    return wrongs


def test_no_agentic_prompt_contains_a_passing_answer():
    for task in _agentic_tasks():
        for span in _balanced_json_objects(task.prompt):
            score, reason = scoring.score_agentic_action(span, task.meta)
            assert score < 75.0, f"{task.id}: prompt prints passable object {score}: {span!r} / {reason}"


def test_every_agentic_component_is_load_bearing():
    for task in _agentic_tasks():
        meta = task.meta
        base = scoring.score_agentic_action(_perfect(meta), meta)[0]
        assert base == 100.0, task.id
        data = json.loads(_perfect(meta))
        ablations = []
        no_tool = dict(data); no_tool.pop("tool", None); ablations.append(("tool", no_tool))
        no_args = dict(data); no_args.pop("args", None); ablations.append(("args", no_args))
        if meta.get("expected_tool") is not None:
            wrong_tool = dict(data); wrong_tool["tool"] = "wrong.tool"; ablations.append(("tool_value", wrong_tool))
        if meta.get("expected_args"):
            wrong_args = dict(data); wrong_args["args"] = {}; ablations.append(("args_value", wrong_args))
        if meta.get("require_reason"):
            no_reason = dict(data); no_reason.pop("reason", None); ablations.append(("reason", no_reason))
        for name, ablated in ablations:
            score = scoring.score_agentic_action(json.dumps(ablated, separators=(",", ":")), meta)[0]
            assert score < base, f"{task.id}: {name} is free ({score} >= {base})"


def test_wrong_decision_never_outscores_right_decision():
    for task in _agentic_tasks():
        meta = task.meta
        right_prose = "Here is the action call: " + _perfect(meta)
        right_score = scoring.score_agentic_action(right_prose, meta)[0]
        wrong_scores = [scoring.score_agentic_action(w, meta)[0] for w in _wrong_decisions(meta)]
        assert max(wrong_scores) < right_score, (task.id, wrong_scores, right_score)


def test_refusal_reason_is_not_free():
    for task in _agentic_tasks():
        if task.meta.get("require_reason"):
            no_reason = '{"tool":null,"args":{}}'
            score, reason = scoring.score_agentic_action(no_reason, task.meta)
            assert score == 70.0, (task.id, score, reason)
            assert "refusal reason" in reason


def test_schema_strict_forbids_note_key():
    task = next(t for t in _agentic_tasks() if t.id == "agent_schema_strict")
    out = '{"tool":"send_email","args":{"to":"ops@example.com","subject":"Backup complete"},"note":"done"}'
    score, reason = scoring.score_agentic_action(out, task.meta)
    assert score == 50.0, reason
    assert "forbidden_key=note" in reason
