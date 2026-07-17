import json
from pathlib import Path

from llm_modelbench import report


def _run_dir(tmp_path, task_hash, error_kind=None):
    d = tmp_path / "run"
    (d / "raw" / "agent_tool_select").mkdir(parents=True)
    (d / "raw" / "agent_tool_select" / "m.txt").write_text(
        '{"tool":"read_file","args":{"path":"README.md"},"comment":"x"}', encoding="utf-8")
    row = {"model": "m", "task": "agent_tool_select", "category": "agentic_tool",
           "score": 100.0, "reason": "agentic action ok", "task_hash": task_hash,
           "raw_path": "raw/agent_tool_select/m.txt", "error_kind": error_kind}
    (d / "raw_results.jsonl").write_text(json.dumps(row) + "\n")
    return d


def _enrich(d):
    from llm_modelbench.tasks import TASKS
    rows = [json.loads(line) for line in (d / "raw_results.jsonl").read_text().splitlines()]
    report._enrich_agentic_rows(d, rows, [t for t in TASKS if t.category == "agentic_tool"])
    return rows[0]


def test_report_refuses_to_rescore_a_row_from_a_different_task_version():
    """A stale rebuild would put a v9.5.17 decision_score next to a v9.5.15 score_blended."""
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        r = _enrich(_run_dir(Path(t), task_hash="deadbeef"))
    assert r.get("decision_score") is None
    assert "stale" in (r.get("enrichment") or "")


def test_report_does_not_rescore_a_model_failure_row():
    """An empty raw file is a thinking-budget failure, not malformed JSON."""
    import tempfile
    from llm_modelbench.runner import _task_hash
    from llm_modelbench.tasks import TASKS
    th = _task_hash(next(t for t in TASKS if t.id == "agent_tool_select"))
    with tempfile.TemporaryDirectory() as t:
        r = _enrich(_run_dir(Path(t), task_hash=th, error_kind="thinking_only"))
    assert r.get("caps_fired") in (None, [])
    assert "thinking_only" in (r.get("enrichment") or "")


def test_report_does_rescore_a_matching_row():
    import tempfile
    from llm_modelbench.runner import _task_hash
    from llm_modelbench.tasks import TASKS
    th = _task_hash(next(t for t in TASKS if t.id == "agent_tool_select"))
    with tempfile.TemporaryDirectory() as t:
        r = _enrich(_run_dir(Path(t), task_hash=th))
    assert r["decision_score"] == 50.0          # extra top-level key `comment`
    assert "extra_top_level_key=comment" in r["caps_fired"]
