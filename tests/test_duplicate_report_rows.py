import pytest

from llm_modelbench import report


def _row(model="m", task="t", score=100.0, reason="ok", tps=1.0):
    return {
        "model": model,
        "task": task,
        "task_hash": "abc",
        "category": "coding_python",
        "benchmark_version": "9.5.18",
        "score": score,
        "reason": reason,
        "tps": tps,
        "error_kind": None,
        "warning_kind": None,
        "done_reason": "stop",
    }


def test_report_deduplicates_non_conflicting_result_rows_and_keeps_latest():
    rows = [_row(tps=1.0), _row(tps=2.0)]
    deduped, dropped = report._dedupe_report_rows(rows)
    assert len(deduped) == 1
    assert len(dropped) == 1
    assert deduped[0]["tps"] == 2.0
    assert dropped[0]["task"] == "t"


def test_report_refuses_conflicting_duplicate_result_rows():
    rows = [_row(score=100.0, reason="ok"), _row(score=0.0, reason="wrong")]
    with pytest.raises(SystemExit, match="conflicting duplicate"):
        report._dedupe_report_rows(rows)


def test_report_context_records_duplicate_row_drop():
    rows = [_row()]
    context = report._report_context(
        __import__("pathlib").Path("."),
        rows,
        type("Cfg", (), {"min_report_tasks_per_category": 2})(),
        raw_row_count=2,
        duplicate_rows=[{"model": "m", "task": "t"}],
    )
    assert context["raw_row_count"] == 2
    assert context["report_row_count"] == 1
    assert context["duplicate_rows_dropped"] == 1
