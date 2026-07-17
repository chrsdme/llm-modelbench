import json

from llm_modelbench.runner import assess_run_validity


def _write_rows(path, rows):
    path.mkdir()
    (path / "raw_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_run_validity_rejects_empty_or_all_harness_error_runs(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert assess_run_validity(empty)["status"] == "invalid"

    broken = tmp_path / "broken"
    _write_rows(broken, [{"error_kind": "harness_error", "reason": "fixture missing"}])
    result = assess_run_validity(broken)
    assert result["status"] == "invalid"
    assert result["ranking_eligible"] is False
    assert (broken / "run_validity.json").exists()


def test_run_validity_distinguishes_partial_and_valid(tmp_path):
    partial = tmp_path / "partial"
    _write_rows(partial, [
        {"score": 100, "reason": "exact"},
        {"error_kind": "harness_error", "reason": "fixture missing"},
    ])
    assert assess_run_validity(partial)["status"] == "partial"

    valid = tmp_path / "valid"
    _write_rows(valid, [{"score": 0, "error_kind": "empty_output", "reason": "model failed"}])
    assert assess_run_validity(valid)["status"] == "valid"
