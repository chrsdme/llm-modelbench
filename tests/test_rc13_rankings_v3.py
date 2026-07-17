import json
from pathlib import Path

from llm_modelbench import rankings
from llm_modelbench.freeze import create_freeze
from llm_modelbench.rankings_v3 import SCHEMA_VERSION, build_v3_payload


def _write_run(runs_dir: Path, run_id: str, rows):
    run = runs_dir / run_id
    run.mkdir(parents=True)
    (run / "raw_results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": "full"}))
    (run / "model_identities.json").write_text(json.dumps({"coder:latest": {"digest": "digest-coder"}}))
    return run


def test_rankings_v3_artifacts_and_use_cases(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs, "profile", [{
        "model": "coder:latest",
        "task": "needle",
        "category": "long_context",
        "family": "text",
        "score": 100.0,
        "needle_coverage": 1.0,
        "max_verified_ctx": 64773,
        "needle_target_ctx": 64000,
        "needle_target_status": "ready",
        "needle_target_tps": 12.0,
        "needle_target_prompt_tps": 200.0,
        "needle_attempted": [{"size": 65536, "num_ctx": 65093, "found": True, "tps": 12.0}],
        "task_hash": rankings._CURRENT_HASHES["needle"],
        "timestamp": "2026-07-17T00:00:00Z",
        "benchmark_version": "1.0.0rc13",
    }])
    out = tmp_path / "rankings"
    result = rankings.write_rankings(runs, out, html_template="<html>__MASTER_SUMMARY_JSON__</html>")

    assert (out / "master_report_v3_data.json").exists()
    assert (out / "master_report_v3.html").exists()
    assert result["v3_schema_version"] == SCHEMA_VERSION

    data = json.loads((out / "master_report_v3_data.json").read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert "long_context_64k" in data["use_case_rankings"]
    assert data["use_case_rankings"]["long_context_64k"]["rows"][0]["model"] == "coder:latest"
    assert data["models"][0]["badges"]
    assert "Manual rescan" in (out / "master_report_v3.html").read_text()


def test_v3_payload_keeps_capability_and_recovery_labels():
    master_payload = {
        "models": [{
            "display_name": "vision:latest",
            "digest": "d1",
            "class": "vision",
            "families": ["vision", "text"],
            "quality_status": "complete",
            "overall_mean_score": 50.0,
            "coverage_ratio": 1.0,
            "capability_limited": True,
            "capability_measured_failure": True,
            "recovery_limited": False,
            "size_gb": 4.0,
            "tok_s": 20.0,
            "quality_status_reasons": ["vision unavailable", "fim measured zero"],
            "categories": {"ocr": {"score": 0.0}, "coding": {"score": 50.0}},
            "long_context_profile": {"target_status": "not_verified", "max_verified_ctx": 32768},
        }]
    }
    data = build_v3_payload(master_payload)
    model = data["models"][0]
    labels = {badge["label"] for badge in model["badges"]}
    assert "capability-limited" in labels
    assert "measured capability failure" in labels
    assert "64k:not_verified" in labels
    assert data["summary"]["capability_limited_models"] == 1


def test_freeze_copies_rankings_v3_artifacts(tmp_path):
    repo = tmp_path / "repo"
    (repo / "llm_modelbench").mkdir(parents=True)
    (repo / "llm_modelbench" / "x.py").write_text("x = 1\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_x.py").write_text("def test_x(): assert True\n")
    rankings_dir = repo / "rankings"
    rankings_dir.mkdir()
    (rankings_dir / "master_summary.json").write_text(json.dumps([{
        "display_name": "m", "digest": "d", "quality_status": "complete",
    }]))
    (rankings_dir / "master_raw.jsonl").write_text("{}\n")
    (rankings_dir / "master_report_data.json").write_text("{}")
    (rankings_dir / "master_report.html").write_text("html")
    (rankings_dir / "master_report_v3_data.json").write_text("{}")
    (rankings_dir / "master_report_v3.html").write_text("v3")

    result = create_freeze(repo, repo / "runs", rankings_dir, repo / "snapshots" / "rv3")
    copied = {Path(path).name for path in result["copied_rankings"]}
    assert "master_report_v3_data.json" in copied
    assert "master_report_v3.html" in copied
