import json
from pathlib import Path

from llm_modelbench import rankings
from llm_modelbench.freeze import create_freeze
from llm_modelbench.rankings_v31 import SCHEMA_VERSION, build_v31_payload


def _write_run(runs_dir: Path, run_id: str, rows):
    run = runs_dir / run_id
    run.mkdir(parents=True)
    (run / "raw_results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": "full"}))
    (run / "model_identities.json").write_text(json.dumps({"coder:latest": {"digest": "digest-coder"}}))
    return run


def test_rankings_v31_split_site_and_model_detail(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs, "full", [{
        "model": "coder:latest",
        "task": "needle",
        "category": "long_context",
        "family": "text",
        "score": 100.0,
        "reason": "4k:ok 64k:ok",
        "needle_coverage": 1.0,
        "max_verified_ctx": 64773,
        "needle_target_ctx": 64000,
        "needle_target_status": "ready",
        "needle_target_tps": 12.0,
        "task_hash": rankings._CURRENT_HASHES["needle"],
        "timestamp": "2026-07-17T00:00:00Z",
        "benchmark_version": "1.0.0rc14",
        "prompt": "Find the needle.",
    }])
    out = tmp_path / "rankings"
    result = rankings.write_rankings(runs, out, html_template="<html>__MASTER_SUMMARY_JSON__</html>")

    assert result["v31_schema_version"] == SCHEMA_VERSION
    assert (out / "master_report_v3_1_data.json").exists()
    assert (out / "master_report_v3_1.html").exists()
    assert (out / "v3_1" / "index.html").exists()
    assert (out / "v3_1" / "compare.html").exists()
    assert (out / "v3_1" / "top5.html").exists()
    assert (out / "v3_1" / "help.html").exists()
    assert (out / "v3_1" / "assets" / "report.css").exists()
    model_pages = list((out / "v3_1" / "models").glob("*.html"))
    assert len(model_pages) == 1
    index = (out / "v3_1" / "index.html").read_text()
    model = model_pages[0].read_text()
    assert "Decision dashboard" in index
    assert "{'complete':" not in index
    assert "Operational fit score" in index
    assert "../master_report_v3_1_data.json" in index
    assert "The Best 5" in (out / "v3_1" / "top5.html").read_text()
    assert "Help / About" in (out / "v3_1" / "help.html").read_text()
    js = (out / "v3_1" / "assets" / "report.js").read_text()
    assert "Open model page" in js
    assert "modelName" in js
    assert "Operational fit" in js
    assert "Current selected task evidence" in model
    assert "Long-context needle retrieval" in model
    assert "4k:ok 64k:ok" in model


def test_v31_payload_has_pages_and_human_labels():
    payload = build_v31_payload({"models": [{
        "display_name": "InternVL fixture",
        "digest": "d1",
        "class": "vision",
        "families": ["vision", "text"],
        "quality_status": "complete",
        "overall_mean_score": 0.0,
        "coverage_ratio": 1.0,
        "capability_limited": True,
        "capability_measured_failure": True,
        "capability_measured_failure_tasks": ["fim_suffix_assertion"],
        "capability_unavailable_tasks": ["ocr_invoice"],
        "categories": {"coding": {"score": 0.0, "coverage": 1.0, "tasks": [{"task": "fim_suffix_assertion", "score": 0.0, "reason": "empty"}]}},
    }]})
    model = payload["models"][0]
    assert payload["schema_version"] == SCHEMA_VERSION
    assert model["page"].startswith("models/")
    assert any(group["title"] == "Capability unavailable on this installed build" for group in model["evidence_groups"])
    assert any(group["title"] == "Measured capability-quality failure" for group in model["evidence_groups"])


def test_freeze_copies_rankings_v31_split_site(tmp_path):
    repo = tmp_path / "repo"
    (repo / "llm_modelbench").mkdir(parents=True)
    (repo / "llm_modelbench" / "x.py").write_text("x = 1\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_x.py").write_text("def test_x(): assert True\n")
    rankings_dir = repo / "rankings"
    rankings_dir.mkdir()
    (rankings_dir / "master_summary.json").write_text(json.dumps([{"display_name": "m", "digest": "d", "quality_status": "complete"}]))
    (rankings_dir / "master_raw.jsonl").write_text("{}\n")
    for name in ["master_report_data.json", "master_report.html", "master_report_v3_data.json", "master_report_v3.html", "master_report_v3_1_data.json", "master_report_v3_1.html"]:
        (rankings_dir / name).write_text("{}" if name.endswith(".json") else "html")
    (rankings_dir / "v3_1" / "models").mkdir(parents=True)
    (rankings_dir / "v3_1" / "assets").mkdir(parents=True)
    (rankings_dir / "v3_1" / "index.html").write_text("index")
    (rankings_dir / "v3_1" / "top5.html").write_text("top5")
    (rankings_dir / "v3_1" / "help.html").write_text("help")
    (rankings_dir / "v3_1" / "assets" / "report.css").write_text("css")
    (rankings_dir / "v3_1" / "models" / "m.html").write_text("model")

    result = create_freeze(repo, repo / "runs", rankings_dir, repo / "snapshots" / "rv31")
    copied = set(result["copied_rankings"])
    assert any(path.endswith("master_report_v3_1_data.json") for path in copied)
    assert any(path.endswith("v3_1/index.html") for path in copied)
    assert any(path.endswith("v3_1/top5.html") for path in copied)
    assert any(path.endswith("v3_1/help.html") for path in copied)
    assert any(path.endswith("v3_1/models/m.html") for path in copied)


def test_v31_compare_uses_display_name_and_class_filter_text():
    payload = build_v31_payload({"models": [{
        "display_name": "gemma3:12b",
        "digest": "digest-gemma",
        "class": "vision",
        "families": ["vision", "text"],
        "quality_status": "complete",
        "overall_mean_score": 95.0,
        "coverage_ratio": 1.0,
        "tok_s": 44.8,
        "size_gb": 8.15,
        "categories": {"ocr": {"score": 100.0, "coverage": 1.0, "tasks": []}},
    }, {
        "display_name": "llama3.1:8b",
        "digest": "digest-llama",
        "class": "general",
        "families": ["text"],
        "quality_status": "complete",
        "overall_mean_score": 93.0,
        "coverage_ratio": 1.0,
        "tok_s": 74.5,
        "size_gb": 4.92,
        "categories": {"reasoning": {"score": 90.0, "coverage": 1.0, "tasks": []}},
    }]})
    assert payload["models"][0]["display_name"] == "gemma3:12b"
    from llm_modelbench.rankings_v31 import _compare_html, _js
    compare = _compare_html(payload)
    js = _js()
    assert "modelName" in js
    assert "Showing ${cls}-class models" in js
    assert "Operational fit score" in js
    assert "gemma3:12b" in compare
