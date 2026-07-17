import json
from pathlib import Path

from llm_modelbench import sensitivity
from llm_modelbench.report import _rank_cells


def test_sensitivity_plan_auto_promotes_needle_to_full():
    script = sensitivity.plan_commands(
        run_prefix="needle_probe",
        include_regex="m1|m2",
        tasks="needle",
        level="short",
        ctx_values="default",
        num_predict_values="512",
    )
    assert "auto-promoted: short -> full" in script
    assert "--level full" in script
    assert "--needle-max-ctx 40960" in script
    assert "needle_probe_default_np512" in script


def test_sensitivity_report_summarises_score_null_needle_rows(tmp_path):
    run = tmp_path / "needle_default_np512"
    run.mkdir()
    (run / "summary_meta.json").write_text(json.dumps({"ctx_override": None, "num_predict": 512}))
    row = {
        "model": "m",
        "task": "needle",
        "score": None,
        "max_verified_ctx": 31723,
        "needle_coverage": 0.75,
        "needle_skipped": [
            {"size": 65536, "reason": "needle_max_ctx", "needle_max_ctx": 40960},
        ],
    }
    (run / "raw_results.jsonl").write_text(json.dumps(row) + "\n")
    text = sensitivity.report([run])
    assert "Runs read: 1" in text
    assert "no numeric-score rows" in text
    assert "Needle / long-context sensitivity" in text
    assert "31723" in text
    assert "0.75" in text
    assert "65536:needle_max_ctx" in text


def test_sensitivity_report_shows_kv_skip_margin(tmp_path):
    run = tmp_path / "needle_ctx4096_np512"
    run.mkdir()
    row = {
        "model": "qwen",
        "task": "needle",
        "score": None,
        "max_verified_ctx": 15752,
        "needle_coverage": 0.5,
        "needle_skipped": [
            {"size": 32000, "reason": "kv_cache_exceeds_vram_budget", "estimated_total_gb": 14.546, "vram_budget_gb": 14.4},
        ],
    }
    (run / "raw_results.jsonl").write_text(json.dumps(row) + "\n")
    text = sensitivity.report([run])
    assert "32000:kv_cache_exceeds_vram_budget (+0.146GB)" in text


def test_scorecard_rank_cells_blank_for_undercovered_diagnostics():
    lb = [
        {"model": "a", "quality": 100.0},
        {"model": "b", "quality": 100.0},
        {"model": "c", "quality": 20.0},
    ]
    context = {"min_tasks_per_model": 1, "min_report_tasks_per_category": 2, "category_task_counts": {"coding_web": 1}}
    assert _rank_cells(lb, context) == {"a": "", "b": "", "c": ""}


def test_scorecard_rank_cells_share_tie_ranks_when_covered():
    lb = [
        {"model": "a", "quality": 100.0},
        {"model": "b", "quality": 100.0},
        {"model": "c", "quality": 80.0},
    ]
    context = {"min_tasks_per_model": 2, "min_report_tasks_per_category": 2, "category_task_counts": {"coding_web": 2}}
    assert _rank_cells(lb, context) == {"a": "1", "b": "1", "c": "3"}


def test_update_sh_prefers_venv_and_ignores_runs_for_status():
    text = Path("update.sh").read_text()
    assert '.venv/bin/python' in text
    assert ':(exclude)runs/**' in text
    assert 'git clean' not in text
    assert 'rm -rf runs' not in text


def test_update_sh_rejects_untracked_source_contamination():
    text = Path("update.sh").read_text()
    assert "git ls-files --others --exclude-standard" in text
    assert "runs|rankings|rankings-separate|model_cards|snapshots" in text
