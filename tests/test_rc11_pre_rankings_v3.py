import io
import json
from pathlib import Path

from llm_modelbench.cli import build_parser
from llm_modelbench.context_profile import validate_context_telemetry
from llm_modelbench.freeze import create_freeze
from llm_modelbench.model_cards import generate_model_cards
from llm_modelbench import rankings, watch
from llm_modelbench.runner import _task_hash
from llm_modelbench.tasks import TASKS
from llm_modelbench.watch_fixtures import replay_repair_watch


def _task(task_id):
    return next(task for task in TASKS if task.id == task_id)


def test_exact_repair_watch_command_parses():
    args = build_parser().parse_args([
        "simulate", "repair-watch", "--scenario", "capability-repair", "--speed", "1.0"
    ])
    assert args.cmd == "simulate"
    assert args.simulate_cmd == "repair-watch"
    assert args.scenario == "capability-repair"
    assert args.speed == 1.0


def test_repair_watch_fixture_writes_discoverable_parent_and_linked_child(tmp_path):
    stream = io.StringIO()
    result = replay_repair_watch(
        tmp_path, scenario="capability-repair", speed=0, run_id="fixture",
        render=True, screen="scroll", stream=stream,
    )
    campaign = Path(result["campaign_dir"])
    child = Path(result["child_dir"])
    assert (campaign / "status.json").exists()
    assert (child / "repair_link.json").exists()
    assert (child / "status.json").exists()
    status = watch._load_repair_status_for_run(child)
    assert status["plan_id"] == result["plan_id"]
    assert status["phase"] == "complete"
    text = stream.getvalue()
    assert "LLM MODELBENCH REPAIR" in text
    assert "FUNCTIONAL CAPABILITY PROBE" in text
    assert "SCORED REPAIR TASK" in text
    assert "models 0/1" not in text.lower()
    assert "CTX ?" not in text


def test_needle_fixture_exposes_context_speed_memory_and_offload(tmp_path):
    stream = io.StringIO()
    replay_repair_watch(
        tmp_path, scenario="needle-current", speed=0, run_id="needle_fixture",
        render=True, screen="scroll", stream=stream,
    )
    text = stream.getvalue()
    assert "target=65536" in text
    assert "decode=11.8 tok/s" in text
    assert "offload=52.3%" in text
    assert "current/default service configuration" in text


def test_context_telemetry_validation_accepts_complete_64k_probe(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    row = {
        "model": "model", "task": "needle", "score": 100.0,
        "needle_coverage": 1.0, "max_verified_ctx": 64773,
        "needle_target_status": "ready",
        "needle_attempted": [{
            "size": 65536, "num_ctx": 65093, "prompt_tokens_actual": 64773,
            "found": True, "elapsed_seconds": 123.0, "request_elapsed_seconds": 122.0,
            "telemetry_samples": 400, "vram_peak_mb": 15526.0,
            "offload_fraction": 0.523, "ollama_pss_peak_mb": 16400.0,
            "ollama_pss_delta_peak_mb": 7400.0, "tps": 11.8, "prompt_tps": 241.7,
            "ttft_ms": 2400.0, "gpu_util_mean_pct": 91.0,
            "gpu_util_peak_pct": 100.0, "power_mean_w": 120.0,
            "power_peak_w": 132.0, "temp_peak_c": 68.0,
            "cpu_util_mean_pct": 45.0, "cpu_util_peak_pct": 82.0,
            "ram_available_min_mb": 12000.0, "swap_delta_peak_mb": 0.0,
            "model_host_bytes": 7600000000, "model_vram_bytes": 8500000000,
            "needle_response_exact": True, "needle_response_suspect": False,
        }],
    }
    (run / "raw_results.jsonl").write_text(json.dumps(row) + "\n")
    result = validate_context_telemetry(run, target_ctx=64000)
    assert result["passed"] is True
    assert result["critical_missing"] == []
    assert result["operating_status"] == "ready"


def test_diagnostic_context_profile_supplies_operating_data_without_replacing_quality_row(tmp_path):
    task = _task("needle")
    base = {
        "model": "model", "model_digest_resolved": "digest", "task": task.id,
        "task_hash": _task_hash(task), "category": task.category, "family": task.family,
        "level": "full", "capability_families": ["text"], "score": 100.0,
        "needle_coverage": 1.0, "max_verified_ctx": 32000,
        "needle_attempted": [{"size": 32000, "num_ctx": 32000, "found": True}],
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    profile = {
        **base,
        "run_id": "context_profile",
        "context_profile_run": True,
        "max_verified_ctx": 64773,
        "timestamp": "2026-01-02T00:00:00+00:00",
        "needle_attempted": [{
            "size": 65536, "num_ctx": 65093, "prompt_tokens_actual": 64773,
            "found": True, "elapsed_seconds": 120.0, "request_elapsed_seconds": 119.0,
            "tps": 12.0, "prompt_tps": 250.0, "vram_peak_mb": 15500.0,
            "ollama_pss_peak_mb": 16000.0, "offload_fraction": 0.5,
        }],
        "needle_target_ctx": 64000,
        "needle_target_status": "ready",
        "needle_target_tps": 12.0,
    }
    selected = {**base, "run_id": "quality"}
    summary = rankings.build_summary([selected], [selected, profile], runs_dir=tmp_path)["digest"]
    assert summary["overall_mean_score"] == 100.0
    assert summary["long_context_profile"]["run_id"] == "context_profile"
    assert summary["long_context_profile"]["max_verified_ctx"] == 64773
    assert summary["long_context_profile"]["target_status"] == "ready"


def test_model_cards_generate_standalone_json_and_markdown(tmp_path):
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    (rankings_dir / "master_summary.json").write_text(json.dumps([{
        "display_name": "model:latest", "digest": "d1", "quality_status": "complete",
        "overall_mean_score": 88.0, "coverage_ratio": 1.0, "completion_rate": 1.0,
        "families": ["text"], "class": "general", "quality_status_reasons": ["complete"],
        "capability_limited": False, "capability_measured_failure": False,
        "recovery_limited": False, "long_context_profile": {
            "target_ctx": 64000, "target_status": "ready", "max_verified_ctx": 64773,
            "score": 100.0, "coverage": 1.0, "target_tps": 12.0,
            "target_prompt_tps": 220.0, "depths": [],
        },
    }]))
    out = tmp_path / "cards"
    result = generate_model_cards(rankings_dir, out)
    assert result["models"] == 1
    assert (out / "model_latest.json").exists()
    assert "64k `ready`" in (out / "README.md").read_text()


def test_freeze_captures_complete_rankings_and_contract_hashes(tmp_path):
    repo = tmp_path / "repo"
    (repo / "llm_modelbench").mkdir(parents=True)
    (repo / "llm_modelbench" / "x.py").write_text("x = 1\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_x.py").write_text("def test_x(): assert True\n")
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    (rankings_dir / "master_summary.json").write_text(json.dumps([
        {"display_name": "a", "digest": "d1", "quality_status": "complete"},
        {"display_name": "b", "digest": "d2", "quality_status": "complete"},
    ]))
    (rankings_dir / "master_raw.jsonl").write_text("{}\n{}\n")
    (rankings_dir / "master_report_data.json").write_text("{}")
    result = create_freeze(repo, tmp_path / "runs", rankings_dir, tmp_path / "freeze")
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["rankings"]["models"] == 2
    assert manifest["rankings"]["status_counts"] == {"complete": 2}
    assert "needle" in manifest["task_contract_hashes"]
    assert Path(result["checksums_path"]).exists()


def test_follow_queue_treats_terminal_repair_parent_as_authoritative_for_stale_child(tmp_path):
    replay_repair_watch(
        tmp_path, scenario="kv-cascade", speed=0, run_id="fixture",
        render=False,
    )
    candidates = watch.discover_runs(tmp_path)
    by_id = {item["run_id"]: item for item in candidates}
    assert by_id["fixture"]["in_progress"] is False
    assert "fixture_child" not in by_id  # linked children are collapsed into the parent campaign
