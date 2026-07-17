import io
import json
import subprocess

import pytest

from llm_modelbench import watch
from llm_modelbench.config import Config
from llm_modelbench.context_profile import run_context_profile, validate_context_telemetry
from llm_modelbench.freeze import create_freeze, verify_freeze
from llm_modelbench.inline_ui import InlineUI
from llm_modelbench.model_cards import generate_model_cards
from llm_modelbench.runner import _run_once
from llm_modelbench.tasks import TASKS
from llm_modelbench.watch_fixtures import replay_repair_watch


def _task(task_id):
    return next(task for task in TASKS if task.id == task_id)


class _LargeContextClient:
    def __init__(self):
        self.last_ctx = 0

    def context_length(self, model):
        return 163840

    def model_size_bytes(self, model):
        return 30 * 1024**3  # deliberately over any small pre-flight budget

    def model_info(self, model):
        return {
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
            "llama.attention.key_length": 128,
            "llama.attention.value_length": 128,
        }

    def chat(self, model, prompt, **kwargs):
        self.last_ctx = int(kwargs.get("num_ctx") or 0)
        prompt_tokens = max(1, int(len(prompt) / 6.85))
        return {
            "ok": True,
            "text": "SECRET_CODE_77",
            "prompt_eval_count": prompt_tokens,
            "eval_count": 4,
            "tokens": 4,
            "tps": 20.0,
            "prompt_tps": 200.0,
            "ttft_ms": 100.0,
            "request_elapsed_seconds": 1.0,
            "done_reason": "stop",
            "num_ctx": kwargs.get("num_ctx"),
            "num_predict": kwargs.get("num_predict"),
        }

    def loaded_model_stats(self, model):
        return {
            "size_bytes": 30 * 1024**3,
            "size_vram_bytes": 15 * 1024**3,
            "size_host_bytes": 15 * 1024**3,
            "offload_fraction": 0.5,
            "context_length": self.last_ctx,
        }

    def offload_fraction(self, model, exact=True):
        return 0.5


class _Telemetry:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        return {
            "elapsed_seconds": 1.0,
            "telemetry_samples": 4,
            "vram_peak_mb": 15000.0,
            "ram_peak_mb": 12000.0,
            "ram_delta_peak_mb": 1000.0,
            "ram_available_min_mb": 12000.0,
            "ollama_pss_peak_mb": 10000.0,
            "ollama_pss_delta_peak_mb": 1000.0,
            "swap_delta_peak_mb": 0.0,
        }


def test_controlled_profile_advisory_mode_attempts_over_budget_tiers(monkeypatch):
    monkeypatch.setattr("llm_modelbench.runner.ProbeTelemetry", _Telemetry)
    monkeypatch.setattr("llm_modelbench.runner.host_memory_snapshot", lambda: {"ram_available_mb": 12000.0})
    cfg = Config(vram_budget_gb=1.0)
    cfg.needle_preflight_mode = "advisory"
    cfg.needle_max_ctx = 70000
    cfg.long_context_target_ctx = 64000
    result = _run_once(_LargeContextClient(), cfg, "large:model", _task("needle"))
    assert result["needle_coverage"] == 1.0
    assert len(result["needle_attempted"]) == 4
    assert result["max_verified_ctx"] >= 64000
    assert result["needle_attempted"][0].get("preflight_budget_advisory")
    assert result["needle_attempted"][1].get("preflight_budget_advisory")
    assert all(probe.get("found") for probe in result["needle_attempted"])


def test_repair_fixture_restoring_phase_is_not_zero_percent(tmp_path):
    stream = io.StringIO()
    replay_repair_watch(
        tmp_path, scenario="kv-cascade", speed=0, run_id="fixture",
        render=True, screen="scroll", stream=stream,
    )
    text = stream.getvalue()
    restoring = text.split("RESTORING ORIGINAL SERVICE STATE", 1)[1]
    assert "Actions 1/1" in restoring
    assert "lifecycle=8/9" in restoring
    assert "88.9%" in restoring
    assert "last verified KV=q8_0" in restoring


def test_queue_discovery_collapses_linked_repair_child(tmp_path):
    replay_repair_watch(tmp_path, scenario="kv-cascade", speed=0, run_id="fixture", render=False)
    ids = [item["run_id"] for item in watch.discover_runs(tmp_path)]
    assert ids.count("fixture") == 1
    assert "fixture_child" not in ids


def test_context_profile_renderer_is_dedicated_and_evidence_first():
    status = {
        "status_type": "context_profile",
        "run_id": "profile",
        "profile_phase": "needle_profile",
        "elapsed_seconds": 12.0,
        "hardware_config": {"context_profile_target_ctx": 64000},
        "current": {
            "model": "deepseek-coder-v2:16b",
            "task": "needle",
            "state": "task_running",
            "probe_index": 3,
            "probe_total": 4,
            "probe_size": 32000,
            "probe_num_ctx": 32344,
            "probe_state": "running",
            "probe_history": [{
                "probe_size": 16000, "probe_num_ctx": 16172,
                "probe_state": "finished", "prompt_tps": 2200.0,
                "tps": 20.0, "elapsed_seconds": 14.0,
                "vram_peak_mb": 13754.0, "offload_fraction": 0.0,
            }],
        },
    }
    hw = {"vram_used_mb": 14000.0, "vram_total_mb": 16311.0, "gpu_util_pct": 100.0,
          "gpu_temp_c": 55.0, "gpu_power_w": 150.0, "ram_used_mb": 5000.0,
          "ram_total_mb": 30981.0, "swap_used_mb": 0.0}
    text = watch.render_context_profile(status, hw)
    assert "CONTEXT PROFILE" in text
    assert "COMPLETED TIERS" in text
    assert "32000" in text
    assert "Models" not in text
    assert "excluded 59" not in text


def test_validation_can_require_behavior_probe(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    row = {
        "model": "model", "task": "needle", "score": 100.0,
        "needle_coverage": 1.0, "max_verified_ctx": 64773,
        "needle_target_status": "ready",
        "needle_attempted": [{
            "size": 65536, "num_ctx": 65093, "prompt_tokens_actual": 64773,
            "found": True, "elapsed_seconds": 100.0, "request_elapsed_seconds": 99.0,
            "telemetry_samples": 20, "vram_peak_mb": 15500.0, "offload_fraction": 0.5,
            "ram_peak_mb": 10000.0, "ram_delta_peak_mb": 2000.0,
            "tps": 12.0, "prompt_tps": 200.0,
        }],
    }
    (run / "raw_results.jsonl").write_text(json.dumps(row) + "\n")
    missing = validate_context_telemetry(run, target_ctx=64000, require_behavior_probe=True)
    assert missing["passed"] is False
    assert "64k behavior probe" in missing["critical_missing"]
    (run / "context_behavior_probe.json").write_text(json.dumps({
        "model": "model", "operating_status": "ready", "prompt_eval_count": 64500,
        "all_anchors_exact": True, "sequence_ok": True, "tps": 12.0,
    }))
    complete = validate_context_telemetry(run, target_ctx=64000, require_behavior_probe=True)
    assert complete["passed"] is True
    assert complete["operating_status"] == "ready"
    assert complete["agentic_readiness"] == "not_assessed"


def test_model_cards_include_latest_behavior_probe(tmp_path):
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    (rankings_dir / "master_summary.json").write_text(json.dumps([{
        "display_name": "model:latest", "digest": "d1", "quality_status": "complete",
        "overall_mean_score": 90.0, "coverage_ratio": 1.0, "families": ["text"],
        "long_context_profile": {"target_ctx": 64000, "target_status": "ready", "depths": []},
    }]))
    runs = tmp_path / "runs"
    profile = runs / "profile"
    profile.mkdir(parents=True)
    (profile / "context_behavior_probe.json").write_text(json.dumps({
        "model": "model:latest", "validated_at": "2026-07-17T00:00:00+00:00",
        "operating_status": "slow", "prompt_eval_count": 64500, "tps": 7.0,
        "all_anchors_exact": True, "sequence_ok": True,
        "response_repetition_ratio": 0.0, "agentic_readiness": "not_assessed",
    }))
    out = tmp_path / "cards"
    generate_model_cards(rankings_dir, out, runs_dir=runs)
    card = json.loads((out / "model_latest.json").read_text())
    assert card["long_context"]["behavior_probe"]["operating_status"] == "slow"
    assert card["long_context"]["agentic_readiness"] == "not_assessed"
    assert "64k behavior probe" in (out / "model_latest.md").read_text()


def test_freeze_checksums_work_from_repo_root_and_portably(tmp_path):
    repo = tmp_path / "repo"
    (repo / "llm_modelbench").mkdir(parents=True)
    (repo / "llm_modelbench" / "x.py").write_text("x = 1\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_x.py").write_text("def test_x(): assert True\n")
    rankings = repo / "rankings"
    rankings.mkdir()
    (rankings / "master_summary.json").write_text(json.dumps([
        {"display_name": "a", "digest": "d1", "quality_status": "complete"},
    ]))
    (rankings / "master_raw.jsonl").write_text("{}\n")
    (rankings / "master_report_data.json").write_text("{}")
    out = repo / "snapshots" / "freeze"
    create_freeze(repo, repo / "runs", rankings, out)
    assert verify_freeze(out)["passed"] is True
    manifest_text = (out / "SHA256SUMS.txt").read_text()
    assert "snapshots/freeze/README.md" in manifest_text
    completed = subprocess.run(
        ["sha256sum", "-c", "snapshots/freeze/SHA256SUMS.txt"],
        cwd=repo, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    portable = subprocess.run([str(out / "VERIFY.sh")], capture_output=True, text=True)
    assert portable.returncode == 0, portable.stdout + portable.stderr


def test_inline_ui_auto_selects_context_profile_renderer(tmp_path, monkeypatch):
    run = tmp_path / "profile"
    run.mkdir()
    status = {
        "status_type": "context_profile",
        "run_id": "profile",
        "profile_phase": "needle_profile",
        "hardware_config": {"context_profile_target_ctx": 64000},
        "current": {
            "model": "deepseek-coder-v2:16b",
            "task": "needle",
            "state": "task_running",
            "probe_index": 4,
            "probe_total": 4,
            "probe_size": 65536,
            "probe_num_ctx": 65093,
            "probe_state": "running",
        },
    }
    (run / "status.json").write_text(json.dumps(status))
    monkeypatch.setattr(watch, "_load_repair_status_for_run", lambda _run: None)
    monkeypatch.setattr(
        "llm_modelbench.inline_ui.live_snapshot",
        lambda prev: ({"vram_used_mb": 15000.0, "vram_total_mb": 16311.0}, prev),
    )
    ui = InlineUI(run, layout="full", enabled=False)
    text = ui._render_dashboard()
    assert "CONTEXT PROFILE" in text
    assert "65536" in text
    assert "Models" not in text
    assert "excluded 59" not in text


def test_context_profile_refuses_nonempty_run_directory(tmp_path):
    run = tmp_path / "existing"
    run.mkdir()
    (run / "raw_results.jsonl").write_text("{}\n")
    with pytest.raises(FileExistsError, match="use a new --run-id"):
        run_context_profile(
            object(), Config(), model="model:latest", run_dir=run,
            target_ctx=64000, gpu_vram_gb=16.0, behavior_probe=False,
        )
