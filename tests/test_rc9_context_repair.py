import json
from pathlib import Path
from types import SimpleNamespace


from llm_modelbench import rankings, repair, watch
from llm_modelbench.config import Config
from llm_modelbench.ollama import MockClient
from llm_modelbench.runner import (
    _measured_memory_estimate,
    _needle_environment_skip,
    _needle_kv_estimate,
    _task_hash,
)
from llm_modelbench.tasks import TASKS


def _task(task_id):
    return next(t for t in TASKS if t.id == task_id)


def test_dynamic_offload_without_ram_telemetry_rejects_vram_slope_for_skip():
    measured = _measured_memory_estimate([
        {"num_ctx": 16172, "vram_peak_mb": 13754, "offload_fraction": 0.0},
        {"num_ctx": 31971, "vram_peak_mb": 15278, "offload_fraction": 0.364},
    ])
    assert measured["valid_for_skip"] is False
    assert measured["dynamic_offload_observed"] is True

    cfg = Config(vram_budget_gb=18.18)
    estimate = _needle_kv_estimate(MockClient(), cfg, "qwen2.5-coder:14b", 65093, measured)
    assert estimate["kv_estimate_valid_for_skip"] is False
    assert estimate["kv_estimate_confidence"] == "invalid_for_skip_dynamic_offload"
    assert _needle_environment_skip(estimate, 65093) is None


def test_system_ram_slope_is_diagnostic_only_when_offload_changes():
    measured = _measured_memory_estimate([
        {"num_ctx": 16172, "vram_peak_mb": 13754, "ram_delta_peak_mb": 500, "offload_fraction": 0.0},
        {"num_ctx": 31971, "vram_peak_mb": 15278, "ram_delta_peak_mb": 2200, "offload_fraction": 0.364},
    ])
    assert measured["valid_for_skip"] is False
    assert measured["method"] == "diagnostic_process_or_system_resident_slope"
    assert measured["bytes_per_token"] > 0
    assert measured["dynamic_offload_observed"] is True
    assert measured["host_memory_signal"] == "system_ram_delta"


def test_process_pss_slope_is_diagnostic_only_when_offload_changes():
    measured = _measured_memory_estimate([
        {"num_ctx": 16172, "vram_peak_mb": 13754, "ollama_pss_delta_peak_mb": 500, "offload_fraction": 0.0},
        {"num_ctx": 31971, "vram_peak_mb": 15278, "ollama_pss_delta_peak_mb": 2200, "offload_fraction": 0.364},
    ])
    assert measured["valid_for_skip"] is False
    assert measured["method"] == "diagnostic_process_or_system_resident_slope"
    assert measured["host_memory_signal"] == "ollama_process_pss_delta"
    assert measured["bytes_per_token"] > 0


def test_apply_plan_writes_repair_link_before_runner_starts(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    source = runs / "fleet"
    source.mkdir(parents=True)
    (source / "capability_report.json").write_text("{}")

    task = _task("py_anagram")
    action = repair.RepairAction(
        action_id="action123", kind="run_missing_task", model="qwen2.5-coder:14b",
        model_digest="digest", source_run_id="fleet", tasks=[task.id],
        reason="missing", automatic=True, overrides={}, source_row_hashes={}, details={},
    )
    plan = repair.RepairPlan(
        schema_version=1, repair_policy_version=repair.POLICY_VERSION,
        plan_id="phaseplan", created_at="now", runs_dir=str(runs),
        selected_runs=["fleet"], actions=[action], observations=[], counts={}, options={},
    )
    seen = {}

    def fake_run(client, cfg, *, out_dir, **kwargs):
        link = Path(out_dir) / "repair_link.json"
        seen["link_existed_during_run"] = link.exists()
        seen["link"] = json.loads(link.read_text())
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "raw_results.jsonl").write_text(json.dumps({
            "model": action.model, "task": task.id, "category": task.category,
            "family": task.family, "task_hash": _task_hash(task), "score": 100.0,
            "reason": "ok", "timestamp": "2026-07-16T00:00:00Z",
        }) + "\n")

    monkeypatch.setattr("llm_modelbench.runner.run", fake_run)
    monkeypatch.setattr("llm_modelbench.report.build", lambda *a, **k: None)
    result = repair.apply_plan(
        MockClient(), Config(vram_budget_gb=12), plan,
        parent_repair_plan_id="parent123", parent_repair_phase="q8_0",
    )
    assert result["child_runs"]
    assert seen["link_existed_during_run"] is True
    assert seen["link"]["repair_plan_id"] == "parent123"
    assert seen["link"]["repair_phase"] == "q8_0"


def test_watch_promotes_child_to_repair_renderer_even_with_default_full_layout(tmp_path, capsys):
    runs = tmp_path / "runs"
    child = runs / "repair_child"
    child.mkdir(parents=True)
    (child / "status.json").write_text(json.dumps({
        "run_id": child.name, "models_done": 0, "models_total": 1,
    }))
    (child / "repair_link.json").write_text(json.dumps({"repair_plan_id": "abc"}))
    (runs / "repair_status_abc.json").write_text(json.dumps({
        "status_type": "repair", "plan_id": "abc", "phase": "running_q8_action",
        "actions_total": 1, "actions_completed": 0,
    }))
    assert watch.watch(child, layout="full", refresh=0.01, clear=False, once=True) == 0
    output = capsys.readouterr().out
    assert "LLM MODELBENCH REPAIR abc" in output
    assert "CURRENT MODEL" not in output


def test_q8_flash_attention_incompatibility_is_persisted_and_q4_is_not_run(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    source = runs / "fleet"
    source.mkdir(parents=True)
    action = repair.RepairAction(
        action_id="needle123", kind="retry_needle_guarded", model="deepseek-coder-v2:16b",
        model_digest="digest", source_run_id="fleet", tasks=["needle"],
        reason="guarded", automatic=True, overrides={"vram_budget_gb": 18.18},
        source_row_hashes={}, details={},
    )
    plan = repair.RepairPlan(
        schema_version=1, repair_policy_version=repair.POLICY_VERSION,
        plan_id="parent", created_at="now", runs_dir=str(runs), selected_runs=["fleet"],
        actions=[action], observations=[], counts={}, options={},
    )
    phase_calls = []

    def fake_apply(client, cfg, phase_plan, **kwargs):
        kv = phase_plan.options.get("kv_type")
        phase_calls.append(kv)
        if kv == "current":
            return {
                "outcome": "PARTIAL", "completed": 1, "recovered": 0, "unresolved": 1,
                "errors": 0, "child_runs": [],
                "actions": [{"action_id": phase_plan.actions[0].action_id, "status": "unresolved", "attempts": []}],
            }
        assert kv == "q8_0", "q4 must not run after definitive Flash Attention incompatibility"
        child = runs / "q8_child"
        child.mkdir(exist_ok=True)
        error = "quantized V cache was requested, but this requires Flash Attention"
        (child / "raw_results.jsonl").write_text(json.dumps({
            "task": "needle", "needle_attempted": [
                {"size": 4000, "harness_error_detail": error},
                {"size": 16000, "http_error_body": error},
            ],
        }) + "\n")
        return {
            "outcome": "PARTIAL", "completed": 1, "recovered": 0, "unresolved": 1,
            "errors": 0, "child_runs": ["q8_child"],
            "actions": [{"action_id": phase_plan.actions[0].action_id, "status": "unresolved",
                         "attempts": [{"child_run_id": "q8_child"}]}],
        }

    class Controller:
        unit = "ollama-gpu0.service"
        auto_confirm = True
        events = []
        mutation_started = False
        def require_supervised_tty(self): pass
        def confirm(self, *a, **k): pass
        def authorise_sudo(self): pass
        def set_kv_type(self, kv_type, *, phase):
            return SimpleNamespace(kv_type=kv_type, observed_kv_type=kv_type, verified=True)
        def restore(self, *a, **k): pass

    monkeypatch.setattr(repair, "apply_plan", fake_apply)
    result = repair.apply_plan_with_managed_kv_cascade(
        MockClient(), Config(vram_budget_gb=18.18), plan, Controller(), rankings_dir=None,
    )
    assert phase_calls == ["current", "q8_0"]
    assert result["outcome"] == "PARTIAL"
    compat = json.loads((source / "kv_compatibility.json").read_text())
    entry = compat[action.model]
    assert entry["avoid_quantized_kv"] is True
    assert entry["preferred_kv_type"] == "current"
    assert "kv_quantization_requires_flash_attention" in entry["kv_modes"]["q8_0"]["error_kinds"]


def test_missing_fim_task_is_functionally_gated_before_scored_run(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    run = runs / "fleet"
    run.mkdir(parents=True)
    model = "hf.co/atahmih/InternVL3-8B-Q4_K_M-GGUF:latest"
    task = _task("py_anagram")
    (run / "raw_results.jsonl").write_text(json.dumps({
        "model": model, "task": task.id, "category": task.category, "family": task.family,
        "task_hash": _task_hash(task), "score": 100.0, "reason": "ok",
        "timestamp": "2026-07-16T00:00:00Z",
    }) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": "full"}))
    (run / "filters.json").write_text(json.dumps({"level": "full"}))
    (run / "model_identities.json").write_text(json.dumps({model: {"digest": "d1"}}))
    (run / "capability_report.json").write_text(json.dumps({model: {
        "declared_capabilities": ["completion", "insert"],
        "supported_families": ["insert"],
    }}))
    monkeypatch.setattr(repair, "detect_gpu", lambda: SimpleNamespace(total_vram_gb=16.0))
    plan = repair.build_plan(runs, run_id="fleet", include_missing=True)
    fim = [a for a in plan.actions if "fim_suffix_assertion" in a.tasks]
    assert len(fim) == 1
    assert fim[0].kind == "capability_gate"
    assert fim[0].family == "insert"


def test_recovery_exhaustion_is_adopted_as_zero_quality_terminal_evidence(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    source = runs / "fleet"
    child = runs / "repair_child"
    source.mkdir(parents=True)
    child.mkdir(parents=True)
    task = _task("py_anagram")
    model = "reasoner"
    (child / "raw_results.jsonl").write_text(json.dumps({
        "model": model, "task": task.id, "score": 0.0,
        "error_kind": "thinking_only", "think_ineffective": True,
    }) + "\n")
    record = {
        "status": "unresolved",
        "action": {"kind": "retry_generation", "model": model, "tasks": [task.id],
                   "details": {"attempt_limit": 1}},
        "attempts": [{"child_run_id": "repair_child"}],
    }
    (source / "repair_results.jsonl").write_text(json.dumps(record) + "\n")
    row = {
        "model": model, "model_digest_resolved": "digest", "run_id": "fleet",
        "task": task.id, "category": task.category, "family": "text",
        "task_hash": _task_hash(task), "score": 0.0, "error_kind": "thinking_only",
        "reason": "hidden reasoning only", "level": "full", "timestamp": "2026-07-16T00:00:00Z",
        "capability_families": ["text"], "class": "reasoning",
    }
    monkeypatch.setattr(rankings, "_required_quality_tasks", lambda families: [task.id])
    summary = rankings.build_summary([row], [row], runs_dir=runs)["digest"]
    assert summary["quality_status"] == "complete"
    assert summary["overall_mean_score"] == 0.0
    assert summary["recovery_exhausted_tasks"] == [task.id]
    assert summary["recovery_limited"] is True


def test_watch_merges_live_child_status_into_repair_view(tmp_path):
    runs = tmp_path / "runs"
    child = runs / "repair_child"
    child.mkdir(parents=True)
    (child / "repair_link.json").write_text(json.dumps({"repair_plan_id": "abc"}))
    (child / "status.json").write_text(json.dumps({
        "run_id": child.name,
        "current": {
            "model": "deepseek-coder-v2:16b", "task": "needle",
            "state": "task_running", "task_index": 1, "tasks_total": 1,
            "context_length": 65093,
        },
        "last_result": {"tps": 8.5, "prompt_tps": 120.0, "num_ctx": 31971},
    }))
    (runs / "repair_status_abc.json").write_text(json.dumps({
        "status_type": "repair", "plan_id": "abc", "phase": "running_current_kv_action",
        "actions_total": 1, "actions_completed": 0, "current_child_run": child.name,
    }))
    status = watch._load_repair_status_for_run(child)
    assert status["child_status"]["current"]["state"] == "task_running"
    rendered = watch.render_repair(status, {})
    assert "deepseek-coder-v2:16b" in rendered
    assert "needle" in rendered
    assert "task_running" in rendered
    assert "65093" in rendered
    assert "prefill=120.0" in rendered


def test_current_kv_recovery_avoids_all_service_mutation(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    (runs / "fleet").mkdir(parents=True)
    action = repair.RepairAction(
        action_id="needle123", kind="retry_needle_guarded", model="model",
        model_digest="digest", source_run_id="fleet", tasks=["needle"],
        reason="guarded", automatic=True,
    )
    plan = repair.RepairPlan(1, repair.POLICY_VERSION, "parent", "now", str(runs),
                             ["fleet"], [action], [], {}, {})
    phases = []

    def fake_apply(client, cfg, phase_plan, **kwargs):
        phases.append(phase_plan.options.get("kv_type"))
        return {
            "outcome": "COMPLETE", "completed": 1, "recovered": 1,
            "unresolved": 0, "errors": 0, "child_runs": [],
            "actions": [{"action_id": phase_plan.actions[0].action_id, "status": "recovered"}],
        }

    factory_calls = []
    def controller_factory():
        factory_calls.append(True)
        raise AssertionError("no sudo/service discovery expected when current KV recovers")

    monkeypatch.setattr(repair, "apply_plan", fake_apply)
    result = repair.apply_plan_with_managed_kv_cascade(
        MockClient(), Config(), plan, None, controller_factory=controller_factory, auto_confirm=True,
    )
    assert phases == ["current"]
    assert factory_calls == []
    assert result["outcome"] == "COMPLETE"
    assert result["restored_original_service_state"] is False
    compat = json.loads((runs / "fleet" / "kv_compatibility.json").read_text())
    assert compat["model"]["current_kv_supported"] is True


def test_inconclusive_capability_probe_does_not_persist_unavailable(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    source = runs / "fleet"
    source.mkdir(parents=True)
    (source / "capability_report.json").write_text("{}")

    action = repair.RepairAction(
        action_id="gate123", kind="capability_gate", model="model-under-test",
        model_digest="digest", source_run_id="fleet", tasks=["fim_suffix_assertion"],
        family="insert", reason="probe required", automatic=True,
        overrides={}, source_row_hashes={}, details={},
    )
    plan = repair.RepairPlan(
        schema_version=1, repair_policy_version=repair.POLICY_VERSION,
        plan_id="phaseplan", created_at="now", runs_dir=str(runs),
        selected_runs=["fleet"], actions=[action], observations=[], counts={}, options={},
    )

    def transient_profile(client, model, *, functional=False, **kwargs):
        if not functional:
            return {
                "model": model,
                "supported_families": [],
                "probe_states": {},
                "probes": {},
            }
        return {
            "model": model,
            "supported_families": [],
            "probe_states": {"insert": "transient_failure"},
            "probes": {
                "insert": {
                    "responded": False,
                    "ok": False,
                    "state": "transient_failure",
                    "error": "temporary connection reset",
                }
            },
        }

    monkeypatch.setattr(repair, "interrogate_model", transient_profile)
    result = repair.apply_plan(MockClient(), Config(), plan)

    assert result["actions"][0]["status"] == "capability_probe_inconclusive"
    assert result["actions"][0]["gate"]["probe_state"] == "transient_failure"
    assert not (source / "capability_repair.json").exists()
