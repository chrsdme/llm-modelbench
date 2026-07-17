import json
from pathlib import Path
from types import SimpleNamespace

from llm_modelbench import repair, rankings, watch
from llm_modelbench.config import Config
from llm_modelbench.ollama import MockClient
from llm_modelbench.runner import _task_hash
from llm_modelbench.tasks import TASKS


def _task(task_id):
    return next(task for task in TASKS if task.id == task_id)


def _write_full_run(run: Path, model: str, rows):
    run.mkdir(parents=True)
    (run / "raw_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (run / "summary_meta.json").write_text(json.dumps({"level": "full"}))
    (run / "filters.json").write_text(json.dumps({"level": "full"}))
    (run / "model_identities.json").write_text(json.dumps({model: {"digest": "d1"}}))
    (run / "capability_report.json").write_text(json.dumps({model: {
        "declared_capabilities": ["completion", "insert"],
        "supported_families": ["vision", "text", "insert"],
    }}))


def test_force_does_not_reprobe_terminal_unavailable_vision_lane(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    source = runs / "fleet"
    model = "hf.co/atahmih/InternVL3-8B-Q4_K_M-GGUF:latest"
    fim = _task("fim_suffix_assertion")
    vision = _task("ocr_receipt_total")
    rows = [
        {
            "model": model, "model_digest_resolved": "d1", "run_id": "fleet",
            "task": fim.id, "category": fim.category, "family": fim.family,
            "task_hash": _task_hash(fim), "score": 0.0, "error_kind": "empty_output",
            "reason": "FIM returned no insertion", "level": "full",
        },
        {
            "model": model, "model_digest_resolved": "d1", "run_id": "fleet",
            "task": vision.id, "category": vision.category, "family": vision.family,
            "task_hash": _task_hash(vision), "score": None, "error_kind": "harness_error",
            "reason": "HTTP 400 Bad Request", "level": "full",
        },
    ]
    _write_full_run(source, model, rows)
    (source / "capability_repair.json").write_text(json.dumps({model: {
        "unavailable_families": {"vision": {"reason": "confirmed unavailable"}},
        "history": [],
    }}))
    monkeypatch.setattr(repair, "detect_gpu", lambda: SimpleNamespace(total_vram_gb=16.0))

    plan = repair.build_plan(runs, run_id="fleet", include_missing=False, force=True)
    assert [(action.family, action.tasks) for action in plan.actions] == [
        ("insert", ["fim_suffix_assertion"])
    ]
    assert any(item.get("kind") == "capability_already_unavailable" for item in plan.observations)


def test_capability_profiles_are_functionally_probed_once_per_model(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    source = runs / "fleet"
    source.mkdir(parents=True)
    (source / "capability_report.json").write_text("{}")
    model = "model"
    actions = [
        repair.RepairAction("a1", "capability_gate", model, "d1", "fleet",
                            ["fim_suffix_assertion"], "gate", True, family="insert", overrides={}),
        repair.RepairAction("a2", "capability_gate", model, "d1", "fleet",
                            ["agent_native_tool_call"], "gate", True, family="tools", overrides={}),
    ]
    plan = repair.RepairPlan(1, repair.POLICY_VERSION, "plan", "now", str(runs),
                             ["fleet"], actions, [], {}, {})
    calls = {"functional": 0}

    def fake_interrogate(client, name, functional=False, probe_families=None):
        if functional:
            calls["functional"] += 1
        return {
            "probe_states": {"insert": "confirmed_supported", "tools": "confirmed_supported"},
            "probes": {"insert": {"responded": True}, "tools": {"responded": True}},
            "supported_families": ["insert", "tools", "text"],
        }

    def fake_run(client, cfg, *, out_dir, task_ids, **kwargs):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        rows = []
        for task_id in task_ids:
            task = _task(task_id)
            rows.append({
                "model": model, "model_digest_resolved": "d1", "run_id": Path(out_dir).name,
                "task": task_id, "category": task.category, "family": task.family,
                "task_hash": _task_hash(task), "score": 100.0, "reason": "ok", "level": "full",
            })
        (Path(out_dir) / "raw_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    monkeypatch.setattr(repair, "interrogate_model", fake_interrogate)
    monkeypatch.setattr("llm_modelbench.runner.run", fake_run)
    monkeypatch.setattr("llm_modelbench.report.build", lambda *args, **kwargs: None)
    result = repair.apply_plan(MockClient(), Config(), plan)
    assert result["outcome"] == "COMPLETE"
    assert calls["functional"] == 1


def test_direct_repair_campaign_is_visible_before_child_finishes(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    source = runs / "fleet"
    source.mkdir(parents=True)
    (source / "capability_report.json").write_text("{}")
    task = _task("py_anagram")
    action = repair.RepairAction("a1", "run_missing_task", "model", "d1", "fleet",
                                 [task.id], "missing", True, overrides={})
    plan = repair.RepairPlan(1, repair.POLICY_VERSION, "live123", "now", str(runs),
                             ["fleet"], [action], [], {}, {})
    seen = {}

    def fake_run(client, cfg, *, out_dir, **kwargs):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "status.json").write_text(json.dumps({
            "run_id": out_dir.name, "models_done": 0, "models_total": 1,
            "current": {"model": "model", "task": task.id, "state": "task_running"},
        }))
        campaign = runs / "repair_campaign_live123"
        seen["campaign_exists"] = (campaign / "status.json").exists()
        seen["link_exists"] = (out_dir / "repair_link.json").exists()
        discovered = watch.discover_runs(runs)
        seen["campaign_active"] = any(
            item["run_id"] == "repair_campaign_live123" and item["in_progress"]
            for item in discovered
        )
        status = watch._load_repair_status_for_run(campaign)
        seen["child_merged"] = (status or {}).get("child_status", {}).get("current", {}).get("task")
        (out_dir / "raw_results.jsonl").write_text(json.dumps({
            "model": "model", "model_digest_resolved": "d1", "run_id": out_dir.name,
            "task": task.id, "category": task.category, "family": task.family,
            "task_hash": _task_hash(task), "score": 100.0, "reason": "ok", "level": "full",
        }) + "\n")

    monkeypatch.setattr("llm_modelbench.runner.run", fake_run)
    monkeypatch.setattr("llm_modelbench.report.build", lambda *args, **kwargs: None)
    result = repair.apply_plan_with_live_status(MockClient(), Config(), plan)
    assert result["outcome"] == "COMPLETE"
    assert seen == {
        "campaign_exists": True,
        "link_exists": True,
        "campaign_active": True,
        "child_merged": task.id,
    }
    final_status = json.loads((runs / "repair_campaign_live123" / "status.json").read_text())
    assert final_status["phase"] == "complete"


def test_legacy_rc9_fim_empty_output_is_adopted_as_measured_zero(tmp_path):
    runs = tmp_path / "runs"
    source = runs / "fleet"
    child = runs / "repair_child"
    source.mkdir(parents=True)
    child.mkdir(parents=True)
    model = "fim-model"
    task = _task("fim_suffix_assertion")
    source_row = {
        "model": model, "model_digest_resolved": "d1", "run_id": "fleet",
        "task": task.id, "category": task.category, "family": task.family,
        "task_hash": _task_hash(task), "score": None, "error_kind": "harness_error",
        "reason": "old API failure", "level": "full", "capability_families": ["insert"],
    }
    child_row = {
        "model": model, "model_digest_resolved": "d1", "run_id": "repair_child",
        "task": task.id, "category": task.category, "family": task.family,
        "task_hash": _task_hash(task), "score": 0.0, "error_kind": "empty_output",
        "reason": "FIM returned no insertion", "level": "full", "capability_families": ["insert"],
    }
    (source / "repair_results.jsonl").write_text(json.dumps({
        "status": "unresolved",
        "kind": "capability_gate",
        "action": {"kind": "capability_gate", "model": model, "tasks": [task.id]},
        "gate": {"probe_state": "responded_contract_failed", "responded": True},
        "attempts": [{"child_run_id": "repair_child", "status": "unresolved"}],
    }) + "\n")
    (child / "raw_results.jsonl").write_text(json.dumps(child_row) + "\n")

    summary = rankings.build_summary([child_row], [source_row, child_row], runs_dir=runs)["d1"]
    assert summary["quality_status"] == "complete"
    assert summary["missing_quality_tasks"] == []
    assert summary["capability_measured_failure_tasks"] == [task.id]
    assert summary["capability_measured_failure"] is True
    assert summary["recovery_limited"] is False
    assert summary["overall_mean_score"] == 0.0


def test_force_does_not_repeat_legacy_terminal_fim_measurement(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    source = runs / "fleet"
    child = runs / "repair_child"
    model = "fim-model"
    task = _task("fim_suffix_assertion")
    row = {
        "model": model, "model_digest_resolved": "d1", "run_id": "fleet",
        "task": task.id, "category": task.category, "family": task.family,
        "task_hash": _task_hash(task), "score": 0.0, "error_kind": "empty_output",
        "reason": "FIM returned no insertion", "level": "full",
    }
    _write_full_run(source, model, [row])
    child.mkdir(parents=True)
    (child / "raw_results.jsonl").write_text(json.dumps({**row, "run_id": "repair_child"}) + "\n")
    monkeypatch.setattr(repair, "detect_gpu", lambda: SimpleNamespace(total_vram_gb=16.0))

    initial = repair.build_plan(runs, run_id="fleet", include_missing=False, force=True)
    assert len(initial.actions) == 1
    action = initial.actions[0]
    (source / "repair_results.jsonl").write_text(json.dumps({
        "action_id": action.action_id,
        "status": "unresolved",
        "kind": "capability_gate",
        "action": {
            "kind": "capability_gate", "model": model, "family": "insert",
            "tasks": [task.id],
        },
        "gate": {"probe_state": "responded_contract_failed", "responded": True},
        "attempts": [{"child_run_id": "repair_child", "status": "unresolved"}],
    }) + "\n")

    repeated = repair.build_plan(runs, run_id="fleet", include_missing=False, force=True)
    assert repeated.actions == []
    assert any(
        item.get("kind") == "terminal_capability_failure_not_repeated"
        for item in repeated.observations
    )
