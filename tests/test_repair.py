import hashlib
import json
from pathlib import Path

from llm_modelbench import repair
from llm_modelbench.config import Config
from llm_modelbench.hardware import GPUInfo
from llm_modelbench.ollama import MockClient, _exception_payload
from llm_modelbench.runner import _task_hash
from llm_modelbench.tasks import TASKS


def _task(task_id):
    return next(t for t in TASKS if t.id == task_id)


def _write_run(root: Path, run_id: str, rows, *, model="qwen2.5-coder:14b", families=None):
    run = root / run_id
    run.mkdir(parents=True)
    (run / "raw_results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": "full"}))
    (run / "filters.json").write_text(json.dumps({"level": "full", "think": "auto"}))
    (run / "model_identities.json").write_text(json.dumps({model: {"digest": "digest-1", "size": 9_000_000_000}}))
    (run / "capability_report.json").write_text(json.dumps({model: {
        "model": model,
        "declared_capabilities": ["completion", "tools", "insert"],
        "supported_families": families or ["text", "tools", "insert"],
        "functional_probes_enabled": False,
    }}))
    return run


def test_plan_groups_thinking_only_into_bounded_retry(tmp_path):
    runs = tmp_path / "runs"; runs.mkdir()
    task = _task("py_anagram")
    _write_run(runs, "fleet", [{
        "model": "qwen2.5-coder:14b", "task": task.id, "category": task.category,
        "family": task.family, "task_hash": _task_hash(task), "score": 0.0,
        "error_kind": "thinking_only", "reason": "ERROR_THINKING_ONLY",
        "timestamp": "2026-07-15T01:00:00Z",
    }])
    plan = repair.build_plan(runs, run_id="fleet", include_missing=False)
    assert len(plan.actions) == 1
    action = plan.actions[0]
    assert action.kind == "retry_generation"
    assert action.tasks == ["py_anagram"]
    assert action.overrides == {
        "retry_profiles": [
            {"think": "off", "num_predict": 2048},
            {"think": "off", "num_predict": 4096},
        ]
    }


def test_plan_uses_one_capability_gate_for_repeated_vision_api_failures(tmp_path):
    runs = tmp_path / "runs"; runs.mkdir()
    model = "hf.co/atahmih/InternVL3-8B-Q4_K_M-GGUF:latest"
    rows = []
    for task_id in ("ocr_invoice", "ocr_table_cell"):
        task = _task(task_id)
        rows.append({
            "model": model, "task": task.id, "category": task.category,
            "family": "vision", "task_hash": _task_hash(task), "score": None,
            "error_kind": "harness_error", "reason": "<HTTPError 400: 'Bad Request'>",
            "timestamp": "2026-07-15T01:00:00Z",
        })
    _write_run(runs, "fleet", rows, model=model, families=["vision", "text"])
    plan = repair.build_plan(runs, run_id="fleet", include_missing=False)
    assert len(plan.actions) == 1
    action = plan.actions[0]
    assert action.kind == "capability_gate"
    assert action.family == "vision"
    assert action.tasks == ["ocr_invoice", "ocr_table_cell"]


def test_marginal_needle_skip_becomes_guarded_retry_with_spill_policy(tmp_path, monkeypatch):
    runs = tmp_path / "runs"; runs.mkdir()
    task = _task("needle")
    _write_run(runs, "fleet", [{
        "model": "qwen2.5-coder:14b", "task": "needle", "category": task.category,
        "family": "text", "task_hash": _task_hash(task), "score": 100.0,
        "needle_coverage": 0.75,
        "needle_skipped": [{
            "size": 32768, "reason": "kv_cache_exceeds_vram_budget", "skip_class": "environment",
            "estimated_total_gb": 14.497, "vram_budget_gb": 14.4,
            "kv_estimate_source": "metadata;kv=q4_0",
        }],
        "timestamp": "2026-07-15T01:00:00Z",
    }], families=["text"])
    monkeypatch.setattr(repair, "detect_gpu", lambda: GPUInfo("nvidia", "test", 15.9, True, True))
    plan = repair.build_plan(runs, run_id="fleet", include_missing=False,
                             emergency_headroom_gb=0.25, max_spill_gb=2.0, kv_type="q4_0")
    assert len(plan.actions) == 1
    action = plan.actions[0]
    assert action.kind == "retry_needle_guarded"
    assert action.details["actionable_skips"][0]["soft_budget_overage_gb"] == 0.097
    assert action.overrides["vram_budget_gb"] == 17.65


def test_apply_writes_child_run_and_never_mutates_source_raw(tmp_path):
    runs = tmp_path / "runs"; runs.mkdir()
    task = _task("py_anagram")
    source = _write_run(runs, "fleet", [{
        "model": "qwen2.5-coder:14b", "task": task.id, "category": task.category,
        "family": task.family, "task_hash": _task_hash(task), "score": 0.0,
        "error_kind": "thinking_only", "reason": "ERROR_THINKING_ONLY",
        "timestamp": "2026-07-15T01:00:00Z",
    }])
    before = hashlib.sha256((source / "raw_results.jsonl").read_bytes()).hexdigest()
    plan = repair.build_plan(runs, run_id="fleet", include_missing=False)
    client = MockClient(seed=42, temperature=0.0, timeout=30)
    result = repair.apply_plan(client, Config(vram_budget_gb=12.0), plan)
    after = hashlib.sha256((source / "raw_results.jsonl").read_bytes()).hexdigest()
    assert before == after
    assert result["child_runs"]
    child = runs / result["child_runs"][0]
    repaired = [json.loads(line) for line in (child / "raw_results.jsonl").read_text().splitlines() if line]
    assert repaired[0]["repair_parent_run_id"] == "fleet"
    assert repaired[0]["repair_policy_version"] == repair.POLICY_VERSION
    assert (source / "repair_results.jsonl").exists()


def test_http_error_payload_preserves_body():
    import io
    import urllib.error
    exc = urllib.error.HTTPError(
        "http://localhost/api/chat", 400, "Bad Request", {}, io.BytesIO(b'{"error":"model does not support images"}')
    )
    payload = _exception_payload(exc)
    assert payload["http_status"] == 400
    assert "does not support images" in payload["http_error_body"]


def test_repair_cli_accepts_documented_dry_run_gpu_and_kv_flags():
    from llm_modelbench.cli import build_parser
    args = build_parser().parse_args([
        "repair", "--run-prefix", "overnight_v2", "--dry-run",
        "--gpu-vram-gb", "15.93", "--kv-type", "q8_0",
        "--confirm-kv-server", "--force",
    ])
    assert args.dry_run is True
    assert args.apply is False
    assert args.gpu_vram_gb == 15.93
    assert args.kv_type == "q8_0"
    assert args.confirm_kv_server is True
    assert args.force is True


def test_repair_cli_rejects_apply_and_dry_run_together():
    import pytest
    from llm_modelbench.cli import build_parser
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "repair", "--run-id", "r1", "--apply", "--dry-run",
        ])


def test_measured_vram_slope_divergence_does_not_claim_an_effective_kv_type():
    row = {
        "needle_attempted": [{
            "kv_estimate_source": "derived_embedding_per_head;kv=q4_0",
            "kv_bytes_per_token": 49152,
        }]
    }
    item = {
        "kv_estimate_source": "measured_vram_slope",
        "kv_bytes_per_token": 203369,
    }
    result = repair._needle_measurement_analysis(row, item)
    assert result["estimator_divergence"] is True
    assert result["measured_to_metadata_ratio"] == 4.138
    assert "does not prove" in result["estimator_divergence_note"]
    assert "effective_kv_type" not in result


def test_gpu_override_makes_realistic_marginal_needle_skip_actionable_offline(tmp_path, monkeypatch):
    runs = tmp_path / "runs"; runs.mkdir()
    task = _task("needle")
    _write_run(runs, "fleet", [{
        "model": "gpt-oss:20b", "task": "needle", "category": task.category,
        "family": "text", "task_hash": _task_hash(task), "score": None,
        "needle_coverage": 0.75,
        "needle_attempted": [{
            "size": 32000, "kv_estimate_source": "metadata;kv=q4_0",
            "kv_bytes_per_token": 12288,
        }],
        "needle_skipped": [{
            "size": 65536, "reason": "kv_cache_exceeds_vram_budget",
            "skip_class": "environment", "estimated_total_gb": 14.487,
            "estimated_kv_gb": 1.641, "vram_budget_gb": 14.4,
            "kv_estimate_source": "measured_vram_slope",
            "kv_bytes_per_token": 26742,
        }],
        "timestamp": "2026-07-15T01:00:00Z",
    }], model="gpt-oss:20b", families=["text"])
    monkeypatch.setattr(repair, "detect_gpu", lambda: GPUInfo())
    monkeypatch.setattr(repair, "inspect_ollama_kv_environment", lambda: {
        "effective_kv_type": None, "verified": False, "notes": [],
    })
    plan = repair.build_plan(
        runs, run_id="fleet", include_missing=False, gpu_total_gb=15.93,
        emergency_headroom_gb=0.25, max_spill_gb=2.0, kv_type="q8_0",
    )
    action = next(a for a in plan.actions if a.kind == "retry_needle_guarded")
    detail = action.details["actionable_skips"][0]
    assert detail["classification"] == "MARGINAL_SOFT_LIMIT"
    assert detail["soft_budget_overage_gb"] == 0.087
    assert plan.options["gpu_total_source"] == "cli_override"
    assert action.overrides["vram_budget_gb"] == 17.68


def test_explicit_kv_apply_requires_matching_shell_and_verified_or_confirmed_server(monkeypatch):
    monkeypatch.delenv("OLLAMA_KV_CACHE_TYPE", raising=False)
    ok, reason = repair._kv_environment_check(
        "q8_0", server_confirmed=True,
        server_inspection={"verified": False, "effective_kv_type": None},
    )
    assert ok is False
    assert "repair process" in reason

    monkeypatch.setenv("OLLAMA_KV_CACHE_TYPE", "q8_0")
    ok, reason = repair._kv_environment_check(
        "q8_0", server_confirmed=False,
        server_inspection={"verified": True, "effective_kv_type": "q8_0"},
    )
    assert ok is True
    assert "verified" in reason


def test_plan_reports_unknown_gpu_instead_of_generic_needle_failure(tmp_path, monkeypatch):
    runs = tmp_path / "runs"; runs.mkdir()
    task = _task("needle")
    _write_run(runs, "fleet", [{
        "model": "m", "task": "needle", "category": task.category,
        "family": "text", "task_hash": _task_hash(task), "score": None,
        "needle_coverage": 0.75,
        "needle_skipped": [{
            "size": 32768, "reason": "kv_cache_exceeds_vram_budget",
            "skip_class": "environment", "estimated_total_gb": 14.5,
            "vram_budget_gb": 14.4,
        }],
        "timestamp": "2026-07-15T01:00:00Z",
    }], model="m", families=["text"])
    monkeypatch.setattr(repair, "detect_gpu", lambda: GPUInfo())
    monkeypatch.setattr(repair, "inspect_ollama_kv_environment", lambda: {
        "effective_kv_type": None, "verified": False, "notes": [],
    })
    plan = repair.build_plan(runs, run_id="fleet", include_missing=False)
    obs = next(o for o in plan.observations if o["kind"] == "needle_not_automatically_repairable")
    assert obs["classifications"] == ["GPU_CAPACITY_UNKNOWN"]
    assert "--gpu-vram-gb" in obs["reason"]


def test_kv_environment_inspection_discovers_the_real_active_unit_not_a_hardcoded_one(monkeypatch):
    """This is the actual bug: inspection used to always ask about
    'ollama.service'/'ollama' no matter what, even when a completely
    different unit (e.g. ollama-gpu0.service) was the one really serving
    requests. It must now discover the real active unit first."""

    class Result:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(argv, **kwargs):
        joined = " ".join(argv)
        if "ss" in argv and "-tlnp" in argv:
            return Result(stdout='LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=2119,fd=4))\n')
        if "list-units" in joined:
            return Result(
                stdout=(
                    "ollama-gpu0.service loaded active running Ollama GPU0\n"
                    "ollama.service loaded inactive dead Ollama Service\n"
                )
            )
        if "show" in argv and "ollama-gpu0.service" in argv and "--property=MainPID" in joined:
            return Result(stdout="2119\n")
        if "show" in argv and "ollama.service" in argv and "--property=MainPID" in joined:
            return Result(stdout="0\n")
        if "show" in argv and "ollama-gpu0.service" in argv and "--property=Environment" in joined:
            return Result(stdout='OLLAMA_KV_CACHE_TYPE=q8_0 OPENAI_API_KEY=secret-value OTHER=private\n')
        return Result()

    monkeypatch.setattr(repair.shutil, "which", lambda name: None if name == "pgrep" else "/usr/bin/systemctl")
    monkeypatch.setattr(repair.subprocess, "run", fake_run)
    result = repair.inspect_ollama_kv_environment()
    encoded = json.dumps(result)
    assert result["systemd_unit"] == "ollama-gpu0.service", "must report the real active unit, not a hardcoded guess"
    assert result["systemd_kv_type"] == "q8_0"
    assert "secret-value" not in encoded
    assert "OPENAI_API_KEY" not in encoded
    assert "systemd_environment" not in result


def test_kv_environment_inspection_reports_not_inspected_when_discovery_is_unsafe(monkeypatch):
    """Planning/dry-run must never escalate privileges or guess a unit just
    to describe current state. If the active unit genuinely can't be
    determined without a privileged pass, report it plainly instead of
    silently describing some other unit's configuration."""

    class Result:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(argv, **kwargs):
        # No listener visible at all without sudo -- simulates the socket
        # being owned by a different user, unreadable without privilege.
        return Result(stdout="")

    monkeypatch.setattr(repair.shutil, "which", lambda name: None if name == "pgrep" else "/usr/bin/systemctl")
    monkeypatch.setattr(repair.subprocess, "run", fake_run)
    monkeypatch.delenv("OLLAMA_KV_CACHE_TYPE", raising=False)
    result = repair.inspect_ollama_kv_environment()
    assert result["systemd_unit"] is None
    assert result["effective_kv_type"] is None
    assert result["effective_source"] == "not inspected"
    assert result["verified"] is False


def test_kv_environment_inspection_never_reports_a_disabled_units_kv_as_the_live_state(monkeypatch):
    """Regression guard for the exact real-host bug: a disabled/inactive
    ollama.service must never be reported as if it described the live
    server's KV state just because its name matches a hardcoded guess."""

    class Result:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(argv, **kwargs):
        joined = " ".join(argv)
        if "ss" in argv and "-tlnp" in argv:
            return Result(stdout='LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=133664,fd=4))\n')
        if "list-units" in joined:
            return Result(
                stdout=(
                    "ollama-gpu0.service loaded active running Ollama GPU0\n"
                    "ollama.service loaded inactive dead Ollama Service\n"
                )
            )
        if "show" in argv and "ollama-gpu0.service" in argv and "--property=MainPID" in joined:
            return Result(stdout="133664\n")
        if "show" in argv and "ollama.service" in argv and "--property=MainPID" in joined:
            return Result(stdout="0\n")
        if "show" in argv and "ollama.service" in argv and "--property=Environment" in joined:
            # If the old hardcoded path were still in effect, this stale
            # q4_0 value would leak through as if it were the live state.
            return Result(stdout="OLLAMA_KV_CACHE_TYPE=q4_0\n")
        if "show" in argv and "ollama-gpu0.service" in argv and "--property=Environment" in joined:
            return Result(stdout="OLLAMA_MODELS=/srv/ollama-models\n")  # no KV var set on the real unit
        return Result()

    monkeypatch.setattr(repair.shutil, "which", lambda name: None if name == "pgrep" else "/usr/bin/systemctl")
    monkeypatch.setattr(repair.subprocess, "run", fake_run)
    monkeypatch.delenv("OLLAMA_KV_CACHE_TYPE", raising=False)
    result = repair.inspect_ollama_kv_environment()
    assert result["systemd_unit"] == "ollama-gpu0.service"
    assert result["systemd_kv_type"] is None, "the real active unit has no KV override -- must not borrow ollama.service's stale q4_0"


def test_repair_plan_rejects_unsafe_negative_capacity_settings(tmp_path):
    import pytest
    runs = tmp_path / "runs"; runs.mkdir()
    task = _task("py_anagram")
    _write_run(runs, "fleet", [{
        "model": "m", "task": task.id, "category": task.category,
        "family": task.family, "task_hash": _task_hash(task), "score": 100.0,
    }], model="m", families=["text"])
    with pytest.raises(ValueError, match="headroom"):
        repair.build_plan(runs, run_id="fleet", emergency_headroom_gb=-0.1)
    with pytest.raises(ValueError, match="spill"):
        repair.build_plan(runs, run_id="fleet", max_spill_gb=-1)
    with pytest.raises(ValueError, match="gpu-vram"):
        repair.build_plan(runs, run_id="fleet", gpu_total_gb=0)
    with pytest.raises(ValueError, match="num-predict"):
        repair.build_plan(runs, run_id="fleet", think_retry_num_predict=0)


def test_rendered_plan_exposes_needle_classification_and_guard_values(tmp_path, monkeypatch):
    runs = tmp_path / "runs"; runs.mkdir()
    task = _task("needle")
    _write_run(runs, "fleet", [{
        "model": "gpt-oss:20b", "task": "needle", "category": task.category,
        "family": "text", "task_hash": _task_hash(task), "score": None,
        "needle_coverage": 0.75,
        "needle_skipped": [{
            "size": 65536, "reason": "kv_cache_exceeds_vram_budget",
            "skip_class": "environment", "estimated_total_gb": 14.487,
            "vram_budget_gb": 14.4,
        }],
    }], model="gpt-oss:20b", families=["text"])
    monkeypatch.setattr(repair, "inspect_ollama_kv_environment", lambda: {
        "effective_kv_type": None, "verified": False, "effective_source": None,
    })
    plan = repair.build_plan(
        runs, run_id="fleet", include_missing=False, gpu_total_gb=15.93,
        emergency_headroom_gb=0.25, max_spill_gb=2.0, kv_type="q8_0",
    )
    text = repair.render_plan(plan)
    assert "MARGINAL_SOFT_LIMIT" in text
    assert "overage=0.087 GB" in text
    assert "guarded total=17.680 GB" in text
    assert "requested=q8_0" in text


def test_judge_repairs_record_each_action_and_count_actions_not_runs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"; runs.mkdir()
    source = runs / "fleet"; source.mkdir()
    (source / "raw_results.jsonl").write_text("")
    actions = []
    for index, task_id in enumerate(("wr_rag", "kb_taxonomy"), start=1):
        actions.append(repair.RepairAction(
            action_id=f"judge-{index}", kind="judge_existing_dump", model="m",
            model_digest="d", source_run_id="fleet", tasks=[task_id],
            reason="awaiting judge", automatic=True,
        ))
    plan = repair.RepairPlan(
        schema_version=1, repair_policy_version=repair.POLICY_VERSION,
        plan_id="plan", created_at="now", runs_dir=str(runs),
        selected_runs=["fleet"], actions=actions, observations=[], counts={}, options={"kv_type": "current"},
    )
    monkeypatch.setattr(repair, "judge_run", lambda *args, **kwargs: {"judged": 2, "written": 2})
    result = repair.apply_plan(object(), Config(), plan, judge_mode="single", judge_model="judge")
    assert result["completed"] == 2
    assert result["recovered"] == 2
    records = [json.loads(line) for line in (source / "repair_results.jsonl").read_text().splitlines()]
    assert {record["action_id"] for record in records} == {"judge-1", "judge-2"}


def test_repair_cli_accepts_human_supervised_kv_cascade_flags():
    from llm_modelbench.cli import build_parser
    args = build_parser().parse_args([
        "repair", "--run-prefix", "overnight_v2", "--apply", "--yes",
        "--kv-cascade", "--restart-ollama", "--ollama-service", "ollama.service",
        "--gpu-vram-gb", "15.93",
    ])
    assert args.kv_cascade is True
    assert args.restart_ollama is True
    assert args.ollama_service == "ollama.service"
    assert args.keep_final_kv is False
    assert args.reuse_sudo_credentials is False


def test_repair_cli_defaults_to_active_service_discovery():
    from llm_modelbench.cli import build_parser
    args = build_parser().parse_args([
        "repair", "--run-prefix", "overnight_v2",
        "--kv-cascade", "--restart-ollama",
    ])
    assert args.ollama_service == "auto"


def test_needle_numeric_score_is_not_recovered_when_coverage_remains_incomplete(tmp_path, monkeypatch):
    runs = tmp_path / "runs"; runs.mkdir()
    _write_run(runs, "fleet", [{
        "model": "m", "task": "needle", "category": "long_context", "family": "text",
        "task_hash": _task_hash(_task("needle")), "score": 50.0, "needle_coverage": 0.75,
        "needle_skipped": [{
            "size": 32768, "skip_class": "environment", "reason": "kv_cache_exceeds_vram_budget",
            "estimated_total_gb": 14.5, "vram_budget_gb": 14.4,
        }],
    }], model="m", families=["text"])
    action = repair.RepairAction(
        action_id="needle-action", kind="retry_needle_guarded", model="m",
        model_digest="digest-1", source_run_id="fleet", tasks=["needle"],
        reason="guarded", automatic=True, overrides={"vram_budget_gb": 17.0},
    )
    plan = repair.RepairPlan(
        schema_version=1, repair_policy_version=repair.POLICY_VERSION, plan_id="plan",
        created_at="now", runs_dir=str(runs), selected_runs=["fleet"], actions=[action],
        observations=[], counts={}, options={"kv_type": "current"},
    )

    from llm_modelbench import runner
    def fake_run(*args, **kwargs):
        out = kwargs["out_dir"]
        out.mkdir(parents=True)
        (out / "raw_results.jsonl").write_text(json.dumps({
            "model": "m", "task": "needle", "category": "long_context", "family": "text",
            "score": 100.0, "needle_coverage": 0.75, "error_kind": None,
        }) + "\n")
        (out / "summary_meta.json").write_text(json.dumps({"level": "full"}))
        (out / "model_identities.json").write_text(json.dumps({"m": {"digest": "digest-1"}}))
    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(repair, "interrogate_model", lambda *a, **k: {
        "model": "m", "supported_families": ["text"], "declared_capabilities": ["completion"]
    })
    monkeypatch.setattr("llm_modelbench.report.build", lambda *a, **k: None)

    result = repair.apply_plan(MockClient(), Config(), plan)
    assert result["recovered"] == 0
    assert result["unresolved"] == 1
    assert result["actions"][0]["status"] == "unresolved"


def test_managed_kv_cascade_runs_current_then_q8_then_only_unresolved_q4_and_restores(monkeypatch, tmp_path):
    source = tmp_path / "r"
    source.mkdir()
    needle_a = repair.RepairAction(
        action_id="a", kind="retry_needle_guarded", model="m1", model_digest="d1",
        source_run_id="r", tasks=["needle"], reason="x", automatic=True,
    )
    needle_b = repair.RepairAction(
        action_id="b", kind="retry_needle_guarded", model="m2", model_digest="d2",
        source_run_id="r", tasks=["needle"], reason="x", automatic=True,
    )
    ordinary = repair.RepairAction(
        action_id="c", kind="retry_generation", model="m3", model_digest="d3",
        source_run_id="r", tasks=["py_anagram"], reason="x", automatic=True,
    )
    plan = repair.RepairPlan(
        schema_version=1, repair_policy_version=repair.POLICY_VERSION, plan_id="base",
        created_at="now", runs_dir=str(tmp_path), selected_runs=["r"],
        actions=[needle_a, needle_b, ordinary], observations=[], counts={},
        options={"kv_type": "current"},
    )

    class FakeController:
        unit = "ollama.service"
        auto_confirm = False
        mutation_started = False
        def __init__(self):
            self.events = []
            self.confirmed = []
            self.authorised = 0
            self.restored = 0
        def require_supervised_tty(self): pass
        def confirm(self, phase, message): self.confirmed.append(phase)
        def authorise_sudo(self): self.authorised += 1
        def set_kv_type(self, kv, phase):
            from llm_modelbench.ollama_service import ServicePhaseResult
            self.mutation_started = True
            item = ServicePhaseResult(phase, self.unit, kv, True, True, kv, "verified")
            self.events.append(item.to_dict())
            return item
        def restore(self, phase="restore"):
            from llm_modelbench.ollama_service import ServicePhaseResult
            self.restored += 1
            self.mutation_started = False
            item = ServicePhaseResult(phase, self.unit, None, True, True, None, "restored")
            self.events.append(item.to_dict())
            return item

    calls = []
    def fake_apply(client, cfg, phase_plan, **kwargs):
        phase = phase_plan.options.get("service_phase")
        parents = [a.details.get("parent_action_id") for a in phase_plan.actions]
        calls.append((phase, parents))
        if phase == "standard":
            return {"outcome": "COMPLETE", "actions": [{"action_id": a.action_id, "status": "recovered"} for a in phase_plan.actions]}
        if phase == "current":
            return {"outcome": "PARTIAL", "completed": 2, "recovered": 0, "unresolved": 2, "errors": 0,
                    "child_runs": [], "actions": [{"action_id": a.action_id, "status": "unresolved"} for a in phase_plan.actions]}
        if phase == "q8_0":
            entries = []
            for action in phase_plan.actions:
                parent = action.details["parent_action_id"]
                entries.append({"action_id": action.action_id, "status": "recovered" if parent == "a" else "unresolved"})
            return {"outcome": "PARTIAL", "completed": 2, "recovered": 1, "unresolved": 1,
                    "errors": 0, "child_runs": [], "actions": entries}
        return {"outcome": "COMPLETE", "completed": 1, "recovered": 1, "unresolved": 0,
                "errors": 0, "child_runs": [],
                "actions": [{"action_id": a.action_id, "status": "recovered"} for a in phase_plan.actions]}
    monkeypatch.setattr(repair, "apply_plan", fake_apply)
    controller = FakeController()
    result = repair.apply_plan_with_managed_kv_cascade(object(), Config(), plan, controller)
    assert calls[0] == ("standard", ["c"])
    assert calls[1] == ("current", ["a", "b"])
    assert calls[2] == ("q8_0", ["a", "b"])
    assert calls[3] == ("q4_0", ["b"])
    assert controller.confirmed == ["q8_0", "q4_0", "restore"]
    assert controller.restored == 1
    assert result["restored_original_service_state"] is True
    assert result["unresolved_needle_parent_actions"] == []


def test_managed_kv_cascade_does_not_restore_when_current_phase_fails_before_mutation(monkeypatch, tmp_path):
    action = repair.RepairAction(
        action_id="a", kind="retry_needle_guarded", model="m", model_digest="d",
        source_run_id="r", tasks=["needle"], reason="x", automatic=True,
    )
    plan = repair.RepairPlan(1, repair.POLICY_VERSION, "p", "now", str(tmp_path), ["r"], [action], [], {}, {"kv_type": "current"})
    class Controller:
        unit = "ollama.service"
        events = []
        restored = 0
        def require_supervised_tty(self): pass
        def confirm(self, *a): pass
        def authorise_sudo(self): pass
        def set_kv_type(self, kv, phase):
            from llm_modelbench.ollama_service import ServicePhaseResult
            return ServicePhaseResult(phase, self.unit, kv, True, True, kv, "ok")
        def restore(self, phase="restore"):
            self.restored += 1
            from llm_modelbench.ollama_service import ServicePhaseResult
            return ServicePhaseResult(phase, self.unit, None, True, True, None, "ok")
    controller = Controller()
    monkeypatch.setattr(repair, "apply_plan", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    import pytest
    with pytest.raises(RuntimeError, match="boom"):
        repair.apply_plan_with_managed_kv_cascade(object(), Config(), plan, controller)
    assert controller.restored == 0


def test_service_controller_uses_sudo_password_prompt_and_dedicated_dropin(monkeypatch, tmp_path):
    from llm_modelbench import ollama_service
    state = {"kv": None}
    commands = []

    class Result:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode = returncode; self.stdout = stdout; self.stderr = stderr

    def fake_run(argv, **kwargs):
        commands.append(list(argv))
        if argv[:3] == ["sudo", "test", "-e"]:
            return Result(returncode=1)
        if "install" in argv and argv[-1].endswith("zzzz-llmb-repair-kv.conf"):
            content = Path(argv[-2]).read_text()
            state["kv"] = "q8_0" if "q8_0" in content else "q4_0"
        if "ss" in argv and "-H" in argv:
            return Result(stdout='LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=123,fd=4))\n')
        if "list-units" in argv:
            return Result(stdout="ollama.service loaded active running Ollama\n")
        if "--property=Environment" in " ".join(argv):
            return Result(stdout=f"OLLAMA_KV_CACHE_TYPE={state['kv']}\n")
        if "--property=DropInPaths" in " ".join(argv):
            return Result(stdout="/etc/systemd/system/ollama.service.d/zzzz-llmb-repair-kv.conf\n")
        if argv[:3] == ["systemctl", "show", "ollama.service"]:
            return Result(stdout="123\n")
        if argv and argv[0] == "nvidia-smi":
            return Result(stdout="GPU-test\n")
        if argv[:2] == ["sudo", "sh"]:
            return Result(stdout=f"OLLAMA_KV_CACHE_TYPE={state['kv']}\n")
        return Result()

    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = ollama_service.OllamaServiceController(
        "ollama.service", run=fake_run, input_fn=lambda prompt: "RESTART",
        isatty_fn=lambda: True, sleep_fn=lambda _: None,
        kv_read_helper_exists_fn=lambda p: False,
    )
    controller.confirm("q8_0", "test")
    controller.authorise_sudo()
    phase = controller.set_kv_type("q8_0", phase="q8_0")
    assert phase.verified is True
    assert ["sudo", "-k"] in commands
    assert ["sudo", "-v"] in commands
    install_commands = [cmd for cmd in commands if "install" in cmd and cmd[-1].endswith("zzzz-llmb-repair-kv.conf")]
    assert install_commands
    assert str(controller.dropin_path) == "/etc/systemd/system/ollama.service.d/zzzz-llmb-repair-kv.conf"


def test_phase_plans_keep_original_action_id_for_future_repair_suppression():
    action = repair.RepairAction(
        action_id="stable-action", kind="retry_needle_guarded", model="m", model_digest="d",
        source_run_id="r", tasks=["needle"], reason="x", automatic=True,
    )
    base = repair.RepairPlan(1, repair.POLICY_VERSION, "p", "now", "runs", ["r"], [action], [], {}, {"kv_type": "current"})
    q8 = repair._derived_plan(base, [action], phase="q8_0", kv_type="q8_0")
    q4 = repair._derived_plan(base, [action], phase="q4_0", kv_type="q4_0")
    assert q8.actions[0].action_id == "stable-action"
    assert q4.actions[0].action_id == "stable-action"
    assert q8.actions[0].details["service_phase"] == "q8_0"
    assert q4.actions[0].details["service_phase"] == "q4_0"


def test_service_controller_refuses_to_overwrite_unmanaged_existing_dropin(monkeypatch):
    from llm_modelbench import ollama_service
    class Result:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode = returncode; self.stdout = stdout; self.stderr = stderr
    def fake_run(argv, **kwargs):
        if argv[:3] == ["sudo", "test", "-e"]:
            return Result(returncode=0)
        if argv[:2] == ["sudo", "cat"]:
            return Result(stdout=b"[Service]\nEnvironment=OTHER=value\n")
        return Result()
    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = ollama_service.OllamaServiceController(
        run=fake_run, input_fn=lambda _: "RESTART", isatty_fn=lambda: True,
    )
    import pytest
    with pytest.raises(ollama_service.ServiceControlError, match="unmanaged"):
        controller.snapshot_dropin()


def test_service_controller_rejects_non_tty_privileged_execution(monkeypatch):
    from llm_modelbench import ollama_service
    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = ollama_service.OllamaServiceController(isatty_fn=lambda: False)
    import pytest
    with pytest.raises(ollama_service.ServiceControlError, match="interactive terminal"):
        controller.require_supervised_tty()


def test_managed_kv_cascade_does_not_restart_when_pre_mutation_guard_fails(tmp_path):
    import pytest
    from llm_modelbench.ollama_service import ServiceControlError

    action = repair.RepairAction(
        action_id="needle-guard", kind="retry_needle_guarded", model="m", model_digest="d",
        source_run_id="source", tasks=["needle"], reason="guarded", automatic=True,
    )
    plan = repair.RepairPlan(
        schema_version=1, repair_policy_version=repair.POLICY_VERSION,
        plan_id="pre-mutation", created_at="now", runs_dir=str(tmp_path),
        selected_runs=["source"], actions=[action], observations=[], counts={},
        options={"kv_type": "current"},
    )

    class Controller:
        unit = "ollama-gpu0.service"
        events = []
        mutation_started = False
        restore_calls = 0

        def require_supervised_tty(self):
            return None

        def confirm(self, *args, **kwargs):
            return None

        def authorise_sudo(self):
            return None

        def set_kv_type(self, *args, **kwargs):
            raise ServiceControlError("stale CUDA_VISIBLE_DEVICES")

        def restore(self, *args, **kwargs):
            self.restore_calls += 1

    controller = Controller()
    with pytest.raises(ServiceControlError, match="stale CUDA"):
        repair.apply_plan_with_managed_kv_cascade(object(), Config(), plan, controller)
    assert controller.restore_calls == 0


def test_kv_environment_inspection_distinguishes_checked_unset_from_could_not_check(monkeypatch):
    """The exact real-host case: ollama-gpu0.service is discovered fine and
    its environment is genuinely queried successfully, but it has no
    OLLAMA_KV_CACHE_TYPE override at all. That's a real, checked answer --
    it must not be reported the same generic way as 'could not inspect'."""

    class Result:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(argv, **kwargs):
        joined = " ".join(argv)
        if "ss" in argv and "-tlnp" in argv:
            return Result(stdout='LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=133664,fd=4))\n')
        if "list-units" in joined:
            return Result(stdout="ollama-gpu0.service loaded active running Ollama GPU0\n")
        if "show" in argv and "--property=MainPID" in joined:
            return Result(stdout="133664\n")
        if "show" in argv and "--property=Environment" in joined:
            return Result(stdout="OLLAMA_MODELS=/srv/ollama-models OLLAMA_HOST=127.0.0.1:11434\n")
        return Result()

    monkeypatch.setattr(repair.shutil, "which", lambda name: None if name == "pgrep" else "/usr/bin/systemctl")
    monkeypatch.setattr(repair.subprocess, "run", fake_run)
    monkeypatch.delenv("OLLAMA_KV_CACHE_TYPE", raising=False)
    result = repair.inspect_ollama_kv_environment()
    assert result["systemd_unit"] == "ollama-gpu0.service"
    assert result["systemd_kv_type"] is None
    assert result["effective_source"] is not None, "must not be blank/unavailable -- a real check happened"
    assert result["effective_source"] != "not inspected", "the unit WAS inspected, this isn't the same as a failed discovery"
    assert "ollama-gpu0.service" in result["effective_source"]
    assert "checked" in result["effective_source"].lower() or "no" in result["effective_source"].lower()


def test_auto_confirm_skips_typed_prompt_and_sudo_revalidation(monkeypatch):
    """--auto-confirm must never call input() or require a TTY, and must
    skip sudo -k/-v (relying on a scoped NOPASSWD sudoers rule instead)."""
    from llm_modelbench import ollama_service

    calls = []
    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    def input_that_must_never_be_called(prompt):
        raise AssertionError("auto_confirm must never call input()")

    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = ollama_service.OllamaServiceController(
        "ollama-gpu0.service", run=fake_run, input_fn=input_that_must_never_be_called,
        isatty_fn=lambda: False,  # no TTY at all -- must still work
        auto_confirm=True,
    )
    controller.confirm("q8_0", "test message")  # must not raise, must not call input
    controller.authorise_sudo()
    assert not any(c[:2] == ["sudo", "-k"] or c[:2] == ["sudo", "-v"] for c in calls), \
        "auto_confirm must skip sudo -k/-v and rely on scoped NOPASSWD instead"


def test_without_auto_confirm_behavior_is_unchanged(monkeypatch):
    """Default behavior (auto_confirm=False) must be identical to before."""
    from llm_modelbench import ollama_service
    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = ollama_service.OllamaServiceController(
        "ollama-gpu0.service", input_fn=lambda p: "RESTART", isatty_fn=lambda: True,
    )
    assert controller.auto_confirm is False
    controller.confirm("q8_0", "test message")  # should succeed via typed RESTART, no error


def test_rankings_capability_unavailable_stops_blocking_completeness(tmp_path):
    """Real schema match for repair.py's _record_unavailable_capability output.
    A model whose vision capability gate genuinely failed on this build must
    not show OCR/vision tasks as forever-missing once every other applicable
    task is covered -- it should reach complete, flagged capability_limited."""
    from llm_modelbench import rankings

    run_dir = tmp_path / "overnight_v2_some_vision_model"
    run_dir.mkdir()
    (run_dir / "capability_repair.json").write_text(json.dumps({
        "some-vision-model:latest": {
            "unavailable_families": {
                "vision": {
                    "family": "vision",
                    "action_id": "abc123",
                    "recorded_at": "2026-07-16T00:00:00+00:00",
                    "reason": "functional capability gate failed on the current Ollama/model build",
                    "gate": {"http_status": 400},
                }
            },
            "history": [],
        }
    }))

    # Minimal row set: one real, current, non-vision task result plus the
    # model's identity/family info. Real vision-family tasks (ocr_*, pdf_*)
    # are deliberately absent -- that's the real-world shape being tested.
    rows = [{
        "model": "some-vision-model:latest",
        "model_digest_resolved": "digestA",
        "task": "py_good" if "py_good" in rankings._CURRENT_HASHES else next(iter(rankings._CURRENT_HASHES)),
        "task_hash": None,
        "level": "full",
        "score": 90.0,
        "family": "vision",
        "capability_families": ["vision", "text"],
        "run_id": run_dir.name,
    }]
    rows[0]["task_hash"] = rankings._CURRENT_HASHES.get(rows[0]["task"])

    summary = rankings.build_summary(rows, rows, runs_dir=tmp_path)
    entry = summary["digestA"]
    assert "capability_unavailable_tasks" in entry
    assert entry["capability_unavailable_tasks"], "vision-family tasks should be classified capability_unavailable"
    for task_id in entry["capability_unavailable_tasks"]:
        assert task_id not in entry["missing_quality_tasks"], (
            "a capability-unavailable task must not also count as forever-missing"
        )


def test_auto_confirm_uses_sudo_n_not_plain_sudo(monkeypatch):
    """Every privileged call in auto_confirm mode must use 'sudo -n', which
    fails immediately if NOPASSWD isn't configured, instead of plain sudo
    which would hang or block waiting for a password with no TTY attached."""
    from llm_modelbench import ollama_service
    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = ollama_service.OllamaServiceController("ollama-gpu0.service", auto_confirm=True)
    assert controller.privileged_prefix == ["sudo", "-n"]

    normal_controller = ollama_service.OllamaServiceController("ollama-gpu0.service", auto_confirm=False)
    assert normal_controller.privileged_prefix == ["sudo"]


def test_preflight_fails_closed_when_nopasswd_not_configured(monkeypatch):
    """The exact scenario GPT flagged: auto-confirm must not silently fall
    back to interactive mode or hang. If the NOPASSWD rule isn't actually
    installed, this must fail immediately and say so clearly."""
    import pytest
    from llm_modelbench import ollama_service

    def fake_run(argv, **kwargs):
        class R:
            returncode = 1  # sudo -n fails: a password would have been required
            stdout = ""
            stderr = "sudo: a password is required\n"
        return R()

    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = ollama_service.OllamaServiceController(
        "ollama-gpu0.service", run=fake_run, auto_confirm=True,
    )
    with pytest.raises(ollama_service.ServiceControlError, match="NOPASSWD"):
        controller.verify_noninteractive_sudo_ready()


def test_preflight_error_reports_exact_command_path_and_stderr(monkeypatch):
    import pytest
    from llm_modelbench import ollama_service

    def fake_run(argv, **kwargs):
        class R:
            stdout = ""
            stderr = "sudo: a password is required\n"
            returncode = 0 if argv == ["sudo", "-k"] else 1
        return R()

    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = ollama_service.OllamaServiceController(
        "ollama-gpu0.service", run=fake_run, auto_confirm=True,
        ss_command="/usr/sbin/ss",
    )
    with pytest.raises(ollama_service.ServiceControlError) as excinfo:
        controller.verify_noninteractive_sudo_ready()
    message = str(excinfo.value)
    assert "/usr/sbin/ss -H -tlnp" in message
    assert "sudo: a password is required" in message


def test_ss_resolution_is_isolated_from_shared_shutil_which_monkeypatch(tmp_path, monkeypatch):
    from llm_modelbench import ollama_service, repair

    fake_ss = tmp_path / "ss"
    fake_ss.write_text("#!/bin/sh\nexit 0\n")
    fake_ss.chmod(0o755)
    monkeypatch.setattr(ollama_service.os, "get_exec_path", lambda: [str(tmp_path)])
    monkeypatch.setattr(repair.shutil, "which", lambda name: "/usr/bin/systemctl")

    assert ollama_service._resolve_executable("ss") == str(fake_ss)


def test_preflight_passes_when_nopasswd_actually_works(monkeypatch):
    from llm_modelbench import ollama_service

    calls = []
    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = ollama_service.OllamaServiceController(
        "ollama-gpu0.service", run=fake_run, auto_confirm=True,
        ss_command="/usr/bin/ss",
    )
    controller.verify_noninteractive_sudo_ready()  # must not raise
    assert calls == [
        ["sudo", "-k"],
        ["sudo", "-n", "/usr/bin/ss", "-H", "-tlnp"],
    ], (
        "preflight must invalidate cached credentials, then prove the exact "
        "NOPASSWD command path works without an interactive timestamp"
    )


def test_preflight_is_a_noop_without_auto_confirm(monkeypatch):
    """Must never run (or require) this check for the default interactive path."""
    from llm_modelbench import ollama_service
    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = ollama_service.OllamaServiceController("ollama-gpu0.service", auto_confirm=False)
    controller.verify_noninteractive_sudo_ready()  # must not raise, must not run anything


def test_install_uses_fixed_predictable_temp_path_not_random(monkeypatch):
    """Sudoers on some systems rejects wildcards in command arguments
    entirely. A random tempfile name would make the install step impossible
    to authorize with an exact-match NOPASSWD rule. Must be fixed and
    predictable per unit instead."""
    from llm_modelbench import ollama_service

    calls = []
    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = ollama_service.OllamaServiceController("ollama-gpu0.service", run=fake_run)
    controller._install_bytes(b"test content")

    install_calls = [c for c in calls if "install" in c and "0644" in c]
    assert install_calls, "expected an install -m 0644 call"
    source_path = install_calls[0][install_calls[0].index("0644") + 1]
    assert source_path == "/tmp/llmb-ollama-kv-pending-ollama-gpu0.service.conf", (
        f"expected a fixed, predictable path, got {source_path!r}"
    )
    assert "*" not in source_path and not any(c in source_path for c in ("?", "[", "]"))


def test_observed_process_kv_uses_fixed_helper_script_when_present(monkeypatch):
    """When the sudoers-friendly helper script is installed, use it (no PID
    wildcard needed in sudoers at all) instead of the inline sh -c string
    that this exact host's sudo build rejected with two separate errors
    (wildcard + illegal escape sequence)."""
    from llm_modelbench import ollama_service

    calls = []
    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        class R:
            returncode = 0
            stdout = "OLLAMA_KV_CACHE_TYPE=q8_0\n"
            stderr = ""
        return R()

    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = ollama_service.OllamaServiceController(
        "ollama-gpu0.service", run=fake_run,
        kv_read_helper_exists_fn=lambda p: True,
    )
    monkeypatch.setattr(controller, "_main_pid", lambda: 2119)

    value = controller.observed_process_kv()
    assert value == "q8_0"
    matching = [c for c in calls if "/usr/local/libexec/llmb-read-kv-env.sh" in c]
    assert matching, f"expected the helper script to be invoked, got: {calls}"
    assert matching[0][-1] == "2119", f"expected pid as trailing arg: {matching[0]}"
    assert not any("sh" in c and "-c" in c for c in calls), "should not fall back to inline sh -c when the script exists"


def test_observed_process_kv_falls_back_when_helper_script_absent(monkeypatch):
    """Default behavior for anyone who hasn't installed the helper script
    must be unchanged -- the inline sh -c fallback still works."""
    from llm_modelbench import ollama_service

    calls = []
    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        class R:
            returncode = 0
            stdout = "OLLAMA_KV_CACHE_TYPE=q4_0\n"
            stderr = ""
        return R()

    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = ollama_service.OllamaServiceController(
        "ollama-gpu0.service", run=fake_run,
        kv_read_helper_exists_fn=lambda p: False,
    )
    monkeypatch.setattr(controller, "_main_pid", lambda: 2119)

    value = controller.observed_process_kv()
    assert value == "q4_0"
    assert any("sh" in c and "-c" in c for c in calls), "expected fallback to inline sh -c"


def test_helper_script_default_checks_the_real_documented_path():
    """The default (no override) must point at the exact documented
    location so the install instructions and the code agree."""
    from llm_modelbench import ollama_service
    controller = ollama_service.OllamaServiceController("ollama-gpu0.service")
    assert str(controller.kv_read_helper_path) == "/usr/local/libexec/llmb-read-kv-env.sh"
