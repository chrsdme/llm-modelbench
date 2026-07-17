"""Deterministic repair-watcher replay fixtures.

The fixtures exercise the exact on-disk contract consumed by ``llmb-watch``.
They never call Ollama, touch systemd, mutate rankings, or create benchmark
rows.  They are safe for visual regression work and CI.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def available_scenarios() -> List[str]:
    return [
        "capability-repair",
        "needle-current",
        "kv-cascade",
        "interrupted-child",
        "failed-child",
    ]


def _fixture_hardware() -> Dict[str, Any]:
    return {
        "gpu_name": "SIMULATED RTX 5060 Ti",
        "gpu_temp_c": 64.0,
        "gpu_power_w": 118.0,
        "gpu_util_pct": 92.0,
        "vram_used_mb": 15120.0,
        "vram_total_mb": 16311.0,
        "vram_used_pct": 92.7,
        "cpu_usage_pct": 47.0,
        "cpu_temp_c": 58.0,
        "ram_used_mb": 16420.0,
        "ram_total_mb": 30976.0,
        "ram_used_pct": 53.0,
        "swap_used_mb": 0.0,
    }


def _child_status(
    run_id: str,
    *,
    model: str,
    task: str,
    state: str,
    task_index: int = 1,
    tasks_total: int = 1,
    probe_index: Optional[int] = None,
    probe_total: Optional[int] = None,
    probe_size: Optional[int] = None,
    probe_num_ctx: Optional[int] = None,
    probe_state: Optional[str] = None,
    prompt_tps: Optional[float] = None,
    tps: Optional[float] = None,
    elapsed_seconds: float = 0.0,
    models_done: int = 0,
    last_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    current: Dict[str, Any] = {
        "model": model,
        "model_index": 1,
        "class": "vision" if "InternVL" in model else "coding",
        "size_gb": 4.68 if "InternVL" in model else 8.91,
        "offload_fraction": 0.0 if probe_size in (None, 4000, 16000) else 0.42,
        "context_length": probe_num_ctx or 32768,
        "task": task,
        "task_index": task_index,
        "tasks_total": tasks_total,
        "sample_index": 1,
        "samples_for_task": 1,
        "state": state,
    }
    if probe_index is not None:
        current.update({
            "probe_index": probe_index,
            "probe_total": probe_total,
            "probe_size": probe_size,
            "probe_num_ctx": probe_num_ctx,
            "probe_state": probe_state,
            "probe_prompt_tps": prompt_tps,
            "probe_tps": tps,
        })
    return {
        "run_id": run_id,
        "level": "full",
        "samples": 1,
        "sample_mode": "smart",
        "models_done": models_done,
        "models_total": 1,
        "models_skipped": 60,
        "tasks_done": models_done,
        "tasks_total": 1,
        "samples_done": models_done,
        "samples_total": 1,
        "elapsed_seconds": elapsed_seconds,
        "current": current,
        "last_result": last_result or {},
        "hardware_config": {
            "gpu_name": "SIMULATED RTX 5060 Ti",
            "gpu_pause_temp_c": 85,
            "gpu_resume_temp_c": 75,
        },
        "updated_at": _now(),
    }


def _base_parent(plan_id: str, campaign_id: str, *, actions_total: int = 1) -> Dict[str, Any]:
    return {
        "status_type": "repair",
        "plan_id": plan_id,
        "run_id": campaign_id,
        "phase": "planning",
        "actions_total": actions_total,
        "actions_completed": 0,
        "recovered": 0,
        "terminal_failures": 0,
        "unresolved": 0,
        "errors": 0,
        "simulation": True,
        "simulated_hardware": _fixture_hardware(),
        "updated_at": _now(),
    }


def _scenario_steps(scenario: str, plan_id: str, campaign_id: str, child_id: str) -> List[Dict[str, Any]]:
    if scenario not in available_scenarios():
        raise ValueError(f"unknown repair-watch scenario {scenario!r}")

    if scenario == "capability-repair":
        model = "hf.co/atahmih/InternVL3-8B-Q4_K_M-GGUF:latest"
        task = "fim_suffix_assertion"
        return [
            {"parent": {"phase": "planning", "current_model": model, "current_task": task,
                        "current_action_index": 1, "current_action_id": "sim-capability-1"}},
            {"parent": {"phase": "probing_capability", "current_model": model, "current_task": task,
                        "current_family": "insert", "current_action_index": 1,
                        "current_action_id": "sim-capability-1", "probe_state": "running",
                        "probe_detail": "task-equivalent suffix insertion probe"}},
            {"parent": {"phase": "running_action", "current_model": model, "current_task": task,
                        "current_family": "insert", "current_child_run": child_id,
                        "current_action_index": 1, "current_action_id": "sim-capability-1"},
             "child": _child_status(child_id, model=model, task=task, state="model_loading", elapsed_seconds=3.2)},
            {"parent": {"phase": "running_action", "current_model": model, "current_task": task,
                        "current_family": "insert", "current_child_run": child_id,
                        "current_action_index": 1, "current_action_id": "sim-capability-1",
                        "probe_state": "responded_contract_failed",
                        "probe_detail": "endpoint responded; scored task is authoritative"},
             "child": _child_status(
                 child_id, model=model, task=task, state="task_running", elapsed_seconds=8.5,
                 tps=119.33, prompt_tps=445.2,
                 last_result={"task": task, "score": None, "tps": 119.33, "prompt_tps": 445.2},
             )},
            {"parent": {"phase": "refreshing_rankings", "current_model": model, "current_task": task,
                        "current_child_run": child_id, "current_action_index": 1,
                        "current_action_id": "sim-capability-1", "actions_completed": 1,
                        "terminal_failures": 1},
             "child": _child_status(
                 child_id, model=model, task=task, state="complete", elapsed_seconds=10.1,
                 models_done=1,
                 last_result={"task": task, "score": 0.0, "tps": 119.33,
                              "reason": "ERROR_EMPTY_OUTPUT: FIM returned no insertion"},
             )},
            {"parent": {"phase": "complete", "actions_completed": 1, "terminal_failures": 1,
                        "outcome": "COMPLETE", "current_model": model, "current_task": task,
                        "current_child_run": child_id}},
        ]

    if scenario == "needle-current":
        model = "deepseek-coder-v2:16b"
        task = "needle"
        steps: List[Dict[str, Any]] = [
            {"parent": {"phase": "planning", "current_model": model, "current_task": task,
                        "current_action_index": 1, "current_action_id": "sim-needle-1",
                        "requested_kv_type": "current"}},
        ]
        probes = [
            (1, 4000, 4320, 628.4, 42.1, 10176.0, 0.0),
            (2, 16000, 16172, 502.2, 28.8, 13754.0, 0.0),
            (3, 32000, 31971, 385.0, 18.4, 15278.0, 0.364),
            (4, 65536, 65093, 241.7, 11.8, 15526.0, 0.523),
        ]
        for index, size, ctx, prefill, decode, vram, offload in probes:
            hw = _fixture_hardware()
            hw.update({"vram_used_mb": vram, "vram_used_pct": round(vram / hw["vram_total_mb"] * 100, 1)})
            child = _child_status(
                child_id, model=model, task=task, state="task_running",
                probe_index=index, probe_total=4, probe_size=size, probe_num_ctx=ctx,
                probe_state="running", prompt_tps=prefill, tps=decode,
                elapsed_seconds=float(index * 15),
            )
            child["current"]["offload_fraction"] = offload
            steps.append({
                "parent": {"phase": "running_current_kv_action", "requested_kv_type": "current",
                           "current_model": model, "current_task": task, "current_child_run": child_id,
                           "current_action_index": 1, "current_action_id": "sim-needle-1",
                           "simulated_hardware": hw},
                "child": child,
            })
        steps.append({
            "parent": {"phase": "complete", "actions_completed": 1, "recovered": 1,
                       "outcome": "COMPLETE", "requested_kv_type": "current",
                       "current_model": model, "current_task": task, "current_child_run": child_id},
            "child": _child_status(
                child_id, model=model, task=task, state="complete", models_done=1,
                elapsed_seconds=66.0,
                last_result={"task": task, "score": 100.0, "tps": 11.8,
                             "prompt_tps": 241.7, "reason": "65k verified"},
            ),
        })
        return steps

    if scenario == "kv-cascade":
        model = "quantized-kv-fixture:latest"
        task = "needle"
        phases = [
            ("planning", None, None, False),
            ("running_current_kv_action", "current", None, False),
            ("discovering_service", "q8_0", None, False),
            ("restarting_q8", "q8_0", None, False),
            ("verifying_q8", "q8_0", "q8_0", True),
            ("running_q8_action", "q8_0", "q8_0", True),
            ("q8_complete", "q8_0", "q8_0", True),
            ("restoring", None, None, True),
            ("complete", None, None, True),
        ]
        out = []
        for idx, (phase, req, obs, verified) in enumerate(phases):
            parent = {
                "phase": phase,
                "service_unit": "ollama-gpu0.service",
                "requested_kv_type": req,
                "effective_kv_type": obs,
                "observed_kv_type": obs,
                "last_verified_kv_type": "q8_0" if idx >= 4 else None,
                "service_verified": verified,
                "current_model": model,
                "current_task": task,
                "current_action_index": 1,
                "current_action_id": "sim-kv-1",
                "current_child_run": child_id if idx >= 1 else None,
                "last_child_run": child_id if idx >= 1 else None,
            }
            if phase in {"q8_complete", "restoring", "complete"}:
                parent.update({"actions_completed": 1, "recovered": 1})
            if phase == "restoring":
                parent.update({"restoring_original_service_state": True})
            if phase == "complete":
                parent.update({"outcome": "COMPLETE", "restored_original_service_state": True})
            step: Dict[str, Any] = {"parent": parent}
            if phase in {"running_current_kv_action", "running_q8_action"}:
                step["child"] = _child_status(
                    child_id, model=model, task=task, state="task_running",
                    probe_index=2, probe_total=4, probe_size=16000, probe_num_ctx=16344,
                    probe_state="running", prompt_tps=300.0, tps=20.0,
                    elapsed_seconds=float(idx * 4),
                )
            elif phase in {"q8_complete", "restoring", "complete"}:
                step["child"] = _child_status(
                    child_id, model=model, task=task, state="complete",
                    models_done=1, elapsed_seconds=float(idx * 4),
                    last_result={"task": task, "score": 100.0, "tps": 20.0,
                                 "prompt_tps": 300.0, "reason": "simulated recovery"},
                )
            out.append(step)
        return out

    model = "fixture-model:latest"
    task = "needle"
    terminal_phase = "failed" if scenario == "failed-child" else "partial"
    terminal_state = "error" if scenario == "failed-child" else "interrupted"
    return [
        {"parent": {"phase": "planning", "current_model": model, "current_task": task,
                    "current_action_index": 1, "current_action_id": "sim-failure-1"}},
        {"parent": {"phase": "running_action", "current_model": model, "current_task": task,
                    "current_child_run": child_id, "current_action_index": 1,
                    "current_action_id": "sim-failure-1"},
         "child": _child_status(child_id, model=model, task=task, state="task_running",
                                elapsed_seconds=5.0)},
        {"parent": {"phase": terminal_phase, "actions_completed": 1, "unresolved": 1,
                    "errors": 1 if scenario == "failed-child" else 0,
                    "outcome": "FAILED" if scenario == "failed-child" else "PARTIAL",
                    "current_model": model, "current_task": task, "current_child_run": child_id,
                    "error": "simulated child failure" if scenario == "failed-child" else None},
         "child": _child_status(child_id, model=model, task=task, state=terminal_state,
                                elapsed_seconds=6.0)},
    ]


def replay_repair_watch(
    runs_dir: Path,
    *,
    scenario: str = "capability-repair",
    speed: float = 1.0,
    run_id: Optional[str] = None,
    render: bool = True,
    screen: str = "auto",
    keep: bool = True,
    sleep_fn: Callable[[float], None] = time.sleep,
    stream: Any = None,
) -> Dict[str, Any]:
    """Write and optionally render a deterministic repair status sequence."""
    from . import watch

    runs_dir = Path(runs_dir)
    stream = stream or sys.stdout
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_id = run_id or f"sim_repair_watch_{scenario.replace('-', '_')}_{stamp}"
    digest = hashlib.sha256(f"{scenario}:{campaign_id}".encode()).hexdigest()[:16]
    plan_id = f"sim{digest[:13]}"
    child_id = f"{campaign_id}_child"
    campaign_dir = runs_dir / campaign_id
    child_dir = runs_dir / child_id
    legacy_status = runs_dir / f"repair_status_{plan_id}.json"
    plan_path = runs_dir / f"repair_plan_{plan_id}.json"

    base = _base_parent(plan_id, campaign_id)
    steps = _scenario_steps(scenario, plan_id, campaign_id, child_id)
    _atomic_json(plan_path, {
        "schema_version": 1,
        "repair_policy_version": "simulation",
        "plan_id": plan_id,
        "created_at": _now(),
        "simulation": True,
        "scenario": scenario,
        "actions": [],
    })

    use_clear = bool(render and screen != "scroll" and getattr(stream, "isatty", lambda: False)())
    speed = max(0.0, float(speed))
    rendered_frames: List[str] = []
    parent_state = dict(base)
    for index, step in enumerate(steps, start=1):
        parent_state.update(step.get("parent") or {})
        parent = dict(parent_state)
        parent.update({
            "simulation_step": index,
            "simulation_steps": len(steps),
            "scenario": scenario,
            "updated_at": _now(),
        })
        if parent.get("simulated_hardware") is None:
            parent["simulated_hardware"] = _fixture_hardware()
        _atomic_json(campaign_dir / "status.json", parent)
        _atomic_json(legacy_status, parent)

        child_payload = step.get("child")
        if child_payload is not None:
            _atomic_json(child_dir / "repair_link.json", {
                "parent_status_type": "repair",
                "repair_plan_id": plan_id,
                "repair_action_id": parent.get("current_action_id"),
                "repair_phase": parent.get("phase"),
                "simulation": True,
            })
            child_payload = dict(child_payload)
            child_payload["updated_at"] = _now()
            _atomic_json(child_dir / "status.json", child_payload)

        if render:
            status = watch._load_repair_status_for_run(campaign_dir) or parent
            hardware = status.get("simulated_hardware") or _fixture_hardware()
            frame = watch.render_repair(status, hardware)
            rendered_frames.append(frame)
            width, height = watch._terminal_size()
            frame = watch._fit_screen(frame, width, height)
            try:
                if use_clear:
                    stream.write("\033[H\033[J")
                stream.write(frame + "\n")
                stream.flush()
            except BrokenPipeError:
                render = False
        if index < len(steps) and speed > 0:
            sleep_fn(speed)

    result = {
        "scenario": scenario,
        "plan_id": plan_id,
        "campaign_run_id": campaign_id,
        "child_run_id": child_id,
        "campaign_dir": str(campaign_dir),
        "child_dir": str(child_dir),
        "plan_path": str(plan_path),
        "steps": len(steps),
        "final_phase": steps[-1]["parent"].get("phase"),
        "rendered_frames": len(rendered_frames),
        "kept": bool(keep),
    }
    if not keep:
        import shutil
        shutil.rmtree(campaign_dir, ignore_errors=True)
        shutil.rmtree(child_dir, ignore_errors=True)
        for path in (legacy_status, plan_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return result
