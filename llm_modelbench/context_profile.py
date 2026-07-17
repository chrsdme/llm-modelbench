"""Controlled long-context telemetry and behavior profile runner."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .hardware import ProbeTelemetry, detect_gpu


def _read_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _update_status(run_dir: Path, **updates: Any) -> None:
    path = Path(run_dir) / "status.json"
    status = _load_json(path) or {"run_id": Path(run_dir).name}
    status.update(updates)
    status["status_type"] = "context_profile"
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(path, status)


def _extract_json_object(text: str) -> Tuple[Optional[dict], Optional[str]]:
    raw = str(text or "").strip()
    try:
        value = json.loads(raw)
        return (value, None) if isinstance(value, dict) else (None, "response JSON is not an object")
    except Exception as first:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return None, f"no JSON object found: {first}"
        try:
            value = json.loads(match.group(0))
            return (value, None) if isinstance(value, dict) else (None, "embedded JSON is not an object")
        except Exception as second:
            return None, f"invalid JSON: {second}"


def _repetition_ratio(text: str) -> float:
    lines = [line.strip().lower() for line in str(text or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return 0.0
    return round(1.0 - (len(set(lines)) / len(lines)), 4)


def _build_behavior_prompt(target_ctx: int, chars_per_token: float) -> Tuple[str, Dict[str, str]]:
    """Build a synthetic, deterministic long-context operating probe.

    This is intentionally separate from canonical scoring. It checks basic
    long-context instruction retention, exact anchor recall, output structure,
    repetition and sustained decode speed. It does not certify agentic ability.
    """
    anchors = {
        "alpha": "ORCHID-7319",
        "beta": "HARBOR-4826",
        "gamma": "EMBER-9054",
    }
    instruction = (
        "\n\nFINAL INSTRUCTION: Return one JSON object only, with exactly these keys: "
        '"alpha", "beta", "gamma", "sequence", and "assessment". '
        "Copy the three anchor values exactly. Set sequence to [1, 2, 3, 4]. "
        "Write assessment as 120 to 180 words describing a safe four-stage software "
        "deployment workflow. Do not invent additional codes and do not use markdown."
    )
    target_chars = max(12000, int(float(target_ctx) * max(1.0, float(chars_per_token))))
    segments = []
    index = 0
    while sum(len(x) for x in segments) < target_chars - len(instruction) - 512:
        segments.append(
            f"Operational record {index:06d}: retain the ordered deployment notes, "
            "verify prerequisites, preserve rollback evidence, and avoid changing any "
            "unrelated system state. This record is synthetic benchmark filler.\n"
        )
        index += 1
    body = "".join(segments)
    inserts = [
        (int(len(body) * 0.10), f"\nANCHOR ALPHA = {anchors['alpha']}\n"),
        (int(len(body) * 0.50), f"\nANCHOR BETA = {anchors['beta']}\n"),
        (int(len(body) * 0.90), f"\nANCHOR GAMMA = {anchors['gamma']}\n"),
    ]
    offset = 0
    for pos, value in inserts:
        at = pos + offset
        body = body[:at] + value + body[at:]
        offset += len(value)
    return body + instruction, anchors


def run_behavior_probe(
    client: Any,
    *,
    model: str,
    target_ctx: int,
    chars_per_token: float,
    min_tps: float,
    critical_tps: float,
) -> Dict[str, Any]:
    prompt, anchors = _build_behavior_prompt(target_ctx, chars_per_token)
    telemetry = ProbeTelemetry(interval=0.25)
    telemetry.start()
    try:
        response = client.chat(
            model,
            prompt,
            num_predict=384,
            num_ctx=int(target_ctx) + 1024,
            think="off",
        )
    except Exception as exc:
        response = {"ok": False, "error": repr(exc)}
    finally:
        hardware = telemetry.stop()

    text = str(response.get("text") or "")
    payload, parse_error = _extract_json_object(text)
    exact = {
        key: bool(payload and str(payload.get(key) or "") == value)
        for key, value in anchors.items()
    }
    sequence_ok = bool(payload and payload.get("sequence") == [1, 2, 3, 4])
    assessment = str((payload or {}).get("assessment") or "")
    assessment_words = len(re.findall(r"\b\w+\b", assessment))
    assessment_length_ok = 120 <= assessment_words <= 180
    repeated = _repetition_ratio(text)
    response_suspect = bool(repeated > 0.35 or len(text.strip()) == 0)
    effective_ctx = int(response.get("prompt_eval_count") or 0)
    tps = response.get("tps")

    if not response.get("ok"):
        operating_status = "probe_error"
    elif effective_ctx < int(target_ctx):
        operating_status = "target_not_reached"
    elif not all(exact.values()) or not sequence_ok or not assessment_length_ok or response_suspect or parse_error:
        operating_status = "behavior_warning"
    elif isinstance(tps, (int, float)) and float(tps) < float(critical_tps):
        operating_status = "impractical_speed"
    elif isinstance(tps, (int, float)) and float(tps) < float(min_tps):
        operating_status = "slow"
    elif not isinstance(tps, (int, float)):
        operating_status = "verified_speed_unavailable"
    else:
        operating_status = "ready"

    result = {
        "schema_version": 1,
        "model": model,
        "target_ctx": int(target_ctx),
        "prompt_chars": len(prompt),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "effective_context_reached": effective_ctx >= int(target_ctx),
        "num_ctx_used": response.get("num_ctx"),
        "num_predict": response.get("num_predict", 384),
        "ok": bool(response.get("ok")),
        "error": response.get("error"),
        "http_status": response.get("http_status"),
        "http_reason": response.get("http_reason"),
        "http_error_body": response.get("http_error_body"),
        "json_parse_error": parse_error,
        "anchors_exact": exact,
        "all_anchors_exact": all(exact.values()),
        "sequence_ok": sequence_ok,
        "assessment_words": assessment_words,
        "assessment_length_ok": assessment_length_ok,
        "response_repetition_ratio": repeated,
        "response_suspect": response_suspect,
        "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "response_chars": len(text),
        "done_reason": response.get("done_reason"),
        "tps": response.get("tps"),
        "prompt_tps": response.get("prompt_tps"),
        "ttft_ms": response.get("ttft_ms"),
        "request_elapsed_seconds": response.get("request_elapsed_seconds", response.get("elapsed_seconds")),
        "server_total_duration_ms": response.get("server_total_duration_ms"),
        "server_load_duration_ms": response.get("server_load_duration_ms"),
        "server_prompt_eval_duration_ms": response.get("server_prompt_eval_duration_ms"),
        "server_eval_duration_ms": response.get("server_eval_duration_ms"),
        "operating_status": operating_status,
        "agentic_readiness": "not_assessed",
        "agentic_readiness_note": (
            "This synthetic probe checks recall, structure, drift and speed at context. "
            "It does not certify multi-step tool use or long-horizon agentic reliability."
        ),
        "telemetry": hardware,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    return result


def validate_context_telemetry(
    run_dir: Path,
    *,
    target_ctx: int = 64000,
    require_behavior_probe: bool = False,
) -> Dict[str, Any]:
    rows = _read_rows(Path(run_dir) / "raw_results.jsonl")
    needle = next((row for row in rows if row.get("task") == "needle"), None)
    behavior = _load_json(Path(run_dir) / "context_behavior_probe.json")
    result: Dict[str, Any] = {
        "schema_version": 2,
        "run_dir": str(run_dir),
        "target_ctx": int(target_ctx),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "passed": False,
        "critical_missing": [],
        "warnings": [],
        "target_probe": None,
        "behavior_probe": behavior,
    }
    if not needle:
        result["critical_missing"].append("needle row")
        return result
    successful = [p for p in needle.get("needle_attempted") or [] if p.get("found")]
    target_probe = next(
        (p for p in successful if int(p.get("prompt_tokens_actual") or p.get("num_ctx") or 0) >= int(target_ctx)),
        None,
    )
    if target_probe is None and successful:
        target_probe = max(successful, key=lambda p: int(p.get("prompt_tokens_actual") or p.get("num_ctx") or 0))
        result["warnings"].append("target context was not reached; validating the largest successful tier")
    if target_probe is None:
        result["critical_missing"].append("successful target context probe")
        return result

    result["target_probe"] = target_probe
    effective_ctx = int(target_probe.get("prompt_tokens_actual") or target_probe.get("num_ctx") or 0)
    if effective_ctx < int(target_ctx):
        result["critical_missing"].append(f"effective context >= {target_ctx}")

    required_any = {
        "host memory": ["ollama_pss_peak_mb", "ollama_rss_peak_mb", "ram_peak_mb"],
        "host memory delta": ["ollama_pss_delta_peak_mb", "ollama_rss_delta_peak_mb", "ram_delta_peak_mb"],
        "decode speed": ["tps", "server_eval_duration_ms"],
        "prefill speed": ["prompt_tps", "server_prompt_eval_duration_ms"],
    }
    required = [
        "elapsed_seconds",
        "vram_peak_mb",
        "offload_fraction",
        "request_elapsed_seconds",
        "telemetry_samples",
    ]
    for key in required:
        if target_probe.get(key) is None:
            result["critical_missing"].append(key)
    for label, keys in required_any.items():
        if not any(target_probe.get(key) is not None for key in keys):
            result["critical_missing"].append(label)

    optional = [
        "ttft_ms", "gpu_util_mean_pct", "gpu_util_peak_pct", "power_mean_w",
        "power_peak_w", "temp_peak_c", "cpu_util_mean_pct", "cpu_util_peak_pct",
        "ram_available_min_mb", "swap_delta_peak_mb", "model_host_bytes",
        "model_vram_bytes", "needle_response_exact", "needle_response_suspect",
    ]
    for key in optional:
        if target_probe.get(key) is None:
            result["warnings"].append(f"optional telemetry unavailable: {key}")

    if require_behavior_probe:
        if not behavior:
            result["critical_missing"].append("64k behavior probe")
        elif behavior.get("operating_status") in {"probe_error", "target_not_reached"}:
            result["critical_missing"].append(
                f"behavior probe: {behavior.get('operating_status')}"
            )
        elif behavior.get("operating_status") == "behavior_warning":
            result["warnings"].append("behavior probe detected recall, structure or repetition concerns")

    result["passed"] = not result["critical_missing"]
    result["model"] = needle.get("model")
    result["score"] = needle.get("score")
    result["coverage"] = needle.get("needle_coverage")
    result["max_verified_ctx"] = needle.get("max_verified_ctx")
    result["needle_operating_status"] = needle.get("needle_target_status")
    result["operating_status"] = (
        behavior.get("operating_status") if behavior else needle.get("needle_target_status")
    )
    result["agentic_readiness"] = "not_assessed"
    return result


def run_context_profile(
    client: Any,
    cfg: Any,
    *,
    model: str,
    run_dir: Path,
    rankings_dir: Optional[Path] = None,
    cards_dir: Optional[Path] = None,
    target_ctx: int = 64000,
    gpu_vram_gb: Optional[float] = None,
    emergency_headroom_gb: float = 0.25,
    max_spill_gb: float = 2.5,
    min_tps: float = 10.0,
    critical_tps: float = 3.0,
    live_ui: str = "compact",
    behavior_probe: bool = True,
    ranking_scope: str = "canonical",
) -> Dict[str, Any]:
    from . import rankings, report, runner
    from .model_cards import generate_model_cards

    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"diagnostic run directory already exists: {run_dir}; use a new --run-id to preserve prior evidence"
        )

    from .ranking_controls import write_run_scope
    write_run_scope(run_dir, scope=ranking_scope, rankings_dir=rankings_dir)

    gpu = detect_gpu()
    total_gpu = float(gpu_vram_gb) if gpu_vram_gb is not None else float(gpu.total_vram_gb or 0.0)
    if total_gpu <= 0:
        raise ValueError("GPU VRAM could not be detected; pass --gpu-vram-gb")
    cfg.vram_budget_gb = round(max(0.0, total_gpu - float(emergency_headroom_gb)) + float(max_spill_gb), 3)
    cfg.needle_max_ctx = int(target_ctx) + 2048
    cfg.long_context_target_ctx = int(target_ctx)
    cfg.long_context_min_tps = float(min_tps)
    cfg.long_context_critical_tps = float(critical_tps)
    cfg.needle_preflight_mode = "advisory"
    cfg.context_profile_mode = True
    cfg.context_profile_behavior_probe = bool(behavior_probe)
    cfg.samples = 1
    cfg.fingerprint = False
    cfg.dump_raw = True

    runner.run(
        client, cfg,
        level="full", out_dir=run_dir,
        include=None, exclude=None, skip_offload=False,
        categories=None, task_ids=["needle"], task_regex=None,
        family_base_only=False, context_aliases_only=False, context_only=False,
        resume=False, judge_mode="off", dump_subjective=False, dump_raw=True,
        status_interval=1.0, live_ui=live_ui, sample_mode="smart",
        fingerprint_enabled=False, selected_models=[model],
        capability_profiles=None, auto_probe=False,
        row_metadata_by_task={"needle": {
            "context_profile_run": True,
            "context_profile_target_ctx": int(target_ctx),
            "context_profile_vram_budget_gb": cfg.vram_budget_gb,
            "context_profile_max_spill_gb": float(max_spill_gb),
            "context_profile_emergency_headroom_gb": float(emergency_headroom_gb),
            "context_profile_preflight_mode": "advisory",
        }},
    )

    behavior_result = None
    if behavior_probe:
        rows = _read_rows(run_dir / "raw_results.jsonl")
        needle = next((row for row in rows if row.get("task") == "needle"), None) or {}
        successful = [p for p in needle.get("needle_attempted") or [] if p.get("found")]
        cpt = 6.85
        if successful:
            best = max(successful, key=lambda p: int(p.get("prompt_tokens_actual") or p.get("num_ctx") or 0))
            if isinstance(best.get("needle_chars_per_token"), (int, float)):
                cpt = float(best["needle_chars_per_token"])
        _update_status(
            run_dir,
            profile_phase="behavior_probe_running",
            current={
                "model": model,
                "task": "context_behavior_probe",
                "state": "task_running",
                "probe_state": "running synthetic 64k behavior probe",
                "probe_size": int(target_ctx),
                "probe_num_ctx": int(target_ctx) + 1024,
            },
        )
        behavior_result = run_behavior_probe(
            client,
            model=model,
            target_ctx=int(target_ctx),
            chars_per_token=cpt,
            min_tps=float(min_tps),
            critical_tps=float(critical_tps),
        )
        _atomic_json(run_dir / "context_behavior_probe.json", behavior_result)
        _update_status(
            run_dir,
            profile_phase="behavior_probe_complete",
            behavior_probe=behavior_result,
            current={
                "model": model,
                "task": "context_behavior_probe",
                "state": "task_finished",
                "probe_state": behavior_result.get("operating_status"),
                "probe_size": int(target_ctx),
                "probe_num_ctx": behavior_result.get("prompt_eval_count"),
                "probe_tps": behavior_result.get("tps"),
                "probe_prompt_tps": behavior_result.get("prompt_tps"),
            },
        )

    report.build(run_dir, cfg)
    validation = validate_context_telemetry(
        run_dir,
        target_ctx=target_ctx,
        require_behavior_probe=bool(behavior_probe),
    )
    validation_path = run_dir / "telemetry_validation.json"
    _atomic_json(validation_path, validation)
    _update_status(
        run_dir,
        profile_phase="complete" if validation.get("passed") else "validation_failed",
        telemetry_validation=validation,
        profile_outcome="COMPLETE" if validation.get("passed") else "FAILED_VALIDATION",
    )

    ranking_result = None
    cards_result = None
    if rankings_dir is not None:
        template_path = Path(__file__).parent / "rankings_template.html"
        template = template_path.read_text() if template_path.exists() else None
        ranking_result = rankings.write_rankings(
            run_dir.parent, Path(rankings_dir), html_template=template, force_rescan=True,
            include_separate=(ranking_scope == "separate"),
            only_run_ids=([run_dir.name] if ranking_scope == "separate" else None),
        )
        if cards_dir is not None:
            cards_result = generate_model_cards(
                Path(rankings_dir), Path(cards_dir), runs_dir=run_dir.parent,
            )

    return {
        "run_dir": str(run_dir),
        "model": model,
        "target_ctx": int(target_ctx),
        "vram_budget_gb": cfg.vram_budget_gb,
        "behavior_probe": behavior_result,
        "telemetry_validation": validation,
        "telemetry_validation_path": str(validation_path),
        "rankings": ranking_result,
        "model_cards": cards_result,
    }
