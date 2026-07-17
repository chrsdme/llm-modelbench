"""Generate durable per-model operating cards from master rankings evidence."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("._-")
    return text[:120] or "model"


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _gb_from_mb(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value) / 1024:.2f} GB"


def _merge_kv_compatibility(runs_dir: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    if runs_dir is None or not Path(runs_dir).exists():
        return merged
    for run_dir in Path(runs_dir).iterdir():
        if not run_dir.is_dir():
            continue
        payload = _read_json(run_dir / "kv_compatibility.json")
        if not isinstance(payload, dict):
            continue
        for model, entry in payload.items():
            if not isinstance(entry, dict):
                continue
            existing = merged.setdefault(str(model), {"kv_modes": {}, "history": []})
            existing["kv_modes"].update(entry.get("kv_modes") or {})
            existing["history"].extend(entry.get("history") or [])
            for key in (
                "model_digest", "runtime_identity", "preferred_kv_type",
                "avoid_quantized_kv", "current_kv_supported",
            ):
                if key in entry:
                    existing[key] = entry[key]
    return merged




def _merge_behavior_profiles(runs_dir: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    if runs_dir is None or not Path(runs_dir).exists():
        return profiles
    for run_dir in Path(runs_dir).iterdir():
        if not run_dir.is_dir():
            continue
        payload = _read_json(run_dir / "context_behavior_probe.json")
        if not isinstance(payload, dict) or not payload.get("model"):
            continue
        model = str(payload["model"])
        existing = profiles.get(model)
        if existing is None or str(payload.get("validated_at") or "") >= str(existing.get("validated_at") or ""):
            profiles[model] = payload
    return profiles

def build_card(
    model: Dict[str, Any],
    *,
    kv_compatibility: Optional[Dict[str, Any]] = None,
    behavior_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profile = model.get("long_context_profile") or {}
    depths = list(profile.get("depths") or [])
    target_ctx = profile.get("target_ctx") or 64000
    target = None
    successful = [d for d in depths if d.get("found")]
    for depth in successful:
        ctx = depth.get("num_ctx") or 0
        if isinstance(ctx, (int, float)) and int(ctx) >= int(target_ctx):
            target = depth
            break
    if target is None and successful:
        target = max(successful, key=lambda d: int(d.get("num_ctx") or 0))

    host_delta = None
    if target:
        for key in ("ollama_pss_delta_peak_mb", "ollama_rss_delta_peak_mb", "ram_delta_peak_mb"):
            if isinstance(target.get(key), (int, float)):
                host_delta = target.get(key)
                break

    warnings: List[str] = []
    if not profile:
        warnings.append("No long-context operating evidence is available.")
    elif not target:
        warnings.append("No successful context tier is available for operating-profile classification.")
    if profile and profile.get("target_status") in {None, "verified_speed_unavailable"}:
        warnings.append("Context was measured without authoritative target decode-speed evidence.")
    if profile and profile.get("behavior_suspect"):
        warnings.append("At least one successful context tier had suspicious response-shape evidence.")
    if target and target.get("ram_peak_mb") is None and target.get("ollama_pss_peak_mb") is None:
        warnings.append("Host RAM telemetry is incomplete for the target context tier.")
    if target and target.get("elapsed_seconds") is None:
        warnings.append("Elapsed-time telemetry is incomplete for the target context tier.")

    return {
        "schema_version": 1,
        "model": model.get("display_name"),
        "digest": model.get("digest"),
        "identity": {
            "names_seen": model.get("names_seen") or [],
            "class": model.get("class"),
            "families": model.get("families") or [],
            "parameter_size": model.get("parameter_size"),
            "quantization_level": model.get("quantization_level"),
            "architecture_family": model.get("architecture_family"),
            "model_size_bytes": model.get("model_size_bytes"),
            "size_gb": model.get("size_gb"),
        },
        "quality": {
            "status": model.get("quality_status"),
            "overall_mean_score": model.get("overall_mean_score"),
            "overall_rank": model.get("overall_rank"),
            "tie_band": model.get("tie_band"),
            "coverage_ratio": model.get("coverage_ratio"),
            "completion_rate": model.get("completion_rate"),
            "reasons": model.get("quality_status_reasons") or [],
        },
        "limits": {
            "capability_limited": bool(model.get("capability_limited")),
            "capability_unavailable_tasks": model.get("capability_unavailable_tasks") or [],
            "capability_measured_failure": bool(model.get("capability_measured_failure")),
            "capability_measured_failure_tasks": model.get("capability_measured_failure_tasks") or [],
            "recovery_limited": bool(model.get("recovery_limited")),
            "recovery_exhausted_tasks": model.get("recovery_exhausted_tasks") or [],
            "think_ineffective_tasks": model.get("think_ineffective_tasks") or [],
        },
        "long_context": {
            "target_ctx": target_ctx,
            "behavior_probe": behavior_profile or {},
            "agentic_readiness": (behavior_profile or {}).get("agentic_readiness", "not_assessed"),
            "target_status": profile.get("target_status"),
            "max_verified_ctx": profile.get("max_verified_ctx"),
            "coverage": profile.get("coverage"),
            "score": profile.get("score"),
            "target_tps": profile.get("target_tps"),
            "target_prompt_tps": profile.get("target_prompt_tps"),
            "target_elapsed_seconds": profile.get("target_elapsed_seconds"),
            "target_offload_fraction": profile.get("target_offload_fraction"),
            "target_host_delta_mb": host_delta,
            "target_swap_delta_mb": profile.get("target_swap_delta_mb"),
            "min_tps": profile.get("min_tps"),
            "median_tps": profile.get("median_tps"),
            "max_offload_fraction": profile.get("max_offload_fraction"),
            "behavior_suspect": profile.get("behavior_suspect"),
            "slow_depths": profile.get("slow_depths") or [],
            "critical_slow_depths": profile.get("critical_slow_depths") or [],
            "depths": depths,
        },
        "kv_compatibility": kv_compatibility or {},
        "evidence_warnings": warnings,
    }


def _render_markdown(card: Dict[str, Any]) -> str:
    identity = card["identity"]
    quality = card["quality"]
    limits = card["limits"]
    context = card["long_context"]
    kv = card.get("kv_compatibility") or {}
    lines = [
        f"# {card.get('model')}",
        "",
        "## Identity",
        "",
        f"- Digest: `{card.get('digest') or 'n/a'}`",
        f"- Class: `{identity.get('class') or 'n/a'}`",
        f"- Families: {', '.join(identity.get('families') or []) or 'n/a'}",
        f"- Parameters: `{identity.get('parameter_size') or 'n/a'}`",
        f"- Quantization: `{identity.get('quantization_level') or 'n/a'}`",
        f"- Architecture: `{identity.get('architecture_family') or 'n/a'}`",
        f"- Stored size: `{_fmt(identity.get('size_gb'))} GB`",
        "",
        "## Quality and applicability",
        "",
        f"- Status: **{quality.get('status') or 'unknown'}**",
        f"- Overall score: **{_fmt(quality.get('overall_mean_score'))}**",
        f"- Rank / tie band: `{_fmt(quality.get('overall_rank'))}` / `{_fmt(quality.get('tie_band'))}`",
        f"- Coverage: `{_pct(quality.get('coverage_ratio'))}`",
        f"- Capability limited: `{limits.get('capability_limited')}`",
        f"- Measured capability failure: `{limits.get('capability_measured_failure')}`",
        f"- Recovery limited: `{limits.get('recovery_limited')}`",
    ]
    if limits.get("capability_unavailable_tasks"):
        lines.append("- Unavailable tasks: " + ", ".join(limits["capability_unavailable_tasks"]))
    if limits.get("capability_measured_failure_tasks"):
        lines.append("- Measured zero-quality capability tasks: " + ", ".join(limits["capability_measured_failure_tasks"]))
    if limits.get("recovery_exhausted_tasks"):
        lines.append("- Recovery-exhausted tasks: " + ", ".join(limits["recovery_exhausted_tasks"]))

    lines += [
        "",
        "## Long-context operating profile",
        "",
        f"- Target context: `{_fmt(context.get('target_ctx'))}`",
        f"- Operating status: **{context.get('target_status') or 'not_verified'}**",
        f"- Maximum verified effective context: `{_fmt(context.get('max_verified_ctx'))}`",
        f"- Context quality score / coverage: `{_fmt(context.get('score'))}` / `{_pct(context.get('coverage'))}`",
        f"- Target prompt / decode speed: `{_fmt(context.get('target_prompt_tps'))}` / `{_fmt(context.get('target_tps'))}` tok/s",
        f"- Target elapsed time: `{_fmt(context.get('target_elapsed_seconds'))}` seconds",
        f"- Target offload: `{_pct(context.get('target_offload_fraction'))}`",
        f"- Target host-memory delta: `{_gb_from_mb(context.get('target_host_delta_mb'))}`",
        f"- Target swap delta: `{_gb_from_mb(context.get('target_swap_delta_mb'))}`",
        f"- Minimum / median decode speed: `{_fmt(context.get('min_tps'))}` / `{_fmt(context.get('median_tps'))}` tok/s",
        "",
        "| Tier | num_ctx | Result | Prompt tok/s | Decode tok/s | Elapsed | VRAM peak | Host delta | Swap | Offload | Output |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for depth in context.get("depths") or []:
        host = depth.get("ollama_pss_delta_peak_mb")
        if host is None:
            host = depth.get("ollama_rss_delta_peak_mb")
        if host is None:
            host = depth.get("ram_delta_peak_mb")
        result = "pass" if depth.get("found") else (depth.get("error_kind") or "not-run")
        output = "suspect" if depth.get("response_suspect") else ("exact" if depth.get("response_exact") else "n/a")
        lines.append(
            f"| {_fmt(depth.get('size'))} | {_fmt(depth.get('num_ctx'))} | {result} | "
            f"{_fmt(depth.get('prompt_tps'))} | {_fmt(depth.get('tps'))} | "
            f"{_fmt(depth.get('elapsed_seconds'))}s | {_gb_from_mb(depth.get('vram_peak_mb'))} | "
            f"{_gb_from_mb(host)} | {_gb_from_mb(depth.get('swap_delta_peak_mb'))} | "
            f"{_pct(depth.get('offload_fraction'))} | {output} |"
        )

    behavior = context.get("behavior_probe") or {}
    lines += ["", "## 64k behavior probe", ""]
    if behavior:
        lines.extend([
            f"- Operating status: **{behavior.get('operating_status') or 'unknown'}**",
            f"- Effective prompt context: `{_fmt(behavior.get('prompt_eval_count'))}`",
            f"- Prompt / decode speed: `{_fmt(behavior.get('prompt_tps'))}` / `{_fmt(behavior.get('tps'))}` tok/s",
            f"- Exact anchors: `{behavior.get('all_anchors_exact')}`",
            f"- Ordered sequence retained: `{behavior.get('sequence_ok')}`",
            f"- Response repetition ratio: `{_fmt(behavior.get('response_repetition_ratio'))}`",
            f"- Agentic readiness: `{behavior.get('agentic_readiness') or 'not_assessed'}`",
            f"- Scope note: {behavior.get('agentic_readiness_note') or 'Long-horizon agentic reliability was not assessed.'}",
        ])
    else:
        lines.append("No synthetic long-context behavior probe is available.")

    lines += ["", "## KV compatibility", ""]
    if kv:
        lines.append(f"- Preferred KV mode: `{kv.get('preferred_kv_type') or 'n/a'}`")
        lines.append(f"- Current KV supported: `{bool(kv.get('current_kv_supported'))}`")
        lines.append(f"- Avoid quantized KV: `{bool(kv.get('avoid_quantized_kv'))}`")
        for mode, entry in sorted((kv.get("kv_modes") or {}).items()):
            lines.append(
                f"- `{mode}`: `{entry.get('status') or 'unknown'}`"
                + (f" ({', '.join(entry.get('error_kinds') or [])})" if entry.get("error_kinds") else "")
            )
    else:
        lines.append("No build-scoped KV compatibility record is available.")

    if card.get("evidence_warnings"):
        lines += ["", "## Evidence warnings", ""]
        lines.extend(f"- {warning}" for warning in card["evidence_warnings"])

    lines += ["", "## Status reasons", ""]
    lines.extend(f"- {reason}" for reason in quality.get("reasons") or ["No reasons recorded."])
    return "\n".join(lines) + "\n"


def generate_model_cards(
    rankings_dir: Path,
    out_dir: Path,
    *,
    runs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    rankings_dir = Path(rankings_dir)
    out_dir = Path(out_dir)
    summary = _read_json(rankings_dir / "master_summary.json")
    if not isinstance(summary, list):
        raise ValueError(f"cannot read rankings summary: {rankings_dir / 'master_summary.json'}")
    runs_path = Path(runs_dir) if runs_dir else None
    kv_by_model = _merge_kv_compatibility(runs_path)
    behavior_by_model = _merge_behavior_profiles(runs_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_lines = ["# LLM ModelBench operating cards", ""]
    cards: List[Dict[str, Any]] = []
    for model in sorted(summary, key=lambda item: str(item.get("display_name") or "").lower()):
        name = str(model.get("display_name") or model.get("digest") or "model")
        kv = kv_by_model.get(name) or {}
        if not kv:
            for alias in model.get("names_seen") or []:
                if alias in kv_by_model:
                    kv = kv_by_model[alias]
                    break
        behavior = behavior_by_model.get(name) or {}
        if not behavior:
            for alias in model.get("names_seen") or []:
                if alias in behavior_by_model:
                    behavior = behavior_by_model[alias]
                    break
        card = build_card(model, kv_compatibility=kv, behavior_profile=behavior)
        cards.append(card)
        slug = _slug(name)
        json_path = out_dir / f"{slug}.json"
        md_path = out_dir / f"{slug}.md"
        json_path.write_text(json.dumps(card, indent=2, sort_keys=True))
        md_path.write_text(_render_markdown(card))
        status = card["long_context"].get("target_status") or "not_verified"
        index_lines.append(
            f"- [{name}]({md_path.name}) | quality `{card['quality'].get('status')}` | "
            f"64k `{status}` | score `{_fmt(card['quality'].get('overall_mean_score'))}`"
        )
    index_path = out_dir / "README.md"
    index_path.write_text("\n".join(index_lines) + "\n")
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "models": len(cards),
        "complete": sum(1 for card in cards if card["quality"].get("status") == "complete"),
        "context_ready": sum(1 for card in cards if card["long_context"].get("target_status") == "ready"),
        "context_slow": sum(1 for card in cards if card["long_context"].get("target_status") == "slow"),
        "context_impractical": sum(1 for card in cards if card["long_context"].get("target_status") == "impractical_speed"),
        "behavior_ready": sum(1 for card in cards if (card["long_context"].get("behavior_probe") or {}).get("operating_status") == "ready"),
        "agentic_assessed": sum(1 for card in cards if card["long_context"].get("agentic_readiness") != "not_assessed"),
    }, indent=2, sort_keys=True))
    return {
        "models": len(cards),
        "out_dir": str(out_dir),
        "index_path": str(index_path),
        "manifest_path": str(manifest_path),
    }
