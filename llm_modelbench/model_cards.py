"""Generate durable per-model operating cards from master rankings evidence.

Card assembly itself (identity/evidence-trust/human-validation fields) lives
in ``model_card.py`` (Anvil Stage 3.1's ``ModelCard``) -- this module derives
the quality/limits/long_context sections from a master-summary row (logic
unchanged from before Stage 3.1), then builds and renders a ``ModelCard``
from them. ``build_card()``'s return shape is a deliberate golden-file
contract: unchanged pre-existing keys plus the two new Stage 3.1 fields --
see ``tests/test_model_card.py``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .model_card import build_model_card, model_card_to_dict, render_model_card_markdown


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

    quality = {
        "status": model.get("quality_status"),
        "overall_mean_score": model.get("overall_mean_score"),
        "overall_rank": model.get("overall_rank"),
        "tie_band": model.get("tie_band"),
        "coverage_ratio": model.get("coverage_ratio"),
        "completion_rate": model.get("completion_rate"),
        "reasons": model.get("quality_status_reasons") or [],
    }
    limits = {
        "capability_limited": bool(model.get("capability_limited")),
        "capability_unavailable_tasks": model.get("capability_unavailable_tasks") or [],
        "capability_measured_failure": bool(model.get("capability_measured_failure")),
        "capability_measured_failure_tasks": model.get("capability_measured_failure_tasks") or [],
        "recovery_limited": bool(model.get("recovery_limited")),
        "recovery_exhausted_tasks": model.get("recovery_exhausted_tasks") or [],
        "think_ineffective_tasks": model.get("think_ineffective_tasks") or [],
    }
    long_context = {
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
    }
    card = build_model_card(
        model,
        quality=quality,
        limits=limits,
        long_context=long_context,
        kv_compatibility=kv_compatibility,
        evidence_warnings=warnings,
    )
    return model_card_to_dict(card)


def _render_markdown(card: Dict[str, Any]) -> str:
    return render_model_card_markdown(card)


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
