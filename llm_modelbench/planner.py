"""Run planning and preflight presentation for LLM ModelBench.

The planner is the single source of truth for model selection, capability
routing, task selection, sample counts, and the exact proposal shown before a
run. ``run`` receives this accepted plan so capability probes are not repeated
or silently changed between approval and execution.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .capabilities import interrogate_models
from .classify import classify_model, size_gb
from .filters import describe_filters, filter_models, filter_tasks, validate_task_ids
from .runner import _samples_for_task
from .tasks import TASKS, tasks_for
from .progress import seconds_hms


def _rough_seconds(samples_total: int, models_total: int) -> int:
    if samples_total <= 0:
        return 0
    return int(samples_total * 45 + models_total * 75)


def build_plan(
    client: Any,
    cfg: Any,
    *,
    level: str = "smoke",
    include: Optional[str] = None,
    exclude: Optional[str] = None,
    skip_offload: bool = False,
    categories: Optional[List[str]] = None,
    task_ids: Optional[List[str]] = None,
    task_regex: Optional[str] = None,
    family_base_only: bool = False,
    context_aliases_only: bool = False,
    context_only: bool = False,
    sample_mode: str = "smart",
    judge_mode: str = "off",
    selected_models: Optional[List[str]] = None,
    auto_probe: bool = False,
    capability_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    import re

    rows = client.tags()
    models_rows = {m.get("name"): m for m in rows if m.get("name")}
    inventory = list(models_rows.keys())
    models = list(inventory)
    skipped: List[Dict[str, str]] = []

    if selected_models is not None:
        selected_set = set(selected_models)
        missing = [m for m in selected_models if m not in models_rows]
        if missing:
            raise ValueError("selected model(s) are not installed: " + ", ".join(missing))
        models = [m for m in inventory if m in selected_set]
        skipped.extend({"model": m, "reason": "not_selected"} for m in inventory if m not in selected_set)

    if include:
        rx = re.compile(include, re.I)
        before = list(models)
        models = [m for m in models if rx.search(m)]
        skipped.extend({"model": m, "reason": "include_regex_no_match"} for m in before if m not in models)
    if exclude:
        rx = re.compile(exclude, re.I)
        before = list(models)
        models = [m for m in models if not rx.search(m)]
        skipped.extend({"model": m, "reason": "exclude_regex_match"} for m in before if m not in models)
    if skip_offload:
        before = list(models)
        models = [m for m in models if size_gb(models_rows[m]) <= cfg.vram_budget_gb]
        skipped.extend({"model": m, "reason": "size_exceeds_vram_budget"} for m in before if m not in models)

    models, context_skips = filter_models(
        models,
        family_base_only=family_base_only,
        context_aliases_only=context_aliases_only,
    )
    skipped.extend({"model": s.model, "reason": s.reason} for s in context_skips)

    unknown = validate_task_ids(task_ids, (t.id for t in TASKS))
    if unknown:
        known = ", ".join(sorted(t.id for t in TASKS))
        raise ValueError(f"unknown task id(s): {', '.join(unknown)}. Known tasks: {known}")

    profiles = dict(capability_profiles or {})
    missing_profiles = [m for m in models if m not in profiles]
    if missing_profiles:
        profiles.update(interrogate_models(client, missing_profiles, functional=auto_probe))

    task_source_level = "full" if context_only else level
    active: List[Dict[str, Any]] = []
    for model in models:
        profile = profiles[model]
        fams = list(profile.get("supported_families") or [])
        ts = filter_tasks(
            tasks_for(task_source_level, categories, fams),
            task_ids=task_ids,
            task_regex=task_regex,
            context_only=context_only,
        )
        if not ts:
            skipped.append({"model": model, "reason": "no_tasks_after_filter"})
            continue
        samples_total = sum(_samples_for_task(t, cfg, sample_mode, judge_mode) for t in ts)
        active.append({
            "model": model,
            "class": classify_model(model, profile.get("declared_capabilities"), fams),
            "size_gb": size_gb(models_rows[model]),
            "families": fams,
            "declared_capabilities": profile.get("declared_capabilities") or [],
            "capability_warnings": profile.get("warnings") or [],
            "capability_evidence_hash": profile.get("evidence_hash"),
            "tasks_total": len(ts),
            "samples_total": samples_total,
            "tasks": [t.id for t in ts],
        })

    total_tasks = sum(int(m["tasks_total"]) for m in active)
    unique_tasks = len({task_id for model in active for task_id in model.get("tasks", [])})
    total_samples = sum(int(m["samples_total"]) for m in active)
    rough = _rough_seconds(total_samples, len(active))
    reasons: Dict[str, int] = {}
    for item in skipped:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    return {
        "level": level,
        "sample_mode": sample_mode,
        "judge": judge_mode,
        "judge_model": getattr(cfg, "judge_model", None),
        "requested_samples": max(1, int(getattr(cfg, "samples", 1) or 1)),
        "ctx_override": getattr(cfg, "ctx_override", None),
        "auto_probe": bool(auto_probe),
        "selection_mode": "explicit" if selected_models is not None else "all-installed",
        "models_requested": list(selected_models) if selected_models is not None else None,
        "models_total_inventory": len(models_rows),
        "models_active": len(active),
        "models_skipped": len(skipped),
        "tasks_total": total_tasks,
        "tasks_unique": unique_tasks,
        "samples_total": total_samples,
        "rough_eta_seconds": rough,
        "filters": describe_filters(
            task_ids=task_ids,
            task_regex=task_regex,
            family_base_only=family_base_only,
            context_aliases_only=context_aliases_only,
            context_only=context_only,
        ),
        "active_models": active,
        "skipped_models": skipped,
        "skip_reasons": reasons,
        "capability_profiles": {m: profiles[m] for m in models if m in profiles},
    }


def render_plan(plan: Dict[str, Any], *, max_models: int = 80, max_skips: int = 20) -> str:
    lines = [
        "LLM ModelBench run plan",
        "=======================",
        f"Level:        {plan.get('level')}",
        f"Judge:       {plan.get('judge')}" + (f"  model={plan.get('judge_model')}" if plan.get('judge') != 'off' else ""),
        f"Sample mode:  {plan.get('sample_mode')}  requested={plan.get('requested_samples')}",
        f"CTX override: {plan.get('ctx_override') or 'default'}",
        f"Selection:    {plan.get('selection_mode')}",
        f"Auto probe:   {'yes' if plan.get('auto_probe') else 'no (metadata/profile routing only)'}",
        f"Models:       {plan.get('models_active')} active / {plan.get('models_total_inventory')} installed  ({plan.get('models_skipped')} skipped)",
        f"Tasks:        {plan.get('tasks_unique', plan.get('tasks_total'))} unique / {plan.get('tasks_total')} model-task cells",
        f"Generations:  {plan.get('samples_total')}",
        f"Rough ETA:    {seconds_hms(plan.get('rough_eta_seconds'))}  (pre-run estimate; live ETA updates during run)",
    ]
    if plan.get("filters"):
        lines.append("Filters:      " + "; ".join(plan["filters"]))
    if plan.get("skip_reasons"):
        visible_reasons = {k: v for k, v in plan["skip_reasons"].items() if k != "not_selected"}
        if visible_reasons:
            lines.append("Skipped:      " + ", ".join(f"{k}:{v}" for k, v in sorted(visible_reasons.items())))
    lines += ["", "Active models"]
    for i, model in enumerate(plan.get("active_models", [])[:max_models], 1):
        families = ",".join(model.get("families") or [])
        warning = f"  WARN:{len(model.get('capability_warnings') or [])}" if model.get("capability_warnings") else ""
        lines.append(
            f"{i:>3}. {model['class']:<9} {model['size_gb']:>6}GB  "
            f"tasks={model['tasks_total']:<2} gens={model['samples_total']:<2}  "
            f"families={families:<25} {model['model']}{warning}"
        )
    if len(plan.get("active_models", [])) > max_models:
        lines.append(f"... {len(plan['active_models']) - max_models} more active models")
    relevant_skips = [s for s in plan.get("skipped_models", []) if s.get("reason") != "not_selected"]
    if relevant_skips:
        lines += ["", f"Skipped models, first {max_skips}"]
        for item in relevant_skips[:max_skips]:
            lines.append(f"- {item['reason']:<24} {item['model']}")
        if len(relevant_skips) > max_skips:
            lines.append(f"... {len(relevant_skips) - max_skips} more skipped models")
    warnings = [
        (m["model"], warning)
        for m in plan.get("active_models", [])
        for warning in m.get("capability_warnings") or []
    ]
    if warnings:
        lines += ["", "Capability warnings"]
        lines.extend(f"- {model}: {warning}" for model, warning in warnings[:20])
        if len(warnings) > 20:
            lines.append(f"... {len(warnings) - 20} more warnings; see plan JSON/capability report")
    return "\n".join(lines)


def write_plan(out_path: Path, plan: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2))
