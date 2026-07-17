"""Config-sensitivity planning and reporting.

This is diagnostic evidence, not a leaderboard. It measures how much a model/task score or
long-context primitive moves when run-affecting configuration changes while prompt/scorer stay fixed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

SERVER_DEFAULT = {"default", "none", "server", "server-default"}


def _split_csv(value: str | None, default: Sequence[str]) -> List[str]:
    if not value:
        return list(default)
    return [x.strip() for x in value.split(",") if x.strip()]


def _task_set(tasks: str | None) -> set[str]:
    return set(_split_csv(tasks, []))


def _ctx_label(ctx: str) -> str:
    return "default" if ctx.lower() in SERVER_DEFAULT else f"ctx{ctx}"


def _ctx_is_default(ctx: str) -> bool:
    return ctx.lower() in SERVER_DEFAULT


def _ctx_int(ctx: str) -> int | None:
    try:
        return int(ctx)
    except (TypeError, ValueError):
        return None


def _needle_safe_ctx_values(ctxs: Sequence[str]) -> tuple[List[str], List[str]]:
    """Return a probe-aligned needle ctx grid and human-readable notes.

    Needle/RAG explicit-ctx sweeps must use probe-aligned values. Arbitrary values such
    as 8192 or 16384 are measurement-window diagnostics: they make larger probes skip as
    ctx_override_too_small and must not be interpreted as model instability. With the
    default operator cap, use:
      default = server/model max
      20000   = enough for the 16k probe plus output/headroom
      40960   = enough for the 32k probe while staying under the common cap
    """
    out: List[str] = []
    notes: List[str] = []
    wants_16k = False
    wants_32k = False
    for raw in ctxs:
        ctx = str(raw).strip()
        if not ctx:
            continue
        if _ctx_is_default(ctx):
            if not any(_ctx_is_default(c) for c in out):
                out.append(ctx)
            continue
        val = _ctx_int(ctx)
        if val is None:
            notes.append(f"dropped unparseable needle ctx value: {ctx}")
            continue
        if val < 20000:
            wants_16k = True
            notes.append(f"replaced non-probe-aligned needle ctx={val} with ctx=20000 for the 16k probe")
            continue
        if val < 40960:
            wants_16k = True
            notes.append(f"replaced non-probe-aligned needle ctx={val} with ctx=20000; use ctx=40960 for the 32k probe")
            continue
        wants_32k = True
        if "40960" not in out:
            out.append("40960")
            if val != 40960:
                notes.append(f"rounded needle ctx={val} to probe-aligned ctx=40960 under the diagnostic cap")
    if wants_16k and "20000" not in out:
        out.append("20000")
    if wants_32k and "40960" not in out:
        out.append("40960")
    if not out:
        out = ["default", "20000", "40960"]
        notes.append("all requested needle ctx values were invalid; using probe-aligned default,20000,40960")
    defaults = [c for c in out if _ctx_is_default(c)]
    nums = sorted({int(c) for c in out if not _ctx_is_default(c)})
    return defaults + [str(n) for n in nums], notes

def plan_commands(*, run_prefix: str = "v9514_config", include_regex: str,
                  tasks: str = "web_nav,needle", level: str = "short",
                  ctx_values: str = "default,4096,16384",
                  num_predict_values: str = "512,2048",
                  judge: str = "off", fingerprint: bool = False,
                  needle_max_ctx: int | None = None) -> str:
    """Return a copy-pasteable bash plan for the config-sensitivity sweep.

    V9.5.14 refuses to emit broken or non-probe-aligned needle sweeps: if `needle` is selected,
    the generated commands are automatically promoted to `--level full` and stamped with a
    bounded `--needle-max-ctx` unless the caller supplied another cap, and explicit ctx values are aligned to needle probe depths.
    """
    task_ids = _task_set(tasks)
    requested_level = level
    effective_level = "full" if "needle" in task_ids and level != "full" else level
    cap = needle_max_ctx
    if "needle" in task_ids and cap is None:
        cap = 40960

    ctxs = _split_csv(ctx_values, ["default", "4096", "16384"])
    ctx_notes: List[str] = []
    if "needle" in task_ids:
        ctxs, ctx_notes = _needle_safe_ctx_values(ctxs)
    nps = _split_csv(num_predict_values, ["512", "2048"])
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Diagnostic config-sensitivity sweep. Not a leaderboard.",
        "# Compare score ranges and long-context primitives across ctx/num_predict.",
    ]
    if effective_level != requested_level:
        lines.append(f"# NOTE: tasks include needle, so level was auto-promoted: {requested_level} -> {effective_level}.")
    if "needle" in task_ids:
        lines.append(f"# NOTE: needle_max_ctx={cap} keeps this diagnostic inside the operator cap; partial coverage is expected.")
        for note in ctx_notes:
            lines.append(f"# NOTE: {note}.")
    lines.append("")

    run_dirs: List[str] = []
    for ctx in ctxs:
        for np in nps:
            ctx_label = _ctx_label(ctx)
            run_id = f"{run_prefix}_{ctx_label}_np{np}"
            run_dirs.append(f"runs/{run_id}")
            cmd = [
                "./llmb-run",
                f"--level {effective_level}",
                f"--tasks {tasks}",
                f"--include-regex '{include_regex}'",
                f"--judge {judge}",
                "--yes",
                "--live-ui off",
                f"--num-predict {np}",
                f"--run-id {run_id}",
            ]
            if not _ctx_is_default(ctx):
                cmd.insert(-2, f"--ctx {ctx}")
            if cap is not None:
                cmd.insert(-2, f"--needle-max-ctx {cap}")
            if not fingerprint:
                cmd.insert(-2, "--no-fingerprint")
            lines.append(" \\\n  ".join(cmd))
            lines.append("")
    lines += [
        f"python -m llm_modelbench sensitivity-report {' '.join(run_dirs)}",
        "",
    ]
    return "\n".join(lines)


def _load_rows(run_dir: Path) -> List[Dict[str, Any]]:
    raw = run_dir / "raw_results.jsonl"
    if not raw.exists():
        return []
    return [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]


def _load_meta(run_dir: Path) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for name in ("filters.json", "summary_meta.json"):
        path = run_dir / name
        if path.exists():
            try:
                merged.update(json.loads(path.read_text()))
            except Exception:
                pass
    return merged


def _run_label(meta: Dict[str, Any]) -> str:
    ctx = meta.get("ctx_override")
    np = meta.get("num_predict")
    return f"ctx={ctx if ctx is not None else 'default'};num_predict={np if np is not None else 'task/default'}"


def _needle_skip_summary(row: Dict[str, Any]) -> str:
    skipped = row.get("needle_skipped") or []
    parts = []
    for s in skipped:
        reason = s.get("reason") or "unknown"
        size = s.get("size")
        margin = ""
        if reason == "kv_cache_exceeds_vram_budget" and isinstance(s.get("estimated_total_gb"), (int, float)) and isinstance(s.get("vram_budget_gb"), (int, float)):
            margin = f" (+{round(float(s['estimated_total_gb']) - float(s['vram_budget_gb']), 3)}GB)"
        parts.append(f"{size}:{reason}{margin}" if size else f"{reason}{margin}")
    return "; ".join(parts)


def _spread(nums: List[float]) -> Tuple[str, str]:
    if not nums:
        return "", ""
    spread = round(max(nums) - min(nums), 2)
    return str(spread), "stable" if spread == 0 else ("wide" if spread >= 50 else "moving")


def report(run_dirs: Iterable[str | Path]) -> str:
    """Return a Markdown config-sensitivity report for completed runs.

    Numeric tasks are summarised by score range. Needle rows intentionally have score=None, so
    they are summarised by max_verified_ctx, needle_coverage, and skip reasons instead.
    """
    cells: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    run_count = 0
    empty_runs: List[str] = []
    for rd in run_dirs:
        path = Path(rd)
        rows = _load_rows(path)
        if not rows:
            empty_runs.append(path.name)
            continue
        run_count += 1
        meta = _load_meta(path)
        label = _run_label(meta)
        for r in rows:
            key = (str(r.get("model")), str(r.get("task")))
            cells.setdefault(key, []).append({
                "run": path.name,
                "label": label,
                "score": r.get("score"),
                "reason": r.get("reason"),
                "error_kind": r.get("error_kind"),
                "max_verified_ctx": r.get("max_verified_ctx"),
                "needle_coverage": r.get("needle_coverage"),
                "needle_skips": _needle_skip_summary(r),
            })

    lines = [
        "# Config sensitivity report",
        "",
        "Diagnostic only. A wide range means the ranking is config-fragile even if repeatability is perfect.",
        "Needle rows are reported by `max_verified_ctx` and coverage because partial needle measurements intentionally have `score=null`.",
        "Explicit needle ctx sweeps must be probe-aligned; otherwise status may mean measurement-window change, not model instability.",
        "",
        f"Runs read: {run_count}",
    ]
    if empty_runs:
        lines += ["", f"Empty/no-row runs ignored: {', '.join(empty_runs)}"]
    lines += [
        "",
        "## Numeric score sensitivity",
        "",
        "| Model | Task | numeric scores | range | status |",
        "|---|---|---:|---:|---|",
    ]
    any_numeric = False
    for (model, task), vals in sorted(cells.items()):
        scores = [float(v["score"]) for v in vals if isinstance(v.get("score"), (int, float))]
        if not scores:
            continue
        any_numeric = True
        spread_s, status = _spread(scores)
        score_s = ", ".join(str(round(s, 2)) for s in scores)
        lines.append(f"| `{model}` | `{task}` | {score_s} | {spread_s} | {status} |")
    if not any_numeric:
        lines.append("| n/a | n/a |  |  | no numeric-score rows |")

    lines += [
        "",
        "## Needle / long-context sensitivity",
        "",
        "| Model | Task | max_verified_ctx values | ctx range | coverage values | coverage range | skips | status |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    any_needle = False
    for (model, task), vals in sorted(cells.items()):
        ctx_vals = [float(v["max_verified_ctx"]) for v in vals if isinstance(v.get("max_verified_ctx"), (int, float))]
        cov_vals = [float(v["needle_coverage"]) for v in vals if isinstance(v.get("needle_coverage"), (int, float))]
        if not ctx_vals and not cov_vals:
            continue
        any_needle = True
        ctx_spread = round(max(ctx_vals) - min(ctx_vals), 2) if ctx_vals else ""
        cov_spread = round(max(cov_vals) - min(cov_vals), 4) if cov_vals else ""
        skips = sorted({v.get("needle_skips") or "none" for v in vals})
        stable = (ctx_spread == 0 or ctx_spread == "") and (cov_spread == 0 or cov_spread == "")
        status = "stable" if stable else "moving"
        if not stable and any("ctx_override_too_small" in s for s in skips):
            status = "windowed"
        ctx_s = ", ".join(str(int(v)) if float(v).is_integer() else str(round(v, 2)) for v in ctx_vals)
        cov_s = ", ".join(str(round(v, 4)) for v in cov_vals)
        lines.append(f"| `{model}` | `{task}` | {ctx_s} | {ctx_spread} | {cov_s} | {cov_spread} | {' / '.join(skips)} | {status} |")
    if not any_needle:
        lines.append("| n/a | n/a |  |  |  |  |  | no needle rows |")

    if not cells:
        lines += ["", "No rows found in the supplied run directories."]
    return "\n".join(lines) + "\n"
