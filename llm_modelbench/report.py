"""Report generation: HTML, Markdown, CSV, and JSON, plus routing, prune, and clone reports."""
from __future__ import annotations

import csv
import html as htmlmod
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .aggregate import aggregate, prune_recommendations
from .fingerprint import find_clones, find_digest_clones, invalid_probe_models
from . import __version__


def _load(out_dir: Path) -> List[Dict[str, Any]]:
    raw = out_dir / "raw_results.jsonl"
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()] if raw.exists() else []
    if rows and (out_dir / "judge_results.jsonl").exists():
        from .judge_dumps import apply_judgements
        rows = apply_judgements(out_dir, rows)
    return rows


def _duplicate_key(r: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """One report cell is one model/task/task_hash/sample aggregate.

    Runner rows are already aggregated across repeated samples, so absent sample_index
    means the single aggregate cell. Duplicate cells can appear after interrupted/resumed
    runs or repeated model entries. They must not double-weight quality.
    """
    return (
        str(r.get("model") or ""),
        str(r.get("task") or ""),
        str(r.get("task_hash") or ""),
        str(r.get("sample_index") if r.get("sample_index") is not None else ""),
    )


def _duplicate_signature(r: Dict[str, Any]) -> Tuple[Any, ...]:
    """Quality-relevant fields that must agree before a duplicate is collapsible."""
    return (
        r.get("category"),
        r.get("score"),
        r.get("reason"),
        r.get("error_kind"),
        r.get("warning_kind"),
        r.get("done_reason"),
        r.get("benchmark_version"),
        r.get("task_hash"),
    )


def _dedupe_report_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Drop non-conflicting duplicate report cells and hard-fail conflicting ones.

    Telemetry can legitimately differ across duplicate generations. If the scorer result
    and error state match, keep the latest row so the report remains one cell per
    model/task. If quality-relevant fields differ, refusing the report is safer than
    silently choosing a leaderboard number.
    """
    seen: Dict[Tuple[str, str, str, str], Tuple[int, Dict[str, Any]]] = {}
    dropped: List[Dict[str, Any]] = []
    conflicts: List[str] = []
    for idx, row in enumerate(rows):
        key = _duplicate_key(row)
        prev = seen.get(key)
        if prev is None:
            seen[key] = (idx, row)
            continue
        prev_idx, prev_row = prev
        if _duplicate_signature(prev_row) != _duplicate_signature(row):
            conflicts.append(
                f"model={key[0]!r} task={key[1]!r} prev_index={prev_idx} new_index={idx} "
                f"prev_score={prev_row.get('score')!r} new_score={row.get('score')!r}"
            )
            continue
        dropped.append({"model": key[0], "task": key[1], "kept_index": idx, "dropped_index": prev_idx})
        seen[key] = (idx, row)
    if conflicts:
        preview = "; ".join(conflicts[:5])
        more = f"; +{len(conflicts) - 5} more" if len(conflicts) > 5 else ""
        raise SystemExit(f"conflicting duplicate result rows refused: {preview}{more}")
    return [row for _, row in sorted(seen.values(), key=lambda x: x[0])], dropped


def _enrich_agentic_rows(out_dir: Path, rows: List[Dict[str, Any]], tasks) -> None:
    """Backfill V9.5.17 agentic decomposition from persisted raw outputs.

    This lets V9.5.17 rebuild reports from V9.5.16 fleet artifacts without a GPU
    rerun. The legacy `score` field is not changed.
    """
    from . import scoring
    from .runner import _task_hash  # local: avoids a module-level cycle
    task_map = {t.id: t for t in tasks}
    for r in rows:
        if r.get("category") != "agentic_tool":
            continue
        if isinstance(r.get("decision_score"), (int, float)) and isinstance(r.get("format_multiplier"), (int, float)):
            continue
        raw_path = r.get("raw_path")
        task = task_map.get(str(r.get("task")))
        if not raw_path or not task:
            continue
        # Re-scoring uses the CURRENT scorer and meta. If the task changed since the run,
        # the enriched columns would describe a scoring that never happened, sitting next
        # to a score_blended that did. Refuse, and say so.
        if str(r.get("task_hash") or "") != _task_hash(task):
            r["enrichment"] = "stale: task changed since this run"
            continue
        # A ModelFailed row has no action JSON to re-score. Re-scoring its empty raw file
        # mislabels a thinking-budget failure as `invalid_json:empty`.
        if r.get("error_kind"):
            r["enrichment"] = f"skipped: {r['error_kind']}"
            continue
        path = out_dir / str(raw_path)
        if not path.exists():
            continue
        try:
            output = path.read_text(encoding="utf-8")
        except Exception:
            continue
        detail = scoring.score_agentic_action_details(output, task.meta)
        r["decision_score"] = detail.get("decision_score")
        r["format_multiplier"] = detail.get("format_multiplier")
        r["format_deviation"] = detail.get("format_deviation")
        r["caps_fired"] = detail.get("caps_fired") or []

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def build(out_dir: Path, cfg) -> None:
    raw_rows = _load(out_dir)
    rows, duplicate_rows = _dedupe_report_rows(raw_rows)
    if not rows:
        print("no results to report"); return
    from .tasks import TASKS
    _enrich_agentic_rows(out_dir, rows, TASKS)
    difficulty = {t.id: t.difficulty for t in TASKS}
    lb, per_cat = aggregate(rows, cfg.weights, difficulty)

    fp_path = out_dir / "fingerprints.json"
    ident_path = out_dir / "model_identities.json"
    fingerprints = _load_json(fp_path)
    identities = _load_json(ident_path)
    digest_clones = find_digest_clones(identities)
    probe_clones = find_clones(fingerprints) if fingerprints else []
    invalid_probes = invalid_probe_models(fingerprints) if fingerprints else {}
    prune = prune_recommendations(lb, per_cat, digest_clones)
    context = _report_context(out_dir, rows, cfg, raw_row_count=len(raw_rows), duplicate_rows=duplicate_rows)

    _csv(out_dir / "scorecard.csv", lb, context)
    _md(out_dir / "scorecard.md", lb, per_cat, context)
    _routing(out_dir / "routing.md", lb, per_cat, context)
    _prune(out_dir / "prune.md", lb, prune, context)
    _clones(out_dir / "clones.md", digest_clones, probe_clones, invalid_probes, bool(identities), context)
    (out_dir / "summary.json").write_text(json.dumps(lb, indent=2))
    _retrieval_diagnostics(out_dir, rows)
    meta = _metadata(out_dir, rows, cfg, context)
    (out_dir / "summary_meta.json").write_text(json.dumps(meta, indent=2))
    _regression(out_dir, lb, meta)
    _html(out_dir / "report.html", lb, per_cat, rows, cfg, context)
    print(f"reports -> {out_dir}/ (report.html, scorecard.md/.csv, routing.md, prune.md, "
          f"clones.md, summary.json)")


def _retrieval_diagnostics(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    """Write case details when persisted, otherwise an explicit non-invented partial record."""
    retrieval_rows = [row for row in rows if row.get("category") == "retrieval"]
    if not retrieval_rows:
        return
    entries = []
    for row in retrieval_rows:
        base = {key: row.get(key) for key in ("model", "task", "score", "reason")}
        base["embed_model"] = row.get("embed_model") or _embed_model_from_reason(str(row.get("reason") or ""))
        cases = row.get("retrieval_cases")
        if isinstance(cases, list) and cases:
            entries.append({**base, "case_level_available": True, "cases": cases})
        else:
            entries.append({**base, "case_level_available": False,
                            "note": "case-level rankings unavailable in this run; rerun required after diagnostics instrumentation"})
    (out_dir / "retrieval_diagnostics.json").write_text(json.dumps(entries, indent=2))
    lines = ["# Retrieval Diagnostics", ""]
    for entry in entries:
        lines += [f"## {entry.get('model')} | {entry.get('task')}", f"embed_model: `{entry.get('embed_model')}`", ""]
        if not entry["case_level_available"]:
            lines += [entry["note"], ""]
            continue
        for case in entry["cases"]:
            lines.append(f"- query {case.get('query_index')}: gold={case.get('gold_doc_id')} top1={case.get('top1_doc_id')} top3={case.get('top3_doc_ids')} rank={case.get('target_rank')} margin={case.get('margin')}")
        lines.append("")
    (out_dir / "retrieval_diagnostics.md").write_text("\n".join(lines))


def _embed_model_from_reason(reason: str) -> Optional[str]:
    marker = "embed_model="
    return reason.split(marker, 1)[1].split(";", 1)[0].strip() if marker in reason else None


def _report_context(out_dir: Path, rows: List[Dict[str, Any]], cfg, raw_row_count: Optional[int] = None, duplicate_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    filters = _load_json(out_dir / "filters.json")
    status = _load_json(out_dir / "status.json")
    task_ids = sorted({r.get("task") for r in rows if r.get("task")})
    models = sorted({r.get("model") for r in rows if r.get("model")})
    by_cat: Dict[str, set] = {}
    for r in rows:
        by_cat.setdefault(str(r.get("category")), set()).add(str(r.get("task")))
    tasks_per_model = {m: len({r.get("task") for r in rows if r.get("model") == m}) for m in models}
    min_tasks_per_category = int(getattr(cfg, "min_report_tasks_per_category", 2) or 2)
    return {
        "level": filters.get("level") or status.get("level") or "unknown",
        "sample_mode": filters.get("sample_mode") or status.get("sample_mode"),
        "judge_mode": filters.get("judge_mode"),
        "task_ids": task_ids,
        "include_regex": filters.get("include_regex"),
        "task_regex": filters.get("task_regex"),
        "ctx_override": filters.get("ctx_override"),
        "num_predict": filters.get("num_predict"),
        "think": filters.get("think"),
        "needle_max_ctx": filters.get("needle_max_ctx"),
        "num_ctx_used": sorted({r.get("num_ctx_used") for r in rows if r.get("num_ctx_used") is not None}),
        "tasks_per_model": tasks_per_model,
        "min_tasks_per_model": min(tasks_per_model.values()) if tasks_per_model else 0,
        "category_task_counts": {c: len(v) for c, v in by_cat.items()},
        "min_report_tasks_per_category": min_tasks_per_category,
        "raw_row_count": raw_row_count if raw_row_count is not None else len(rows),
        "report_row_count": len(rows),
        "duplicate_rows_dropped": len(duplicate_rows or []),
        "duplicate_cells_deduped": duplicate_rows or [],
        "filters": filters,
        "status": status,
    }


def _header_lines(context: Dict[str, Any]) -> List[str]:
    lines = [
        f"Level: `{context.get('level')}` | sample_mode: `{context.get('sample_mode')}` | judge: `{context.get('judge_mode')}`",
        f"Tasks: {len(context.get('task_ids') or [])} | task_ids: `{', '.join(context.get('task_ids') or [])}`",
        f"num_ctx_used: `{context.get('num_ctx_used') or 'server-default'}` | num_predict: `{context.get('num_predict') or 'task-default'}` | think: `{context.get('think') or 'auto'}` | needle_max_ctx: `{context.get('needle_max_ctx') or 'none'}`",
    ]
    if context.get("duplicate_rows_dropped"):
        lines.append(
            f"Duplicate result rows dropped from report: {context.get('duplicate_rows_dropped')} "
            f"(raw rows {context.get('raw_row_count')}, report rows {context.get('report_row_count')})"
        )
    lines.append("")
    return lines


def _metadata(out_dir: Path, rows: List[Dict[str, Any]], cfg, context: Dict[str, Any]) -> Dict[str, Any]:
    status = context.get("status") or {}
    filters = context.get("filters") or {}
    hc = status.get("hardware_config") or {}
    versions = sorted({str(r.get("benchmark_version")) for r in rows if r.get("benchmark_version")})
    return {
        "benchmark_version": __version__,
        "llm_modelbench_version": __version__,
        "weights": dict(getattr(cfg, "weights", {}) or {}),
        "weight_override": getattr(cfg, "weight_override_spec", None),
        "row_versions": versions,
        "created_from": str(out_dir),
        "ollama_url": getattr(cfg, "ollama_url", None),
        "ollama_version": filters.get("ollama_version"),
        "seed": getattr(cfg, "seed", None),
        "temperature": getattr(cfg, "temperature", None),
        "judge_model": getattr(cfg, "judge_model", None),
        "ctx_override": filters.get("ctx_override"),
        "num_predict": filters.get("num_predict"),
        "think": filters.get("think"),
        "needle_max_ctx": filters.get("needle_max_ctx"),
        "sample_mode": context.get("sample_mode"),
        "level": context.get("level"),
        "judge_mode": context.get("judge_mode"),
        "task_ids": context.get("task_ids"),
        "include_regex": context.get("include_regex"),
        "raw_row_count": context.get("raw_row_count"),
        "report_row_count": context.get("report_row_count"),
        "duplicate_rows_dropped": context.get("duplicate_rows_dropped"),
        "duplicate_cells_deduped": context.get("duplicate_cells_deduped"),
        "gpu_name": hc.get("gpu_name"),
        "gpu_vendor": hc.get("gpu_vendor"),
        "gpu_driver": hc.get("gpu_driver"),
        "vram_budget_gb": hc.get("vram_budget_gb"),
        "note": "Deterministic agentic_tool tasks, repeatability reporting, and probe-aligned needle/RAG ctx guidance are part of this report. Arbitrary explicit ctx values are measurement-window diagnostics, not model-ranking diagnostics.",
        "measurement_rule": "A measurement the harness declined to take must never become a leaderboard number.",
        "scorer_rule": "A check the scorer cannot fail must never become a leaderboard number.",
        "operator_rule": "Before new action: review recent action chain, expected benefit, and blast radius.",
        "deterministic_measurement_noise_points": 0.0,
        "task_sampling_band_points": 17.0,
        "tie_band_rule": "Treat close scores as tied inside task-sampling band plus measured config-sensitivity envelope.",
        "needle_ctx_rule": "Needle/RAG explicit ctx sweeps must use probe-aligned ctx values; arbitrary ctx values change the measurement window.",
    }


def _rank_cells(lb, context):
    """Conservative rank labels. Blank ranks for under-covered diagnostic data; shared ranks for ties."""
    min_tasks = int(context.get("min_report_tasks_per_category") or 2)
    cat_counts = context.get("category_task_counts") or {}
    undercovered = any(n < min_tasks for n in cat_counts.values())
    if int(context.get("min_tasks_per_model") or 0) < min_tasks or undercovered:
        return {r["model"]: "" for r in lb}
    ranked = [r for r in lb if isinstance(r.get("quality"), (int, float))]
    out = {r["model"]: "" for r in lb}
    last_quality = object()
    current_rank = 0
    for idx, r in enumerate(ranked, 1):
        q = r.get("quality")
        if q != last_quality:
            current_rank = idx
            last_quality = q
        out[r["model"]] = str(current_rank)
    return out


def _csv(path: Path, lb, context):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "model", "class", "quality", "score_blended", "tok_s", "offload",
                    "value_per_gb", "value_per_gb_blended", "size_gb", "err", "completion_rate",
                    "format_strict_rate", "format_modal_deviation", "format_mean_multiplier",
                    "over_refusal_count", "disallowed_tool_count", "thinking_only_count", "agentic_caps_fired"])
        ranks = _rank_cells(lb, context)
        for r in lb:
            w.writerow([ranks.get(r["model"], ""), r["model"], r["class"], r["quality"], r.get("score_blended"),
                        r["tok_s"], r["offload"], r["value_per_gb"], r.get("value_per_gb_blended"),
                        r["size_gb"], r["err"], r.get("completion_rate"),
                        r.get("agentic_format_strict_rate"), r.get("agentic_format_modal_deviation"),
                        r.get("agentic_format_mean_multiplier"), r.get("over_refusal_count"),
                        r.get("disallowed_tool_count"), r.get("thinking_only_count"),
                        json.dumps(r.get("agentic_caps_fired") or {}, sort_keys=True)])


def _md(path: Path, lb, per_cat, context):
    L = ["# Scorecard", ""] + _header_lines(context)
    L += ["Quality is pure task correctness. For agentic_tool, quality uses decision_score; the legacy blended score is shown separately. Empty or thinking-only model outputs score 0.0 and remain visible through Err/completion_rate.", "",
          "| # | Model | Class | Quality | Blended | tok/s | Offload | Value/GB | Blended V/GB | Size | Err | Completion | Strict fmt | Modal fmt | Over-refusal | Disallowed tool | Thinking-only |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    ranks = _rank_cells(lb, context)
    for r in lb:
        rank_cell = ranks.get(r["model"], "")
        L.append(f"| {rank_cell} | `{r['model']}` | {r['class']} | {r['quality']} | {r.get('score_blended')} | {r['tok_s']} | "
                 f"{r['offload']} | {r['value_per_gb']} | {r.get('value_per_gb_blended')} | {r['size_gb']} | {r['err']} | "
                 f"{r.get('completion_rate')} | {r.get('agentic_format_strict_rate')} | {r.get('agentic_format_modal_deviation') or ''} | "
                 f"{r.get('over_refusal_count')} | {r.get('disallowed_tool_count')} | {r.get('thinking_only_count')} |")
    L += ["", "## Category leaders"]
    for c, ranked in sorted(per_cat.items()):
        L.append(f"- **{c}**: " + ", ".join(f"{m} ({s})" for m, s in ranked[:3]))
    path.write_text("\n".join(L))


def _routing(path: Path, lb, per_cat, context):
    L = ["# Recommended routing", ""] + _header_lines(context)
    L.append("Routing refuses single winners when category coverage is too small or the top decision score is tied.")
    L.append("Agentic routing ranks decision quality first, then orders tied ceiling bands by VRAM and throughput.")
    L.append("")
    min_tasks = int(context.get("min_report_tasks_per_category") or 2)
    cat_counts = context.get("category_task_counts") or {}
    model_rows = {r.get("model"): r for r in lb}
    for c, ranked in sorted(per_cat.items()):
        if cat_counts.get(c, 0) < min_tasks:
            L.append(f"- **{c}** -> no recommendation, insufficient coverage: {cat_counts.get(c, 0)} task(s), minimum {min_tasks}")
            continue
        if not ranked:
            L.append(f"- **{c}** -> no eligible models")
            continue
        top = ranked[0][1]
        tied = [(m, s) for m, s in ranked if s == top]
        if len(tied) > 1:
            if c == "agentic_tool":
                tied_rows = [model_rows[m] for m, _ in tied if m in model_rows]
                tied_rows.sort(key=lambda r: ((r.get("size_gb") or 9999), -(r.get("tok_s") or 0)))
                L.append(f"- **{c}** -> ceiling band tied at {top}; no exact rank inside band. Route by VRAM, then throughput:")
                for idx, r in enumerate(tied_rows):
                    tag = " recommended" if idx == 0 else ""
                    L.append(f"  - `{r['model']}` — {r.get('size_gb')} GB, {r.get('tok_s')} tok/s{tag}")
            else:
                L.append(f"- **{c}** -> no single winner, tied at {top}: " + ", ".join(f"`{m}`" for m, _ in tied))
        else:
            L.append(f"- **{c}** -> `{ranked[0][0]}` ({ranked[0][1]})")
    if not per_cat:
        L.append("- no eligible categories")
    excluded = [r for r in lb
                if isinstance(r.get("quality"), (int, float))
                and r.get("value_per_gb") is None
                and (r.get("completion_rate") or 1.0) < 1.0]
    if excluded:
        L.append("")
        L.append("## Excluded from value/GB ranking")
        L.append("")
        L.append("These models have a completed quality score, but value/GB is withheld because completion_rate < 1.0. Model failures such as thinking_only still score as failures; this block explains why the value metric is blank.")
        for r in sorted(excluded, key=lambda x: (x.get("completion_rate") or 0, x.get("model") or "")):
            bits = []
            if r.get("thinking_only_count"):
                bits.append(f"thinking_only={r.get('thinking_only_count')}")
            if r.get("err"):
                bits.append(f"err={r.get('err')}")
            why = ", ".join(bits) if bits else "incomplete"
            L.append(f"- `{r['model']}` — completion_rate {r.get('completion_rate')}, quality {r.get('quality')}, {why}")
    L.append("")
    L.append("Note: thinking_only rows at fixed num_predict are model failures for scoring, but budget-limited rows should be read as lower bounds on reasoning-model agentic ability.")
    path.write_text("\n".join(L))


def _prune(path: Path, lb, prune, context):
    L = ["# Prune recommendations", ""] + _header_lines(context)
    min_tasks = int(context.get("min_report_tasks_per_category") or 2)
    undercovered = {c: n for c, n in (context.get("category_task_counts") or {}).items() if n < min_tasks}
    if int(context.get("min_tasks_per_model") or 0) < min_tasks or undercovered:
        reason = f"only {context.get('min_tasks_per_model')} task(s) per model in the smallest model plan; minimum is {min_tasks}."
        if undercovered:
            reason = "category coverage below minimum: " + ", ".join(f"{c}={n}" for c, n in sorted(undercovered.items())) + f"; minimum is {min_tasks}."
        L += [
            "Prune recommendations refused.",
            "",
            f"Reason: {reason}",
            "Digest clone evidence remains safe to inspect in clones.md, but quality/speed prune advice is not emitted from under-covered data.",
        ]
        path.write_text("\n".join(L)); return
    L += ["Strategies are additive. Keep is the union of top-2-per-category and the Pareto frontier.", "", "## Keep"]
    for m in prune["keep"]:
        r = next((x for x in lb if x["model"] == m), None)
        if r:
            L.append(f"- `{m}` (quality {r['quality']}, value/GB {r['value_per_gb']})")
    L += ["", "## On the Pareto frontier (best quality for their VRAM)"]
    L += [f"- `{m}`" for m in prune["pareto"]]
    L += ["", "## Delete first (bottom quartile in BOTH quality and speed)"]
    L += ([f"- `{m}`" for m in prune["delete_first"]] or ["- none"])
    L += ["", "## Redundant (certain digest clones only)"]
    L += ([f"- drop `{m}` ({why})" for m, why in prune["redundant"]] or ["- none detected"])
    path.write_text("\n".join(L))


def _clones(path: Path, digest_clones, probe_clones, invalid_probes, identities_available: bool, context):
    L = ["# Clone detection", ""] + _header_lines(context)
    L += ["Certain clone evidence (same digest) is separated from advisory probe evidence.", "", "## Certain clones, same Ollama digest / ID", ""]
    if not identities_available:
        L.append("- No `model_identities.json` found. Re-run to enable digest clone detection.")
    elif digest_clones:
        L += [f"- `{a}` ~ `{b}` (same digest / ID)" for a, b, _ in digest_clones]
    else:
        L.append("- none detected")
    L += ["", "## Advisory probe matches", "",
          "Probe matches are not used for prune recommendations. Empty or near-empty probe outputs are ignored, "
          "and models with too few valid probe answers are marked non-actionable.", ""]
    skip = ((context.get("filters") or {}).get("fingerprint_skip_reason"))
    if skip:
        L.append(f"- fingerprint probes skipped: {skip}")
    else:
        L += ([f"- `{a}` ~ `{b}` (valid-probe similarity {s})" for a, b, s in probe_clones]
              or ["- none detected"])
    if invalid_probes:
        L += ["", "## Non-actionable probe sets", "",
              "These models returned too few non-empty probe answers for safe probe-based clone detection:"]
        for model, h in sorted(invalid_probes.items()):
            L.append(f"- `{model}` valid {h['valid']}/{h['total']} probes, minimum {h['min_required']}")
    path.write_text("\n".join(L))


def _regression(out_dir: Path, lb, meta: Dict[str, Any]):
    # Automatic mutable-baseline regression is intentionally disabled. Use explicit llmb diff
    # for cross-run comparisons so version/config incompatibilities are visible.
    path = out_dir / "regression.md"
    path.write_text("# Regression\n\nAutomatic mutable-baseline regression is disabled. Use `llm-modelbench diff --a <baseline> --b <run>` with an explicit baseline.\n")


def _html(path: Path, lb, per_cat, rows, cfg, context):
    """Write a self-contained offline HTML report with no third-party requests."""
    def e(value: Any) -> str:
        return htmlmod.escape(str(value))
    cats = sorted(per_cat.keys())
    rows_html = []
    rank = 0
    for row in lb:
        rank_cell = ""
        if isinstance(row.get("quality"), (int, float)):
            rank += 1
            rank_cell = str(rank)
        rows_html.append(
            f"<tr><td>{rank_cell}</td><td><code>{e(row['model'])}</code></td><td>{e(row['class'])}</td>"
            f"<td><b>{row['quality']}</b></td><td>{row['tok_s']}</td><td>{row['offload']}</td>"
            f"<td>{e(row['value_per_gb'])}</td><td>{e(row.get('score_blended'))}</td>"
            f"<td>{e(row['size_gb'])}</td><td>{row['err']}</td><td>{row.get('completion_rate')}</td></tr>"
        )

    category_rows = []
    for category in cats:
        values = [score for _, score in per_cat.get(category, []) if isinstance(score, (int, float))]
        average = round(sum(values) / len(values), 1) if values else 0.0
        category_rows.append(
            f"<tr><td>{e(category)}</td><td>{average:.1f}</td>"
            f"<td><div class='meter'><span style='width:{max(0.0, min(100.0, average)):.1f}%'></span></div></td></tr>"
        )

    top_rows = []
    for row in lb[:5]:
        cells = "".join(f"<td>{e(row.get('categories', {}).get(category, '--'))}</td>" for category in cats)
        top_rows.append(f"<tr><td><code>{e(row['model'])}</code></td>{cells}</tr>")
    top_header = "".join(f"<th>{e(category)}</th>" for category in cats)

    sub = (
        f"rows {len(rows)} | models {len(lb)} | level {e(context.get('level'))} | "
        f"tasks {len(context.get('task_ids') or [])} | "
        f"num_ctx {e(context.get('num_ctx_used') or 'server-default')} | "
        f"num_predict {e(context.get('num_predict') or 'task-default')} | "
        f"think {e(context.get('think') or 'auto')} | "
        "quality is decision-first for agentic_tool; speed, offload and value/GB are separate axes"
    )
    doc = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<meta http-equiv='Content-Security-Policy' content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
<title>LLM ModelBench v{__version__}</title>
<style>
body{{font-family:system-ui,Arial;margin:0;padding:24px;background:#0b1120;color:#e2e8f0}}
h1{{background:linear-gradient(90deg,#38bdf8,#818cf8);-webkit-background-clip:text;color:transparent;margin:0}}
.sub{{color:#94a3b8;font-size:.9rem;margin:4px 0 20px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;background:#111827;border-radius:8px;overflow:hidden}}
td,th{{border-bottom:1px solid #1f2937;padding:8px 10px;font-size:14px;text-align:left}}
th{{background:#0f172a;color:#94a3b8;text-transform:uppercase;font-size:.72rem;letter-spacing:.05em}}
.card{{background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px;margin:12px 0;overflow:auto}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}}
code{{color:#cbd5e1}}
.meter{{height:10px;background:#1f2937;border-radius:999px;overflow:hidden;min-width:140px}}
.meter span{{display:block;height:100%;background:#38bdf8}}
</style></head><body>
<h1>LLM ModelBench v{__version__}</h1>
<div class='sub'>{sub}</div>
<div class='grid'>
<div class='card'><h3>Category averages</h3><table><tr><th>Category</th><th>Average</th><th>Scale</th></tr>{''.join(category_rows)}</table></div>
<div class='card'><h3>Top 5 category matrix</h3><table><tr><th>Model</th>{top_header}</tr>{''.join(top_rows)}</table></div>
</div>
<div class='card'><h3>Leaderboard</h3>
<table><tr><th>#</th><th>Model</th><th>Class</th><th>Quality</th><th>tok/s</th><th>Offload</th>
<th>Value/GB</th><th>Blended</th><th>Size</th><th>Err</th><th>Completion</th></tr>{''.join(rows_html)}</table>
</div></body></html>"""
    path.write_text(doc)

