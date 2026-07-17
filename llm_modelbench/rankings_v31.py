"""Rankings V3.1 calm split-site report.

This renderer is an additive human-facing view over the existing master rankings
payload. It does not alter scores, evidence selection, or the V3 operational
JSON. The goal is readability: one decision dashboard plus one spacious page per
model, with technical details hidden behind disclosures.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .rankings_v3 import SCHEMA_VERSION as V3_SCHEMA_VERSION, build_v3_payload, model_axes, status_badges

SCHEMA_VERSION = "llm-modelbench.rankings.v3.1"

_TASK_LABELS = {
    "needle": "Long-context needle retrieval",
    "fim_suffix_assertion": "Suffix insertion / FIM assertion",
    "kb_taxonomy": "Knowledge-base taxonomy",
    "js_debounce": "JavaScript debounce implementation",
    "py_csv": "Python CSV transformation",
    "reasoning_birthday_twins": "Reasoning: birthday twins",
    "reasoning_bridge_crossing": "Reasoning: bridge crossing",
    "reasoning_monty_hall": "Reasoning: Monty Hall",
    "reasoning_poisoned_wine": "Reasoning: poisoned wine",
    "reasoning_wolf_goat_cabbage": "Reasoning: wolf, goat and cabbage",
    "agent_malformed_repair": "Agentic: repair malformed tool call",
    "agent_nested_args": "Agentic: nested tool arguments",
    "agent_schema_collision": "Agentic: schema collision",
    "agent_schema_strict": "Agentic: strict schema output",
    "agent_state_delta": "Agentic: state delta",
    "agent_tool_refuse": "Agentic: correct refusal",
    "agent_tool_repair": "Agentic: tool-call repair",
    "agent_tool_select": "Agentic: tool selection",
    "agent_tool_state": "Agentic: tool state tracking",
    "agent_unknown_tool_reject": "Agentic: reject unknown tool",
    "ocr_invoice": "OCR: invoice extraction",
    "ocr_receipt_total": "OCR: receipt total",
    "ocr_table_cell": "OCR: table cell lookup",
    "ocr_form_code": "OCR: form code",
    "ocr_noisy": "OCR: noisy text",
    "ocr_noisy_label": "OCR: noisy label",
    "pdf_text": "PDF text extraction",
}

_CAUSE_LABELS = {
    "missing": "Evidence still required",
    "capability_unavailable": "Capability unavailable on this installed build",
    "measured_failure": "Measured capability-quality failure",
    "recovery_exhausted": "Recovery exhausted",
    "stale": "Outdated task evidence",
}


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _num(value: Any, default: Optional[float] = None) -> Optional[float]:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _pct(value: Any) -> str:
    n = _num(value)
    return "-" if n is None else f"{n * 100:.1f}%"


def _fmt(value: Any, digits: int = 2) -> str:
    n = _num(value)
    if n is None:
        return "-"
    if abs(n - round(n)) < 0.0001:
        return str(int(round(n)))
    return f"{n:.{digits}f}"


def _slug(name: Any, digest: Any = None) -> str:
    raw = str(name or digest or "model")
    raw = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-._")[:90] or "model"
    suffix = hashlib.sha1(str(digest or name or raw).encode(), usedforsecurity=False).hexdigest()[:10]
    return f"{raw}-{suffix}.html"


def _task_label(task_id: Any) -> str:
    task = str(task_id or "")
    if task in _TASK_LABELS:
        return _TASK_LABELS[task]
    return task.replace("_", " ").replace("-", " ").strip().title() or "Unknown task"


def _category_label(category: Any) -> str:
    text = str(category or "uncategorised").replace("_", " ").replace("-", " ")
    known = {
        "agentic tool": "Agentic and tool use",
        "long context": "Long context",
        "retrieval rag": "Retrieval / RAG",
        "ocr": "OCR / vision",
        "pdf": "PDF",
    }
    return known.get(text, text.title())


def _model_name(model: Dict[str, Any]) -> str:
    return str(model.get("display_name") or model.get("model") or model.get("digest") or "unknown model")


def _model_link(model: Dict[str, Any]) -> str:
    page = str(model.get("page") or "")
    if page:
        return page
    return "models/" + _slug(_model_name(model), model.get("digest"))


def _badges(model: Dict[str, Any]) -> List[Dict[str, str]]:
    try:
        return status_badges(model)
    except Exception:
        labels = [{"kind": "status", "label": str(model.get("quality_status") or "unknown")}]
        return labels


def _badge_html(badge: Dict[str, str]) -> str:
    label = str(badge.get("label") or badge)
    kind = str(badge.get("kind") or "")
    css = "badge"
    low = label.lower()
    if "complete" in low or kind == "status" and low == "complete":
        css += " badge-good"
    elif "provisional" in low or "slow" in low or "warning" in low:
        css += " badge-warn"
    elif "error" in low or "ineligible" in low:
        css += " badge-bad"
    elif "64k" in low or kind == "context":
        css += " badge-context"
    elif "capability" in low:
        css += " badge-cap"
    return f'<span class="{css}">{_esc(label)}</span>'


def _metric_card(label: str, value: Any, sub: str = "") -> str:
    return f"""<section class="metric-card"><div class="metric-label">{_esc(label)}</div><div class="metric-value">{_esc(value)}</div>{f'<div class="metric-sub">{_esc(sub)}</div>' if sub else ''}</section>"""


def _verdict(model: Dict[str, Any]) -> str:
    status = str(model.get("quality_status") or "unknown")
    coverage = _num(model.get("coverage_ratio"), 0.0) or 0.0
    name = _model_name(model)
    if status == "complete" and model.get("capability_limited"):
        return f"{name} has complete current evidence for every applicable task, with some capability lanes excluded for this installed build."
    if status == "complete":
        return f"{name} has complete current evidence for the applicable benchmark scope in this snapshot."
    if coverage >= 0.9:
        return f"{name} is near-complete but should not be treated as a definitive overall winner until remaining evidence is resolved."
    return f"{name} has partial evidence. Scores are useful diagnostics, not a comparable overall rank."


def _evidence_groups(model: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    groups: List[Tuple[str, List[str]]] = []
    mapping = [
        ("missing", model.get("missing_quality_tasks") or []),
        ("capability_unavailable", model.get("capability_unavailable_tasks") or []),
        ("measured_failure", model.get("capability_measured_failure_tasks") or []),
        ("recovery_exhausted", model.get("recovery_exhausted_tasks") or model.get("think_ineffective_tasks") or []),
        ("stale", model.get("stale_quality_tasks") or []),
    ]
    for key, tasks in mapping:
        if tasks:
            groups.append((_CAUSE_LABELS[key], [_task_label(task) + f" <code>{_esc(task)}</code>" for task in tasks]))
    return groups


def _all_tasks(model: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    categories = model.get("categories") if isinstance(model.get("categories"), dict) else {}
    for category, payload in categories.items():
        if not isinstance(payload, dict):
            continue
        for task in payload.get("tasks") or []:
            if isinstance(task, dict):
                row = dict(task)
                row["category"] = category
                rows.append(row)
    return rows


def _history_rows(model: Dict[str, Any], limit: int = 160) -> List[Dict[str, Any]]:
    hist = model.get("history") if isinstance(model.get("history"), list) else []
    out = [row for row in hist if isinstance(row, dict)]
    out.sort(key=lambda row: str(row.get("timestamp") or row.get("run_id") or ""), reverse=True)
    return out[:limit]


def _score_cell(value: Any) -> str:
    n = _num(value)
    if n is None:
        return '<span class="muted">not scored</span>'
    cls = "score-good" if n >= 90 else "score-warn" if n >= 60 else "score-bad"
    return f'<span class="score {cls}">{_fmt(n, 1)}</span>'


def _task_table(tasks: List[Dict[str, Any]]) -> str:
    if not tasks:
        return '<p class="empty">No current task rows found for this model.</p>'
    rows = []
    for task in tasks:
        detail = {
            "prompt": task.get("prompt"),
            "rubric": task.get("rubric"),
            "expectation": task.get("expectation"),
            "reason": task.get("reason"),
            "run_id": task.get("run_id"),
            "import_tag": task.get("import_tag"),
            "scorer": task.get("scorer"),
            "judge_mode": task.get("judge_mode"),
            "error_kind": task.get("error_kind"),
        }
        rows.append(f"""
<tr>
<td><div class="strong">{_esc(_task_label(task.get('task')))}</div><div class="muted mono">{_esc(task.get('task'))}</div></td>
<td>{_esc(_category_label(task.get('category')))}</td>
<td>{_score_cell(task.get('score'))}</td>
<td>{_esc(task.get('reason') or task.get('error_kind') or 'ok')}</td>
<td class="num">{_fmt(task.get('tps'))}</td>
<td class="num">{_fmt(task.get('ttft_ms'), 1)}</td>
<td><details><summary>Evidence</summary><pre>{_esc(json.dumps(detail, indent=2, ensure_ascii=False))}</pre></details></td>
</tr>""")
    return """<div class="table-wrap"><table class="evidence-table"><thead><tr><th>Task</th><th>Category</th><th>Score</th><th>Outcome</th><th>tok/s</th><th>TTFT ms</th><th>Details</th></tr></thead><tbody>""" + "".join(rows) + "</tbody></table></div>"


def _history_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return '<p class="empty">No historical rows are available in the master payload.</p>'
    body = []
    for row in rows:
        body.append(f"""
<tr><td>{_esc(row.get('run_id'))}</td><td>{_esc(_task_label(row.get('task')))}</td><td>{_score_cell(row.get('score'))}</td><td>{_esc(row.get('reason') or row.get('error_kind') or '')}</td><td>{_esc(row.get('benchmark_version') or '')}</td></tr>""")
    return """<div class="table-wrap"><table class="evidence-table"><thead><tr><th>Run</th><th>Task</th><th>Score</th><th>Reason</th><th>Version</th></tr></thead><tbody>""" + "".join(body) + "</tbody></table></div>"


def _category_cards(model: Dict[str, Any]) -> str:
    cats = model.get("categories") if isinstance(model.get("categories"), dict) else {}
    if not cats:
        return '<p class="empty">No category breakdown found.</p>'
    parts = []
    for key, cat in sorted(cats.items()):
        if not isinstance(cat, dict):
            continue
        score = cat.get("score")
        coverage = cat.get("coverage")
        count = len(cat.get("tasks") or [])
        parts.append(f"""<details class="category-detail"><summary><span>{_esc(_category_label(key))}</span><span>{_fmt(score)} · {_pct(coverage)} · {count} task(s)</span></summary>{_task_table(cat.get('tasks') or [])}</details>""")
    return "".join(parts)


def _model_page(model: Dict[str, Any], payload: Dict[str, Any]) -> str:
    name = _model_name(model)
    axes = model_axes(model)
    profile = model.get("long_context_profile") or {}
    groups = _evidence_groups(model)
    group_html = "".join(
        f"<details class='gap-group' open><summary>{_esc(title)} <span>{len(items)}</span></summary><ul>{''.join(f'<li>{item}</li>' for item in items)}</ul></details>"
        for title, items in groups
    ) or '<p class="empty">No missing or limited evidence remains for applicable tasks.</p>'
    reasons = "".join(f"<li>{_esc(reason)}</li>" for reason in (model.get("quality_status_reasons") or [])) or "<li>No ranking warnings.</li>"
    badges = "".join(_badge_html(badge) for badge in _badges(model))
    return f"""<!doctype html><html lang="en"><head>{_head_model('Model detail - ' + name)}</head><body>
<header class="topbar"><a href="../index.html" class="brand">LLM ModelBench V3.1</a><nav>{_nav("dashboard", model=True)}</nav></header>
<main class="page model-page">
<section class="hero model-hero"><div><p class="eyebrow">Model detail</p><h1>{_esc(name)}</h1><p class="verdict">{_esc(_verdict(model))}</p><div class="badge-row">{badges}</div></div><div class="hero-metrics">{_metric_card('Quality', _fmt(model.get('overall_mean_score')))}{_metric_card('Coverage', _pct(model.get('coverage_ratio')))}{_metric_card('Speed', _fmt(model.get('tok_s')) + ' tok/s')}{_metric_card('Size', _fmt(model.get('size_gb')) + ' GB')}</div></section>
<section class="panel"><h2>Operating profile</h2><div class="metric-grid small-grid">{_metric_card('Max verified context', _fmt(profile.get('max_verified_ctx') or axes.get('max_verified_ctx')))}{_metric_card('64k status', profile.get('target_status') or axes.get('target_context_status') or 'not profiled')}{_metric_card('64k decode', (_fmt(profile.get('target_tps') or axes.get('target_decode_tps')) + ' tok/s') if (profile.get('target_tps') or axes.get('target_decode_tps')) else '-')}{_metric_card('Max offload', _pct(profile.get('max_offload_fraction') or axes.get('max_offload_fraction') or 0))}</div><ul class="notes">{reasons}</ul></section>
<section class="panel"><h2>Evidence still needed or limited</h2>{group_html}</section>
<section class="panel command-panel"><h2>Ranking controls</h2><p class="muted">Static reports cannot change files directly. Copy a command and run it from the repository root if this model should be hidden from the canonical view.</p><pre>./llmb rankings --exclude-model '{_esc(name)}' --reason 'diagnostic or duplicate model' --rescan</pre><pre>./llmb rankings --include-model '{_esc(name)}' --rescan</pre></section>
<section class="panel"><h2>Category breakdown</h2>{_category_cards(model)}</section>
<section class="panel"><h2>Current selected task evidence</h2>{_task_table(_all_tasks(model))}</section>
<section class="panel"><details><summary><h2>Historical rows</h2></summary>{_history_table(_history_rows(model))}</details></section>
<section class="panel"><details><summary><h2>Analyst JSON</h2></summary><pre>{_esc(json.dumps(model, indent=2, ensure_ascii=False))}</pre></details></section>
</main></body></html>"""


def _head(title: str = "LLM ModelBench Rankings V3.1") -> str:
    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_esc(title)}</title><link rel="stylesheet" href="assets/report.css">"""


def _head_model(title: str = "LLM ModelBench Rankings V3.1") -> str:
    return f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_esc(title)}</title><link rel="stylesheet" href="../assets/report.css">"""


def _hero_rows(v3: Dict[str, Any], models: List[Dict[str, Any]]) -> str:
    overall = (v3.get("use_case_rankings", {}).get("overall") or {}).get("rows") or []
    coding = (v3.get("use_case_rankings", {}).get("coding") or {}).get("rows") or []
    multimodal = (v3.get("use_case_rankings", {}).get("multimodal") or {}).get("rows") or []
    small = (v3.get("use_case_rankings", {}).get("small_fast") or {}).get("rows") or []
    rows = [
        ("General use", overall[0] if overall else None, "Best current overall operational score in this evidence snapshot."),
        ("Coding", coding[0] if coding else None, "Best coding-oriented operational score."),
        ("Multimodal", multimodal[0] if multimodal else None, "Top OCR/vision entry; equal-quality ceiling ties are shown below."),
        ("Small fast", small[0] if small else None, "Efficiency-oriented option for lighter daily use."),
    ]
    out = []
    for label, row, note in rows:
        if not row:
            out.append(f"<article class='decision-row'><div><h3>{_esc(label)}</h3><p class='muted'>No eligible model in this snapshot.</p></div></article>")
            continue
        model_name = row.get("model")
        model = next((m for m in models if _model_name(m) == model_name), {})
        out.append(f"""<article class="decision-row"><div><p class="eyebrow">{_esc(label)}</p><h3><a href="{_esc(_model_link(model))}">{_esc(model_name)}</a></h3><p>{_esc(note)}</p><div class="badge-row">{''.join(_badge_html(b) for b in row.get('badges', []))}</div></div><div class="decision-metrics"><span>{_fmt(row.get('quality'))}<small>quality</small></span><span>{_pct(row.get('coverage'))}<small>coverage</small></span><span>{_fmt(row.get('speed_tps'))}<small>tok/s</small></span><span>{_fmt(row.get('size_gb'))}<small>GB</small></span></div></article>""")
    return "".join(out)



def _nav(active: str = "dashboard", *, model: bool = False) -> str:
    prefix = "../" if model else ""
    items = [
        ("dashboard", prefix + "index.html", "Dashboard"),
        ("top5", prefix + "top5.html", "The Best 5"),
        ("compare", prefix + "compare.html", "Compare"),
        ("methodology", prefix + "methodology.html", "Methodology"),
        ("help", prefix + "help.html", "Help / About"),
    ]
    return "".join(
        f'<a class="{"active" if key == active else ""}" href="{_esc(href)}">{_esc(label)}</a>'
        for key, href, label in items
    )


def _index_html(site_payload: Dict[str, Any]) -> str:
    data = json.dumps(site_payload, ensure_ascii=False).replace("</", "<\\/")
    summary = site_payload.get("summary") or {}
    status_counts = summary.get("status_counts") or {}
    context_counts = summary.get("long_context_status_counts") or {}
    complete = status_counts.get("complete", 0)
    not_profiled = context_counts.get("not_profiled", 0)
    return f"""<!doctype html><html lang="en"><head>{_head()}</head><body>
<header class="topbar"><a class="brand" href="index.html">LLM ModelBench V3.1</a><nav>{_nav('dashboard')}</nav></header>
<main class="page">
<section class="hero"><div><p class="eyebrow">Decision dashboard</p><h1>Readable model rankings with drill-down evidence.</h1><p class="lead">Scores are unchanged from the master evidence. This view separates decisions from audit detail and gives every model its own page.</p></div><div class="summary-stack"><span><b>{_esc(summary.get('models', 0))}</b><small>models in this snapshot</small></span><span><b>{_esc(complete)}</b><small>complete terminal classifications</small></span><span><b>{_esc(not_profiled)}</b><small>models without a validated 64k operating profile</small></span></div></section>
<section class="decision-list">{_hero_rows(site_payload['v3'], site_payload['models'])}</section>
<section class="panel explanation-panel"><h2>How to read this page</h2><p><b>Quality</b> is the measured benchmark score. <b>Operational fit score</b> is a V3.1 sorting value that may exceed 100 because it adds coverage, speed, size efficiency and context-readiness bonuses. It is for ordering inside one use case, not a task score.</p><p>The slider filters by measured quality, not by operational fit score. Use <a href="methodology.html">Methodology</a> for details.</p></section>
<section class="toolbar panel"><label>Search <input id="q" placeholder="model name, class, capability"></label><label>Class <select id="classFilter"><option value="">All</option></select></label><label>Use case <select id="caseFilter"></select></label><label>Minimum measured quality <input id="minQuality" type="range" min="0" max="100" value="0"><span id="minQualityLabel">0</span></label></section>
<section class="panel"><div class="section-head"><h2 id="caseTitle">Models</h2><p id="caseNote" class="muted"></p></div><div id="rankRows" class="rank-list"></div></section>
<section class="panel"><details><summary><h2>Analyst details</h2></summary><p>Legacy dense report remains available as <code>../master_report.html</code>. Operational V3 single-file remains <code>../master_report_v3.html</code>. Canonical V3.1 data is available at <a href="../master_report_v3_1_data.json">../master_report_v3_1_data.json</a>.</p><pre>{_esc(json.dumps(summary, indent=2))}</pre></details></section>
</main><script>window.RANKINGS_V31={data};</script><script src="assets/report.js"></script></body></html>"""


def _methodology_html(site_payload: Dict[str, Any]) -> str:
    summary = site_payload.get("summary") or {}
    return f"""<!doctype html><html lang="en"><head>{_head('Methodology - V3.1')}</head><body><header class="topbar"><a class="brand" href="index.html">LLM ModelBench V3.1</a><nav>{_nav('methodology')}</nav></header><main class="page">
<section class="hero"><div><p class="eyebrow">Methodology</p><h1>Decision first, evidence on demand.</h1><p class="lead">V3.1 does not change benchmark scores. It changes how evidence is grouped, explained and linked.</p></div><div class="summary-stack"><span><b>{_esc(summary.get('models', 0))}</b><small>models</small></span><span><b>{_esc((summary.get('status_counts') or {}).get('complete', 0))}</b><small>complete</small></span><span><b>{_esc(summary.get('capability_limited_models', 0))}</b><small>capability-limited</small></span></div></section>
<section class="panel"><h2>Scores shown in V3.1</h2><ul><li><b>Quality</b> is the benchmark evidence score. It stays on the familiar 0-100 scale unless the underlying scorer explicitly says otherwise.</li><li><b>Coverage</b> is the fraction of applicable positive-difficulty tasks with current terminal evidence.</li><li><b>Operational fit score</b> is the right-side number on ranking rows. It is a sorting value, not a raw benchmark score. It can exceed 100 because V3.1 adds small bonuses for coverage, speed, smaller size and context-readiness.</li><li><b>64k status</b> is separate from quality. A model can be high quality but not yet profiled for regular 64k use.</li></ul></section>
<section class="panel"><h2>Evidence tiers</h2><ul><li><b>Complete</b>: all applicable current positive-difficulty tasks have terminal evidence.</li><li><b>Capability-limited</b>: unavailable installed-build lanes are labelled instead of counted as missing work.</li><li><b>Measured failure</b>: the endpoint responded and a real scored task measured zero-quality output.</li><li><b>Recovery-limited</b>: bounded recovery was exhausted. No positive score is fabricated.</li><li><b>Not profiled for 64k</b>: no validated target-context operating profile is present yet. This is not the same as a model failing 64k.</li></ul></section>
<section class="panel"><h2>Use-case leaderboards</h2><p>Each use case has its own ranking view. Filtering by class narrows the same use-case list to a model class such as coding, vision, reasoning or embedding. It does not change the underlying score formula.</p><p>The class filter changes the visible set, so the page text updates to show whether you are looking at all models or one class inside a use case.</p></section>
<section class="panel"><h2>Static report limits</h2><p>This calm site is static HTML. It cannot delete or alter evidence. Model pages provide copyable CLI commands for reversible ranking exclusions. The raw run evidence remains append-only unless a user deliberately deletes files outside the report.</p></section>
<section class="panel"><h2>Data files</h2><p>The calm site is generated from <code>../master_report_v3_1_data.json</code>, which is useful for auditing, automated checks or custom visualisations. Most users do not need to open it directly.</p></section>
</main></body></html>"""


def _compare_html(site_payload: Dict[str, Any]) -> str:
    data = json.dumps(site_payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang="en"><head>{_head('Compare - V3.1')}</head><body><header class="topbar"><a class="brand" href="index.html">LLM ModelBench V3.1</a><nav>{_nav('compare')}</nav></header><main class="page"><section class="hero"><div><p class="eyebrow">Compare</p><h1>Compare two models without the full fleet noise.</h1><p class="lead">Choose two models, then open either model page for full task evidence.</p></div></section><section class="panel compare-controls"><label>Left model<select id="leftModel"></select></label><label>Right model<select id="rightModel"></select></label></section><section id="compareOut" class="compare-grid"></section></main><script>window.RANKINGS_V31={data};</script><script src="assets/report.js"></script></body></html>"""


def _top5_html(site_payload: Dict[str, Any]) -> str:
    sections = []
    for key, use_case in (site_payload.get("use_cases") or {}).items():
        rows = list(use_case.get("rows") or [])[:5]
        if not rows:
            continue
        cards = []
        for row in rows:
            cards.append(f"""<article class="top5-row"><div class="rank-num">#{_esc(row.get('rank'))}</div><div><h3><a href="{_esc(row.get('page') or '#')}">{_esc(row.get('model'))}</a></h3><p class="subtle">{_esc(row.get('class') or 'unknown')} · quality {_fmt(row.get('quality'))} · coverage {_pct(row.get('coverage'))} · speed {_fmt(row.get('speed_tps'))} tok/s · size {_fmt(row.get('size_gb'))} GB</p></div><div class="op-score"><small>Operational fit</small><b>{_fmt(row.get('score'))}</b></div></article>""")
        sections.append(f"""<section class="panel"><div class="section-head"><h2>{_esc(use_case.get('title'))}</h2><p class="muted">Top 5 for this use case</p></div><div class="top5-list">{''.join(cards)}</div></section>""")
    return f"""<!doctype html><html lang="en"><head>{_head('The Best 5 - V3.1')}</head><body><header class="topbar"><a class="brand" href="index.html">LLM ModelBench V3.1</a><nav>{_nav('top5')}</nav></header><main class="page"><section class="hero"><div><p class="eyebrow">The Best 5</p><h1>Top five models for each use case.</h1><p class="lead">This page keeps categories separate. A model that is best for OCR is not automatically best for coding or agentic work.</p></div></section>{''.join(sections)}</main></body></html>"""


def _help_html(site_payload: Dict[str, Any]) -> str:
    return f"""<!doctype html><html lang="en"><head>{_head('Help and commands - V3.1')}</head><body><header class="topbar"><a class="brand" href="index.html">LLM ModelBench V3.1</a><nav>{_nav('help')}</nav></header><main class="page"><section class="hero"><div><p class="eyebrow">Help / About</p><h1>How to use LLM ModelBench.</h1><p class="lead">Common terminal commands and what each report artifact means.</p></div></section>
<section class="panel"><h2>Typical workflow</h2><pre>./llmb doctor
./llmb inventory --auto
./llmb plan --level short --models 'model-a;model-b'
./llmb run --level short --models 'model-a;model-b' --judge off --yes
./llmb rankings --runs-dir runs --out rankings --rescan
./llmb-watch --runs-dir runs --follow-queue</pre></section>
<section class="panel"><h2>Ranking controls</h2><pre>./llmb rankings --exclude-model 'model-name' --reason 'diagnostic duplicate' --rescan
./llmb rankings --include-model 'model-name' --rescan
./llmb rankings --list-excluded
./llmb run ... --no-ranking-update
./llmb run ... --separate-ranking</pre><p class="muted">Exclusions are non-destructive. Raw evidence under <code>runs/</code> is not deleted.</p></section>
<section class="panel"><h2>Reports generated</h2><ul><li><code>rankings/v3_1/index.html</code>: calm day-to-day browsing UI.</li><li><code>rankings/v3_1/models/*.html</code>: individual model detail pages.</li><li><code>rankings/master_report_v3.html</code>: portable operational single-file report.</li><li><code>rankings/master_report_v3_1_data.json</code>: canonical calm-site data payload.</li><li><code>model_cards/</code>: operating cards for model routing decisions.</li></ul></section>
<section class="panel"><h2>Safety model</h2><ul><li>Benchmark evidence is append-only by default.</li><li>Ranking exclusions hide evidence from views; they do not delete raw rows.</li><li>Use <code>--separate-ranking</code> for playground or diagnostic runs that should not touch the main leaderboard.</li><li>Use <code>--no-ranking-update</code> when you want to update rankings later manually.</li></ul></section>
</main></body></html>"""


def _css() -> str:
    return r"""
:root{--bg:#0b1020;--bg2:#121a2a;--panel:#1a2538;--panel2:#21304a;--line:#42506a;--text:#f4f7fb;--muted:#c3cee0;--soft:#e2eaf7;--accent:#96cbff;--cyan:#8be3f2;--green:#55dca0;--amber:#ffc978;--red:#ff8585;--purple:#ceb5ff;--shadow:0 18px 42px rgba(0,0,0,.22)}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top left,#1b2a49 0,#0f1728 420px,#0a0f1b 100%);color:var(--text);font:18px/1.58 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}.topbar{position:sticky;top:0;z-index:10;display:flex;justify-content:space-between;gap:20px;align-items:center;padding:18px 32px;border-bottom:1px solid var(--line);background:rgba(15,23,40,.94);backdrop-filter:blur(10px)}.brand{font-weight:850;color:var(--text)}nav{display:flex;gap:16px;flex-wrap:wrap}nav a.active{color:var(--cyan);font-weight:800}.page{max-width:1220px;margin:0 auto;padding:28px 24px 80px}.hero{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:26px;align-items:stretch;margin-bottom:22px}.hero h1{font-size:42px;line-height:1.06;margin:8px 0 12px}.lead{font-size:20px;color:var(--soft)}.eyebrow{letter-spacing:.12em;text-transform:uppercase;color:var(--cyan);font-weight:850;font-size:13px}.summary-stack,.hero-metrics{display:grid;gap:12px}.summary-stack span,.metric-card,.panel,.decision-row,.rank-row,.top5-row{background:linear-gradient(180deg,var(--panel),#141f32);border:1px solid var(--line);border-radius:22px;box-shadow:var(--shadow)}.summary-stack span{padding:18px;display:grid;gap:4px}.summary-stack b{font-size:26px}.summary-stack small{color:var(--muted)}.metric-card{padding:16px}.metric-label{color:var(--muted);font-size:14px}.metric-value{font-size:28px;font-weight:850}.metric-sub{font-size:13px;color:var(--muted)}.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}.small-grid .metric-value{font-size:22px}.panel{padding:22px;margin:18px 0}.explanation-panel{font-size:17px}.decision-list{display:grid;gap:14px;margin:20px 0}.decision-row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:18px;padding:22px}.decision-row h3{font-size:26px;margin:2px 0}.decision-row p{margin:.2rem 0;color:var(--soft)}.decision-metrics{display:grid;grid-template-columns:repeat(4,88px);gap:10px;align-items:center}.decision-metrics span{display:grid;text-align:center;background:rgba(255,255,255,.055);border:1px solid var(--line);border-radius:16px;padding:12px;font-weight:800}.decision-metrics small{font-size:12px;color:var(--muted);font-weight:600}.toolbar{display:grid;grid-template-columns:2fr 1fr 1.2fr 1.2fr;gap:14px;align-items:end}.toolbar label,.compare-controls label{display:grid;gap:7px;color:var(--muted);font-size:14px}input,select{width:100%;border:1px solid var(--line);border-radius:14px;background:#101827;color:var(--text);padding:12px 14px;font:inherit}select option{background:#101827;color:#f4f7fb}.section-head{display:flex;justify-content:space-between;gap:18px;align-items:end}.section-head h2{font-size:28px;margin:0}.rank-list{display:grid;gap:12px}.rank-row{display:grid;grid-template-columns:70px minmax(0,1.25fr) 1.9fr 150px;gap:18px;padding:18px;align-items:center}.rank-num{font-size:24px;font-weight:850;color:var(--cyan)}.model-title{font-size:20px;font-weight:850}.subtle{color:var(--muted);font-size:14px}.badge-row{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px}.badge{display:inline-flex;align-items:center;border-radius:999px;padding:5px 10px;font-size:13px;border:1px solid var(--line);background:rgba(255,255,255,.055);color:var(--soft)}.badge-good{border-color:rgba(85,220,160,.65);background:rgba(85,220,160,.15);color:#dcffef}.badge-warn{border-color:rgba(255,201,120,.65);background:rgba(255,201,120,.16);color:#fff2d8}.badge-bad{border-color:rgba(255,133,133,.65);background:rgba(255,133,133,.16);color:#ffe3e3}.badge-context{border-color:rgba(150,203,255,.65);background:rgba(150,203,255,.15);color:#deefff}.badge-cap{border-color:rgba(206,181,255,.65);background:rgba(206,181,255,.15);color:#f0e9ff}.score{font-weight:850}.score-good{color:var(--green)}.score-warn{color:var(--amber)}.score-bad{color:var(--red)}.muted{color:var(--muted)}.mono,code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.strong{font-weight:800}.num{text-align:right;font-variant-numeric:tabular-nums}.bar{height:10px;border-radius:20px;background:#111a2c;overflow:hidden;border:1px solid var(--line)}.bar>i{display:block;height:100%;border-radius:20px;background:linear-gradient(90deg,var(--accent),var(--green))}.op-score{display:grid;gap:4px;justify-items:end;text-align:right;background:rgba(255,255,255,.045);border:1px solid var(--line);border-radius:18px;padding:10px 12px}.op-score b{font-size:22px}.op-score small{color:var(--muted);font-size:12px}.model-hero{grid-template-columns:minmax(0,1fr) 420px}.verdict{font-size:20px;color:var(--soft)}.category-detail,.gap-group,details{background:rgba(255,255,255,.03);border:1px solid var(--line);border-radius:16px;padding:12px;margin:10px 0}.category-detail summary,.gap-group summary,details summary{cursor:pointer;font-weight:800;display:flex;justify-content:space-between;gap:16px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:16px}.evidence-table{width:100%;border-collapse:collapse;min-width:820px}.evidence-table th,.evidence-table td{padding:12px;border-bottom:1px solid var(--line);vertical-align:top}.evidence-table th{background:#111a2c;color:var(--soft);text-align:left}.evidence-table pre,pre{white-space:pre-wrap;overflow:auto;background:#0b1220;border:1px solid var(--line);border-radius:14px;padding:14px;font-size:13px}.empty{color:var(--muted);padding:18px}.notes{color:var(--soft)}.compare-controls{display:grid;grid-template-columns:1fr 1fr;gap:14px}.compare-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.top5-list{display:grid;gap:12px}.top5-row{display:grid;grid-template-columns:70px minmax(0,1fr) 150px;gap:16px;align-items:center;padding:16px}.top5-row h3{margin:0 0 4px;font-size:20px}@media(max-width:900px){body{font-size:16px}.topbar{padding:14px 18px}.hero,.model-hero,.decision-row,.rank-row,.toolbar,.compare-grid,.top5-row{grid-template-columns:1fr}.decision-metrics{grid-template-columns:repeat(2,1fr)}.page{padding:18px 14px}.hero h1{font-size:32px}.op-score{justify-items:start;text-align:left}}
"""


def _js() -> str:
    return r"""
const DATA=window.RANKINGS_V31||{};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const n=x=>Number.isFinite(Number(x))?Number(x):null;
const fmt=(x,d=2)=>n(x)===null?'–':(Math.abs(n(x)-Math.round(n(x)))<.001?String(Math.round(n(x))):n(x).toFixed(d));
const pct=x=>n(x)===null?'–':(n(x)*100).toFixed(1)+'%';
const modelName=m=>m?.display_name||m?.model||m?.digest||'unknown model';
function badge(label){let low=String(label||'').toLowerCase();let cls='badge';if(low.includes('complete'))cls+=' badge-good';else if(low.includes('warning')||low.includes('slow')||low.includes('provisional'))cls+=' badge-warn';else if(low.includes('error')||low.includes('ineligible'))cls+=' badge-bad';else if(low.includes('64k'))cls+=' badge-context';else if(low.includes('capability'))cls+=' badge-cap';return `<span class="${cls}">${esc(label)}</span>`}
function rowHtml(r,idx){return `<article class="rank-row"><div class="rank-num">${r.rank?('#'+r.rank):('#'+(idx+1))}</div><div><div class="model-title"><a href="${esc(r.page||'#')}">${esc(r.model)}</a></div><div class="subtle">${esc(r.class||'unknown')} · ${fmt(r.size_gb)} GB</div></div><div><div class="badge-row">${(r.badges||[]).map(b=>badge(b.label||b)).join('')}</div><div class="subtle">quality ${fmt(r.quality)} · coverage ${pct(r.coverage)} · speed ${fmt(r.speed_tps)} tok/s · 64k ${esc(r.target_context_status||'not profiled')}</div><div class="bar" title="Measured quality"><i style="width:${Math.max(0,Math.min(100,n(r.quality)||0))}%"></i></div></div><div class="op-score" title="Operational fit score: quality plus coverage, speed, size efficiency and context-readiness tie-breakers. It can exceed 100."><small>Operational fit</small><b>${fmt(r.score)}</b></div></article>`}
function currentRows(){let key=document.getElementById('caseFilter')?.value||'overall';let rows=(DATA.use_cases?.[key]?.rows||[]);let q=(document.getElementById('q')?.value||'').toLowerCase();let cls=document.getElementById('classFilter')?.value||'';let min=Number(document.getElementById('minQuality')?.value||0);return rows.filter(r=>{let hay=[r.model,r.class,(r.families||[]).join(' '),(r.badges||[]).map(b=>b.label||b).join(' ')].join(' ').toLowerCase();return (!q||hay.includes(q))&&(!cls||r.class===cls)&&((n(r.quality)||0)>=min)})}
function renderRows(){let key=document.getElementById('caseFilter')?.value||'overall';let uc=DATA.use_cases?.[key]||{};let rows=currentRows();let cls=document.getElementById('classFilter')?.value||'';let title=document.getElementById('caseTitle');let note=document.getElementById('caseNote');let out=document.getElementById('rankRows');if(title)title.textContent=cls?`${cls} models in ${uc.title||'selected use case'}`:(uc.title||'Models');if(note){let desc=uc.description||'';if(cls)desc=`Showing ${cls}-class models within this use case. ${desc}`;note.textContent=desc+' · '+rows.length+' shown';}if(out)out.innerHTML=rows.map(rowHtml).join('')||'<p class="empty">No matching models.</p>'}
function initIndex(){let cf=document.getElementById('caseFilter');if(!cf)return;Object.entries(DATA.use_cases||{}).forEach(([k,v])=>cf.innerHTML+=`<option value="${esc(k)}">${esc(v.title)} (${v.count})</option>`);let classes=[...new Set((DATA.models||[]).map(m=>m.class).filter(Boolean))].sort();let cl=document.getElementById('classFilter');classes.forEach(c=>cl.innerHTML+=`<option value="${esc(c)}">${esc(c)}</option>`);['q','classFilter','caseFilter'].forEach(id=>document.getElementById(id)?.addEventListener('input',renderRows));let ms=document.getElementById('minQuality');ms?.addEventListener('input',e=>{document.getElementById('minQualityLabel').textContent=e.target.value;renderRows()});renderRows()}
function initCompare(){let left=document.getElementById('leftModel'),right=document.getElementById('rightModel'),out=document.getElementById('compareOut');if(!left||!right||!out)return;(DATA.models||[]).forEach((m,i)=>{let label=modelName(m);let opt=`<option value="${i}">${esc(label)}</option>`;left.innerHTML+=opt;right.innerHTML+=opt});right.value=(DATA.models||[]).length>1?'1':'0';function card(m){return `<section class="panel"><h2>${esc(modelName(m))}</h2><p>${esc(m.verdict||'')}</p><p>Quality <b>${fmt(m.overall_mean_score)}</b> · Coverage <b>${pct(m.coverage_ratio)}</b> · Speed <b>${fmt(m.tok_s)} tok/s</b> · Size <b>${fmt(m.size_gb)} GB</b></p><div class="badge-row">${(m.badges||[]).map(b=>badge(b.label||b)).join('')}</div><p><a href="${esc(m.page||'#')}">Open model page</a></p></section>`}function render(){out.innerHTML=card(DATA.models[left.value]||{})+card(DATA.models[right.value]||{})}left.oninput=right.oninput=render;render()}
initIndex();initCompare();
"""

def _v31_models(master_payload: Dict[str, Any], v3_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    v3_by_name = {row.get("model"): row for row in v3_payload.get("models") or []}
    for model in master_payload.get("models") or []:
        if not isinstance(model, dict):
            continue
        item = dict(model)
        name = _model_name(item)
        page = _slug(name, item.get("digest"))
        item["page"] = "models/" + page
        item["page_file"] = page
        item["verdict"] = _verdict(item)
        item["badges"] = _badges(item)
        item["axes"] = model_axes(item)
        item["task_count"] = len(_all_tasks(item))
        item["evidence_groups"] = [{"title": title, "tasks": [re.sub(r"<[^>]+>", "", task) for task in tasks]} for title, tasks in _evidence_groups(item)]
        if name in v3_by_name:
            item["v3_inventory"] = v3_by_name[name]
        out.append(item)
    return out


def _use_cases_with_pages(v3_payload: Dict[str, Any], models: List[Dict[str, Any]]) -> Dict[str, Any]:
    page_by_name = {_model_name(model): model.get("page") for model in models}
    out: Dict[str, Any] = {}
    for key, payload in (v3_payload.get("use_case_rankings") or {}).items():
        rows = []
        for row in payload.get("rows") or []:
            r = dict(row)
            r["page"] = page_by_name.get(r.get("model"), "#")
            rows.append(r)
        out[key] = {**payload, "rows": rows}
    return out


def build_v31_payload(master_payload: Dict[str, Any]) -> Dict[str, Any]:
    v3_payload = build_v3_payload(master_payload)
    models = _v31_models(master_payload, v3_payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(),
        "v3_schema_version": V3_SCHEMA_VERSION,
        "summary": v3_payload.get("summary") or {},
        "methodology": v3_payload.get("methodology") or {},
        "models": models,
        "use_cases": _use_cases_with_pages(v3_payload, models),
        "v3": v3_payload,
    }


def write_v31_artifacts(master_payload: Dict[str, Any], rankings_dir: Path) -> Dict[str, str]:
    rankings_dir.mkdir(parents=True, exist_ok=True)
    payload = build_v31_payload(master_payload)
    data_path = rankings_dir / "master_report_v3_1_data.json"
    html_path = rankings_dir / "master_report_v3_1.html"
    site_dir = rankings_dir / "v3_1"
    if site_dir.exists():
        shutil.rmtree(site_dir)
    (site_dir / "assets").mkdir(parents=True)
    (site_dir / "models").mkdir(parents=True)
    (site_dir / "data").mkdir(parents=True)
    (site_dir / "assets" / "report.css").write_text(_css())
    (site_dir / "assets" / "report.js").write_text(_js())
    (site_dir / "index.html").write_text(_index_html(payload))
    (site_dir / "methodology.html").write_text(_methodology_html(payload))
    (site_dir / "compare.html").write_text(_compare_html(payload))
    (site_dir / "top5.html").write_text(_top5_html(payload))
    (site_dir / "help.html").write_text(_help_html(payload))
    (site_dir / "data" / "site_manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "generated_at": payload["generated_at"],
        "models": len(payload["models"]),
        "use_cases": list(payload["use_cases"].keys()),
    }, indent=2, sort_keys=True))
    for model in payload["models"]:
        page = site_dir / "models" / str(model.get("page_file"))
        page.write_text(_model_page(model, payload).replace('href="assets/report.css"', 'href="../assets/report.css"'))
    data_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    html_path.write_text("""<!doctype html><meta charset=\"utf-8\"><meta http-equiv=\"refresh\" content=\"0; url=v3_1/index.html\"><title>LLM ModelBench Rankings V3.1</title><p><a href=\"v3_1/index.html\">Open Rankings V3.1 calm split report</a></p>""")
    return {
        "v31_schema_version": SCHEMA_VERSION,
        "v31_data_path": str(data_path),
        "v31_html_path": str(html_path),
        "v31_site_path": str(site_dir / "index.html"),
        "v31_model_pages": str(len(payload["models"])),
    }
