"""Rankings V3 operational leaderboard payload and standalone HTML.

V3 is deliberately derived from the already-frozen master rankings summary. It
adds operational views, context-readiness semantics, and capability/recovery
badges without changing the underlying task scores or source evidence.
"""
from __future__ import annotations

import html
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_VERSION = "llm-modelbench.rankings.v3"

USE_CASES: Dict[str, Dict[str, Any]] = {
    "overall": {
        "title": "Overall text assistants",
        "description": "Text-capable models ranked by current weighted quality, with coverage and operational tie-breakers.",
        "families_any": ["text"],
        "categories_any": [],
        "require_score": True,
    },
    "coding": {
        "title": "Coding",
        "description": "Code generation, debugging, patching and FIM-adjacent evidence where available.",
        "categories_any": ["coding"],
        "families_any": ["text", "insert"],
        "category_boosts": {"coding": 1.0},
    },
    "reasoning": {
        "title": "Reasoning",
        "description": "Reasoning and deterministic puzzle-style tasks, with terminal recovery evidence preserved.",
        "categories_any": ["reasoning"],
        "families_any": ["text"],
        "category_boosts": {"reasoning": 1.0},
    },
    "agentic_tool": {
        "title": "Agentic and tool use",
        "description": "Tool-call, structured action and agentic workflow tasks. Long-context readiness is shown separately.",
        "categories_any": ["agentic_tool", "tool", "tools"],
        "families_any": ["tools", "text"],
        "category_boosts": {"agentic_tool": 1.0, "tool": 1.0, "tools": 1.0},
    },
    "long_context_64k": {
        "title": "64k context readiness",
        "description": "Models with long-context evidence, ordered by verified context, target-context speed and quality.",
        "families_any": ["text"],
        "categories_any": ["long_context"],
        "long_context": True,
        "min_verified_ctx": 64000,
    },
    "multimodal": {
        "title": "Multimodal / OCR / PDF",
        "description": "Vision, OCR and PDF tasks. Capability-unavailable lanes are labelled instead of hidden.",
        "families_any": ["vision"],
        "categories_any": ["ocr", "pdf"],
        "category_boosts": {"ocr": 1.0, "pdf": 1.0},
    },
    "retrieval_rag": {
        "title": "Retrieval and RAG",
        "description": "Retrieval diagnostics and RAG-like factual lookup tasks.",
        "categories_any": ["retrieval", "rag"],
        "families_any": ["text", "embedding"],
        "category_boosts": {"retrieval": 1.0, "rag": 1.0},
    },
    "embedding": {
        "title": "Embedding specialists",
        "description": "Embedding-only or embedding-capable models shown separately from text assistant rankings.",
        "families_any": ["embedding"],
        "categories_any": ["embedding", "retrieval"],
        "category_boosts": {"embedding": 1.0, "retrieval": 0.5},
        "allow_no_overall": True,
    },
    "small_fast": {
        "title": "Small and fast",
        "description": "Lower-storage models ordered by quality first, then speed and size.",
        "families_any": ["text"],
        "categories_any": [],
        "max_size_gb": 9.5,
        "require_score": True,
    },
}

_STATUS_WEIGHT = {
    "complete": 1.0,
    "complete_for_applicable_capabilities": 0.98,
    "capability_limited": 0.95,
    "recovery_exhausted": 0.9,
    "environment_limited": 0.85,
    "provisional": 0.6,
    "ineligible": 0.1,
}

_LONG_CONTEXT_STATUS_WEIGHT = {
    "ready": 1.0,
    "slow": 0.82,
    "verified_speed_unavailable": 0.72,
    "behavior_warning": 0.55,
    "impractical_speed": 0.45,
    "target_not_reached": 0.2,
    "not_verified": 0.0,
    None: 0.0,
}


def _num(value: Any, default: Optional[float] = None) -> Optional[float]:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _categories(model: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    cats = model.get("categories")
    return cats if isinstance(cats, dict) else {}


def _category_score(model: Dict[str, Any], names: Iterable[str]) -> Optional[float]:
    scores = []
    for name in names:
        score = _num((_categories(model).get(name) or {}).get("score"))
        if score is not None:
            scores.append(score)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 3)


def _max_category_score(model: Dict[str, Any]) -> Optional[float]:
    scores = [_num(cat.get("score")) for cat in _categories(model).values() if isinstance(cat, dict)]
    scores = [score for score in scores if score is not None]
    return round(max(scores), 3) if scores else None


def status_badges(model: Dict[str, Any]) -> List[Dict[str, str]]:
    badges: List[Dict[str, str]] = []
    status = str(model.get("quality_status") or "unknown")
    badges.append({"kind": "status", "label": status})
    if model.get("capability_limited"):
        badges.append({"kind": "capability", "label": "capability-limited"})
    if model.get("capability_measured_failure"):
        badges.append({"kind": "warning", "label": "measured capability failure"})
    if model.get("recovery_limited"):
        badges.append({"kind": "warning", "label": "recovery-limited"})
    profile = model.get("long_context_profile") or {}
    target_status = profile.get("target_status")
    if target_status:
        badges.append({"kind": "context", "label": f"64k:{target_status}"})
    if profile.get("behavior_suspect") or profile.get("target_behavior_suspect"):
        badges.append({"kind": "warning", "label": "behavior warning"})
    if model.get("error_count"):
        badges.append({"kind": "error", "label": f"errors:{model.get('error_count')}"})
    return badges


def model_axes(model: Dict[str, Any]) -> Dict[str, Any]:
    profile = model.get("long_context_profile") or {}
    quality = _num(model.get("overall_mean_score"))
    coverage = _num(model.get("coverage_ratio"), 0.0) or 0.0
    speed = _num(model.get("tok_s"))
    size = _num(model.get("size_gb"))
    max_ctx = _num(profile.get("max_verified_ctx"), 0.0) or 0.0
    target_tps = _num(profile.get("target_tps"))
    max_offload = _num(profile.get("max_offload_fraction"))
    max_ram_delta = _num(profile.get("max_ollama_pss_delta_mb"))
    if max_ram_delta is None:
        max_ram_delta = _num(profile.get("max_ollama_rss_delta_mb"))
    if max_ram_delta is None:
        max_ram_delta = _num(profile.get("max_ram_delta_mb"))
    long_status = profile.get("target_status")
    return {
        "quality": quality,
        "coverage": round(coverage, 4),
        "speed_tps": speed,
        "size_gb": size,
        "efficiency_quality_per_gb": round((quality or 0.0) / size, 4) if quality is not None and size and size > 0 else None,
        "max_verified_ctx": int(max_ctx) if max_ctx else None,
        "target_context_status": long_status,
        "target_decode_tps": target_tps,
        "max_offload_fraction": max_offload,
        "max_process_ram_delta_mb": max_ram_delta,
        "status_weight": _STATUS_WEIGHT.get(str(model.get("quality_status") or ""), 0.5),
        "long_context_weight": _LONG_CONTEXT_STATUS_WEIGHT.get(long_status, 0.0),
    }


def _matches_use_case(model: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    families = set(_list(model.get("families")))
    cats = set(_categories(model).keys())
    if spec.get("require_score") and _num(model.get("overall_mean_score")) is None:
        return False
    max_size = _num(spec.get("max_size_gb"))
    size = _num(model.get("size_gb"))
    if max_size is not None and (size is None or size > max_size):
        return False
    families_any = set(_list(spec.get("families_any")))
    categories_any = set(_list(spec.get("categories_any")))
    family_match = bool(families_any & families) if families_any else True
    cat_match = bool(categories_any & cats) if categories_any else False
    if categories_any and not (family_match or cat_match):
        return False
    if not categories_any and not family_match:
        return False
    if spec.get("long_context"):
        profile = model.get("long_context_profile") or {}
        if not profile:
            return False
    min_ctx = _num(spec.get("min_verified_ctx"))
    if min_ctx is not None:
        ctx = _num((model.get("long_context_profile") or {}).get("max_verified_ctx"), 0.0) or 0.0
        # Keep not-yet-verified models out of the primary 64k table. They still appear in the full inventory.
        if ctx < min_ctx:
            return False
    if _num(model.get("overall_mean_score")) is None and not spec.get("allow_no_overall"):
        return False
    return True


def _use_case_score(model: Dict[str, Any], spec: Dict[str, Any]) -> Optional[float]:
    axes = model_axes(model)
    quality = axes["quality"]
    if spec.get("category_boosts"):
        cat_score = _category_score(model, spec["category_boosts"].keys())
        if cat_score is not None:
            quality = cat_score
    if quality is None:
        quality = _max_category_score(model)
    if quality is None:
        return None
    score = float(quality) * float(axes["status_weight"])
    score += min(float(axes.get("coverage") or 0.0), 1.0) * 3.0
    speed = _num(axes.get("speed_tps"), 0.0) or 0.0
    score += min(speed, 80.0) / 80.0 * 2.0
    size = _num(axes.get("size_gb"))
    if size and size > 0:
        score += min(8.0 / size, 1.5)
    if spec.get("long_context"):
        score += float(axes["long_context_weight"]) * 20.0
        target_tps = _num(axes.get("target_decode_tps"), 0.0) or 0.0
        score += min(target_tps, 30.0) / 30.0 * 5.0
        max_ctx = _num(axes.get("max_verified_ctx"), 0.0) or 0.0
        score += min(max_ctx / 64000.0, 2.0) * 5.0
    return round(score, 4)


def _rank_entry(model: Dict[str, Any], use_case: str, score: float) -> Dict[str, Any]:
    axes = model_axes(model)
    return {
        "model": model.get("display_name"),
        "digest": model.get("digest"),
        "class": model.get("class"),
        "families": model.get("families") or [],
        "score": score,
        "quality": axes["quality"],
        "coverage": axes["coverage"],
        "speed_tps": axes["speed_tps"],
        "size_gb": axes["size_gb"],
        "max_verified_ctx": axes["max_verified_ctx"],
        "target_context_status": axes["target_context_status"],
        "target_decode_tps": axes["target_decode_tps"],
        "capability_limited": bool(model.get("capability_limited")),
        "recovery_limited": bool(model.get("recovery_limited")),
        "capability_measured_failure": bool(model.get("capability_measured_failure")),
        "badges": status_badges(model),
        "reasons": model.get("quality_status_reasons") or [],
        "use_case": use_case,
    }


def use_case_rankings(models: List[Dict[str, Any]], *, limit: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, spec in USE_CASES.items():
        rows: List[Dict[str, Any]] = []
        for model in models:
            if not _matches_use_case(model, spec):
                continue
            score = _use_case_score(model, spec)
            if score is None:
                continue
            rows.append(_rank_entry(model, name, score))
        rows.sort(key=lambda row: (
            -float(row.get("score") or 0.0),
            -float(row.get("quality") or 0.0),
            -float(row.get("coverage") or 0.0),
            -float(row.get("target_decode_tps") or 0.0),
            -float(row.get("speed_tps") or 0.0),
            float(row.get("size_gb") or 1e9),
            str(row.get("model") or "").lower(),
        ))
        for idx, row in enumerate(rows, start=1):
            row["rank"] = idx
        out[name] = {
            "title": spec["title"],
            "description": spec["description"],
            "count": len(rows),
            "rows": rows[:limit] if limit else rows,
        }
    return out


def build_v3_payload(master_payload: Dict[str, Any]) -> Dict[str, Any]:
    models = list(master_payload.get("models") or [])
    status_counts = Counter(str(model.get("quality_status") or "unknown") for model in models)
    capability_limited = sum(1 for model in models if model.get("capability_limited"))
    recovery_limited = sum(1 for model in models if model.get("recovery_limited"))
    measured_failure = sum(1 for model in models if model.get("capability_measured_failure"))
    long_status = Counter(str((model.get("long_context_profile") or {}).get("target_status") or "not_profiled") for model in models)
    inventory = []
    for model in models:
        inventory.append({
            "model": model.get("display_name"),
            "digest": model.get("digest"),
            "class": model.get("class"),
            "families": model.get("families") or [],
            "quality_status": model.get("quality_status"),
            "overall_rank": model.get("overall_rank"),
            "tie_band": model.get("tie_band"),
            "overall_mean_score": model.get("overall_mean_score"),
            "coverage_ratio": model.get("coverage_ratio"),
            "size_gb": model.get("size_gb"),
            "tok_s": model.get("tok_s"),
            "capability_limited": bool(model.get("capability_limited")),
            "capability_measured_failure": bool(model.get("capability_measured_failure")),
            "recovery_limited": bool(model.get("recovery_limited")),
            "capability_unavailable_tasks": model.get("capability_unavailable_tasks") or [],
            "capability_measured_failure_tasks": model.get("capability_measured_failure_tasks") or [],
            "recovery_exhausted_tasks": model.get("recovery_exhausted_tasks") or [],
            "long_context_profile": model.get("long_context_profile"),
            "axes": model_axes(model),
            "badges": status_badges(model),
            "quality_status_reasons": model.get("quality_status_reasons") or [],
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(),
        "methodology": {
            "scope": "V3 is an operational view over the master evidence. It does not rewrite task scores.",
            "ranking_formula": "use-case score = measured task quality, status confidence, coverage, speed, size efficiency, and context readiness where applicable",
            "evidence_policy": "terminal capability and recovery evidence are labelled; unavailable capabilities are not counted as missing work",
            "long_context_policy": "64k readiness requires successful target-context evidence plus usable speed and behavior checks; needle success alone does not certify agentic reliability",
            "manual_rescan_command": "./llmb rankings --runs-dir runs --out rankings --rescan",
        },
        "summary": {
            "models": len(models),
            "status_counts": dict(sorted(status_counts.items())),
            "capability_limited_models": capability_limited,
            "recovery_limited_models": recovery_limited,
            "capability_measured_failure_models": measured_failure,
            "long_context_status_counts": dict(sorted(long_status.items())),
            "use_cases": list(USE_CASES.keys()),
        },
        "use_case_rankings": use_case_rankings(models),
        "models": inventory,
    }


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _json_for_html(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def render_v3_html(payload: Dict[str, Any]) -> str:
    data = _json_for_html(payload)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>LLM ModelBench Rankings V3</title>
<style>
:root{{--bg:#080b10;--panel:#111827;--panel2:#172033;--line:#2b3548;--text:#eef3fb;--muted:#9aa7bd;--accent:#7cb7ff;--green:#22c55e;--red:#ef4444;--orange:#f59e0b;--blue:#60a5fa;--purple:#a78bfa}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(180deg,#05070b,#0b1020 240px,#080b10);color:var(--text);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}a{{color:var(--accent)}}
header{{padding:22px 28px;border-bottom:1px solid var(--line);background:rgba(8,11,16,.92);position:sticky;top:0;z-index:2;backdrop-filter:blur(8px)}}h1{{margin:0;font-size:26px}}h2{{margin:0 0 10px}}.muted{{color:var(--muted)}}.wrap{{display:grid;grid-template-columns:280px 1fr;min-height:100vh}}aside{{border-right:1px solid var(--line);background:#0b111d;padding:16px;position:sticky;top:78px;height:calc(100vh - 78px);overflow:auto}}main{{padding:18px 22px 60px}}.controls{{display:grid;gap:10px}}input,select,button{{background:#0e1523;color:var(--text);border:1px solid var(--line);border-radius:9px;padding:9px}}button{{cursor:pointer}}button.active{{border-color:var(--accent);background:#12223a}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}}.card{{background:linear-gradient(180deg,var(--panel),#0f1624);border:1px solid var(--line);border-radius:14px;padding:14px;box-shadow:0 10px 24px rgba(0,0,0,.18)}}.model-card{{display:grid;gap:9px}}.big{{font-size:28px;font-weight:800}}.row{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}}.pill{{display:inline-flex;gap:4px;align-items:center;border-radius:999px;padding:3px 8px;border:1px solid var(--line);font-size:12px;background:#111827}}.status-complete,.good{{color:#d1fae5;border-color:rgba(34,197,94,.5);background:rgba(34,197,94,.13)}}.status-provisional,.warn{{color:#ffedd5;border-color:rgba(245,158,11,.55);background:rgba(245,158,11,.14)}}.status-ineligible,.bad{{color:#fee2e2;border-color:rgba(239,68,68,.5);background:rgba(239,68,68,.14)}}.context{{color:#dbeafe;border-color:rgba(96,165,250,.55);background:rgba(96,165,250,.13)}}.cap{{color:#ede9fe;border-color:rgba(167,139,250,.55);background:rgba(167,139,250,.13)}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{color:#cbd5e1;background:#101827;position:sticky;top:78px}}.num{{text-align:right;font-variant-numeric:tabular-nums}}.bar{{height:8px;background:#1f2937;border-radius:20px;overflow:hidden}}.bar>i{{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--green));border-radius:20px}}.small{{font-size:12px}}.hidden{{display:none}}.section-title{{display:flex;align-items:end;justify-content:space-between;margin:18px 0 10px}}.tabs{{display:grid;gap:8px}}.tab{{text-align:left}}.kv{{display:grid;grid-template-columns:auto 1fr;gap:6px 12px}}.empty{{padding:26px;text-align:center;color:var(--muted)}}
@media(max-width:900px){{.wrap{{grid-template-columns:1fr}}aside{{position:relative;top:0;height:auto;border-right:0;border-bottom:1px solid var(--line)}}th{{top:0}}}}
</style>
</head>
<body>
<header><h1>LLM ModelBench Rankings V3</h1><div class=\"muted\">Operational rankings, context readiness, capability evidence and terminal repair semantics. Generated {_esc(payload.get('generated_at'))}</div></header>
<div class=\"wrap\"><aside><div class=\"controls\"><input id=\"q\" placeholder=\"Search models\"><select id=\"status\"><option value=\"\">All statuses</option></select><select id=\"ctx\"><option value=\"\">Any context status</option><option value=\"ready\">64k ready</option><option value=\"slow\">64k slow</option><option value=\"not_verified\">not verified</option></select><label class=\"small muted\">Minimum quality <span id=\"minScoreLabel\">0</span></label><input id=\"minScore\" type=\"range\" min=\"0\" max=\"100\" value=\"0\"><button id=\"overall\" class=\"active\">Inventory</button><div id=\"tabs\" class=\"tabs\"></div></div><p class=\"small muted\">Manual rescan: <code>./llmb rankings --runs-dir runs --out rankings --rescan</code></p></aside><main id=\"main\"></main></div>
<script>const DATA={data};
const esc=s=>String(s??'').replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));
const num=x=>Number.isFinite(Number(x))?Number(x):null;const fmt=x=>num(x)===null?'–':Number(x).toFixed(Number(x)%1?2:0);const pct=x=>num(x)===null?'–':(Number(x)*100).toFixed(1)+'%';
let view='inventory';
function badge(b){{let cls='pill ';let lab=b.label||b;if(String(lab).includes('complete'))cls+='good ';else if(String(lab).includes('provisional')||String(lab).includes('warning')||String(lab).includes('slow'))cls+='warn ';else if(String(lab).includes('ineligible')||String(lab).includes('error'))cls+='bad ';else if(String(lab).includes('64k'))cls+='context ';else if(String(lab).includes('capability'))cls+='cap ';return `<span class="${{cls}}">${{esc(lab)}}</span>`}}
function bars(v){{let n=Math.max(0,Math.min(100,Number(v)||0));return `<div class="bar"><i style="width:${{n}}%"></i></div>`}}
function filtered(models){{let q=document.getElementById('q').value.toLowerCase();let st=document.getElementById('status').value;let cs=document.getElementById('ctx').value;let min=Number(document.getElementById('minScore').value||0);return models.filter(m=>{{let name=(m.model||m.display_name||'').toLowerCase();let score=num(m.quality??m.overall_mean_score??m.axes?.quality);let ctx=m.target_context_status??m.axes?.target_context_status??m.long_context_profile?.target_status;return (!q||name.includes(q))&&(!st||m.quality_status===st||m.status===st)&&(!cs||ctx===cs)&&((score??0)>=min)}})}}
function card(m){{let axes=m.axes||{{}};let name=m.model||m.display_name;let score=m.quality??m.overall_mean_score??axes.quality;return `<div class="card model-card"><div class="row"><b>${{esc(name)}}</b></div><div class="row">${{(m.badges||[]).map(badge).join('')}}</div><div class="big">${{fmt(score)}}</div>${{bars(score??0)}}<div class="kv small"><div>Coverage</div><div>${{pct(m.coverage??m.coverage_ratio??axes.coverage)}}</div><div>Speed</div><div>${{fmt(m.speed_tps??m.tok_s??axes.speed_tps)}} tok/s</div><div>Size</div><div>${{fmt(m.size_gb??axes.size_gb)}} GB</div><div>Max ctx</div><div>${{fmt(m.max_verified_ctx??axes.max_verified_ctx)}}</div><div>64k</div><div>${{esc(m.target_context_status??axes.target_context_status??'not_profiled')}}</div></div><details><summary>Evidence notes</summary><ul>${{(m.reasons||m.quality_status_reasons||[]).map(r=>`<li>${{esc(r)}}</li>`).join('')||'<li>No warnings</li>'}}</ul></details></div>`}}
function inventory(){{let models=filtered(DATA.models);document.getElementById('main').innerHTML=`<div class="grid"><div class="card"><h2>${{DATA.summary.models}} models</h2><p class="muted">${{JSON.stringify(DATA.summary.status_counts)}}</p></div><div class="card"><h2>Capability limited</h2><div class="big">${{DATA.summary.capability_limited_models}}</div></div><div class="card"><h2>Measured capability failures</h2><div class="big">${{DATA.summary.capability_measured_failure_models}}</div></div><div class="card"><h2>Long context</h2><p class="muted">${{JSON.stringify(DATA.summary.long_context_status_counts)}}</p></div></div><div class="section-title"><h2>Model inventory</h2><span class="muted">${{models.length}} shown</span></div><div class="grid">${{models.map(card).join('')||'<div class="empty">No matching models</div>'}}</div>`}}
function caseView(name){{let uc=DATA.use_case_rankings[name];let rows=filtered(uc.rows||[]);document.getElementById('main').innerHTML=`<div class="section-title"><div><h2>${{esc(uc.title)}}</h2><p class="muted">${{esc(uc.description)}}</p></div><span class="muted">${{rows.length}} shown / ${{uc.count}}</span></div><div class="grid">${{rows.slice(0,60).map(card).join('')||'<div class="empty">No matching rows</div>'}}</div><div class="card"><h2>Table</h2><table><thead><tr><th>Rank</th><th>Model</th><th class="num">V3 score</th><th class="num">Quality</th><th class="num">Coverage</th><th class="num">tok/s</th><th class="num">Size</th><th>64k</th><th>Badges</th></tr></thead><tbody>${{rows.map(r=>`<tr><td>${{r.rank}}</td><td>${{esc(r.model)}}</td><td class="num">${{fmt(r.score)}}</td><td class="num">${{fmt(r.quality)}}</td><td class="num">${{pct(r.coverage)}}</td><td class="num">${{fmt(r.speed_tps)}}</td><td class="num">${{fmt(r.size_gb)}}</td><td>${{esc(r.target_context_status||'–')}}</td><td>${{(r.badges||[]).map(badge).join('')}}</td></tr>`).join('')}}</tbody></table></div>`}}
function render(){{document.querySelectorAll('button').forEach(b=>b.classList.remove('active'));if(view==='inventory'){{document.getElementById('overall').classList.add('active');inventory()}}else{{let b=document.querySelector(`[data-case="${{CSS.escape(view)}}"]`);if(b)b.classList.add('active');caseView(view)}}}}
function init(){{let statuses=[...new Set(DATA.models.map(m=>m.quality_status).filter(Boolean))].sort();statuses.forEach(s=>document.getElementById('status').innerHTML+=`<option value="${{esc(s)}}">${{esc(s)}}</option>`);document.getElementById('tabs').innerHTML=Object.entries(DATA.use_case_rankings).map(([k,v])=>`<button class="tab" data-case="${{esc(k)}}">${{esc(v.title)}} <span class="muted">(${{v.count}})</span></button>`).join('');document.getElementById('overall').onclick=()=>{{view='inventory';render()}};document.querySelectorAll('[data-case]').forEach(b=>b.onclick=()=>{{view=b.dataset.case;render()}});['q','status','ctx'].forEach(id=>document.getElementById(id).oninput=render);document.getElementById('minScore').oninput=e=>{{document.getElementById('minScoreLabel').textContent=e.target.value;render()}};render()}}
init();
</script>
</body></html>"""


def write_v3_artifacts(master_payload: Dict[str, Any], rankings_dir: Path) -> Dict[str, str]:
    rankings_dir.mkdir(parents=True, exist_ok=True)
    payload = build_v3_payload(master_payload)
    data_path = rankings_dir / "master_report_v3_data.json"
    html_path = rankings_dir / "master_report_v3.html"
    data_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    html_path.write_text(render_v3_html(payload))
    from .rankings_v31 import write_v31_artifacts
    v31_result = write_v31_artifacts(master_payload, rankings_dir)
    return {
        "v3_schema_version": SCHEMA_VERSION,
        "v3_data_path": str(data_path),
        "v3_html_path": str(html_path),
        "v3_use_cases": len(payload["use_case_rankings"]),
        **v31_result,
    }
