"""Aggregation and pruning.

Quality is a difficulty-weighted, category-weighted composite kept strictly separate from
speed, VRAM offload, and value-per-GB. V9.5.13 converts JSON rows into explicit Outcome
objects before averaging so harness/operator failures cannot become leaderboard numbers.
"""
from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Dict, List, Tuple

from .outcome import (
    HARNESS_FAILURE_KINDS,
    MODEL_FAILURE_KINDS,
    HarnessError,
    ModelFailed,
    NotAttempted,
    category_score,
    row_to_outcome,
)

MODEL_ERROR_KINDS = set(MODEL_FAILURE_KINDS)
HARNESS_ERROR_KINDS = set(HARNESS_FAILURE_KINDS)


def _mean(xs, default: Any = 0.0):
    vals = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.mean(vals), 2) if vals else default


def _is_raw_subjective(r: Dict[str, Any]) -> bool:
    return str(r.get("reason") or "").startswith("raw only, judge off")


def _visible_speed_row(r: Dict[str, Any]) -> bool:
    return bool((r.get("output_chars") or 0) > 0 and isinstance(r.get("tps"), (int, float)) and r.get("tps"))


def _outcome_blocker(o) -> str:
    if isinstance(o, HarnessError):
        return f"harness_error:{o.kind}"
    if isinstance(o, NotAttempted):
        return f"not_attempted:{o.kind}"
    return ""



def _row_for_quality(r: Dict[str, Any]) -> Dict[str, Any]:
    """Row view used for leaderboard quality.

    For agentic_tool, V9.5.17 ranks the decision axis. The persisted `score`
    remains the legacy blended score so compare/repeat artifacts stay compatible.
    """
    if r.get("category") == "agentic_tool" and isinstance(r.get("decision_score"), (int, float)):
        q = dict(r)
        q["score"] = r.get("decision_score")
        return q
    return r


def _format_mode(values: List[str]) -> str:
    vals = [str(v) for v in values if v]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]

def aggregate(rows: List[Dict[str, Any]], weights: Dict[str, float],
              difficulty: Dict[str, float]) -> Tuple[List[Dict], Dict[str, List]]:
    by_model: Dict[str, List[Dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    per_cat: Dict[str, List[Tuple[str, float]]] = {}
    leaderboard = []
    for model, rs in by_model.items():
        cat_items: Dict[str, List[Tuple[Any, float]]] = {}
        cat_items_blended: Dict[str, List[Tuple[Any, float]]] = {}
        cat_errors: Dict[str, int] = {}
        cat_ineligible: Dict[str, str] = {}
        gate_failures = 0

        for r in rs:
            outcome = row_to_outcome(_row_for_quality(r))
            blended_outcome = row_to_outcome(r)
            cat = r.get("category")
            d = float(difficulty.get(r.get("task"), 1.0) or 0.0)

            if isinstance(outcome, (ModelFailed, HarnessError)):
                cat_errors[cat] = cat_errors.get(cat, 0) + 1
            if d <= 0:
                # Difficulty-zero tasks are gates. They can surface errors, but they do not
                # contribute positive leaderboard quality.
                if isinstance(outcome, ModelFailed):
                    gate_failures += 1
                continue
            if _is_raw_subjective(r):
                continue
            if cat is None:
                continue
            cat_items.setdefault(str(cat), []).append((outcome, d))
            cat_items_blended.setdefault(str(cat), []).append((blended_outcome, d))

        cat_mean: Dict[str, float] = {}
        cat_coverage: Dict[str, float] = {}
        for c, items in cat_items.items():
            outs = [o for o, _ in items]
            ds = [d for _, d in items]
            score, coverage, blocker = category_score(outs, ds)
            cat_coverage[c] = coverage
            if blocker:
                cat_ineligible[c] = blocker
                continue
            if isinstance(score, (int, float)):
                cat_mean[c] = score

        cat_mean_blended: Dict[str, float] = {}
        for c, items in cat_items_blended.items():
            outs = [o for o, _ in items]
            ds = [d for _, d in items]
            score_blended, _, blocker_blended = category_score(outs, ds)
            if not blocker_blended and isinstance(score_blended, (int, float)):
                cat_mean_blended[c] = score_blended

        for c, v in cat_mean.items():
            if cat_errors.get(c, 0) == 0 and c not in cat_ineligible:
                per_cat.setdefault(c, []).append((model, v))

        wtot = sum(weights.get(c, 0.0) for c in cat_mean)
        if wtot:
            quality = round(sum(cat_mean[c] * weights.get(c, 0.0) for c in cat_mean) / wtot, 2)
        else:
            quality = _mean(list(cat_mean.values()), default=None)
        wtot_blended = sum(weights.get(c, 0.0) for c in cat_mean_blended)
        if wtot_blended:
            quality_blended = round(sum(cat_mean_blended[c] * weights.get(c, 0.0) for c in cat_mean_blended) / wtot_blended, 2)
        else:
            quality_blended = _mean(list(cat_mean_blended.values()), default=None)

        tps = _mean([r.get("tps") for r in rs if _visible_speed_row(r)], default=None)
        offload = _mean([r.get("offload_fraction") for r in rs], default=0.0)
        errors = sum(1 for r in rs if r.get("error_kind") in MODEL_ERROR_KINDS | HARNESS_ERROR_KINDS)
        attempted = len([r for r in rs if not _is_raw_subjective(r)]) or len(rs) or 1
        completed = len([r for r in rs if not _is_raw_subjective(r) and r.get("error_kind") not in MODEL_ERROR_KINDS | HARNESS_ERROR_KINDS])
        completion_rate = round(completed / attempted, 4) if attempted else 1.0
        caps_counter: Counter[str] = Counter()
        for r in rs:
            for cap in r.get("caps_fired") or []:
                caps_counter[str(cap)] += 1
        agentic_rows = [r for r in rs if r.get("category") == "agentic_tool"]
        fmts = [float(r.get("format_multiplier")) for r in agentic_rows if isinstance(r.get("format_multiplier"), (int, float))]
        strict_rate = round(sum(1 for x in fmts if x == 1.0) / len(fmts), 4) if fmts else None
        mean_multiplier = round(statistics.mean(fmts), 4) if fmts else None
        deviations = [str(r.get("format_deviation")) for r in agentic_rows if r.get("format_deviation") and r.get("format_deviation") != "strict_json"]
        size = next((r.get("size_gb") for r in rs if r.get("size_gb")), None)
        leaderboard.append({
            "model": model, "class": rs[0].get("class"), "quality": quality,
            "quality_blended": quality_blended,
            "score_blended": quality_blended,
            "tok_s": tps, "offload": offload,
            "value_per_gb": round(quality / size, 2) if isinstance(quality, (int, float)) and size and completion_rate >= 1.0 else None,
            "value_per_gb_blended": round(quality_blended / size, 2) if isinstance(quality_blended, (int, float)) and size and completion_rate >= 1.0 else None,
            "size_gb": size, "err": errors, "tasks": len(rs),
            "completion_rate": completion_rate,
            "categories": cat_mean,
            "category_errors": cat_errors,
            "category_ineligible": cat_ineligible,
            "category_coverage": cat_coverage,
            "gate_failures": gate_failures,
            "agentic_caps_fired": dict(sorted(caps_counter.items())),
            "over_refusal_count": caps_counter.get("over_refusal", 0),
            "disallowed_tool_count": caps_counter.get("tool not allowed", 0),
            "thinking_only_count": sum(1 for r in rs if r.get("error_kind") == "thinking_only"),
            "agentic_format_strict_rate": strict_rate,
            "agentic_format_mean_multiplier": mean_multiplier,
            "agentic_format_modal_deviation": _format_mode(deviations),
        })
    for c in per_cat:
        per_cat[c].sort(key=lambda x: x[1], reverse=True)
    leaderboard.sort(key=lambda x: (x["quality"] if isinstance(x.get("quality"), (int, float)) else -1.0, x.get("completion_rate", 0)), reverse=True)
    return leaderboard, per_cat


def pareto_frontier(lb: List[Dict]) -> List[str]:
    """Models not dominated on both quality (higher better) and size (lower better)."""
    pts = [(r["model"], r["quality"], r.get("size_gb") or 999) for r in lb if isinstance(r.get("quality"), (int, float))]
    frontier = []
    for name, q, s in pts:
        dominated = any((q2 >= q and s2 <= s and (q2 > q or s2 < s)) for n2, q2, s2 in pts if n2 != name)
        if not dominated:
            frontier.append(name)
    return frontier


def prune_recommendations(lb: List[Dict], per_cat: Dict[str, List],
                          clones: List[Tuple[str, str, float]]) -> Dict[str, List]:
    keep = set()
    for c, ranked in per_cat.items():
        for m, _ in ranked[:2]:
            keep.add(m)
    frontier = set(pareto_frontier(lb))
    keep |= frontier
    useful = [r for r in lb if r.get("completion_rate", 0) >= 1.0 and isinstance(r.get("quality"), (int, float)) and isinstance(r.get("tok_s"), (int, float))]
    if useful:
        qs = sorted(r["quality"] for r in useful)
        ts = sorted(r["tok_s"] for r in useful)
        q25 = qs[len(qs) // 4]
        t25 = ts[len(ts) // 4]
        delete_first = [r["model"] for r in useful
                        if r["quality"] <= q25 and r["tok_s"] <= t25 and r["model"] not in keep]
    else:
        delete_first = []
    redundant = []
    seen_drop = set()
    for a, b, sim in clones:
        ra = next((r for r in lb if r["model"] == a), None)
        rb = next((r for r in lb if r["model"] == b), None)
        if ra and rb:
            drop = a if (ra.get("size_gb") or 0) >= (rb.get("size_gb") or 0) else b
            if drop not in seen_drop:
                seen_drop.add(drop)
                redundant.append((drop, "same Ollama digest / ID"))
    return {"keep": sorted(keep), "pareto": sorted(frontier),
            "delete_first": delete_first, "redundant": redundant}


def regression(prev: List[Dict], curr: List[Dict], drop_threshold: float = 5.0) -> List[Tuple[str, float, float]]:
    """Models whose quality dropped by more than drop_threshold vs a previous run."""
    pj = {r["model"]: r["quality"] for r in prev}
    out = []
    for r in curr:
        old = pj.get(r["model"])
        if isinstance(old, (int, float)) and isinstance(r.get("quality"), (int, float)) and old - r["quality"] > drop_threshold:
            out.append((r["model"], old, r["quality"]))
    return out
