"""Cross-category composite scoring. Prototype for v9.6.5.

Reads the coverage ledger plus each category's own already-computed per_cat
leaderboard (report.py's `per_cat`, unchanged). Adds nothing to scoring math
inside a category -- it only combines category results that already exist,
weighted by declared use-case priority instead of averaged flat.

Composite is computed ONLY over categories the model has non-stale coverage
for. A missing category is omitted from the weighted sum and renormalized,
never treated as a zero. That distinction matters: this project already
refuses to let a harness failure (None) read as a capability failure (0)
inside one category; this is the same rule one level up, across categories.
"""
from __future__ import annotations

from typing import Any, Dict


# Default weights reflect the locked use-case priority order: agentic tool-calling
# for AI Workdesk, then coding/scripting, then UK-doc RAG. Operator-tunable, must
# sum to 1.0 across whatever's declared -- unweighted categories get 0 and are
# excluded from the composite (still shown per-category, just not in "overall").
DEFAULT_CATEGORY_WEIGHTS: Dict[str, float] = {
    "agentic": 0.25,
    "agentic_tool": 0.25,
    "coding_python": 0.15,
    "coding_js": 0.10,
    "coding_web": 0.05,
    "retrieval": 0.10,
    "ocr": 0.05,
    "pdf": 0.05,
}


def validate_weights(weights: Dict[str, float]) -> None:
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"category weights must sum to 1.0, got {total}")


def composite_score(
    digest: str,
    ledger: Dict[str, Any],
    category_quality: Dict[str, float],
    weights: Dict[str, float],
    tasks,
) -> Dict[str, Any]:
    """category_quality: {category: quality_score} for THIS model in THIS category,
    already computed by the existing per-run aggregate() -- not recomputed here."""
    from .coverage import is_stale

    entry = ledger.get(digest, {}).get("categories", {})
    covered, pending, stale = [], [], []
    weighted_sum, weight_used = 0.0, 0.0

    all_weighted_cats = set(weights.keys())
    for cat in sorted(all_weighted_cats):
        cat_entry = entry.get(cat)
        if cat_entry is None:
            pending.append(cat)
            continue
        if is_stale(cat_entry, tasks, cat):
            stale.append(cat)
            continue
        q = category_quality.get(cat)
        if q is None:
            pending.append(cat)  # covered in ledger but this run's per_cat has no score for it
            continue
        covered.append(cat)
        weighted_sum += weights[cat] * q
        weight_used += weights[cat]

    composite = round(weighted_sum / weight_used, 2) if weight_used > 0 else None
    return {
        "composite": composite,
        "covered_categories": covered,
        "pending_categories": pending,
        "stale_categories": stale,
        "coverage_fraction": f"{len(covered)}/{len(all_weighted_cats)}",
        "weight_used": round(weight_used, 4),
    }


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class T:
        id: str
        category: str
        family: str

    TASKS = [
        T("ocr_invoice", "ocr", "vision"),
        T("py_dedupe", "coding_python", "text"),
        T("agent_tool_select", "agentic_tool", "text"),
    ]
    weights = {"ocr": 0.2, "coding_python": 0.3, "agentic_tool": 0.5}
    validate_weights(weights)

    ledger = {
        "digestA": {
            "categories": {
                "ocr": {"task_ids_covered": ["ocr_invoice"]},
                "coding_python": {"task_ids_covered": ["py_dedupe"]},
                # agentic_tool never run for this model
            }
        }
    }
    category_quality = {"ocr": 100.0, "coding_python": 80.0}

    result = composite_score("digestA", ledger, category_quality, weights, TASKS)
    # expected: only ocr+coding_python count, renormalized over their combined weight (0.5)
    expected = round((0.2 * 100.0 + 0.3 * 80.0) / 0.5, 2)
    assert result["composite"] == expected, result
    assert result["pending_categories"] == ["agentic_tool"], result
    assert result["coverage_fraction"] == "2/3", result
    print("OK  partial coverage renormalizes correctly, agentic_tool correctly pending, not zeroed:")
    print(" ", result)

    # weight validation catches a bad config before it silently mis-scores anyone
    try:
        validate_weights({"a": 0.5, "b": 0.4})
        print("FAIL should have raised")
    except ValueError as e:
        print("OK  bad weight config rejected:", e)

    print("\nALL DOSSIER COMPOSITE TESTS PASS")
