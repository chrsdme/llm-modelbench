"""Cross-run model coverage ledger. Prototype for v9.6.3.

Nothing in this file touches runner.py, report.py, or planner.py. It reads the
same artifacts those modules already write (raw_results.jsonl, model_identities.json,
summary_meta.json) and produces one new artifact: a persistent ledger keyed by
model DIGEST (not name/tag), because digest is already this project's identity
key for clone detection (fingerprint.py: find_digest_clones). Re-tagging or
re-pulling a model under a new name should not reset its coverage; a genuinely
new weight file should not inherit old coverage. Digest keying gets both for free.

Eligibility is derived, not hand-maintained: every task already declares
`family` (text/vision/embedding) and `category` (ocr, coding_python, retrieval...).
The family -> category mapping falls out of the existing task registry, so this
module never needs its own copy of "which categories can a vision model do."
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ---------- eligibility, derived from the live task registry, not hand-maintained ----------

def family_categories(tasks) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for t in tasks:
        out.setdefault(t.family, set()).add(t.category)
    return out


def eligible_categories(families: List[str], fam_cats: Dict[str, Set[str]]) -> Set[str]:
    out: Set[str] = set()
    for f in families:
        out |= fam_cats.get(f, set())
    return out


def category_task_ids(tasks, category: str) -> Set[str]:
    return {t.id for t in tasks if t.category == category}


# ---------- ledger I/O ----------

def load_ledger(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_ledger(ledger: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True))


# ---------- updating the ledger from one run's artifacts ----------

def update_ledger_from_run(
    ledger: Dict[str, Any],
    *,
    raw_rows: List[Dict[str, Any]],
    identities: Dict[str, Any],
    tasks,
    benchmark_version: str,
    out_dir: str,
    timestamp: str,
) -> Dict[str, Any]:
    """Merge one run's rows into the ledger. Pure function: takes data in, returns
    the updated ledger, so it's trivially unit-testable without touching disk."""
    tasks_by_id = {t.id: t for t in tasks}
    by_model_cat: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for r in raw_rows:
        task = tasks_by_id.get(r.get("task"))
        if task is None:
            continue  # unknown task id, e.g. a needle synthetic id; skip, not this module's job
        model = r.get("model")
        by_model_cat.setdefault(model, {}).setdefault(task.category, []).append(r)

    for model, cats in by_model_cat.items():
        ident = identities.get(model) or {}
        digest = ident.get("digest")
        if not digest:
            continue  # can't key what we can't identify; skip rather than guess
        entry = ledger.setdefault(digest, {"names_seen": [], "categories": {}})
        if model not in entry["names_seen"]:
            entry["names_seen"].append(model)

        for category, rows in cats.items():
            seen_task_ids = {r.get("task") for r in rows}
            current_task_ids = category_task_ids(tasks, category)
            completed = sum(1 for r in rows if r.get("score") is not None)
            completion_rate = round(completed / len(rows), 4) if rows else 0.0
            cat_entry = {
                "task_ids_covered": sorted(seen_task_ids),
                "task_ids_current": sorted(current_task_ids),
                "benchmark_version": benchmark_version,
                "completion_rate": completion_rate,
                "last_run": timestamp,
                "out_dir": out_dir,
            }
            prior = entry["categories"].get(category)
            # Keep the most recent attempt only. Staleness is computed at read time
            # against the LIVE registry, not frozen at write time, so a ledger entry
            # never needs rewriting just because a later release added a task elsewhere.
            if prior is None or timestamp >= prior.get("last_run", ""):
                entry["categories"][category] = cat_entry
    return ledger


# ---------- reading coverage state back out ----------

def is_stale(cat_entry: Dict[str, Any], tasks, category: str) -> bool:
    """True if the category's live task set has grown/changed since this model was tested."""
    current = category_task_ids(tasks, category)
    covered = set(cat_entry.get("task_ids_covered", []))
    return not current.issubset(covered)


def pending_categories_for_model(
    ledger: Dict[str, Any],
    digest: Optional[str],
    families: List[str],
    tasks,
) -> List[str]:
    fam_cats = family_categories(tasks)
    eligible = eligible_categories(families, fam_cats)
    if not digest or digest not in ledger:
        return sorted(eligible)  # never seen at all: everything eligible is pending
    covered = ledger[digest].get("categories", {})
    pending = []
    for cat in eligible:
        entry = covered.get(cat)
        if entry is None or is_stale(entry, tasks, cat):
            pending.append(cat)
    return sorted(pending)


if __name__ == "__main__":
    # Self-test using the project's real category/family shape, not a toy example.
    from dataclasses import dataclass

    @dataclass
    class T:
        id: str
        category: str
        family: str

    TASKS = [
        T("ocr_invoice", "ocr", "vision"),
        T("ocr_form_code", "ocr", "vision"),
        T("pdf_text", "pdf", "vision"),
        T("py_dedupe", "coding_python", "text"),
        T("ret_ukdocs", "retrieval", "embedding"),
        T("ret_uk_services_hard", "retrieval", "embedding"),  # arrives in a later release
    ]

    # --- scenario 1: brand new model, never in the ledger ---
    ledger: Dict[str, Any] = {}
    pending = pending_categories_for_model(ledger, "digestA", ["vision", "text"], TASKS)
    assert pending == ["coding_python", "ocr", "pdf"], pending
    print("OK  new model -> all eligible categories pending:", pending)

    # --- scenario 2: run OCR+PDF only, ledger updates, coding still pending ---
    raw_rows = [
        {"model": "gemma3:12b", "task": "ocr_invoice", "score": 100.0},
        {"model": "gemma3:12b", "task": "ocr_form_code", "score": 100.0},
        {"model": "gemma3:12b", "task": "pdf_text", "score": 100.0},
    ]
    identities = {"gemma3:12b": {"digest": "digestA"}}
    ledger = update_ledger_from_run(
        ledger, raw_rows=raw_rows, identities=identities, tasks=TASKS,
        benchmark_version="9.6.3", out_dir="runs/x", timestamp="2026-07-10T00:00:00",
    )
    pending = pending_categories_for_model(ledger, "digestA", ["vision", "text"], TASKS)
    assert pending == ["coding_python"], pending
    print("OK  after OCR+PDF run -> only coding_python pending:", pending)

    # --- scenario 3: rename to a new tag, same digest -> coverage must carry over ---
    pending_after_rename = pending_categories_for_model(ledger, "digestA", ["vision", "text"], TASKS)
    assert pending_after_rename == pending
    print("OK  same digest under a different name would keep coverage (keyed by digest, not name)")

    # --- scenario 4: retrieval task set grows (the real ret_uk_services_hard event) ---
    embed_rows = [{"model": "bge-m3:latest", "task": "ret_ukdocs", "score": 100.0}]
    embed_identities = {"bge-m3:latest": {"digest": "digestB"}}
    ledger = update_ledger_from_run(
        ledger, raw_rows=embed_rows, identities=embed_identities, tasks=TASKS[:5],  # old registry, no hard task yet
        benchmark_version="9.6.0", out_dir="runs/rag_v960", timestamp="2026-06-01T00:00:00",
    )
    pending_before_growth = pending_categories_for_model(ledger, "digestB", ["embedding"], TASKS[:5])
    assert pending_before_growth == [], pending_before_growth
    print("OK  fully covered against the OLD registry:", pending_before_growth)

    pending_after_growth = pending_categories_for_model(ledger, "digestB", ["embedding"], TASKS)  # new registry, hard task added
    assert pending_after_growth == ["retrieval"], pending_after_growth
    print("OK  same model, same ledger entry, but registry grew -> retrieval flagged stale/pending:", pending_after_growth)

    print("\nALL COVERAGE LEDGER TESTS PASS")
