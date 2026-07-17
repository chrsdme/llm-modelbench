"""Gap scheduling. Prototype for v9.6.4.

Deliberately does NOT touch planner.build_plan(). It's a read-only companion:
given a live model roster (same client.tags()/client.capabilities() shape
build_plan already uses) and the coverage ledger, it reports which models are
missing which categories. Nothing here runs inference or writes results --
same operator-in-the-loop posture as prune.md: advisory, reviewed, then acted
on manually via the EXISTING --categories flag on a normal run. No new
auto-run behavior in this release, on purpose, to keep blast radius at zero.
"""
from __future__ import annotations

from typing import Any, Dict, List


def gap_report(client: Any, ledger: Dict[str, Any], tasks, classify_model, families_for) -> Dict[str, List[str]]:
    from .coverage import pending_categories_for_model

    rows = client.tags()
    gaps: Dict[str, List[str]] = {}
    for row in rows:
        model = row.get("name")
        if not model:
            continue
        caps = client.capabilities(model) if hasattr(client, "capabilities") else None
        fams = families_for(model, caps)
        digest = row.get("digest")
        pending = pending_categories_for_model(ledger, digest, fams, tasks)
        if pending:
            gaps[model] = pending
    return gaps


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class T:
        id: str
        category: str
        family: str

    TASKS = [T("ocr_invoice", "ocr", "vision"), T("py_dedupe", "coding_python", "text")]

    def families_for(name, caps):
        return ["vision", "text"] if caps and "vision" in caps else ["text"]

    def classify_model(name, caps):
        return "vision" if caps and "vision" in caps else "general"

    class FakeClient:
        def tags(self):
            return [{"name": "gemma3:12b"}, {"name": "qwen2.5-coder:7b"}]
        def capabilities(self, m):
            return ["completion", "vision"] if m == "gemma3:12b" else ["completion"]
        def show(self, m):
            return {"digest": "digestA"} if m == "gemma3:12b" else {"digest": "digestC"}

    # gemma3:12b is vision+text eligible (ocr AND coding_python), only ocr covered so far
    ledger = {"digestA": {"categories": {"ocr": {"task_ids_covered": ["ocr_invoice"]}}}}
    gaps = gap_report(FakeClient(), ledger, TASKS, classify_model, families_for)
    assert gaps == {"gemma3:12b": ["coding_python"], "qwen2.5-coder:7b": ["coding_python"]}, gaps
    print("OK  gemma3:12b correctly still shows coding_python pending (vision models are text-eligible too):")
    print(" ", gaps)

    # now cover gemma3:12b's coding_python too -> it should drop off the gap report entirely
    ledger["digestA"]["categories"]["coding_python"] = {"task_ids_covered": ["py_dedupe"]}
    gaps2 = gap_report(FakeClient(), ledger, TASKS, classify_model, families_for)
    assert gaps2 == {"qwen2.5-coder:7b": ["coding_python"]}, gaps2
    print("OK  once fully covered, gemma3:12b drops off the gap report:")
    print(" ", gaps2)
    print("\nALL GAP PLANNER TESTS PASS")
