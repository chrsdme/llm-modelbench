"""Offline self-test. Exercises every pure-logic path without Ollama or a GPU, so CI and a
fresh clone can verify the scoring engine in seconds. `llm-modelbench selftest`."""
from __future__ import annotations

from . import scoring, fingerprint
from .aggregate import aggregate, pareto_frontier
from .tasks import TASKS, make_needle_prompt


def run() -> int:
    fails = 0

    def ck(name, got, want, tol=1e-6):
        nonlocal fails
        ok = (got is want) or (isinstance(got, (int, float)) and abs(got - want) < tol) or got == want
        print(f"  {'PASS' if ok else 'FAIL'} {name}: {got} vs {want}")
        fails += 0 if ok else 1

    good = ("```python\ndef group_anagrams(words):\n from collections import defaultdict\n"
            " d=defaultdict(list)\n for w in words: d[''.join(sorted(w))].append(w)\n"
            " return list(d.values())\n```")
    checks = TASKS[0].meta["checks"]
    ck("py_good", scoring.score_python(good, {"checks": checks})[0], 100.0)
    ck("py_bad", scoring.score_python("```python\ndef group_anagrams(w): return []\n```",
                                      {"checks": checks})[0], 100.0 / 3)
    ck("py_unsafe", scoring.score_python("```python\nimport subprocess\n```", {"checks": checks})[0], 0.0)
    ck("ocr_perfect", scoring.score_ocr("INVOICE 2026-0042", {"reference": "INVOICE 2026-0042"})[0], 100.0)
    ck("tokens", scoring.score_tokens_in_code("<nav> display:flex @media 600px",
        {"required": ["<nav", "display", "flex", "@media", "600px"], "all_required": False})[0], 100.0)
    ck("web_nav_spacing", scoring.score_web_nav("<nav></nav><style>nav { display : flex } @media (max-width: 600 px){nav{flex-direction: column}}</style>", {})[0], 100.0)
    ck("contains", scoring.score_contains("ada@calc.io hopper@navy.mil",
        {"needles": ["ada@calc.io", "hopper@navy.mil"], "all_required": True})[0], 100.0)
    ck("lineset", scoring.score_lineset("apple\napricot\nbanana\ncherry\ndate",
        {"expected_lines": ["apple", "apricot", "banana", "cherry", "date"]})[0], 100.0)
    ck("regex", scoring.score_regex("answer 8", {"pattern": r"\b8\b"})[0], 100.0)
    ck("exact", scoring.score_exact("John Smith", {"expected": "John Smith"})[0], 100.0)
    ck("json_ok", scoring.score_json_schema('{"server":"a","ip":"b","status":"c"}',
        {"required_keys": ["server", "ip", "status"]})[0], 100.0)
    ck("json_missing", scoring.score_json_schema('{"server":"a"}',
        {"required_keys": ["server", "ip"]})[0], 50.0)

    fs = ("```python\nimport os, shutil\nfor f in os.listdir('.'):\n"
          "  if f.endswith('.txt'): os.makedirs('text', exist_ok=True); shutil.move(f, 'text/'+f)\n"
          "  elif f.endswith('.md'): os.makedirs('docs', exist_ok=True); shutil.move(f, 'docs/'+f)\n```")
    file_task = next(t for t in TASKS if t.id == "file_ext")
    ck("filesort", scoring.score_filesort(fs, file_task.meta)[0], 100.0)

    # aggregation: quality must not be contaminated by speed
    rows = [
        {"model": "m1", "task": "py_anagram", "category": "coding_python", "score": 100.0, "tps": 5.0},
        {"model": "m2", "task": "py_anagram", "category": "coding_python", "score": 100.0, "tps": 60.0},
    ]
    lb, _ = aggregate(rows, {"coding_python": 1.0}, {"py_anagram": 1.0})
    q = {r["model"]: r["quality"] for r in lb}
    ck("quality_pure", q["m1"], q["m2"])

    # aggregation: completed quality must not be contaminated by VRAM/context metadata.
    # These assertions are deliberately non-vacuous: both quality values must be real numbers.
    rows = [
        {"model": "same_on_16gb", "task": "needle", "category": "long_context", "score": 100.0,
         "needle_coverage": 1.0, "vram_budget_gb": 14.4, "num_ctx_used": 4096},
        {"model": "same_on_48gb", "task": "needle", "category": "long_context", "score": 100.0,
         "needle_coverage": 1.0, "vram_budget_gb": 43.0, "num_ctx_used": 65536},
    ]
    lb, _ = aggregate(rows, {"long_context": 1.0}, {"needle": 1.0})
    q = {r["model"]: r["quality"] for r in lb}
    ck("quality_vram_pure", q["same_on_16gb"], q["same_on_48gb"])
    ck("quality_ctx_pure_not_none", q["same_on_16gb"] is not None, True)

    # Partial environment/operator coverage must not turn into a quality number.
    rows = [
        {"model": "small_card", "task": "needle", "category": "long_context", "score": None,
         "needle_coverage": 0.5, "vram_budget_gb": 14.4},
    ]
    lb, _ = aggregate(rows, {"long_context": 1.0}, {"needle": 1.0})
    q = {r["model"]: r["quality"] for r in lb}
    ck("quality_partial_coverage_null", q["small_card"], None)

    # pareto: a small high-quality model dominates a big low-quality one
    lb2 = [{"model": "big_bad", "quality": 40, "size_gb": 14, "tok_s": 5},
           {"model": "small_good", "quality": 90, "size_gb": 5, "tok_s": 60}]
    front = pareto_frontier(lb2)
    ck("pareto_small_good", "small_good" in front, True)

    # clone detection: empty/short probe outputs must not create false clones, but valid
    # identical outputs may still be reported as advisory probe matches.
    valid = [
        "A benchmark can overfit when prompts are too narrow. Mitigate it with held-out tasks.",
        "A namespace with functions cannot be JSON serialized. Store primitive config fields only.",
        "def stable_slug(text): return text.lower().strip('-')",
        "Long context is slower because the KV cache grows with sequence length.",
    ]
    fps = {"a": valid, "b": list(valid), "empty1": ["", "", "", ""], "empty2": ["", "", "", ""]}
    clones = fingerprint.find_clones(fps)
    ck("clone_found", any({"a", "b"} == {x, y} for x, y, _ in clones), True)
    ck("empty_not_clone", any({"empty1", "empty2"} == {x, y} for x, y, _ in clones), False)

    # needle prompt contains the needle
    ck("needle_prompt", "SECRET_CODE_77" in make_needle_prompt(2000, "SECRET_CODE_77"), True)

    print("\nSELFTEST:", "ALL GOOD" if fails == 0 else f"{fails} FAILED")
    return 1 if fails else 0
