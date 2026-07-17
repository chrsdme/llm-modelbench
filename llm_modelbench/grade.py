"""Blind subjective grading workflow."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

CRITERIA = [
    ("accuracy", 0.30),
    ("usefulness", 0.25),
    ("structure", 0.15),
    ("reasoning", 0.15),
    ("brevity", 0.10),
    ("safety", 0.05),
]


def _subjective_files(out_dir: Path) -> List[Path]:
    base = out_dir / "subjective"
    if not base.is_dir():
        return []
    return sorted(p for p in base.glob("*/*.md") if not p.name.startswith("_paste_"))


def _parse_header(text: str) -> Tuple[str, str]:
    first = text.splitlines()[0] if text.splitlines() else ""
    m = re.match(r"#\s*(.*?)\s*\|\s*(.*?)(?:\s*\||$)", first)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "unknown", "unknown"


def export_blind(out_dir: Path) -> Path:
    files = _subjective_files(out_dir)
    if not files:
        raise SystemExit(f"no subjective markdown files found in {out_dir}/subjective")
    mapping: Dict[str, Dict[str, str]] = {}
    chunks: List[str] = ["# Blind subjective grading pack", "", "Model names are hidden. Grade before opening blind_mapping.json.", ""]
    for idx, path in enumerate(files, 1):
        text = path.read_text()
        task, model = _parse_header(text)
        anon = f"M{idx:03d}"
        mapping[anon] = {"task": task, "model": model, "file": str(path.relative_to(out_dir))}
        blinded = text.replace(model, anon)
        chunks += [f"---\n\n## {anon} | {task}\n", blinded]
    pack = out_dir / "subjective" / "blind_grade_pack.md"
    map_path = out_dir / "subjective" / "blind_mapping.json"
    pack.write_text("\n\n".join(chunks))
    map_path.write_text(json.dumps(mapping, indent=2))
    return pack


def interactive_grade(out_dir: Path) -> None:
    files = _subjective_files(out_dir)
    if not files:
        print("no subjective outputs")
        return
    grades: List[Dict[str, object]] = []
    print("Blind grading. Enter 1-5 for each criterion, s=skip, q=quit.")
    print("Criteria: " + ", ".join(f"{n}({int(w*100)}%)" for n, w in CRITERIA))
    for idx, path in enumerate(files, 1):
        text = path.read_text()
        task, model = _parse_header(text)
        anon = f"M{idx:03d}"
        blinded = text.replace(model, anon)
        print("\n" + "=" * 80)
        print(f"{anon} | {task}")
        print("=" * 80)
        # Show prompt/output but keep within reason. User can open the file for full text.
        print(blinded[:6000])
        item_scores: Dict[str, float] = {}
        total = 0.0
        skipped = False
        for name, weight in CRITERIA:
            while True:
                val = input(f"{name} 1-5 (s skip, q quit): ").strip().lower()
                if val == "q":
                    _write_grades(out_dir, grades)
                    print("saved partial grades")
                    return
                if val == "s":
                    skipped = True; break
                try:
                    score = float(val)
                    if 1 <= score <= 5:
                        item_scores[name] = score
                        total += score * weight
                        break
                except Exception:
                    pass
                print("enter 1, 2, 3, 4, 5, s, or q")
            if skipped:
                break
        if not skipped:
            grades.append({"anon": anon, "task": task, "model": model, "file": str(path.relative_to(out_dir)),
                           "criteria": item_scores, "weighted_score_1_5": round(total, 3)})
            _write_grades(out_dir, grades)
    _write_grades(out_dir, grades)
    print(f"wrote {out_dir/'human_grades.json'} and human_grades.md")


def _write_grades(out_dir: Path, grades: List[Dict[str, object]]) -> None:
    (out_dir / "human_grades.json").write_text(json.dumps(grades, indent=2))
    lines = ["# Human blind grades", "", "| anon | task | score | model |", "|---|---|---:|---|"]
    for g in grades:
        lines.append(f"| {g['anon']} | {g['task']} | {g['weighted_score_1_5']} | `{g['model']}` |")
    (out_dir / "human_grades.md").write_text("\n".join(lines))
