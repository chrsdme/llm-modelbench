"""Compare two benchmark runs."""
from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Dict, List


def _load_meta(run: Path) -> dict:
    p = run / "summary_meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _benchmark_version(meta: dict) -> str:
    """Read both metadata keys so historical run artifacts remain comparable."""
    return str(meta.get("llm_modelbench_version") or "unknown")


def _load_summary(run: Path) -> List[dict]:
    p = run / "summary.json"
    if p.exists():
        return json.loads(p.read_text())
    c = run / "scorecard.csv"
    if c.exists():
        with c.open(newline="") as f:
            return list(csv.DictReader(f))
    raise SystemExit(f"no summary.json or scorecard.csv in {run}")


def diff_runs(a: Path, b: Path, out: Path | None = None, noise_band: float | None = None) -> str:
    aa = {r["model"]: r for r in _load_summary(a)}
    bb = {r["model"]: r for r in _load_summary(b)}
    ma, mb = _load_meta(a), _load_meta(b)
    va, vb = _benchmark_version(ma), _benchmark_version(mb)
    models = sorted(set(aa) | set(bb))
    lines = ["# Run diff", "", f"A: `{a}`", f"B: `{b}`", "", f"A version: `{va}`", f"B version: `{vb}`"]
    if va != vb:
        lines += ["", "> WARNING: benchmark versions differ. Scorer changes can make some category scores non-comparable."]
    for key in ("ctx_override", "num_predict", "think", "needle_max_ctx", "sample_mode", "level", "judge_mode", "ollama_version", "gpu_driver", "gpu_name"):
        if ma.get(key) != mb.get(key):
            lines += ["", f"> WARNING: `{key}` differs: A=`{ma.get(key)}` B=`{mb.get(key)}`."]
    lines += ["", "| model | A quality | B quality | delta | interpretation | A tok/s | B tok/s |", "|---|---:|---:|---:|---|---:|---:|"]
    for m in models:
        ra, rb = aa.get(m), bb.get(m)
        def _q(row):
            if not row:
                return None
            val = row.get("quality")
            if val in (None, "", "None", "null"):
                return None
            try:
                return float(val)
            except Exception:
                return None
        qa = _q(ra)
        qb = _q(rb)
        delta = "n/a" if qa is None or qb is None else round(qb - qa, 2)
        interpretation = "n/a" if delta == "n/a" else ("tied/noise-band" if noise_band is not None and abs(delta) <= noise_band else "meaningful")
        lines.append(f"| `{m}` | {qa if qa is not None else 'missing'} | {qb if qb is not None else 'missing'} | {delta} | {interpretation} | {ra.get('tok_s') if ra else 'missing'} | {rb.get('tok_s') if rb else 'missing'} |")
    text = "\n".join(lines)
    if out:
        out.write_text(text)
    return text



def _load_raw_rows(run: Path) -> List[dict]:
    p = run / "raw_results.jsonl"
    if not p.exists():
        raise SystemExit(f"no raw_results.jsonl in {run}")
    rows: List[dict] = []
    for line in p.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def repeatability_report(run_dirs: List[Path], out: Path | None = None) -> str:
    """Compare repeated runs at the per-model/per-task score level."""
    if len(run_dirs) < 2:
        raise SystemExit("repeat-report needs at least two run directories")
    indexed: List[Dict[tuple, dict]] = []
    versions = []
    for run in run_dirs:
        versions.append(_benchmark_version(_load_meta(run)))
        m: Dict[tuple, dict] = {}
        for row in _load_raw_rows(run):
            key = (str(row.get("model")), str(row.get("task")))
            m[key] = row
        indexed.append(m)
    keys = sorted(set().union(*(set(x) for x in indexed)))
    lines = [
        "# Repeatability report",
        "",
        "Runs:",
    ]
    for run, ver in zip(run_dirs, versions):
        lines.append(f"- `{run}` version `{ver}`")
    if len(set(versions)) > 1:
        lines += ["", "> WARNING: benchmark versions differ. Repeatability may be confounded by scorer changes."]
    lines += ["", "| model | task | scores | range | reasons changed | status |", "|---|---|---:|---:|---:|---|"]
    from .reliability import cell_summary, empirical_noise_band
    unstable = 0
    missing = 0
    summaries = []
    for model, task in keys:
        vals = []
        reasons = []
        present = 0
        for m in indexed:
            row = m.get((model, task))
            if row is None:
                vals.append(None)
                reasons.append("<missing>")
                continue
            present += 1
            vals.append(row.get("score"))
            reasons.append(str(row.get("reason") or ""))
        summary = cell_summary([m.get((model, task)) for m in indexed])
        summaries.append(summary)
        rng = summary["range"] if summary["range"] is not None else "n/a"
        reason_changed = summary["reason_changed"]
        status = summary["status"]
        if status == "missing":
            missing += 1
        if status == "reason-moving": status = "score-stable reason-moving"
        if status == "moving":
            unstable += 1
        vals_s = ", ".join("null" if v is None else str(v) for v in vals)
        lines.append(f"| `{model}` | `{task}` | {vals_s} | {rng} | {str(reason_changed).lower()} | {status} |")
    band = empirical_noise_band(summaries)
    evidence = "insufficient repeat evidence; no empirical noise band" if band is None else f"empirical noise band (max observed repeat range): {band}"
    lines += ["", f"Summary: cells={len(keys)} moving={unstable} missing={missing}", f"Reliability: {evidence}", "Single-run cells are insufficient-repeats, not stable."]
    text = "\n".join(lines)
    if out:
        out.write_text(text)
    return text

def export_review(run_dirs: List[Path], zip_path: Path) -> Path:
    keep_names = {"scorecard.csv", "scorecard.md", "routing.md", "prune.md", "clones.md", "raw_results.jsonl", "summary.json", "status.json", "summary_meta.json", "filters.json", "model_identities.json", "fingerprints.json", "human_grades.json", "human_grades.md", "regression.md", "retrieval_diagnostics.json", "retrieval_diagnostics.md"}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for run in run_dirs:
            if not run.exists():
                continue
            for p in run.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(run.parent)
                if p.name in keep_names or "subjective" in p.parts or "raw" in p.parts:
                    z.write(p, rel.as_posix())
    return zip_path
