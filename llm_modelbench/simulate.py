"""Read-only replay of collected environment-class needle skips."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_rows(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "raw_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no raw_results.jsonl in {run_dir}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def simulate(rows: List[Dict[str, Any]], new_budget_gb: float) -> List[Dict[str, Any]]:
    """Reapply a VRAM budget only to stored environment skips with estimates."""
    results = []
    for row in rows:
        for skip in row.get("needle_skipped") or []:
            if skip.get("skip_class") != "environment":
                continue
            total = skip.get("estimated_total_gb")
            if not isinstance(total, (int, float)):
                continue
            results.append({
                "model": row.get("model"),
                "context_size": skip.get("size"),
                "estimated_total_gb": total,
                "original_budget_gb": skip.get("vram_budget_gb"),
                "simulated_budget_gb": new_budget_gb,
                "would_fit_at_simulated_budget": total <= new_budget_gb,
                "kv_estimate_method": skip.get("kv_estimate_method"),
                "margin_gb": round(new_budget_gb - total, 3),
            })
    return results


def report(results: List[Dict[str, Any]], new_budget_gb: float) -> str:
    if not results:
        return "No environment-class skips with a usable estimate in this run - nothing to simulate."
    lines = [f"Simulated VRAM budget: {new_budget_gb} GB", ""]
    for row in sorted(results, key=lambda item: (not item["would_fit_at_simulated_budget"], item["model"] or "")):
        flag = "YES" if row["would_fit_at_simulated_budget"] else "no"
        lines.append(f"{row['model']} ctx={row['context_size']} needed={row['estimated_total_gb']:.2f}GB fits={flag}")
    return "\n".join(lines)
