"""Post-hoc automated judging of existing subjective dumps.

The tested model is never called. Existing ``raw_results.jsonl`` remains
immutable; judgements are appended to ``judge_results.jsonl`` and overlaid by
reports/rankings at read time.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from . import judge as judge_mod
from .runner import _task_hash
from .tasks import TASKS, Task

_TASKS = {task.id: task for task in TASKS}


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def source_row_hash(row: Dict[str, Any]) -> str:
    stable = {k: v for k, v in row.items() if not str(k).startswith("_judge_")}
    return hashlib.sha256(json.dumps(stable, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def _safe_model(model: str) -> str:
    return str(model).replace("/", "_").replace(":", "_")


def _parse_dump(path: Path) -> Optional[str]:
    try:
        text = path.read_text()
    except Exception:
        return None
    marker = "## OUTPUT\n"
    if marker not in text:
        return None
    return text.split(marker, 1)[1]


def _outputs_for_row(run_dir: Path, row: Dict[str, Any], task: Task) -> List[Tuple[str, str]]:
    candidates: List[Path] = []
    rel = row.get("subjective_path")
    if rel:
        candidates.append(run_dir / str(rel))
    task_dir = run_dir / "subjective" / task.id
    if task_dir.is_dir():
        safe = _safe_model(str(row.get("model") or ""))
        candidates.extend(sorted(task_dir.glob(f"{safe}*.md")))
    seen = set()
    outputs: List[Tuple[str, str]] = []
    for path in candidates:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        output = _parse_dump(path)
        if output is not None and output.strip():
            try:
                display = str(path.relative_to(run_dir))
            except Exception:
                display = str(path)
            outputs.append((display.replace("\\", "/"), output))
    return outputs


def discover_runs(runs_dir: Path) -> List[Path]:
    if not runs_dir.exists():
        return []
    return sorted(
        path for path in runs_dir.iterdir()
        if path.is_dir() and (path / "raw_results.jsonl").is_file()
    )


def scan_run(run_dir: Path, *, judge_model: str, judge_mode: str, force: bool = False) -> Dict[str, Any]:
    raw_rows = _jsonl(run_dir / "raw_results.jsonl")
    existing = _jsonl(run_dir / "judge_results.jsonl")
    prior_keys = {
        (entry.get("source_row_hash"), entry.get("judge_model"), entry.get("judge_mode"))
        for entry in existing if entry.get("status") == "judged"
    }
    eligible: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for row_index, row in enumerate(raw_rows):
        task = _TASKS.get(str(row.get("task") or ""))
        if task is None or task.scorer != "subjective":
            continue
        row_hash = source_row_hash(row)
        key = (row_hash, judge_model, judge_mode)
        if key in prior_keys and not force:
            skipped.append({"row_index": row_index, "model": row.get("model"), "task": row.get("task"), "reason": "already_judged"})
            continue
        if row.get("task_hash") and row.get("task_hash") != _task_hash(task):
            skipped.append({"row_index": row_index, "model": row.get("model"), "task": row.get("task"), "reason": "stale_task_hash"})
            continue
        if row.get("error_kind"):
            skipped.append({"row_index": row_index, "model": row.get("model"), "task": row.get("task"), "reason": f"source_error:{row.get('error_kind')}"})
            continue
        outputs = _outputs_for_row(run_dir, row, task)
        if not outputs:
            skipped.append({"row_index": row_index, "model": row.get("model"), "task": row.get("task"), "reason": "missing_or_empty_dump"})
            continue
        eligible.append({"row_index": row_index, "row": row, "row_hash": row_hash, "task": task, "outputs": outputs})
    return {
        "run_dir": str(run_dir),
        "raw_rows": len(raw_rows),
        "eligible": eligible,
        "skipped": skipped,
        "already_recorded": len(existing),
    }


def judge_run(
    client: Any,
    run_dir: Path,
    *,
    judge_model: str,
    judge_mode: str = "single",
    num_ctx: Optional[int] = None,
    think: str = "auto",
    dry_run: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    if judge_mode not in {"single", "panel"}:
        raise ValueError("judge mode must be single or panel")
    scan = scan_run(run_dir, judge_model=judge_model, judge_mode=judge_mode, force=force)
    eligible = scan.pop("eligible")
    result = {
        **scan,
        "judge_model": judge_model,
        "judge_mode": judge_mode,
        "eligible": len(eligible),
        "attempted": 0,
        "judged": 0,
        "judge_errors": 0,
        "written": 0,
        "dry_run": bool(dry_run),
        "entries": [],
    }
    if dry_run:
        result["entries"] = [
            {"model": item["row"].get("model"), "task": item["task"].id,
             "samples": len(item["outputs"]), "source_row_hash": item["row_hash"]}
            for item in eligible
        ]
        return result
    if not eligible:
        return result

    sidecar = run_dir / "judge_results.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    with sidecar.open("a") as handle:
        for item in eligible:
            row = item["row"]
            task: Task = item["task"]
            result["attempted"] += 1
            sample_results = []
            valid_scores: List[float] = []
            started = time.perf_counter()
            for path, output in item["outputs"]:
                try:
                    if judge_mode == "panel":
                        score, reason = judge_mod.judge_panel(
                            client, judge_model, task.prompt, output, task.rubric,
                            num_ctx=num_ctx, think=think,
                        )
                    else:
                        score, reason = judge_mod.judge_single(
                            client, judge_model, task.prompt, output, task.rubric,
                            num_ctx=num_ctx, think=think,
                        )
                except Exception as exc:
                    score, reason = None, f"judge exception: {exc!r}"
                sample = {
                    "subjective_path": path,
                    "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                    "score": score,
                    "reason": reason,
                }
                sample_results.append(sample)
                if isinstance(score, (int, float)):
                    valid_scores.append(float(score))
            final_score = round(float(statistics.median(valid_scores)), 2) if valid_scores else None
            status = "judged" if final_score is not None else "judge_error"
            entry = {
                "schema_version": 1,
                "applied_at": datetime.now(timezone.utc).isoformat(),
                "run_id": run_dir.name,
                "source_row_index": item["row_index"],
                "source_row_hash": item["row_hash"],
                "source_model": row.get("model"),
                "source_model_digest": row.get("model_digest"),
                "task": task.id,
                "task_hash": row.get("task_hash") or _task_hash(task),
                "judge_model": judge_model,
                "judge_mode": judge_mode,
                "status": status,
                "score": final_score,
                "reason": (
                    f"posthoc {judge_mode} judge median over {len(valid_scores)} valid sample(s)"
                    if final_score is not None else
                    "judge_error: no valid scores; " + "; ".join(str(s["reason"]) for s in sample_results[:3])
                ),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "samples": sample_results,
            }
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            result["written"] += 1
            if status == "judged":
                result["judged"] += 1
            else:
                result["judge_errors"] += 1
            result["entries"].append(entry)
    summary_path = run_dir / "judge_dumps_summary.json"
    summary_path.write_text(json.dumps({k: v for k, v in result.items() if k != "entries"}, indent=2))
    return result


def judge_everything(
    client: Any,
    runs_dir: Path,
    *,
    judge_model: str,
    judge_mode: str = "single",
    num_ctx: Optional[int] = None,
    think: str = "auto",
    dry_run: bool = False,
    force: bool = False,
    progress: Optional[Callable[[int, int, Path, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    runs = discover_runs(runs_dir)
    results = []
    for index, run_dir in enumerate(runs, start=1):
        result = judge_run(
            client, run_dir, judge_model=judge_model, judge_mode=judge_mode,
            num_ctx=num_ctx, think=think, dry_run=dry_run, force=force,
        )
        results.append(result)
        if progress is not None:
            progress(index, len(runs), run_dir, result)
    return {
        "runs_dir": str(runs_dir),
        "runs_scanned": len(runs),
        "runs_with_eligible": sum(1 for r in results if r.get("eligible")),
        "eligible": sum(int(r.get("eligible") or 0) for r in results),
        "attempted": sum(int(r.get("attempted") or 0) for r in results),
        "judged": sum(int(r.get("judged") or 0) for r in results),
        "judge_errors": sum(int(r.get("judge_errors") or 0) for r in results),
        "skipped": sum(len(r.get("skipped") or []) for r in results),
        "dry_run": bool(dry_run),
        "runs": results,
    }


def latest_judgements(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for entry in _jsonl(run_dir / "judge_results.jsonl"):
        row_hash = entry.get("source_row_hash")
        if not row_hash or entry.get("status") != "judged":
            continue
        previous = latest.get(row_hash)
        if previous is None or str(entry.get("applied_at") or "") >= str(previous.get("applied_at") or ""):
            latest[row_hash] = entry
    return latest


def apply_judgements(run_dir: Path, rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    overlays = latest_judgements(run_dir)
    out: List[Dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        overlay = overlays.get(source_row_hash(raw))
        if overlay:
            row["score"] = overlay.get("score")
            row["reason"] = overlay.get("reason")
            row["judge_mode"] = overlay.get("judge_mode")
            row["judge_model"] = overlay.get("judge_model")
            row["posthoc_judged"] = True
            row["judge_applied_at"] = overlay.get("applied_at")
            row["judge_source_row_hash"] = overlay.get("source_row_hash")
            row["judge_elapsed_seconds"] = overlay.get("elapsed_seconds")
        out.append(row)
    return out
