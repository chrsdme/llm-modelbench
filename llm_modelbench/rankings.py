"""Persistent cross-run rankings database and model-card report.

Source run evidence is accumulated in ``rankings/master_raw.jsonl``. Deleting a
run directory does not erase previously imported evidence. Current rankings use
one selected row per model digest/task, while every historical attempt remains
available in each model card.
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics
import string
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .aggregate import aggregate
from .classify import families_for
from .config import DEFAULT_WEIGHTS
from .runner import _task_hash
from .tasks import LEVELS, TASKS
from .ranking_controls import (
    SCOPE_CANONICAL, excluded_model_keys, excluded_run_ids,
    load_exclusions, model_matches, read_run_scope, summarize_exclusions,
)

LEVEL_RANK = {name: index for index, name in enumerate(LEVELS)}
_TASK_DIFFICULTY = {task.id: task.difficulty for task in TASKS}
_TASKS = {task.id: task for task in TASKS}
_CURRENT_HASHES = {task.id: _task_hash(task) for task in TASKS}
_TAG_ALPHABET = string.ascii_uppercase + string.digits
TIE_EPSILON = 0.5


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def discover_run_dirs(runs_dir: Path) -> List[Path]:
    if not runs_dir.exists():
        return []
    return sorted(path for path in runs_dir.iterdir() if path.is_dir() and (path / "raw_results.jsonl").exists())


def _row_level(row: Dict[str, Any], run_meta: Dict[str, Any]) -> str:
    return str(row.get("level") or run_meta.get("level") or "unknown")


def _source_signature(run_dir: Path) -> str:
    parts = []
    for name in ("raw_results.jsonl", "judge_results.jsonl", "repair_results.jsonl", "model_identities.json", "filters.json", "capability_report.json"):
        path = run_dir / name
        if path.exists():
            st = path.stat()
            parts.append(f"{name}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def new_import_tag(existing_tags: set) -> str:
    while True:
        tag = "".join(random.choices(_TAG_ALPHABET, k=6))
        if tag not in existing_tags:
            return tag


def load_accumulated(rankings_dir: Path) -> List[Dict[str, Any]]:
    return _read_jsonl(rankings_dir / "master_raw.jsonl")


def imported_run_ids(accumulated: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in accumulated:
        run_id = row.get("run_id")
        if run_id and run_id not in out:
            out[str(run_id)] = str(row.get("_source_signature") or row.get("_source_mtime") or "")
    return out


def _run_configuration(run_dir: Path) -> Dict[str, Any]:
    filters = _read_json(run_dir / "filters.json") or {}
    config = _read_json(run_dir / "config.json") or {}
    return {
        "level": filters.get("level"),
        "sample_mode": filters.get("sample_mode"),
        "requested_samples": filters.get("requested_samples") or config.get("samples"),
        "judge_mode": filters.get("judge_mode"),
        "ctx_override": filters.get("ctx_override") or config.get("ctx_override"),
        "num_predict_override": filters.get("num_predict") or config.get("num_predict_override"),
        "think": filters.get("think") or config.get("think"),
        "needle_max_ctx": filters.get("needle_max_ctx") or config.get("needle_max_ctx"),
        "include_regex": filters.get("include_regex"),
        "exclude_regex": filters.get("exclude_regex"),
        "task_regex": filters.get("task_regex"),
        "selected_models": filters.get("selected_models"),
        "auto_probe": filters.get("auto_probe"),
    }


def import_new_runs(
    runs_dir: Path,
    accumulated: List[Dict[str, Any]],
    *,
    force_rescan: bool = False,
    rankings_dir: Optional[Path] = None,
    include_separate: bool = False,
    only_run_ids: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    already = imported_run_ids(accumulated)
    existing_tags = {row["import_tag"] for row in accumulated if row.get("import_tag")}
    result = list(accumulated)
    only = set(str(x) for x in only_run_ids) if only_run_ids is not None else None
    exclusions = load_exclusions(rankings_dir or Path("rankings"))
    excluded_runs = excluded_run_ids(exclusions)

    for run_dir in discover_run_dirs(runs_dir):
        if only is not None and run_dir.name not in only:
            continue
        scope_info = read_run_scope(run_dir)
        run_scope = str(scope_info.get("ranking_scope") or SCOPE_CANONICAL)
        if run_dir.name in excluded_runs and not include_separate:
            result = [row for row in result if row.get("run_id") != run_dir.name]
            continue
        if run_scope != SCOPE_CANONICAL and not include_separate:
            result = [row for row in result if row.get("run_id") != run_dir.name]
            continue
        signature = _source_signature(run_dir)
        if not force_rescan and run_dir.name in already and already[run_dir.name] == signature:
            continue
        result = [row for row in result if row.get("run_id") != run_dir.name]

        run_meta = _read_json(run_dir / "summary_meta.json") or {}
        identities = _read_json(run_dir / "model_identities.json") or {}
        capabilities = _read_json(run_dir / "capability_report.json") or {}
        run_config = _run_configuration(run_dir)
        raw_rows = _read_jsonl(run_dir / "raw_results.jsonl")
        if (run_dir / "judge_results.jsonl").exists():
            from .judge_dumps import apply_judgements
            raw_rows = apply_judgements(run_dir, raw_rows)

        for row_index, row in enumerate(raw_rows):
            model = row.get("model")
            digest = (identities.get(model) or {}).get("digest") or row.get("model_digest") or model
            candidate = dict(row)
            candidate["run_id"] = run_dir.name
            candidate["level"] = _row_level(row, run_meta)
            candidate["model_digest_resolved"] = digest
            identity = identities.get(model) or {}
            candidate["model_identity"] = identity
            candidate.setdefault("parameter_size", identity.get("parameter_size"))
            candidate.setdefault("quantization_level", identity.get("quantization_level"))
            candidate.setdefault("architecture_family", identity.get("family"))
            candidate.setdefault("architecture_families", identity.get("families"))
            candidate["_source_signature"] = signature
            candidate["_source_row_index"] = row_index
            candidate["run_configuration"] = run_config
            candidate["ranking_scope"] = run_scope
            candidate["canonical_rankings"] = bool(scope_info.get("canonical_rankings", run_scope == SCOPE_CANONICAL))
            if model in capabilities:
                candidate["capability_profile"] = capabilities[model]
                candidate.setdefault("capability_families", capabilities[model].get("supported_families"))
                candidate.setdefault("capabilities_declared", capabilities[model].get("declared_capabilities"))
            tag = new_import_tag(existing_tags)
            existing_tags.add(tag)
            candidate["import_tag"] = tag
            result.append(candidate)
    return result


def _canonical_run_score(row: Dict[str, Any]) -> int:
    cfg = row.get("run_configuration") or {}
    diagnostic = any([
        cfg.get("ctx_override") is not None,
        cfg.get("num_predict_override") is not None,
        cfg.get("task_regex"),
        cfg.get("think") not in {None, "auto"},
        row.get("repair_kind"),
        row.get("context_profile_run"),
    ])
    return 0 if diagnostic else 1


def _selection_key(row: Dict[str, Any]) -> Tuple[int, int, int, int, str, str]:
    task_id = str(row.get("task") or "")
    current_hash = _CURRENT_HASHES.get(task_id)
    hash_match = 1 if current_hash and row.get("task_hash") == current_hash else 0
    # A row with no valid, scoreable outcome (None: awaiting judge, harness
    # error, interrupted run, etc.) must never outrank a row that already has
    # a real result for the same task, no matter which is more recent. Without
    # this, a later but incomplete/timed-out run (e.g. a --think off reprobe
    # that never finished) can silently displace a genuinely valid, already
    # judged earlier result, since recency alone was the only tiebreaker left
    # once both rows shared the same task hash and neither used a ctx/
    # num_predict/task_regex override that _canonical_run_score() checks for
    # (think overrides alone aren't covered there).
    has_valid_score = 1 if (
        isinstance(row.get("score"), (int, float))
        and not isinstance(row.get("score"), bool)
        and not row.get("error_kind")
    ) else 0
    return (
        hash_match,
        has_valid_score,
        _canonical_run_score(row),
        LEVEL_RANK.get(str(row.get("level")), -1),
        str(row.get("timestamp") or ""),
        str(row.get("run_id") or ""),
    )


def rank_for_output(accumulated: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select one defensible current row per ``(digest, task)``.

    Preference: current task hash, canonical configuration, highest cumulative
    level, then latest timestamp. Historical rows remain untouched in the
    accumulated database and model-card history.
    """
    best: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    for row in accumulated:
        key = (row.get("model_digest_resolved"), row.get("task"))
        existing = best.get(key)
        if existing is None or _selection_key(row) >= _selection_key(existing):
            best[key] = row
    return list(best.values())


def _parse_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _task_seconds(rows: Iterable[Dict[str, Any]]) -> Tuple[Optional[float], str]:
    values = [float(row["task_wall_seconds"]) for row in rows if isinstance(row.get("task_wall_seconds"), (int, float))]
    if values:
        return round(sum(values), 3), "measured task wall time"
    times = [t for t in (_parse_time(row.get("timestamp")) for row in rows) if t is not None]
    if len(times) >= 2:
        return round((max(times) - min(times)).total_seconds(), 3), "legacy timestamp span"
    return None, "unavailable in legacy row"


def _families_for_entry(name: str, rows: List[Dict[str, Any]]) -> List[str]:
    family_set = set()
    declared: List[str] = []
    for row in rows:
        family_set.update(row.get("capability_families") or [])
        if row.get("family"):
            family_set.add(str(row.get("family")))
        declared.extend(row.get("capabilities_declared") or [])
        profile = row.get("capability_profile") or {}
        family_set.update(profile.get("supported_families") or [])
        declared.extend(profile.get("declared_capabilities") or [])

    # Re-evaluate accumulated legacy profiles through the current classifier.
    # This corrects old fleet rows where embedding-only models were persisted as
    # text+embedding+tools and prevents those obsolete routes from keeping the
    # model provisionally incomplete forever after the classifier is fixed.
    current_route = families_for(name, declared or None)
    if current_route == ["embedding"]:
        return ["embedding"]
    if not family_set:
        family_set.update(current_route)
    if "vision" in family_set:
        family_set.add("text")
    order = ("vision", "text", "embedding", "tools", "insert")
    return [family for family in order if family in family_set]


def _required_quality_tasks(families: List[str]) -> List[str]:
    return [task.id for task in TASKS if task.difficulty > 0 and task.family in families]


def _row_evidence_key(row: Dict[str, Any]) -> Tuple[str, str, int]:
    return (
        str(row.get("run_id") or ""),
        str(row.get("import_tag") or ""),
        int(row.get("_source_row_index") or 0),
    )


def _history_item(row: Dict[str, Any], *, selected_keys: set[Tuple[str, str, int]]) -> Dict[str, Any]:
    task = _TASKS.get(str(row.get("task") or ""))
    return {
        "used_for_current_ranking": _row_evidence_key(row) in selected_keys,
        "run_id": row.get("run_id"),
        "import_tag": row.get("import_tag"),
        "timestamp": row.get("timestamp"),
        "level": row.get("level"),
        "model_name": row.get("model"),
        "task": row.get("task"),
        "category": row.get("category"),
        "family": row.get("family"),
        "scorer": task.scorer if task else None,
        "difficulty": task.difficulty if task else _TASK_DIFFICULTY.get(str(row.get("task")), 1.0),
        "score": row.get("score"),
        "decision_score": row.get("decision_score"),
        "reason": row.get("reason"),
        "error_kind": row.get("error_kind"),
        "warning_kind": row.get("warning_kind"),
        "task_hash": row.get("task_hash"),
        "current_task_hash": _CURRENT_HASHES.get(str(row.get("task") or "")),
        "posthoc_judged": bool(row.get("posthoc_judged")),
        "judge_mode": row.get("judge_mode"),
        "judge_model": row.get("judge_model"),
        "samples_used": row.get("samples_used"),
        "num_ctx_used": row.get("num_ctx_used"),
        "context_length": row.get("context_length"),
        "num_predict": row.get("num_predict"),
        "think_sent": row.get("think_sent"),
        "tps": row.get("tps"),
        "ttft_ms": row.get("ttft_ms"),
        "task_wall_seconds": row.get("task_wall_seconds"),
        "vram_peak_mb": row.get("vram_peak_mb"),
        "offload_fraction": row.get("offload_fraction"),
        "benchmark_version": row.get("benchmark_version"),
        "output_chars": row.get("output_chars"),
        "thinking_chars": row.get("thinking_chars"),
        "done_reason": row.get("done_reason"),
        "raw_path": row.get("raw_path"),
        "subjective_path": row.get("subjective_path"),
        "run_configuration": row.get("run_configuration") or {},
    }


def _load_capability_unavailable_families(
    runs_dir: Optional[Path], run_ids: Iterable[str]
) -> Dict[str, set]:
    """Read capability_repair.json from each referenced run dir.

    Returns {model_name: {unavailable_family, ...}}. A model repeatedly
    failing the same functional capability gate (e.g. vision on an
    incompatible GGUF/mmproj build) should not display as "missing" those
    tasks forever -- the harness already determined the capability is
    genuinely unavailable on this build, which is a different, resolved
    finding, not an unattempted task.
    """
    result: Dict[str, set] = {}
    if runs_dir is None:
        return result
    for run_id in run_ids:
        if not run_id:
            continue
        path = Path(runs_dir) / str(run_id) / "capability_repair.json"
        data = _read_json_safe(path)
        if not data:
            continue
        for model_name, entry in data.items():
            families = set((entry or {}).get("unavailable_families") or {})
            if families:
                result.setdefault(model_name, set()).update(families)
    return result


def _load_recovery_exhausted_tasks(
    runs_dir: Optional[Path], run_ids: Iterable[str], model_names: Iterable[str]
) -> set[str]:
    """Adopt bounded generation-repair exhaustion as terminal model evidence.

    A retry_generation action that completed all configured attempts and remained
    unresolved because the model kept producing empty/thinking-only output is a
    measured task failure, not an unattempted cell.  The raw rows remain
    immutable; rankings use an effective score of zero for that task.
    """
    if runs_dir is None:
        return set()
    names = set(str(n) for n in model_names)
    exhausted: set[str] = set()
    for run_id in run_ids:
        path = Path(runs_dir) / str(run_id) / "repair_results.jsonl"
        for record in _read_jsonl(path):
            action = record.get("action") or {}
            if str(action.get("model") or "") not in names:
                continue
            if str(action.get("kind") or record.get("kind") or "") != "retry_generation":
                continue
            if str(record.get("status") or "") != "unresolved":
                continue
            attempts = list(record.get("attempts") or [])
            attempt_limit = int((action.get("details") or {}).get("attempt_limit") or len(attempts) or 0)
            if not attempts or len(attempts) < attempt_limit:
                continue
            terminal_behavior = False
            for attempt in attempts:
                child = str(attempt.get("child_run_id") or "")
                if not child:
                    continue
                for row in _read_jsonl(Path(runs_dir) / child / "raw_results.jsonl"):
                    if row.get("think_ineffective") or str(row.get("error_kind") or "") in {"thinking_only", "empty_output"}:
                        terminal_behavior = True
            if terminal_behavior:
                exhausted.update(str(t) for t in (action.get("tasks") or record.get("tasks") or []) if t)
    return exhausted


def _load_capability_measured_failure_tasks(
    runs_dir: Optional[Path], run_ids: Iterable[str], model_names: Iterable[str]
) -> set[str]:
    """Adopt scored zero-quality capability-gated outcomes as terminal evidence.

    RC10 writes ``measured_failure`` explicitly. For RC9 evidence already
    produced in the field, infer the same terminal outcome only when the
    task-equivalent probe responded and the scored insert task produced a
    numeric zero with ``empty_output``. Transport/runtime failures remain
    missing and provisional.
    """
    if runs_dir is None:
        return set()
    names = {str(name) for name in model_names}
    measured: set[str] = set()
    for run_id in run_ids:
        for record in _read_jsonl(Path(runs_dir) / str(run_id) / "repair_results.jsonl"):
            action = record.get("action") or {}
            if str(action.get("model") or "") not in names:
                continue
            if str(action.get("kind") or record.get("kind") or "") != "capability_gate":
                continue
            status = str(record.get("status") or "")
            gate_state = str(((record.get("gate") or {}).get("probe_state") or ""))
            explicit = status == "measured_failure"
            infer_legacy = status == "unresolved" and gate_state in {
                "confirmed_supported", "responded_contract_failed"
            }
            if not explicit and not infer_legacy:
                continue
            for attempt in record.get("attempts") or []:
                for task_id in attempt.get("terminal_failure_tasks") or []:
                    measured.add(str(task_id))
                if explicit:
                    continue
                child = str(attempt.get("child_run_id") or "")
                if not child:
                    continue
                for row in _read_jsonl(Path(runs_dir) / child / "raw_results.jsonl"):
                    task_id = str(row.get("task") or "")
                    task = _TASKS.get(task_id)
                    if task_id not in set(action.get("tasks") or []) or not task:
                        continue
                    if (task.family == "insert"
                            and isinstance(row.get("score"), (int, float))
                            and float(row.get("score") or 0.0) <= 0.0
                            and str(row.get("error_kind") or "") == "empty_output"):
                        measured.add(task_id)
    return measured


def _read_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def build_summary(
    ranked_rows: List[Dict[str, Any]],
    history_rows: Optional[List[Dict[str, Any]]] = None,
    *,
    runs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    history_rows = history_rows if history_rows is not None else ranked_rows
    selected_by_digest: Dict[str, List[Dict[str, Any]]] = {}
    history_by_digest: Dict[str, List[Dict[str, Any]]] = {}
    for row in ranked_rows:
        selected_by_digest.setdefault(str(row.get("model_digest_resolved")), []).append(row)
    for row in history_rows:
        history_by_digest.setdefault(str(row.get("model_digest_resolved")), []).append(row)

    summary: Dict[str, Any] = {}
    for digest, rows in selected_by_digest.items():
        all_history = history_by_digest.get(digest, rows)
        names = sorted({str(row.get("model")) for row in all_history if row.get("model")})
        run_ids_for_digest = {row.get("run_id") for row in all_history if row.get("run_id")}
        unavailable_by_name = _load_capability_unavailable_families(runs_dir, run_ids_for_digest)
        unavailable_families: set = set()
        for candidate_name in names:
            unavailable_families.update(unavailable_by_name.get(candidate_name) or set())
        recovery_exhausted_tasks = _load_recovery_exhausted_tasks(
            runs_dir, run_ids_for_digest, names
        )
        capability_measured_failure_tasks = _load_capability_measured_failure_tasks(
            runs_dir, run_ids_for_digest, names
        )
        display_name = names[0] if names else str(digest)
        levels = sorted({str(row.get("level")) for row in all_history if row.get("level")}, key=lambda level: LEVEL_RANK.get(level, -1))
        fully_tested = "full" in levels
        families = _families_for_entry(display_name, rows)

        scoring_rows: List[Dict[str, Any]] = []
        for row in rows:
            adjusted = dict(row)
            task_id = str(adjusted.get("task") or "")
            if task_id in recovery_exhausted_tasks and (
                not isinstance(adjusted.get("score"), (int, float)) or adjusted.get("error_kind")
            ):
                adjusted["score"] = 0.0
                adjusted["error_kind"] = None
                adjusted["terminal_failure_kind"] = "recovery_exhausted"
                adjusted["reason"] = (str(adjusted.get("reason") or "") + "; bounded recovery exhausted").strip("; ")
            if task_id in capability_measured_failure_tasks and (
                not isinstance(adjusted.get("score"), (int, float)) or adjusted.get("error_kind")
            ):
                adjusted["score"] = 0.0
                adjusted["error_kind"] = None
                adjusted["terminal_failure_kind"] = "capability_measured_failure"
                adjusted["reason"] = (str(adjusted.get("reason") or "") + "; capability-gated scored task measured zero quality").strip("; ")
            scoring_rows.append(adjusted)
        canonical_rows = [{**row, "model": digest} for row in scoring_rows]
        leaderboard, per_cat = aggregate(canonical_rows, DEFAULT_WEIGHTS, _TASK_DIFFICULTY)
        aggregate_row = leaderboard[0] if leaderboard else {}
        overall = aggregate_row.get("quality")
        category_scores = aggregate_row.get("categories") or {}
        category_errors = aggregate_row.get("category_errors") or {}
        category_coverage = aggregate_row.get("category_coverage") or {}
        category_ineligible = aggregate_row.get("category_ineligible") or {}

        by_category: Dict[str, List[Dict[str, Any]]] = {}
        for row in scoring_rows:
            by_category.setdefault(str(row.get("category") or "unknown"), []).append(row)
        category_summary: Dict[str, Any] = {}
        for category, category_rows in sorted(by_category.items()):
            seconds, timing_basis = _task_seconds(category_rows)
            category_summary[category] = {
                "score": category_scores.get(category),
                "task_count": len(category_rows),
                "error_count": int(category_errors.get(category, 0)),
                "coverage": category_coverage.get(category),
                "ineligible_reason": category_ineligible.get(category),
                "wall_seconds": seconds,
                "timing_basis": timing_basis,
                "tasks": [],
            }
            for row in sorted(category_rows, key=lambda item: str(item.get("task") or "")):
                task = _TASKS.get(str(row.get("task") or ""))
                category_summary[category]["tasks"].append({
                    "task": row.get("task"),
                    "score": row.get("score"),
                    "decision_score": row.get("decision_score"),
                    "reason": row.get("reason"),
                    "error_kind": row.get("error_kind"),
                    "warning_kind": row.get("warning_kind"),
                    "level": row.get("level"),
                    "run_id": row.get("run_id"),
                    "import_tag": row.get("import_tag"),
                    "family": row.get("family"),
                    "scorer": task.scorer if task else None,
                    "difficulty": task.difficulty if task else _TASK_DIFFICULTY.get(str(row.get("task")), 1.0),
                    "prompt": task.prompt if task else None,
                    "rubric": task.rubric if task else None,
                    "expectation": task.meta if task else None,
                    "samples_used": row.get("samples_used"),
                    "num_ctx_used": row.get("num_ctx_used"),
                    "num_predict": row.get("num_predict"),
                    "think_sent": row.get("think_sent"),
                    "judge_mode": row.get("judge_mode"),
                    "judge_model": row.get("judge_model"),
                    "posthoc_judged": bool(row.get("posthoc_judged")),
                    "task_wall_seconds": row.get("task_wall_seconds"),
                    "tps": row.get("tps"),
                    "ttft_ms": row.get("ttft_ms"),
                    "vram_peak_mb": row.get("vram_peak_mb"),
                })

        required = _required_quality_tasks(families)
        rows_by_task = {str(row.get("task")): row for row in scoring_rows}
        missing: List[str] = []
        stale: List[str] = []
        capability_unavailable: List[str] = []
        think_ineffective_tasks: List[str] = []
        for task_id in required:
            row = rows_by_task.get(task_id)
            task_def = _TASKS.get(task_id)
            task_family = task_def.family if task_def else None
            if task_family and task_family in unavailable_families:
                # The harness already ran this model's functional capability
                # gate for this family and it genuinely failed on the
                # installed build (e.g. vision on an incompatible GGUF/mmproj
                # combination). That's a resolved, diagnosed finding -- it
                # must not display identically to "never attempted".
                capability_unavailable.append(task_id)
                continue
            if task_id in recovery_exhausted_tasks or task_id in capability_measured_failure_tasks:
                continue
            if row is None:
                missing.append(task_id)
                continue
            current_hash = _CURRENT_HASHES.get(task_id)
            if current_hash and row.get("task_hash") != current_hash:
                stale.append(task_id)
                missing.append(task_id)
                continue
            if row.get("think_ineffective") and (not isinstance(row.get("score"), (int, float)) or row.get("error_kind")):
                # The server accepted think=off but the model kept producing
                # hidden-only reasoning anyway. Also diagnosed and resolved,
                # not an unattempted task -- more retries won't change it.
                think_ineffective_tasks.append(task_id)
                continue
            if not isinstance(row.get("score"), (int, float)) or row.get("error_kind"):
                missing.append(task_id)
                continue
            if task_id == "needle" and float(row.get("needle_coverage") or 0.0) < 1.0:
                missing.append(task_id)

        coverage_ratio = round((len(required) - len(set(missing))) / len(required), 4) if required else 0.0
        status_reasons: List[str] = []
        if not fully_tested:
            status_reasons.append("no cumulative full-level run is present")
        if stale:
            status_reasons.append(f"{len(set(stale))} applicable task result(s) use a missing or outdated task hash")
        missing_non_stale = set(missing) - set(stale)
        if missing_non_stale:
            status_reasons.append(f"{len(missing_non_stale)} applicable positive-difficulty task result(s) are missing, errored, unjudged, or incomplete")
        if capability_unavailable:
            status_reasons.append(
                f"{len(set(capability_unavailable))} registered task(s) excluded because the required capability is "
                f"confirmed unavailable on the installed build: {', '.join(sorted(set(capability_unavailable)))}"
            )
        if think_ineffective_tasks:
            status_reasons.append(
                f"{len(set(think_ineffective_tasks))} applicable task(s) hit persistent think_ineffective "
                f"behavior (think=off accepted but hidden reasoning continued): {', '.join(sorted(set(think_ineffective_tasks)))}"
            )
        if recovery_exhausted_tasks:
            status_reasons.append(
                f"{len(set(recovery_exhausted_tasks))} task(s) recorded as zero-quality terminal failures after bounded recovery was exhausted: "
                f"{', '.join(sorted(set(recovery_exhausted_tasks)))}"
            )
        if capability_measured_failure_tasks:
            status_reasons.append(
                f"{len(set(capability_measured_failure_tasks))} task(s) recorded as measured zero-quality outcomes after a responding capability gate: "
                f"{', '.join(sorted(set(capability_measured_failure_tasks)))}"
            )
        if not required:
            status_reasons.append("no positive-difficulty tasks are applicable to the detected capability families")
        if not isinstance(overall, (int, float)):
            quality_status = "ineligible"
            status_reasons.insert(0, "no eligible numeric positive-difficulty quality score exists")
        elif fully_tested and required and not missing:
            quality_status = "complete"
            base_reason = "current full-level evidence covers every applicable positive-difficulty task"
            if capability_unavailable or think_ineffective_tasks or capability_measured_failure_tasks:
                status_reasons = [base_reason] + [
                    r for r in status_reasons
                    if ("confirmed unavailable" in r or "think_ineffective" in r
                        or "bounded recovery" in r or "responding capability gate" in r)
                ]
            else:
                status_reasons = [base_reason]
        else:
            quality_status = "provisional"
        capability_limited = bool(capability_unavailable)
        capability_measured_failure = bool(capability_measured_failure_tasks)
        # Recovery-limited means a bounded retry/thinking policy was exhausted.
        # A responding capability gate followed by a real scored zero is a
        # measured capability-quality outcome, not retry exhaustion.
        recovery_limited = bool(recovery_exhausted_tasks or think_ineffective_tasks)

        total_seconds, total_timing_basis = _task_seconds(rows)
        if not fully_tested:
            total_seconds = None
            total_timing_basis = "partial level coverage"

        tps_values = [float(row["tps"]) for row in rows if isinstance(row.get("tps"), (int, float))]
        size_values = [float(row["size_gb"]) for row in rows if isinstance(row.get("size_gb"), (int, float))]
        class_values = [str(row.get("class")) for row in rows if row.get("class")]
        selected_keys = {_row_evidence_key(row) for row in rows}
        history = [_history_item(row, selected_keys=selected_keys) for row in sorted(all_history, key=lambda item: str(item.get("timestamp") or ""), reverse=True)]
        identities = [row.get("model_identity") or {} for row in all_history]
        parameter_sizes = [str(identity.get("parameter_size")) for identity in identities if identity.get("parameter_size")]
        quantizations = [str(identity.get("quantization_level")) for identity in identities if identity.get("quantization_level")]
        architecture_families = [str(identity.get("family")) for identity in identities if identity.get("family")]
        model_size_bytes = [int(identity.get("size")) for identity in identities if isinstance(identity.get("size"), (int, float))]

        # Dedicated context-profile rows are diagnostic and do not replace
        # canonical quality rows, but they may provide richer operating data.
        needle_rows = [
            row for row in all_history
            if str(row.get("task")) == "needle"
            and str(row.get("task_hash") or "") == str(_CURRENT_HASHES.get("needle") or "")
            and isinstance(row.get("score"), (int, float))
            and not row.get("error_kind")
        ]
        long_context_profile = None
        if needle_rows:
            def telemetry_completeness(row: Dict[str, Any]) -> int:
                fields = (
                    "elapsed_seconds", "request_elapsed_seconds", "tps", "prompt_tps",
                    "vram_peak_mb", "ram_peak_mb", "ollama_rss_peak_mb",
                    "ollama_pss_peak_mb", "offload_fraction",
                )
                return sum(
                    1 for probe in (row.get("needle_attempted") or [])
                    for field in fields if probe.get(field) is not None
                )

            best_needle = max(
                needle_rows,
                key=lambda row: (float(row.get("needle_coverage") or 0.0),
                                 int(row.get("max_verified_ctx") or 0),
                                 telemetry_completeness(row),
                                 str(row.get("timestamp") or "")),
            )
            depths = []
            for probe in best_needle.get("needle_attempted") or []:
                depths.append({
                    "size": probe.get("size"),
                    "num_ctx": probe.get("num_ctx"),
                    "prompt_tokens_actual": probe.get("prompt_tokens_actual"),
                    "found": probe.get("found"),
                    "tps": probe.get("tps"),
                    "prompt_tps": probe.get("prompt_tps"),
                    "ttft_ms": probe.get("ttft_ms"),
                    "ttft_visible_ms": probe.get("ttft_visible_ms"),
                    "request_elapsed_seconds": probe.get("request_elapsed_seconds"),
                    "elapsed_seconds": probe.get("elapsed_seconds"),
                    "server_total_duration_ms": probe.get("server_total_duration_ms"),
                    "server_load_duration_ms": probe.get("server_load_duration_ms"),
                    "server_prompt_eval_duration_ms": probe.get("server_prompt_eval_duration_ms"),
                    "server_eval_duration_ms": probe.get("server_eval_duration_ms"),
                    "vram_start_mb": probe.get("vram_start_mb"),
                    "vram_peak_mb": probe.get("vram_peak_mb"),
                    "vram_delta_peak_mb": probe.get("vram_delta_peak_mb"),
                    "ram_start_mb": probe.get("ram_start_mb"),
                    "ram_peak_mb": probe.get("ram_peak_mb"),
                    "ram_delta_peak_mb": probe.get("ram_delta_peak_mb"),
                    "ram_available_min_mb": probe.get("ram_available_min_mb"),
                    "swap_peak_mb": probe.get("swap_peak_mb"),
                    "swap_delta_peak_mb": probe.get("swap_delta_peak_mb"),
                    "ollama_rss_peak_mb": probe.get("ollama_rss_peak_mb"),
                    "ollama_rss_delta_peak_mb": probe.get("ollama_rss_delta_peak_mb"),
                    "ollama_pss_peak_mb": probe.get("ollama_pss_peak_mb"),
                    "ollama_pss_delta_peak_mb": probe.get("ollama_pss_delta_peak_mb"),
                    "ollama_swap_delta_peak_mb": probe.get("ollama_swap_delta_peak_mb"),
                    "host_memory_signal": probe.get("host_memory_signal"),
                    "offload_fraction": probe.get("offload_fraction"),
                    "model_offloaded_gb": probe.get("model_offloaded_gb"),
                    "model_loaded_size_bytes": probe.get("model_loaded_size_bytes"),
                    "model_vram_bytes": probe.get("model_vram_bytes"),
                    "model_host_bytes": probe.get("model_host_bytes"),
                    "gpu_util_mean_pct": probe.get("gpu_util_mean_pct"),
                    "gpu_util_peak_pct": probe.get("gpu_util_peak_pct"),
                    "power_mean_w": probe.get("power_mean_w"),
                    "power_peak_w": probe.get("power_peak_w"),
                    "temp_peak_c": probe.get("temp_peak_c"),
                    "cpu_util_mean_pct": probe.get("cpu_util_mean_pct"),
                    "cpu_util_peak_pct": probe.get("cpu_util_peak_pct"),
                    "cpu_temp_peak_c": probe.get("cpu_temp_peak_c"),
                    "estimated_total_gb": probe.get("estimated_total_gb"),
                    "estimated_gpu_peak_gb": probe.get("estimated_gpu_peak_gb"),
                    "estimated_host_increment_gb": probe.get("estimated_host_increment_gb"),
                    "kv_estimate_method": probe.get("kv_estimate_method"),
                    "kv_estimate_confidence": probe.get("kv_estimate_confidence"),
                    "kv_estimate_warning": probe.get("kv_estimate_warning"),
                    "response_exact": probe.get("needle_response_exact"),
                    "response_suspect": probe.get("needle_response_suspect"),
                    "done_reason": probe.get("done_reason"),
                    "error_kind": probe.get("error_kind"),
                })
            long_context_profile = {
                "score": best_needle.get("score"),
                "coverage": best_needle.get("needle_coverage"),
                "max_verified_ctx": best_needle.get("max_verified_ctx"),
                "max_requested_size": best_needle.get("needle_max_requested_size"),
                "min_tps": best_needle.get("needle_min_tps"),
                "median_tps": best_needle.get("needle_median_tps"),
                "max_depth_tps": best_needle.get("needle_max_depth_tps"),
                "max_depth_prompt_tps": best_needle.get("needle_max_depth_prompt_tps"),
                "max_depth_elapsed_seconds": best_needle.get("needle_max_depth_elapsed_seconds"),
                "min_prompt_tps": best_needle.get("needle_min_prompt_tps"),
                "max_offload_fraction": best_needle.get("needle_max_offload_fraction"),
                "max_ram_delta_mb": best_needle.get("needle_max_ram_delta_mb"),
                "max_ollama_rss_delta_mb": best_needle.get("needle_max_ollama_rss_delta_mb"),
                "max_ollama_pss_delta_mb": best_needle.get("needle_max_ollama_pss_delta_mb"),
                "max_swap_delta_mb": best_needle.get("needle_max_swap_delta_mb"),
                "behavior_suspect": best_needle.get("needle_behavior_suspect"),
                "target_ctx": best_needle.get("needle_target_ctx"),
                "target_min_tps": best_needle.get("needle_target_min_tps"),
                "target_critical_tps": best_needle.get("needle_target_critical_tps"),
                "target_status": best_needle.get("needle_target_status"),
                "target_size": best_needle.get("needle_target_size"),
                "target_num_ctx": best_needle.get("needle_target_num_ctx"),
                "target_tps": best_needle.get("needle_target_tps"),
                "target_prompt_tps": best_needle.get("needle_target_prompt_tps"),
                "target_elapsed_seconds": best_needle.get("needle_target_elapsed_seconds"),
                "target_offload_fraction": best_needle.get("needle_target_offload_fraction"),
                "target_ram_delta_mb": best_needle.get("needle_target_ram_delta_mb"),
                "target_ollama_rss_delta_mb": best_needle.get("needle_target_ollama_rss_delta_mb"),
                "target_ollama_pss_delta_mb": best_needle.get("needle_target_ollama_pss_delta_mb"),
                "target_swap_delta_mb": best_needle.get("needle_target_swap_delta_mb"),
                "target_behavior_suspect": best_needle.get("needle_target_behavior_suspect"),
                "slow_depths": best_needle.get("needle_slow_depths") or [],
                "critical_slow_depths": best_needle.get("needle_critical_slow_depths") or [],
                "run_id": best_needle.get("run_id"),
                "depths": depths,
            }

        summary[digest] = {
            "digest": digest,
            "names_seen": names,
            "display_name": display_name,
            "class": Counter(class_values).most_common(1)[0][0] if class_values else None,
            "families": families,
            "declared_capabilities": sorted({cap for row in rows for cap in (row.get("capabilities_declared") or [])}),
            "levels_seen": levels,
            "fully_tested": fully_tested,
            "quality_status": quality_status,
            "quality_status_reasons": status_reasons,
            "overall_mean_score": overall,
            "quality_blended": aggregate_row.get("quality_blended"),
            "coverage_ratio": coverage_ratio,
            "required_quality_tasks": required,
            "missing_quality_tasks": sorted(set(missing)),
            "capability_unavailable_tasks": sorted(set(capability_unavailable)),
            "think_ineffective_tasks": sorted(set(think_ineffective_tasks)),
            "recovery_exhausted_tasks": sorted(set(recovery_exhausted_tasks)),
            "capability_measured_failure_tasks": sorted(set(capability_measured_failure_tasks)),
            "capability_limited": capability_limited,
            "capability_measured_failure": capability_measured_failure,
            "recovery_limited": recovery_limited,
            "stale_quality_tasks": sorted(set(stale)),
            "total_wall_seconds": total_seconds,
            "total_timing_basis": total_timing_basis,
            "tok_s": round(statistics.mean(tps_values), 2) if tps_values else None,
            "size_gb": round(statistics.median(size_values), 2) if size_values else None,
            "model_size_bytes": int(statistics.median(model_size_bytes)) if model_size_bytes else None,
            "parameter_size": Counter(parameter_sizes).most_common(1)[0][0] if parameter_sizes else None,
            "quantization_level": Counter(quantizations).most_common(1)[0][0] if quantizations else None,
            "architecture_family": Counter(architecture_families).most_common(1)[0][0] if architecture_families else None,
            "error_count": int(aggregate_row.get("err") or 0),
            "gate_failures": int(aggregate_row.get("gate_failures") or 0),
            "completion_rate": aggregate_row.get("completion_rate"),
            "categories": category_summary,
            "long_context_profile": long_context_profile,
            "history_count": len(history),
            "history": history,
        }
    _assign_overall_ranks(summary)
    return summary


def _tie_break_key(model: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        -float(model.get("coverage_ratio") or 0.0),
        int(model.get("error_count") or 0),
        int(model.get("gate_failures") or 0),
        -float(model.get("tok_s") or 0.0),
        float(model.get("size_gb") or 1e9),
        str(model.get("display_name") or "").lower(),
    )


def _assign_overall_ranks(summary: Dict[str, Any]) -> None:
    # The master overall table compares text-capable assistants. Embedding-only
    # specialists are not commensurate with chat/coding models and remain in
    # their category/class tables instead of winning an artificial cross-class tie.
    for model in summary.values():
        model["overall_comparable"] = "text" in (model.get("families") or [])
        model["overall_rank"] = None
        model["tie_band"] = None
    models = [
        model for model in summary.values()
        if model.get("overall_comparable") and isinstance(model.get("overall_mean_score"), (int, float))
    ]
    models.sort(key=lambda model: (-float(model["overall_mean_score"]),) + _tie_break_key(model))
    band = 0
    band_top: Optional[float] = None
    display_position = 0
    for model in models:
        display_position += 1
        score = float(model["overall_mean_score"])
        if band_top is None or band_top - score > TIE_EPSILON:
            band += 1
            band_top = score
        model["overall_rank"] = display_position
        model["tie_band"] = band
        model["tie_note"] = "quality scores within 0.5 points share a band; coverage/errors/gates/speed/size only order the display"


def _top_by_category(models: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    categories = sorted({category for model in models for category in model.get("categories", {})})
    out: Dict[str, List[Dict[str, Any]]] = {}
    for category in categories:
        rows = []
        for model in models:
            score = (model.get("categories", {}).get(category) or {}).get("score")
            if isinstance(score, (int, float)):
                rows.append({"model": model["display_name"], "digest": model["digest"], "score": score,
                             "status": model["quality_status"], "coverage": model["coverage_ratio"],
                             "tok_s": model.get("tok_s"), "size_gb": model.get("size_gb")})
        rows.sort(key=lambda row: (-float(row["score"]), -float(row.get("coverage") or 0), -float(row.get("tok_s") or 0), float(row.get("size_gb") or 1e9)))
        out[category] = rows[:5]
    return out


def _top_by_class(models: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    classes = sorted({str(model.get("class") or "unclassified") for model in models})
    out: Dict[str, List[Dict[str, Any]]] = {}
    for cls in classes:
        rows = [model for model in models if str(model.get("class") or "unclassified") == cls and isinstance(model.get("overall_mean_score"), (int, float))]
        rows.sort(key=lambda model: (-float(model["overall_mean_score"]),) + _tie_break_key(model))
        out[cls] = [{"model": model["display_name"], "digest": model["digest"], "score": model["overall_mean_score"],
                     "status": model["quality_status"], "coverage": model["coverage_ratio"],
                     "tok_s": model.get("tok_s"), "size_gb": model.get("size_gb")} for model in rows[:5]]
    return out


def _multimodal(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for model in models:
        if "vision" not in model.get("families", []) and not ({"ocr", "pdf"} & set(model.get("categories", {}))):
            continue
        scores = []
        weights = []
        for category in ("ocr", "pdf"):
            score = (model.get("categories", {}).get(category) or {}).get("score")
            if isinstance(score, (int, float)):
                scores.append(float(score) * DEFAULT_WEIGHTS.get(category, 1.0))
                weights.append(DEFAULT_WEIGHTS.get(category, 1.0))
        mm_score = round(sum(scores) / sum(weights), 2) if weights else None
        out.append({"model": model["display_name"], "digest": model["digest"], "score": mm_score,
                    "status": model["quality_status"], "coverage": model["coverage_ratio"],
                    "tok_s": model.get("tok_s"), "size_gb": model.get("size_gb")})
    out.sort(key=lambda row: (-(float(row["score"]) if isinstance(row.get("score"), (int, float)) else -1), -float(row.get("coverage") or 0), -float(row.get("tok_s") or 0), float(row.get("size_gb") or 1e9)))
    return out


def build_report_payload(summary: Dict[str, Any]) -> Dict[str, Any]:
    models = list(summary.values())
    models.sort(key=lambda model: (model.get("overall_rank") or 10**9, str(model.get("display_name") or "")))
    status_counts = Counter(model.get("quality_status") for model in models)
    return {
        "generated_at": datetime.now().isoformat(),
        "models": models,
        "status_counts": dict(status_counts),
        "top_by_category": _top_by_category(models),
        "top_by_class": _top_by_class(models),
        "multimodal": _multimodal(models),
        "methodology": {
            "category_score": "difficulty-weighted task outcomes; difficulty-zero tasks are gates and contribute no positive quality",
            "overall_score": "category-weighted mean renormalised over eligible measured categories",
            "overall_scope": "master overall ranks compare text-capable assistants only; embedding-only specialists remain in category and class rankings",
            "selection": "current task hash, valid non-error outcome, canonical configuration, highest cumulative level, latest timestamp",
            "tie_band_points": TIE_EPSILON,
            "tie_breakers": ["coverage", "fewer errors", "fewer failed gates", "higher measured speed", "smaller model size"],
            "weights": DEFAULT_WEIGHTS,
            "status": {
                "complete": "full-level evidence and every currently applicable positive-difficulty task has a current, numeric, non-error result",
                "provisional": "a numeric score exists but current applicable quality scope is incomplete, stale, unjudged, or not full-level",
                "ineligible": "no eligible numeric positive-difficulty quality score exists",
            },
        },
    }


def write_rankings(
    runs_dir: Path,
    rankings_dir: Path,
    html_template: Optional[str] = None,
    *,
    force_rescan: bool = False,
    include_separate: bool = False,
    only_run_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    rankings_dir.mkdir(parents=True, exist_ok=True)
    accumulated = load_accumulated(rankings_dir)
    accumulated = import_new_runs(
        runs_dir, accumulated, force_rescan=force_rescan, rankings_dir=rankings_dir,
        include_separate=include_separate, only_run_ids=only_run_ids,
    )
    exclusions = load_exclusions(rankings_dir)
    excluded_models = excluded_model_keys(exclusions)
    rankable = [
        row for row in accumulated
        if (include_separate or row.get("ranking_scope", SCOPE_CANONICAL) == SCOPE_CANONICAL)
        and not model_matches(row, excluded_models)
        and (include_separate or str(row.get("run_id") or "") not in excluded_run_ids(exclusions))
    ]
    ranked = rank_for_output(rankable)
    summary = build_summary(ranked, rankable, runs_dir=runs_dir)
    payload = build_report_payload(summary)
    payload["ranking_controls"] = {
        "exclusions": summarize_exclusions(exclusions),
        "include_separate": bool(include_separate),
        "only_run_ids": list(only_run_ids or []),
    }

    raw_path = rankings_dir / "master_raw.jsonl"
    raw_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in accumulated) + ("\n" if accumulated else ""))

    summary_path = rankings_dir / "master_summary.json"
    summary_path.write_text(json.dumps(list(summary.values()), indent=2, sort_keys=True))

    payload_path = rankings_dir / "master_report_data.json"
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    from .rankings_v3 import write_v3_artifacts
    v3_result = write_v3_artifacts(payload, rankings_dir)

    html_path = rankings_dir / "master_report.html"
    if html_template is not None:
        html_path.write_text(html_template.replace("__MASTER_SUMMARY_JSON__", json.dumps(payload).replace("</", "<\\/")))

    return {
        "raw_rows_total": len(accumulated),
        "models": len(summary),
        "exclusions": summarize_exclusions(exclusions),
        "include_separate": bool(include_separate),
        "only_run_ids": list(only_run_ids or []),
        "complete_models": int(payload["status_counts"].get("complete", 0)),
        "provisional_models": int(payload["status_counts"].get("provisional", 0)),
        "raw_path": str(raw_path),
        "summary_path": str(summary_path),
        "payload_path": str(payload_path),
        "html_path": str(html_path),
        "v3_payload_path": v3_result["v3_data_path"],
        "v3_html_path": v3_result["v3_html_path"],
        "v3_schema_version": v3_result["v3_schema_version"],
        "v3_use_cases": v3_result["v3_use_cases"],
        "v31_payload_path": v3_result.get("v31_data_path"),
        "v31_html_path": v3_result.get("v31_html_path"),
        "v31_site_path": v3_result.get("v31_site_path"),
        "v31_schema_version": v3_result.get("v31_schema_version"),
        "v31_model_pages": v3_result.get("v31_model_pages"),
    }
