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

from . import campaign, judge as judge_mod
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


class ManualJudgeIneligibleError(Exception):
    """The explicitly-named ``--judge-model`` candidate does not satisfy the
    typed capability-authority and qualification gates every other judge
    candidate must pass (Anvil Stage 2.6E closure fix). Named selection
    chooses which candidate to evaluate; it does not manufacture capability
    or bypass qualification -- see ``campaign.build_manual_judge_candidate()``.
    """


def _capability_ledger_for_runs_dir(runs_dir: Path) -> "campaign.EvidenceLedger":
    """The native capability ``EvidenceLedger`` for a runs *root* directory
    (Anvil Stage 3.5) -- ``default_ledger_path(runs_dir)`` directly, unlike
    :func:`campaign._run_capability_ledger` which takes a single
    ``<runs_dir>/<run_id>`` run directory."""
    from .capability_reprobe_execute import default_ledger_path

    return campaign.EvidenceLedger(default_ledger_path(Path(runs_dir)))


def _resolve_manual_judge_pool(
    client: Any,
    judge_model: str,
    *,
    ledger: Optional["campaign.EvidenceLedger"] = None,
) -> List[Dict[str, Any]]:
    """Resolve the operator-named ``--judge-model`` into a real qualified-
    judge pool of at most one entry.

    ``ledger`` (Anvil Stage 3.5): when supplied, the capability-eligibility
    and qualification gates below prefer native identity-compatible
    ``EvidenceLedger`` evidence over the legacy adapter. This is the
    operator-named judge path -- unlike the automatic campaign path it
    never flows through ``build_judge_selection``, so this is its *only*
    capability gate. When ``ledger`` is ``None`` (or absent/empty) the
    behaviour is the unchanged pre-3.5 legacy-adapter path.

    Before Anvil Stage 2.6E this bypassed the capability-eligibility and
    qualification gates entirely (a bare dict tagged
    ``manual_unqualified_designation``, handed straight to
    ``campaign.resolve_independent_judge_for_row()``) -- confirmed by
    tracing that neither ``campaign.build_judge_selection()`` nor
    ``campaign.qualify_judge()`` was ever reached on this path. That was a
    real, live judge-authority bypass (category C, "positive judge
    authority outside the typed Stage 2 path" in the 2.6E audit), not a
    cosmetic gap. Fixed per the go-ahead advice's preferred "Option A":
    named selection still chooses the candidate, but the candidate must
    independently pass measured capability eligibility and qualification,
    identically to an automatically-selected candidate. Raises
    :class:`ManualJudgeIneligibleError` with a clear reason rather than
    silently returning an empty pool (which would otherwise surface only
    as an opaque ``awaiting_independent_judge`` state with no indication
    the operator's explicit choice was the reason).
    """
    if client is None:
        raise ManualJudgeIneligibleError(
            f"cannot resolve manual judge {judge_model!r}: a live client is required to interrogate and qualify it"
        )
    candidate = campaign.build_manual_judge_candidate(client, judge_model)
    capability_rejection = campaign._judge_capability_rejection(candidate, ledger=ledger)
    if capability_rejection:
        raise ManualJudgeIneligibleError(
            f"--judge-model {judge_model!r} is not capability-eligible to judge: {capability_rejection}"
        )
    qualification = campaign.qualify_judge(client, candidate, ledger=ledger)
    if not qualification.get("qualified"):
        raise ManualJudgeIneligibleError(
            f"--judge-model {judge_model!r} failed judge qualification: {qualification.get('aggregate_disposition')}"
        )
    return [{
        **candidate,
        "qualified": True,
        "qualification": qualification,
        "manual_designation": True,
        "roles": campaign._model_roles(candidate),
        "stable_identity": campaign.stable_model_identity(candidate),
    }]


def _entry_anchor_semantics_are_current(entry: Dict[str, Any]) -> bool:
    """Anvil Stage 3.5D fail-closed check: may a prior ``judge_results.jsonl``
    entry be treated as *current* judged evidence for skip purposes?

    - Field absent -> legacy entry (pre-3.5D). Preserve the existing
      treatment: the entry counts as current (§7 -- historical evidence is
      not retroactively invalidated, no current hash is invented for it).
    - Field present and equal to ``judge.JUDGE_ANCHOR_POLICY_HASH`` -> the
      scoring anchors / panel personas / request contract version that
      produced it match the current judge semantics. Current.
    - Field present and different -> proven judge-anchor drift. NOT current:
      the row is re-judged rather than the drifted score standing in as
      identical current evidence (§8 -- drift cannot masquerade as current).

    Only scoring semantics are gated here. Structural-capability facts
    (whether a judge 415s) and not-yet-scored states are unaffected by
    anchors, so ``_matching_exhausted_execution_entry`` /
    ``_prior_structural_failure`` / ``_matching_pending_entry`` are
    deliberately left untouched.
    """
    recorded = entry.get("judge_anchor_policy_hash")
    if recorded is None:
        return True
    return recorded == judge_mod.JUDGE_ANCHOR_POLICY_HASH


def _matching_judged_entry(entry: Dict[str, Any], resolution: Dict[str, Any], judge_mode: str) -> bool:
    if entry.get("status") != "judged" or entry.get("judge_mode") != judge_mode:
        return False
    if not _entry_anchor_semantics_are_current(entry):
        return False
    judge = resolution.get("judge") or {}
    expected = {
        "model": resolution.get("judge_model"),
        "digest": resolution.get("judge_digest") or judge.get("digest"),
    }
    actual = {
        "model": entry.get("judge_model"),
        "digest": entry.get("judge_model_digest"),
    }
    if expected["digest"] or actual["digest"]:
        return bool(expected["digest"] and actual["digest"] and expected["digest"] == actual["digest"])
    return campaign.same_stable_model_identity(expected, actual)


def _matching_pending_entry(entry: Dict[str, Any], pool_signature: str, judge_mode: str) -> bool:
    return (
        entry.get("status") == "awaiting_independent_judge"
        and entry.get("judge_mode") == judge_mode
        and entry.get("judge_pool_signature") == pool_signature
    )


def _matching_judge_error_entry(entry: Dict[str, Any], resolution: Dict[str, Any], judge_mode_configuration: Dict[str, Any]) -> bool:
    if entry.get("status") != "judge_error":
        return False
    if not _entry_anchor_semantics_are_current(entry):
        return False
    judge = resolution.get("judge") or {}
    expected = _compatibility_fingerprint(judge, judge_mode_configuration)
    attempts = entry.get("judgement_attempts") if isinstance(entry.get("judgement_attempts"), list) else []
    return any(
        attempt.get("status") == "judge_error"
        and attempt.get("compatibility_fingerprint") == expected
        for attempt in attempts
    )


def _pool_evidence(qualified_judges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evidence = []
    for index, judge in enumerate(qualified_judges):
        qualification = judge.get("qualification") if isinstance(judge.get("qualification"), dict) else {}
        evidence.append({
            "index": index,
            "name": judge.get("name") or judge.get("model"),
            "digest": judge.get("digest") or judge.get("model_digest_resolved"),
            "roles": judge.get("roles") or [],
            "qualified": judge.get("qualified"),
            "qualification_state": judge.get("qualification_state"),
            "qualification_protocol_version": qualification.get("protocol_version"),
            "qualification_disposition": qualification.get("aggregate_disposition"),
            "stable_identity": campaign.stable_model_identity(judge),
        })
    return evidence


def _judge_config(judge_mode: str, num_ctx: Optional[int], think: str) -> Dict[str, Any]:
    return {"judge_mode": judge_mode, "num_ctx": num_ctx, "think": think}


def _source_identity(row: Dict[str, Any]) -> Dict[str, Any]:
    return campaign.stable_model_identity(row)


def _judge_identity_key(judge: Dict[str, Any]) -> str:
    return str(campaign.stable_model_identity(judge).get("identity_key") or "")


def _qualification_protocol_version(judge: Dict[str, Any]) -> Optional[str]:
    qualification = judge.get("qualification") if isinstance(judge.get("qualification"), dict) else {}
    return qualification.get("protocol_version")


def _compatibility_fingerprint(judge: Dict[str, Any], judge_mode_configuration: Dict[str, Any]) -> str:
    qualification = judge.get("qualification") if isinstance(judge.get("qualification"), dict) else {}
    payload = {
        "judge_identity": campaign.stable_model_identity(judge),
        "runtime_identity": judge.get("runtime_identity") or qualification.get("runtime_identity") or {},
        "qualification_protocol_version": qualification.get("protocol_version"),
        "qualification_disposition": qualification.get("aggregate_disposition"),
        "judge_mode_configuration": judge_mode_configuration,
        "request_contract_version": judge_mod.JUDGE_REQUEST_CONTRACT_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def _execution_fingerprint(qualified_judges: List[Dict[str, Any]], judge_mode_configuration: Dict[str, Any]) -> str:
    payload = {
        "judge_compatibility_fingerprints": [
            _compatibility_fingerprint(judge, judge_mode_configuration)
            for judge in qualified_judges
        ],
        "judge_mode_configuration": judge_mode_configuration,
        "request_contract_version": judge_mod.JUDGE_REQUEST_CONTRACT_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def _matching_exhausted_execution_entry(entry: Dict[str, Any], execution_fingerprint: str) -> bool:
    return (
        entry.get("status") == "judge_exhausted_unavailable"
        and entry.get("judge_execution_fingerprint") == execution_fingerprint
    )


def _prior_structural_failure(
    previous: List[Dict[str, Any]],
    compatibility_fingerprint: str,
) -> Optional[Dict[str, Any]]:
    for entry in reversed(previous):
        attempts = entry.get("judgement_attempts") if isinstance(entry.get("judgement_attempts"), list) else []
        for attempt in reversed(attempts):
            if (
                attempt.get("status") == "rejected_structural_incompatibility"
                and attempt.get("compatibility_fingerprint") == compatibility_fingerprint
            ):
                reused = dict(attempt)
                reused["status"] = "reused_structural_incompatibility"
                reused["reused_from_source_row_hash"] = entry.get("source_row_hash")
                return reused
    return None


def scan_run(
    run_dir: Path,
    *,
    judge_model: str,
    qualified_judges: Optional[List[Dict[str, Any]]] = None,
    judge_mode: str,
    num_ctx: Optional[int] = None,
    think: str = "auto",
    force: bool = False,
    client: Any = None,
) -> Dict[str, Any]:
    raw_rows = _jsonl(run_dir / "raw_results.jsonl")
    existing = _jsonl(run_dir / "judge_results.jsonl")
    qualified_judges = (
        qualified_judges
        if qualified_judges is not None
        else _resolve_manual_judge_pool(
            client, judge_model, ledger=campaign._run_capability_ledger(run_dir)
        )
    )
    pool_signature = campaign.judge_pool_signature(qualified_judges)
    judge_mode_configuration = _judge_config(judge_mode, num_ctx, think)
    execution_fingerprint = _execution_fingerprint(qualified_judges, judge_mode_configuration)
    existing_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for entry in existing:
        source = str(entry.get("source_row_hash") or "")
        if source:
            existing_by_source.setdefault(source, []).append(entry)
    eligible: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for row_index, row in enumerate(raw_rows):
        task = _TASKS.get(str(row.get("task") or ""))
        if task is None or task.scorer != "subjective":
            continue
        row_hash = source_row_hash(row)
        resolution = campaign.resolve_independent_judge_for_row(row, qualified_judges)
        previous = existing_by_source.get(row_hash, [])
        if not force:
            if resolution.get("status") == "selected_independent_judge":
                if any(_matching_judged_entry(entry, resolution, judge_mode) for entry in previous):
                    skipped.append({"row_index": row_index, "model": row.get("model"), "task": row.get("task"), "reason": "already_judged_by_resolved_independent_judge"})
                    continue
                if any(_matching_judge_error_entry(entry, resolution, judge_mode_configuration) for entry in previous):
                    skipped.append({"row_index": row_index, "model": row.get("model"), "task": row.get("task"), "reason": "already_judge_error_for_resolved_judge"})
                    continue
                if any(_matching_exhausted_execution_entry(entry, execution_fingerprint) for entry in previous):
                    skipped.append({"row_index": row_index, "model": row.get("model"), "task": row.get("task"), "reason": "already_judge_exhausted_unavailable"})
                    continue
            elif any(_matching_pending_entry(entry, pool_signature, judge_mode) for entry in previous):
                skipped.append({"row_index": row_index, "model": row.get("model"), "task": row.get("task"), "reason": "already_awaiting_independent_judge"})
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
        eligible.append({"row_index": row_index, "row": row, "row_hash": row_hash, "task": task, "outputs": outputs, "judge_resolution": resolution, "prior_sidecars": previous})
    return {
        "run_dir": str(run_dir),
        "raw_rows": len(raw_rows),
        "eligible": eligible,
        "skipped": skipped,
        "already_recorded": len(existing),
        "judge_pool_signature": pool_signature,
        "judge_execution_fingerprint": execution_fingerprint,
    }


def judge_run(
    client: Any,
    run_dir: Path,
    *,
    judge_model: str,
    qualified_judges: Optional[List[Dict[str, Any]]] = None,
    judge_mode: str = "single",
    num_ctx: Optional[int] = None,
    think: str = "auto",
    dry_run: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    if judge_mode not in {"single", "panel"}:
        raise ValueError("judge mode must be single or panel")
    qualified_judges = (
        qualified_judges
        if qualified_judges is not None
        else _resolve_manual_judge_pool(
            client, judge_model, ledger=campaign._run_capability_ledger(run_dir)
        )
    )
    scan = scan_run(
        run_dir,
        judge_model=judge_model,
        qualified_judges=qualified_judges,
        judge_mode=judge_mode,
        num_ctx=num_ctx,
        think=think,
        force=force,
        client=client,
    )
    eligible = scan.pop("eligible")
    pool_signature = str(scan.get("judge_pool_signature") or "")
    result = {
        **scan,
        "judge_model": judge_model,
        "judge_mode": judge_mode,
        "eligible": len(eligible),
        "attempted": 0,
        "judged": 0,
        "pending": 0,
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
    pool_evidence = _pool_evidence(qualified_judges)
    judge_mode_configuration = _judge_config(judge_mode, num_ctx, think)
    execution_fingerprint = _execution_fingerprint(qualified_judges, judge_mode_configuration)
    with sidecar.open("a") as handle:
        for item in eligible:
            row = item["row"]
            task: Task = item["task"]
            resolution = item["judge_resolution"]
            structural_failures: List[Dict[str, Any]] = []
            exhausted = False
            while True:
                selected_judge_model = str(resolution.get("judge_model") or "")
                selected_judge_digest = resolution.get("judge_digest")
                if selected_judge_model:
                    break
                exhausted = bool(structural_failures)
                entry = {
                    "schema_version": 1,
                    "applied_at": datetime.now(timezone.utc).isoformat(),
                    "run_id": run_dir.name,
                    "source_row_index": item["row_index"],
                    "source_row_hash": item["row_hash"],
                    "source_model": row.get("model"),
                    "source_model_digest": row.get("model_digest_resolved") or row.get("model_digest"),
                    "source_identity": _source_identity(row),
                    "task": task.id,
                    "task_hash": row.get("task_hash") or _task_hash(task),
                    "judge_model": None,
                    "judge_model_digest": None,
                    "judge_identity": None,
                    "judge_mode": judge_mode,
                    "status": "judge_exhausted_unavailable" if exhausted else "awaiting_independent_judge",
                    "score": None,
                    "reason": (
                        "judge exhausted/unavailable: qualified independent candidates failed structurally"
                        if exhausted else
                        "awaiting independent judge: no qualified non-self judge available"
                    ),
                    "elapsed_seconds": 0.0,
                    "samples": [],
                    "judge_resolution": resolution,
                    "judgement_attempts": structural_failures,
                    "failure_disposition": "structural_incompatibility" if exhausted else "awaiting_independent_judge",
                    "judge_pool_signature": pool_signature,
                    "judge_execution_fingerprint": execution_fingerprint,
                    "qualified_judge_pool": pool_evidence,
                    "judge_mode_configuration": judge_mode_configuration,
                    "request_contract_version": judge_mod.JUDGE_REQUEST_CONTRACT_VERSION,
                    "judge_anchor_policy_hash": judge_mod.JUDGE_ANCHOR_POLICY_HASH,
                    "judge_policy_version": campaign.JUDGE_POLICY_VERSION,
                    "model_role_policy_version": campaign.MODEL_ROLE_POLICY_VERSION,
                }
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
                handle.flush()
                if exhausted:
                    result["judge_errors"] += 1
                else:
                    result["pending"] += 1
                result["written"] += 1
                result["entries"].append(entry)
                break
            if not selected_judge_model:
                continue

            while selected_judge_model:
                selected_judge = resolution.get("judge") or {}
                compatibility_fingerprint = _compatibility_fingerprint(selected_judge, judge_mode_configuration)
                reused_failure = _prior_structural_failure(item.get("prior_sidecars") or [], compatibility_fingerprint)
                if reused_failure is not None:
                    structural_failures.append(reused_failure)
                    failed_keys = {
                        str((failure.get("judge_identity") or {}).get("identity_key") or "")
                        for failure in structural_failures
                    }
                    remaining_judges = [
                        judge for judge in qualified_judges
                        if _judge_identity_key(judge) not in failed_keys
                    ]
                    resolution = campaign.resolve_independent_judge_for_row(row, remaining_judges)
                    selected_judge_model = str(resolution.get("judge_model") or "")
                    selected_judge_digest = resolution.get("judge_digest")
                    if selected_judge_model:
                        continue
                    break
                result["attempted"] += 1
                sample_results = []
                valid_scores: List[float] = []
                failure_dispositions: List[str] = []
                started = time.perf_counter()
                structural_failure: Optional[Dict[str, Any]] = None
                for path, output in item["outputs"]:
                    try:
                        if judge_mode == "panel":
                            judge_result = judge_mod.judge_panel_result(
                                client, selected_judge_model, task.prompt, output, task.rubric,
                                num_ctx=num_ctx, think=think,
                            )
                        else:
                            judge_result = judge_mod.judge_single_result(
                                client, selected_judge_model, task.prompt, output, task.rubric,
                                num_ctx=num_ctx, think=think,
                            )
                        score = judge_result.get("score")
                        reason = str(judge_result.get("reason") or "")
                    except Exception as exc:
                        score, reason = None, f"judge exception: {exc!r}"
                        judge_result = {
                            "score": None,
                            "reason": reason,
                            "failure_disposition": "backend_failure",
                            "backend": {"ok": False, "error": repr(exc), "failure_disposition": "backend_failure"},
                            "request_contract_version": judge_mod.JUDGE_REQUEST_CONTRACT_VERSION,
                        }
                    failure_disposition = judge_result.get("failure_disposition")
                    sample = {
                        "subjective_path": path,
                        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                        "score": score,
                        "reason": reason,
                        "failure_disposition": failure_disposition,
                        "backend": judge_result.get("backend"),
                        "request_contract_version": judge_result.get("request_contract_version"),
                    }
                    sample_results.append(sample)
                    if isinstance(score, (int, float)):
                        valid_scores.append(float(score))
                    elif failure_disposition:
                        failure_dispositions.append(str(failure_disposition))
                    if failure_disposition == "structural_incompatibility":
                        structural_failure = {
                            "status": "rejected_structural_incompatibility",
                            "judge_model": selected_judge_model,
                            "judge_model_digest": selected_judge_digest,
                            "judge_identity": campaign.stable_model_identity(selected_judge),
                            "identity_relation": campaign.stable_model_identity_relation(row, selected_judge),
                            "reason": reason,
                            "sample": sample,
                            "compatibility_fingerprint": compatibility_fingerprint,
                            "request_contract_version": judge_mod.JUDGE_REQUEST_CONTRACT_VERSION,
                            "elapsed_seconds": round(time.perf_counter() - started, 3),
                        }
                        break
                if structural_failure is not None:
                    structural_failures.append(structural_failure)
                    failed_keys = {
                        str((failure.get("judge_identity") or {}).get("identity_key") or "")
                        for failure in structural_failures
                    }
                    remaining_judges = [
                        judge for judge in qualified_judges
                        if _judge_identity_key(judge) not in failed_keys
                    ]
                    resolution = campaign.resolve_independent_judge_for_row(row, remaining_judges)
                    selected_judge_model = str(resolution.get("judge_model") or "")
                    selected_judge_digest = resolution.get("judge_digest")
                    if selected_judge_model:
                        continue
                    break
                final_score = round(float(statistics.median(valid_scores)), 2) if valid_scores else None
                status = "judged" if final_score is not None else "judge_error"
                final_failure_disposition = None
                if status != "judged":
                    final_failure_disposition = (
                        failure_dispositions[0]
                        if failure_dispositions and all(item == failure_dispositions[0] for item in failure_dispositions)
                        else "backend_failure"
                    )
                break
            if not selected_judge_model:
                if structural_failures:
                    entry = {
                        "schema_version": 1,
                        "applied_at": datetime.now(timezone.utc).isoformat(),
                        "run_id": run_dir.name,
                        "source_row_index": item["row_index"],
                        "source_row_hash": item["row_hash"],
                        "source_model": row.get("model"),
                        "source_model_digest": row.get("model_digest_resolved") or row.get("model_digest"),
                        "source_identity": _source_identity(row),
                        "task": task.id,
                        "task_hash": row.get("task_hash") or _task_hash(task),
                        "judge_model": None,
                        "judge_model_digest": None,
                        "judge_identity": None,
                        "judge_mode": judge_mode,
                        "status": "judge_exhausted_unavailable",
                        "score": None,
                        "reason": "judge exhausted/unavailable: qualified independent candidates failed structurally",
                        "elapsed_seconds": 0.0,
                        "samples": [],
                        "judge_resolution": resolution,
                        "judgement_attempts": structural_failures,
                        "failure_disposition": "structural_incompatibility",
                        "judge_pool_signature": pool_signature,
                        "judge_execution_fingerprint": execution_fingerprint,
                        "qualified_judge_pool": pool_evidence,
                        "judge_mode_configuration": judge_mode_configuration,
                        "request_contract_version": judge_mod.JUDGE_REQUEST_CONTRACT_VERSION,
                        "judge_anchor_policy_hash": judge_mod.JUDGE_ANCHOR_POLICY_HASH,
                        "judge_policy_version": campaign.JUDGE_POLICY_VERSION,
                        "model_role_policy_version": campaign.MODEL_ROLE_POLICY_VERSION,
                    }
                    handle.write(json.dumps(entry, sort_keys=True) + "\n")
                    handle.flush()
                    result["judge_errors"] += 1
                    result["written"] += 1
                    result["entries"].append(entry)
                continue
            if structural_failure is not None:
                continue
            entry = {
                "schema_version": 1,
                "applied_at": datetime.now(timezone.utc).isoformat(),
                "run_id": run_dir.name,
                "source_row_index": item["row_index"],
                "source_row_hash": item["row_hash"],
                "source_model": row.get("model"),
                "source_model_digest": row.get("model_digest_resolved") or row.get("model_digest"),
                "source_identity": _source_identity(row),
                "task": task.id,
                "task_hash": row.get("task_hash") or _task_hash(task),
                "judge_model": selected_judge_model,
                "judge_model_digest": selected_judge_digest,
                "judge_identity": campaign.stable_model_identity(resolution.get("judge") or {}),
                "identity_relation": campaign.stable_model_identity_relation(row, resolution.get("judge") or {}),
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
                "judge_resolution": resolution,
                "judgement_attempts": [
                    *structural_failures,
                    {
                        "status": status,
                        "judge_model": selected_judge_model,
                        "judge_model_digest": selected_judge_digest,
                        "judge_identity": campaign.stable_model_identity(resolution.get("judge") or {}),
                        "identity_relation": campaign.stable_model_identity_relation(row, resolution.get("judge") or {}),
                        "sample_count": len(sample_results),
                        "valid_score_count": len(valid_scores),
                        "failure_disposition": final_failure_disposition,
                        "compatibility_fingerprint": _compatibility_fingerprint(resolution.get("judge") or {}, judge_mode_configuration),
                    },
                ],
                "failure_disposition": final_failure_disposition,
                "judge_pool_signature": pool_signature,
                "judge_execution_fingerprint": execution_fingerprint,
                "qualified_judge_pool": pool_evidence,
                "judge_mode_configuration": judge_mode_configuration,
                "request_contract_version": judge_mod.JUDGE_REQUEST_CONTRACT_VERSION,
                "judge_anchor_policy_hash": judge_mod.JUDGE_ANCHOR_POLICY_HASH,
                "judge_policy_version": campaign.JUDGE_POLICY_VERSION,
                "model_role_policy_version": campaign.MODEL_ROLE_POLICY_VERSION,
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
    qualified_judges: Optional[List[Dict[str, Any]]] = None,
    judge_mode: str = "single",
    num_ctx: Optional[int] = None,
    think: str = "auto",
    dry_run: bool = False,
    force: bool = False,
    progress: Optional[Callable[[int, int, Path, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    runs = discover_runs(runs_dir)
    # Resolve the manual judge pool once, up front, rather than once per run
    # directory: interrogation + qualification are real live-client calls,
    # and re-running them per run_dir would repeat the same expensive work
    # for the same judge_model (same discipline as Stage 1.3's GPU-inventory
    # reuse -- resolve once, thread the result through).
    if qualified_judges is None:
        qualified_judges = _resolve_manual_judge_pool(
            client, judge_model, ledger=_capability_ledger_for_runs_dir(runs_dir)
        )
    results = []
    for index, run_dir in enumerate(runs, start=1):
        result = judge_run(
            client, run_dir, judge_model=judge_model, qualified_judges=qualified_judges, judge_mode=judge_mode,
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


def latest_judge_sidecars(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for entry in _jsonl(run_dir / "judge_results.jsonl"):
        row_hash = entry.get("source_row_hash")
        if not row_hash or entry.get("status") not in {"judged", "judge_error", "awaiting_independent_judge", "judge_exhausted_unavailable"}:
            continue
        previous = latest.get(row_hash)
        if previous is None or str(entry.get("applied_at") or "") >= str(previous.get("applied_at") or ""):
            latest[row_hash] = entry
    return latest


def apply_judgements(run_dir: Path, rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    overlays = latest_judge_sidecars(run_dir)
    out: List[Dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        overlay = overlays.get(source_row_hash(raw))
        if overlay and overlay.get("status") == "judged":
            row["score"] = overlay.get("score")
            row["reason"] = overlay.get("reason")
            row["judge_mode"] = overlay.get("judge_mode")
            row["judge_model"] = overlay.get("judge_model")
            row["judge_model_digest"] = overlay.get("judge_model_digest")
            row["posthoc_judged"] = True
            row["judge_applied_at"] = overlay.get("applied_at")
            row["judge_source_row_hash"] = overlay.get("source_row_hash")
            row["judge_elapsed_seconds"] = overlay.get("elapsed_seconds")
        elif overlay and overlay.get("status") == "awaiting_independent_judge":
            row["score"] = None
            row["reason"] = overlay.get("reason")
            row["judge_mode"] = overlay.get("judge_mode")
            row["judge_model"] = None
            row["judge_model_digest"] = None
            row["posthoc_judged"] = False
            row["judge_pending_status"] = "awaiting_independent_judge"
            row["disposition"] = "awaiting_independent_judge"
            row["judge_applied_at"] = overlay.get("applied_at")
            row["judge_source_row_hash"] = overlay.get("source_row_hash")
            row["judge_elapsed_seconds"] = overlay.get("elapsed_seconds")
        elif overlay and overlay.get("status") == "judge_error":
            row["score"] = None
            row["reason"] = overlay.get("reason")
            row["judge_mode"] = overlay.get("judge_mode")
            row["judge_model"] = overlay.get("judge_model")
            row["judge_model_digest"] = overlay.get("judge_model_digest")
            row["posthoc_judged"] = False
            row["judge_pending_status"] = overlay.get("failure_disposition") or "judge_error"
            row["disposition"] = overlay.get("failure_disposition") or "judge_output_failure"
            row["judge_applied_at"] = overlay.get("applied_at")
            row["judge_source_row_hash"] = overlay.get("source_row_hash")
            row["judge_elapsed_seconds"] = overlay.get("elapsed_seconds")
        elif overlay and overlay.get("status") == "judge_exhausted_unavailable":
            row["score"] = None
            row["reason"] = overlay.get("reason")
            row["judge_mode"] = overlay.get("judge_mode")
            row["judge_model"] = None
            row["judge_model_digest"] = None
            row["posthoc_judged"] = False
            row["judge_pending_status"] = "judge_exhausted_unavailable"
            row["disposition"] = "judge_exhausted_unavailable"
            row["judge_applied_at"] = overlay.get("applied_at")
            row["judge_source_row_hash"] = overlay.get("source_row_hash")
            row["judge_elapsed_seconds"] = overlay.get("elapsed_seconds")
        out.append(row)
    return out
