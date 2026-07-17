"""Post-run evidence repair planning and targeted recovery.

``llmb repair`` scans completed or interrupted run artifacts, classifies every
unresolved current result, and applies only bounded, cause-specific recovery.
It never overwrites source ``raw_results.jsonl``.  Generation retries are
written to new child run directories; post-hoc judgements remain sidecars.

The first repair policy intentionally stays conservative:

* thinking-only / empty visible output -> one think-off retry with a bounded
  output budget;
* transient HTTP 5xx / timeout -> one unload/reload retry;
* failed vision/tool/FIM lanes -> one functional capability gate before any
  lane retry;
* subjective raw output -> post-hoc judge, no source-model rerun;
* incomplete needle coverage -> guarded retry only when the requested
  VRAM+spill policy permits it;
* stale or genuinely absent current tasks -> exact-task rerun.

Ollama KV-cache quantisation is global to the server process. This module can
require and record a requested ``OLLAMA_KV_CACHE_TYPE``. The CLI may also use
the separate, explicit human-supervised service controller; no service change
occurs unless the operator selected that mode and confirms each phase.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .capabilities import interrogate_model
from .classify import families_for
from .hardware import detect_gpu
from .judge_dumps import apply_judgements, judge_run
from .rankings import _CURRENT_HASHES, rank_for_output
from .tasks import TASKS

POLICY_VERSION = "2"
_TASKS = {task.id: task for task in TASKS}
_TRANSIENT_RE = re.compile(
    r"(?:http(?:error)?\s*5\d\d|internal server error|timed?\s*out|timeout|"
    r"connection reset|connection refused|temporar(?:y|ily) unavailable|eof)",
    re.I,
)
_HTTP_400_RE = re.compile(r"(?:http(?:error)?\s*400|bad request)", re.I)


@dataclass
class RepairAction:
    action_id: str
    kind: str
    model: str
    model_digest: str
    source_run_id: str
    tasks: List[str]
    reason: str
    automatic: bool
    family: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = None
    source_row_hashes: Optional[Dict[str, str]] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class RepairPlan:
    schema_version: int
    repair_policy_version: str
    plan_id: str
    created_at: str
    runs_dir: str
    selected_runs: List[str]
    actions: List[RepairAction]
    observations: List[Dict[str, Any]]
    counts: Dict[str, int]
    options: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["actions"] = [asdict(action) for action in self.actions]
        return out


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


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


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _known_unavailable_families(run_dirs: Sequence[Path]) -> Dict[str, set[str]]:
    """Collect terminal capability exclusions already proven for this source set.

    ``--force`` repeats unresolved recovery attempts, but it must not erase a
    terminal build-capability finding. Re-probing a confirmed unavailable lane
    requires an explicit future invalidation/reprobe workflow, not the generic
    force flag.
    """
    result: Dict[str, set[str]] = {}
    for run_dir in run_dirs:
        data = _read_json(run_dir / "capability_repair.json") or {}
        for model, entry in data.items():
            families = set((entry or {}).get("unavailable_families") or {})
            if families:
                result.setdefault(str(model), set()).update(str(f) for f in families)
    return result


def _latest_repair_records(run_dirs: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    """Return the latest source-side repair record per deterministic action ID."""
    latest: Dict[str, Dict[str, Any]] = {}
    for run_dir in run_dirs:
        for record in _read_jsonl(run_dir / "repair_results.jsonl"):
            action_id = str(record.get("action_id") or "")
            if not action_id:
                continue
            previous = latest.get(action_id)
            if previous is None or str(record.get("recorded_at") or "") >= str(previous.get("recorded_at") or ""):
                latest[action_id] = record
    return latest


def _record_is_terminal_capability_failure(record: Dict[str, Any], runs_dir: Path) -> bool:
    action = record.get("action") or {}
    if str(action.get("kind") or record.get("kind") or "") != "capability_gate":
        return False
    if str(record.get("status") or "") == "measured_failure":
        return True
    if str(record.get("status") or "") != "unresolved":
        return False
    gate_state = str(((record.get("gate") or {}).get("probe_state") or ""))
    if gate_state not in {"confirmed_supported", "responded_contract_failed"}:
        return False
    if str(action.get("family") or "") != "insert":
        return False
    required = set(str(task) for task in (action.get("tasks") or []))
    if not required:
        return False
    resolved: set[str] = set()
    for attempt in record.get("attempts") or []:
        resolved.update(str(task) for task in (attempt.get("terminal_failure_tasks") or []))
        child = str(attempt.get("child_run_id") or "")
        if not child:
            continue
        for row in _read_jsonl(Path(runs_dir) / child / "raw_results.jsonl"):
            task_id = str(row.get("task") or "")
            task = _TASKS.get(task_id)
            if (task_id in required and task and task.family == "insert"
                    and _is_numeric(row.get("score"))
                    and float(row.get("score") or 0.0) <= 0.0
                    and str(row.get("error_kind") or "") == "empty_output"):
                resolved.add(task_id)
    return required.issubset(resolved)


def _source_num_predict(row: Dict[str, Any]) -> int:
    for value in (row.get("num_predict"), row.get("num_predict_used"),
                  (row.get("run_configuration") or {}).get("num_predict")):
        if isinstance(value, (int, float)) and int(value) > 0:
            return int(value)
    return 2048


def _thinking_retry_profiles(row: Dict[str, Any], recovery_budget: int) -> List[Dict[str, Any]]:
    """Canonical bounded recovery: think-off at the original budget, then 4096/default requested."""
    original = _source_num_predict(row)
    profiles = [
        {"think": "off", "num_predict": original},
        {"think": "off", "num_predict": max(original, int(recovery_budget))},
    ]
    unique: List[Dict[str, Any]] = []
    for profile in profiles:
        if profile not in unique:
            unique.append(profile)
    return unique


def _kv_scalar(label: Optional[str]) -> Optional[float]:
    value = str(label or "").lower()
    if value.startswith("q4"):
        return 0.5
    if value.startswith("q8"):
        return 1.0
    if value in {"f16", "fp16", "bf16"}:
        return 2.0
    if value in {"f32", "fp32"}:
        return 4.0
    return None


def inspect_ollama_kv_environment() -> Dict[str, Any]:
    """Best-effort inspection of the *running* Ollama server KV environment.

    A shell export does not reconfigure an already-running Ollama service.  We
    therefore distinguish four evidence sources:

    * ``/proc/<pid>/environ`` for a running ``ollama serve`` process (strongest);
    * ``systemctl show ... Environment`` for configured unit environment;
    * the current repair-process environment (weak, client-side only);
    * unavailable.

    This never claims that VRAM-slope arithmetic proves an effective KV type.
    Activation/workspace buffers can also scale with context, so measured VRAM
    divergence is reported separately as an estimator limitation.
    """
    result: Dict[str, Any] = {
        "requested_by_shell": str(os.environ.get("OLLAMA_KV_CACHE_TYPE") or "").strip().lower() or None,
        "running_processes": [],
        "systemd_unit": None,
        "systemd_kv_type": None,
        "effective_kv_type": None,
        "effective_source": None,
        "verified": False,
        "notes": [],
    }

    if shutil.which("pgrep"):
        try:
            proc = subprocess.run(["pgrep", "-x", "ollama"], capture_output=True, text=True, timeout=3)
            pids = [part for part in proc.stdout.split() if part.isdigit()] if proc.returncode == 0 else []
        except Exception as exc:
            pids = []
            result["notes"].append(f"pgrep failed: {exc!r}")
        for pid in pids:
            item: Dict[str, Any] = {"pid": int(pid), "command": None, "kv_type": None, "readable": False}
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
                item["command"] = cmd
                if "serve" not in cmd:
                    result["running_processes"].append(item)
                    continue
                env_bytes = Path(f"/proc/{pid}/environ").read_bytes()
                env = {}
                for chunk in env_bytes.split(b"\0"):
                    if b"=" in chunk:
                        key, value = chunk.split(b"=", 1)
                        env[key.decode(errors="replace")] = value.decode(errors="replace")
                item["readable"] = True
                item["kv_type"] = str(env.get("OLLAMA_KV_CACHE_TYPE") or "").strip().lower() or None
                if item["kv_type"]:
                    result["effective_kv_type"] = item["kv_type"]
                    result["effective_source"] = f"/proc/{pid}/environ"
                    result["verified"] = True
            except PermissionError:
                result["notes"].append(f"permission denied reading /proc/{pid}/environ")
            except Exception as exc:
                result["notes"].append(f"could not inspect ollama pid {pid}: {exc!r}")
            result["running_processes"].append(item)

    active_unit: Optional[str] = None
    try:
        from .ollama_service import discover_active_service
        active = discover_active_service(run=subprocess.run, use_sudo=False)
        active_unit = active.unit
    except Exception as exc:
        result["notes"].append(
            f"active Ollama service unit could not be determined without privileged "
            f"access; systemd inspection skipped ({exc})"
        )

    if active_unit and shutil.which("systemctl"):
        try:
            proc = subprocess.run(
                ["systemctl", "show", active_unit, "--property=Environment", "--value"],
                capture_output=True, text=True, timeout=4,
            )
            if proc.returncode == 0:
                text = proc.stdout.strip()
                # Never persist the full service Environment property: it can
                # contain unrelated credentials. Extract only the one setting
                # needed by repair planning.
                match = re.search(r"(?:^|\s)OLLAMA_KV_CACHE_TYPE=([^\s]+)", text)
                result["systemd_unit"] = active_unit
                result["systemd_kv_type"] = (match.group(1).strip('\"\'').lower() if match else None)
                if result["systemd_kv_type"] and not result["effective_kv_type"]:
                    result["effective_kv_type"] = result["systemd_kv_type"]
                    result["effective_source"] = (
                        f"systemctl:{active_unit} (configured; process restart not proven)"
                    )
                elif not result["systemd_kv_type"] and not result["effective_kv_type"]:
                    # This is a genuine, checked finding, not a failure to
                    # inspect: the real active unit was queried successfully
                    # and confirmed to have no explicit KV override at all.
                    # Reporting this the same way as "couldn't check" would
                    # hide a real, useful answer behind a generic unknown.
                    result["effective_source"] = (
                        f"systemctl:{active_unit} (checked: no OLLAMA_KV_CACHE_TYPE override configured)"
                    )
            else:
                result["notes"].append(
                    f"systemctl show {active_unit} returned exit code {proc.returncode}; "
                    "active unit was identified but its environment could not be queried"
                )
        except Exception as exc:
            result["notes"].append(f"systemctl inspection failed: {exc!r}")
    elif not active_unit:
        result["notes"].append(
            "skipping systemd inspection: no active Ollama service unit could be "
            "safely identified without a privileged discovery pass"
        )

    if not result["effective_kv_type"] and result["requested_by_shell"]:
        result["effective_kv_type"] = result["requested_by_shell"]
        result["effective_source"] = "repair-process environment only"
        result["notes"].append("client shell value does not prove the running Ollama server was restarted")
    if not result["effective_kv_type"] and not active_unit:
        result["effective_source"] = "not inspected"
    return result


def _needle_measurement_analysis(row: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    """Describe estimator-vs-measurement divergence without inferring KV type.

    Measured VRAM slope includes more than pure KV cache: activations, compute
    workspaces, allocator behaviour and offload decisions may all vary with
    context.  The ratio is useful evidence that the simple estimator is weak,
    but it is not proof that Ollama ignored q4/q8 configuration.
    """
    source = str(item.get("kv_estimate_source") or "")
    assumed = source.split("kv=", 1)[1].split(";", 1)[0] if "kv=" in source else None
    measured_bpt = item.get("kv_bytes_per_token")
    metadata_values = []
    for attempted in row.get("needle_attempted") or []:
        attempted_source = str(attempted.get("kv_estimate_source") or "")
        if "measured" not in attempted_source and isinstance(attempted.get("kv_bytes_per_token"), (int, float)):
            metadata_values.append(float(attempted["kv_bytes_per_token"]))
    metadata_bpt = metadata_values[-1] if metadata_values else None
    ratio = None
    if isinstance(measured_bpt, (int, float)) and metadata_bpt and metadata_bpt > 0:
        ratio = float(measured_bpt) / metadata_bpt
    divergence = bool(ratio is not None and (ratio >= 1.5 or ratio <= 0.67))
    return {
        "estimator_assumed_kv_type": assumed,
        "metadata_bytes_per_token": metadata_bpt,
        "measured_slope_bytes_per_token": measured_bpt,
        "measured_to_metadata_ratio": round(ratio, 3) if ratio is not None else None,
        "estimator_divergence": divergence,
        "estimator_divergence_note": (
            "measured VRAM slope differs materially from pure KV math; this can include activation/workspace/offload overhead and does not prove the server KV type"
            if divergence else None
        ),
    }


def _append_jsonl(path: Path, entry: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()


def _row_hash(row: Dict[str, Any]) -> str:
    stable = {
        key: value for key, value in row.items()
        if key not in {"import_tag", "_source_signature", "_source_row_index", "run_configuration"}
    }
    return hashlib.sha256(json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()


def discover_runs(
    runs_dir: Path,
    *,
    run_id: Optional[str] = None,
    run_prefix: Optional[str] = None,
    everything: bool = False,
) -> List[Path]:
    if not runs_dir.exists():
        return []
    candidates = sorted(
        path for path in runs_dir.iterdir()
        if path.is_dir() and (path / "raw_results.jsonl").is_file()
    )
    if run_id:
        return [path for path in candidates if path.name == run_id]
    if run_prefix:
        return [path for path in candidates if path.name.startswith(run_prefix)]
    if everything:
        return candidates
    raise ValueError("choose --run-id, --run-prefix, or --everything")


def _run_config(run_dir: Path) -> Dict[str, Any]:
    filters = _read_json(run_dir / "filters.json") or {}
    config = _read_json(run_dir / "config.json") or {}
    return {
        "level": filters.get("level") or config.get("level"),
        "think": filters.get("think") if "think" in filters else config.get("think"),
        "num_predict": filters.get("num_predict") or config.get("num_predict_override"),
        "ctx_override": filters.get("ctx_override") or config.get("ctx_override"),
        "judge_mode": filters.get("judge_mode"),
        "auto_probe": filters.get("auto_probe"),
        "run_complete": bool((_read_json(run_dir / "status.json") or {}).get("finished_at")),
    }


def _load_selected_rows(run_dirs: Sequence[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    model_context: Dict[str, Dict[str, Any]] = {}
    for run_dir in run_dirs:
        identities = _read_json(run_dir / "model_identities.json") or {}
        profiles = _read_json(run_dir / "capability_report.json") or {}
        run_cfg = _run_config(run_dir)
        raw_rows = apply_judgements(run_dir, _read_jsonl(run_dir / "raw_results.jsonl"))
        for index, raw in enumerate(raw_rows):
            row = dict(raw)
            model = str(row.get("model") or "")
            identity = identities.get(model) or {}
            digest = str(identity.get("digest") or row.get("model_digest") or model)
            row["run_id"] = run_dir.name
            row["model_digest_resolved"] = digest
            row["level"] = str(row.get("level") or run_cfg.get("level") or "unknown")
            row["run_configuration"] = run_cfg
            row["_source_row_index"] = index
            row["_source_run_dir"] = str(run_dir)
            row["_source_row_hash"] = _row_hash(raw)
            if model in profiles:
                profile = profiles[model]
                row["capability_profile"] = profile
                row.setdefault("capability_families", profile.get("supported_families"))
                row.setdefault("capabilities_declared", profile.get("declared_capabilities"))
            rows.append(row)
            ctx = model_context.setdefault(digest, {
                "model": model,
                "digest": digest,
                "profiles": [],
                "runs": set(),
                "levels": set(),
            })
            ctx["model"] = model or ctx["model"]
            ctx["runs"].add(run_dir.name)
            ctx["levels"].add(row["level"])
            if model in profiles:
                ctx["profiles"].append(profiles[model])
    return rows, model_context


def _best_profile(context: Dict[str, Any]) -> Dict[str, Any]:
    profiles = list(context.get("profiles") or [])
    if not profiles:
        return {"model": context.get("model"), "supported_families": ["text"], "declared_capabilities": []}
    profiles.sort(key=lambda p: (bool(p.get("functional_probes_enabled")), len(p.get("supported_families") or [])), reverse=True)
    return copy.deepcopy(profiles[0])


def _action_id(
    kind: str,
    digest: str,
    tasks: Iterable[str],
    source_run_id: str,
    *,
    variant: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "kind": kind,
        "digest": digest,
        "source_run_id": source_run_id,
        "tasks": sorted(set(tasks)),
        "variant": variant or {},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _error_text(row: Dict[str, Any]) -> str:
    return " ".join(str(row.get(key) or "") for key in ("reason", "error", "error_detail", "http_error_body"))


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _needle_observation(
    row: Dict[str, Any],
    *,
    gpu_total_gb: float,
    emergency_headroom_gb: float,
    max_spill_gb: float,
    requested_kv_type: str,
    kv_server_confirmed: bool,
    kv_server_inspection: Dict[str, Any],
    marginal_overage_gb: float = 0.25,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if str(row.get("task")) != "needle" or float(row.get("needle_coverage") or 0.0) >= 1.0:
        return None, None
    skipped = list(row.get("needle_skipped") or [])
    if not skipped:
        return None, {
            "kind": "needle_incomplete_unclassified", "model": row.get("model"),
            "run_id": row.get("run_id"), "task": "needle",
            "reason": "needle coverage is incomplete but no structured skipped-depth evidence is present",
        }

    requested_explicit = requested_kv_type in {"q8_0", "q4_0"}
    observed_kv = str(kv_server_inspection.get("effective_kv_type") or "").lower() or None
    observed_verified = bool(kv_server_inspection.get("verified"))
    if requested_explicit and observed_verified and observed_kv != requested_kv_type:
        return {
            "kind": "kv_reconfigure_required",
            "tasks": ["needle"],
            "automatic": False,
            "details": {
                "classification": "KV_SERVER_MISMATCH",
                "requested_kv_type": requested_kv_type,
                "observed_kv_type": observed_kv,
                "server_inspection": kv_server_inspection,
                "recommended_sequence": ["q8_0", "q4_0"],
                "note": "the running Ollama process reports a different KV type; restart/reconfigure it before retrying",
            },
        }, None

    actionable: List[Dict[str, Any]] = []
    terminal: List[Dict[str, Any]] = []
    allowed_gpu_gb = max(0.0, gpu_total_gb - emergency_headroom_gb) if gpu_total_gb > 0 else 0.0
    allowed_total = allowed_gpu_gb + max(0.0, max_spill_gb) if allowed_gpu_gb else 0.0

    for item in skipped:
        reason = str(item.get("reason") or "")
        skip_class = str(item.get("skip_class") or "")
        if skip_class in {"operator", "model_capability"} or "exceeds_context_length" in reason or "needle_max_ctx" in reason:
            terminal.append({
                **item,
                "classification": "MODEL_OR_OPERATOR_LIMIT",
                "classification_reason": "model context maximum or operator cap; VRAM/KV retries cannot change this depth",
            })
            continue

        estimate = item.get("estimated_total_gb")
        budget = item.get("vram_budget_gb")
        overage = (
            round(float(estimate) - float(budget), 3)
            if isinstance(estimate, (int, float)) and isinstance(budget, (int, float)) else None
        )
        measurement = _needle_measurement_analysis(row, item)
        entry = {
            **item,
            **measurement,
            "gpu_total_gb": gpu_total_gb or None,
            "gpu_only_guarded_limit_gb": round(allowed_gpu_gb, 3) if allowed_gpu_gb else None,
            "allowed_total_with_spill_gb": round(allowed_total, 3) if allowed_total else None,
            "max_spill_gb": round(max(0.0, max_spill_gb), 3),
            "soft_budget_overage_gb": overage,
            "requested_kv_type": requested_kv_type,
            "kv_server_verified": observed_verified or bool(kv_server_confirmed),
            "kv_server_observed_type": observed_kv,
        }

        if not isinstance(estimate, (int, float)):
            entry["classification"] = "ESTIMATE_UNAVAILABLE"
            entry["classification_reason"] = "no usable total-memory estimate was stored for this depth"
            terminal.append(entry)
            continue
        if gpu_total_gb <= 0:
            entry["classification"] = "GPU_CAPACITY_UNKNOWN"
            entry["classification_reason"] = "GPU VRAM was not detectable; pass --gpu-vram-gb to classify guarded retries offline"
            terminal.append(entry)
            continue

        marginal = overage is not None and overage > 0 and overage <= float(marginal_overage_gb)
        fits_guarded = bool(allowed_total and float(estimate) <= allowed_total)
        if fits_guarded:
            entry["classification"] = "MARGINAL_SOFT_LIMIT" if marginal else "GUARDED_RETRY_AVAILABLE"
            entry["classification_reason"] = (
                f"estimate exceeded the old soft budget by only {overage:.3f} GB and is within the guarded GPU+spill allowance"
                if marginal else
                "estimate is within the configured guarded GPU plus system-RAM spill allowance"
            )
            actionable.append(entry)
        else:
            entry["classification"] = "HARD_VRAM_OR_SPILL_LIMIT"
            entry["classification_reason"] = (
                f"estimated {float(estimate):.3f} GB exceeds guarded allowance {allowed_total:.3f} GB"
                if allowed_total else "guarded allowance could not be established"
            )
            terminal.append(entry)

    if actionable:
        note = "retry exact missing needle depth(s) under the configured guard"
        if requested_explicit and not (observed_verified or kv_server_confirmed):
            note += "; application remains blocked until the running Ollama KV setting is verified or --confirm-kv-server is supplied"
        return {
            "kind": "retry_needle_guarded",
            "tasks": ["needle"],
            "automatic": True,
            "details": {
                "classification": "GUARDED_NEEDLE_REPAIR",
                "actionable_skips": actionable,
                "terminal_skips": terminal,
                "requested_kv_type": requested_kv_type,
                "server_inspection": kv_server_inspection,
                "recommended_sequence": ["q8_0", "q4_0"],
                "note": note,
            },
        }, None

    classes = sorted({str(item.get("classification")) for item in terminal})
    reason = "; ".join(
        sorted({str(item.get("classification_reason")) for item in terminal if item.get("classification_reason")})
    ) or "no skipped depth qualifies for automatic repair"
    return None, {
        "kind": "needle_not_automatically_repairable",
        "model": row.get("model"), "run_id": row.get("run_id"),
        "task": "needle", "reason": reason,
        "classifications": classes, "details": terminal,
    }


def build_plan(
    runs_dir: Path,
    *,
    run_id: Optional[str] = None,
    run_prefix: Optional[str] = None,
    everything: bool = False,
    think_retry_num_predict: int = 4096,
    retry_transient: bool = True,
    include_missing: bool = True,
    judge_mode: str = "off",
    judge_model: Optional[str] = None,
    emergency_headroom_gb: float = 0.25,
    max_spill_gb: float = 2.0,
    kv_type: str = "current",
    kv_server_confirmed: bool = False,
    gpu_total_gb: Optional[float] = None,
    force: bool = False,
) -> RepairPlan:
    if int(think_retry_num_predict) <= 0:
        raise ValueError("think retry --num-predict must be greater than zero")
    if float(emergency_headroom_gb) < 0:
        raise ValueError("--emergency-headroom-gb cannot be negative")
    if float(max_spill_gb) < 0:
        raise ValueError("--max-spill-gb cannot be negative")
    if gpu_total_gb is not None and float(gpu_total_gb) <= 0:
        raise ValueError("--gpu-vram-gb must be greater than zero")

    run_dirs = discover_runs(runs_dir, run_id=run_id, run_prefix=run_prefix, everything=everything)
    if not run_dirs:
        raise ValueError("no matching run directories with raw_results.jsonl")
    all_rows, contexts = _load_selected_rows(run_dirs)
    current = rank_for_output(all_rows)
    gpu = detect_gpu()
    resolved_gpu_total_gb = float(gpu_total_gb) if gpu_total_gb is not None else float(gpu.total_vram_gb or 0.0)
    kv_server_inspection = inspect_ollama_kv_environment()
    previous_repairs = _latest_repair_records(run_dirs)
    unavailable_families_by_model = _known_unavailable_families(run_dirs)
    actions: List[RepairAction] = []
    observations: List[Dict[str, Any]] = []
    grouped_gate: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    seen: set[Tuple[str, str, str]] = set()

    def add_action(kind: str, row: Dict[str, Any], tasks: List[str], reason: str,
                   *, automatic: bool = True, family: Optional[str] = None,
                   overrides: Optional[Dict[str, Any]] = None,
                   details: Optional[Dict[str, Any]] = None,
                   source_hashes: Optional[Dict[str, str]] = None) -> None:
        digest = str(row.get("model_digest_resolved") or row.get("model") or "")
        source_run = str(row.get("run_id") or "")
        key = (kind, digest, ",".join(sorted(set(tasks))))
        if key in seen:
            return
        seen.add(key)
        action_id = _action_id(
            kind, digest, tasks, source_run,
            variant={"family": family, "overrides": overrides or {}},
        )
        previous = previous_repairs.get(action_id)
        previous_status = str((previous or {}).get("status") or "")
        if previous and _record_is_terminal_capability_failure(previous, runs_dir):
            observations.append({
                "kind": "terminal_capability_failure_not_repeated",
                "model": row.get("model"), "run_id": source_run,
                "task": ",".join(sorted(set(tasks))),
                "reason": (f"action {action_id} already produced a task-equivalent, measured "
                           "zero-quality insert result; generic --force does not repeat terminal evidence"),
                "previous_record": previous,
            })
            return
        if previous and not force:
            observations.append({
                "kind": "previous_repair_not_repeated",
                "model": row.get("model"), "run_id": source_run,
                "task": ",".join(sorted(set(tasks))),
                "reason": f"action {action_id} already recorded status={previous_status or 'unknown'}; use --force to repeat",
                "previous_record": previous,
            })
            return
        hashes = source_hashes or {task: str(row.get("_source_row_hash") or "") for task in tasks}
        actions.append(RepairAction(
            action_id=action_id,
            kind=kind,
            model=str(row.get("model") or contexts.get(digest, {}).get("model") or digest),
            model_digest=digest,
            source_run_id=source_run,
            tasks=sorted(set(tasks)),
            reason=reason,
            automatic=automatic,
            family=family,
            overrides=overrides or {},
            source_row_hashes=hashes,
            details=details or {},
        ))

    rows_by_digest_task = {(str(row.get("model_digest_resolved")), str(row.get("task"))): row for row in current}

    for row in current:
        task_id = str(row.get("task") or "")
        task = _TASKS.get(task_id)
        if task is None or task.difficulty <= 0:
            continue
        error_kind = str(row.get("error_kind") or "")
        text = _error_text(row)
        digest = str(row.get("model_digest_resolved") or row.get("model") or "")
        context = contexts.get(digest) or {}
        profile = _best_profile(context)
        declared = profile.get("declared_capabilities") or row.get("capabilities_declared") or []
        model_name = str(row.get("model") or "")
        current_families = set(families_for(model_name, list(declared) or None))
        known_unavailable = unavailable_families_by_model.get(model_name, set())
        if task.family in known_unavailable:
            observations.append({
                "kind": "capability_already_unavailable", "model": model_name,
                "run_id": row.get("run_id"), "task": task_id,
                "reason": (f"{task.family} is already confirmed unavailable for the installed "
                           "model build; generic --force does not re-probe terminal capability evidence"),
            })
            continue
        if error_kind and task.family not in current_families:
            observations.append({
                "kind": "obsolete_misrouted_task", "model": row.get("model"),
                "run_id": row.get("run_id"), "task": task_id,
                "reason": f"historical route included {task.family!r}, current route is {sorted(current_families)}; do not retry",
            })
            continue
        if error_kind == "thinking_only":
            profiles = _thinking_retry_profiles(row, int(think_retry_num_predict))
            add_action(
                "retry_generation", row, [task_id],
                "thinking_only: bounded visible-answer recovery",
                overrides={"retry_profiles": profiles},
                details={"attempt_limit": len(profiles)},
            )
            continue
        if error_kind == "empty_output":
            if task.family in {"vision", "tools", "insert"}:
                key = (str(row.get("model_digest_resolved")), task.family, str(row.get("run_id")))
                group = grouped_gate.setdefault(key, {"row": row, "tasks": [], "reasons": [], "hashes": {}})
                group["tasks"].append(task_id)
                group["reasons"].append(text[:300] or "empty output")
                group["hashes"][task_id] = str(row.get("_source_row_hash") or "")
            else:
                profiles = _thinking_retry_profiles(row, int(think_retry_num_predict))
                add_action(
                    "retry_generation", row, [task_id],
                    "empty_output: bounded visible-answer recovery",
                    overrides={"retry_profiles": profiles},
                    details={"attempt_limit": len(profiles)},
                )
            continue
        if error_kind == "harness_error":
            if task.family in {"vision", "tools", "insert"} and (_HTTP_400_RE.search(text) or _TRANSIENT_RE.search(text)):
                key = (str(row.get("model_digest_resolved")), task.family, str(row.get("run_id")))
                group = grouped_gate.setdefault(key, {"row": row, "tasks": [], "reasons": [], "hashes": {}})
                group["tasks"].append(task_id)
                group["reasons"].append(text[:300])
                group["hashes"][task_id] = str(row.get("_source_row_hash") or "")
                continue
            if retry_transient and _TRANSIENT_RE.search(text):
                add_action("retry_transient", row, [task_id], "transient runtime/API failure; unload and retry once")
                continue
            observations.append({
                "kind": "manual_harness_triage", "model": row.get("model"),
                "run_id": row.get("run_id"), "task": task_id, "reason": text[:500],
            })
            continue
        if task.scorer == "subjective" and row.get("score") is None and not error_kind:
            if judge_mode != "off" and judge_model:
                add_action("judge_existing_dump", row, [task_id], "subjective output exists without a judge score")
            else:
                observations.append({
                    "kind": "awaiting_judge", "model": row.get("model"),
                    "run_id": row.get("run_id"), "task": task_id,
                })
            continue
        needle_action, needle_observation = _needle_observation(
            row,
            gpu_total_gb=resolved_gpu_total_gb,
            emergency_headroom_gb=emergency_headroom_gb,
            max_spill_gb=max_spill_gb,
            requested_kv_type=kv_type,
            kv_server_confirmed=kv_server_confirmed,
            kv_server_inspection=kv_server_inspection,
        )
        if needle_action:
            kind = str(needle_action["kind"])
            automatic = bool(needle_action.get("automatic", True))
            reason = (
                "KV-cache configuration mismatch is suspected; reconfigure/restart Ollama for q8_0, then q4_0 if required"
                if kind == "kv_reconfigure_required" else
                "incomplete needle coverage is within the configured guarded VRAM+RAM spill allowance"
            )
            add_action(
                kind, row, ["needle"], reason, automatic=automatic,
                overrides={"vram_budget_gb": round(max(0.0, resolved_gpu_total_gb - emergency_headroom_gb) + max_spill_gb, 3),
                           "kv_type": kv_type},
                details=needle_action.get("details"),
            )
        elif needle_observation:
            observations.append(needle_observation)

    for (_digest, family, _run_id), group in grouped_gate.items():
        row = group["row"]
        add_action(
            "capability_gate", row, group["tasks"],
            f"{family} lane produced API failures; run one functional gate before any task retry",
            family=family,
            overrides={"think": "off"},
            details={"sample_errors": group["reasons"][:3]},
            source_hashes=group.get("hashes"),
        )

    if include_missing:
        for digest, context in contexts.items():
            if "full" not in set(context.get("levels") or []):
                continue
            profile = _best_profile(context)
            families = set(profile.get("supported_families") or [])
            model_name = str(context.get("model") or "")
            known_unavailable = unavailable_families_by_model.get(model_name, set())
            for task in TASKS:
                if task.difficulty <= 0 or task.family not in families or task.family in known_unavailable:
                    continue
                row = rows_by_digest_task.get((digest, task.id))
                if row is None:
                    pseudo = {
                        "model": context.get("model"), "model_digest_resolved": digest,
                        "run_id": sorted(context.get("runs") or [""])[-1], "_source_row_hash": "",
                    }
                    if task.family in {"vision", "tools", "insert", "embedding"}:
                        add_action(
                            "capability_gate", pseudo, [task.id],
                            "missing task requires a task-equivalent functional capability probe before scored execution",
                            family=task.family, overrides={"think": "off"},
                            details={"probe_before_missing_task": True},
                        )
                    else:
                        add_action("run_missing_task", pseudo, [task.id], "applicable current task has no evidence")
                    continue
                current_hash = _CURRENT_HASHES.get(task.id)
                if current_hash and row.get("task_hash") != current_hash:
                    add_action("rerun_stale_task", row, [task.id], "stored task hash does not match the current task definition")

    counts = Counter(action.kind for action in actions)
    counts.update(f"observation:{item.get('kind')}" for item in observations)
    created = datetime.now(timezone.utc).isoformat()
    options = {
        "think_retry_num_predict": int(think_retry_num_predict),
        "retry_transient": bool(retry_transient),
        "include_missing": bool(include_missing),
        "judge_mode": judge_mode,
        "judge_model": judge_model,
        "emergency_headroom_gb": float(emergency_headroom_gb),
        "max_spill_gb": float(max_spill_gb),
        "detected_gpu_total_gb": resolved_gpu_total_gb,
        "gpu_total_source": "cli_override" if gpu_total_gb is not None else "hardware_detection",
        "kv_type": kv_type,
        "kv_server_configuration_confirmed": bool(kv_server_confirmed),
        "kv_server_inspection": kv_server_inspection,
        "force": bool(force),
    }
    plan_seed = json.dumps({
        "runs": [path.name for path in run_dirs],
        "actions": [asdict(action) for action in actions],
        "options": options,
    }, sort_keys=True, default=str)
    return RepairPlan(
        schema_version=1,
        repair_policy_version=POLICY_VERSION,
        plan_id=hashlib.sha256(plan_seed.encode()).hexdigest()[:16],
        created_at=created,
        runs_dir=str(runs_dir),
        selected_runs=[path.name for path in run_dirs],
        actions=actions,
        observations=observations,
        counts=dict(sorted(counts.items())),
        options=options,
    )


def render_plan(plan: RepairPlan) -> str:
    gpu_total = float(plan.options.get("detected_gpu_total_gb") or 0.0)
    headroom = float(plan.options.get("emergency_headroom_gb") or 0.0)
    spill = float(plan.options.get("max_spill_gb") or 0.0)
    allowed = max(0.0, gpu_total - headroom) + max(0.0, spill) if gpu_total else 0.0
    kv = plan.options.get("kv_server_inspection") or {}
    lines = [
        f"Repair plan {plan.plan_id} (policy v{plan.repair_policy_version})",
        f"Runs selected: {len(plan.selected_runs)}",
        f"Automatic actions: {sum(1 for a in plan.actions if a.automatic)}",
        f"Observations/manual items: {len(plan.observations)}",
        (f"GPU planning: {gpu_total:.3f} GB ({plan.options.get('gpu_total_source')}); "
         f"headroom={headroom:.3f} GB, permitted spill={spill:.3f} GB, guarded total={allowed:.3f} GB"
         if gpu_total else "GPU planning: capacity unknown; use --gpu-vram-gb for reproducible offline classification"),
        (f"KV planning: requested={plan.options.get('kv_type')}; "
         f"observed={kv.get('effective_kv_type') or 'unknown'}; "
         f"verified={'yes' if kv.get('verified') else 'no'}; "
         f"source={kv.get('effective_source') or 'unavailable'}"),
        "",
    ]
    if plan.counts:
        width = max(len(key) for key in plan.counts)
        lines.append("Counts")
        for key, value in plan.counts.items():
            lines.append(f"  {key:<{width}}  {value}")
        lines.append("")
    for action in plan.actions:
        lines.append(f"[{action.kind}] {action.model}")
        lines.append(f"  tasks: {', '.join(action.tasks)}")
        lines.append(f"  source: {action.source_run_id}")
        lines.append(f"  reason: {action.reason}")
        if action.family:
            lines.append(f"  family gate: {action.family}")
        if action.overrides:
            lines.append("  overrides: " + json.dumps(action.overrides, sort_keys=True))
        actionable_skips = list((action.details or {}).get("actionable_skips") or [])
        for skip in actionable_skips:
            lines.append(
                "  needle depth: "
                f"ctx={skip.get('size')} classification={skip.get('classification')} "
                f"estimate={skip.get('estimated_total_gb')} GB "
                f"old_budget={skip.get('vram_budget_gb')} GB "
                f"overage={skip.get('soft_budget_overage_gb')} GB"
            )
            if skip.get("estimator_divergence_note"):
                lines.append("    estimator note: " + str(skip.get("estimator_divergence_note")))
        lines.append("")
    if plan.observations:
        lines.append("Not automatically modified")
        for item in plan.observations:
            lines.append(f"  [{item.get('kind')}] {item.get('model', '-')} / {item.get('task', '-')} - {item.get('reason', '')}")
    lines.extend([
        "",
        "Source raw_results.jsonl files are never overwritten.",
        "Generation repairs create child runs; judgements use judge_results.jsonl sidecars.",
    ])
    return "\n".join(lines)


def _slug(value: str, limit: int = 50) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return (text or "model")[:limit]


def _source_run_path(plan: RepairPlan, run_id: str) -> Path:
    return Path(plan.runs_dir) / run_id


def _record_action(plan: RepairPlan, action: RepairAction, payload: Dict[str, Any]) -> None:
    source = _source_run_path(plan, action.source_run_id)
    _append_jsonl(source / "repair_results.jsonl", {
        "schema_version": 1,
        "repair_policy_version": plan.repair_policy_version,
        "plan_id": plan.plan_id,
        "action_id": action.action_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "action": asdict(action),
        **payload,
    })


def _record_unavailable_capability(
    source_run: Path,
    *,
    model: str,
    family: str,
    action_id: str,
    gate_result: Dict[str, Any],
) -> None:
    path = source_run / "capability_repair.json"
    data = _read_json(path) or {}
    model_entry = data.setdefault(model, {"unavailable_families": {}, "history": []})
    item = {
        "family": family,
        "action_id": action_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "reason": "functional capability gate failed on the current Ollama/model build",
        "gate": gate_result,
    }
    model_entry.setdefault("unavailable_families", {})[family] = item
    model_entry.setdefault("history", []).append(item)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


_FLASH_KV_RE = re.compile(r"quantized\s+V\s+cache.*requires\s+Flash\s+Attention", re.I | re.S)
_OOM_RE = re.compile(r"cudaMalloc failed|out of memory|failed to allocate .*buffer", re.I)


def _kv_runtime_identity(client: Any, controller: Any) -> Dict[str, Any]:
    gpu = detect_gpu()
    try:
        ollama_version = client.version() if hasattr(client, "version") else None
    except Exception:
        ollama_version = None
    return {
        "ollama_version": ollama_version,
        "service_unit": getattr(controller, "unit", None),
        "gpu_name": getattr(gpu, "name", None),
        "gpu_vendor": getattr(gpu, "vendor", None),
        "gpu_driver": getattr(gpu, "driver_version", None),
    }


def _record_kv_compatibility(
    source_run: Path, *, model: str, kv_type: str, status: str,
    error_kinds: List[str], child_run_ids: List[str], detail: str,
    model_digest: Optional[str] = None,
    runtime_identity: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist build/runtime-scoped KV compatibility evidence.

    A model name alone is not a stable compatibility key.  Ollama upgrades,
    changed GGUF digests, GPU/backend changes, or a different service can alter
    Flash-Attention support.  The stored identity is therefore checked before
    future plans suppress q8/q4.
    """
    source_run.mkdir(parents=True, exist_ok=True)
    path = source_run / "kv_compatibility.json"
    data = _read_json(path) or {}
    model_entry = data.setdefault(model, {"kv_modes": {}, "history": []})
    if model_digest:
        model_entry["model_digest"] = model_digest
    if runtime_identity:
        model_entry["runtime_identity"] = dict(runtime_identity)
    item = {
        "kv_type": kv_type,
        "status": status,
        "error_kinds": sorted(set(error_kinds)),
        "child_run_ids": list(child_run_ids),
        "detail": detail[:1000],
        "model_digest": model_digest,
        "runtime_identity": dict(runtime_identity or {}),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    model_entry.setdefault("kv_modes", {})[kv_type] = item
    model_entry.setdefault("history", []).append(item)
    if status == "incompatible" and "kv_quantization_requires_flash_attention" in error_kinds:
        model_entry["preferred_kv_type"] = "current"
        model_entry["avoid_quantized_kv"] = True
    if kv_type == "current" and status == "supported":
        model_entry["preferred_kv_type"] = "current"
        model_entry["current_kv_supported"] = True
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _load_model_kv_compatibility(source_run: Path, model: str) -> Dict[str, Any]:
    data = _read_json(source_run / "kv_compatibility.json") or {}
    return dict(data.get(model) or {})


def _kv_identity_matches(
    entry: Dict[str, Any], action: RepairAction,
    runtime_identity: Optional[Dict[str, Any]],
) -> bool:
    stored_digest = str(entry.get("model_digest") or "")
    action_digest = str(action.model_digest or "")
    if stored_digest and action_digest and stored_digest != action_digest:
        return False
    stored_runtime = entry.get("runtime_identity") or {}
    if not stored_runtime or not runtime_identity:
        return True
    for key in ("ollama_version", "service_unit", "gpu_name", "gpu_vendor", "gpu_driver"):
        old = stored_runtime.get(key)
        new = runtime_identity.get(key)
        if old not in (None, "") and new not in (None, "") and str(old) != str(new):
            return False
    return True

def _phase_action_error_evidence(
    result: Dict[str, Any], runs_dir: Path
) -> Dict[str, Dict[str, Any]]:
    """Summarise structured needle failures for each phase action."""
    out: Dict[str, Dict[str, Any]] = {}
    for entry in result.get("actions") or []:
        action_id = str(entry.get("action_id") or "")
        texts: List[str] = []
        child_ids: List[str] = []
        probes = 0
        flash_hits = 0
        oom_hits = 0
        for attempt in entry.get("attempts") or []:
            child = str(attempt.get("child_run_id") or "")
            if not child:
                continue
            child_ids.append(child)
            for row in _read_jsonl(Path(runs_dir) / child / "raw_results.jsonl"):
                if str(row.get("task")) != "needle":
                    continue
                for probe in row.get("needle_attempted") or []:
                    text = " ".join(str(probe.get(k) or "") for k in (
                        "harness_error_detail", "http_error_body", "reason"
                    ))
                    if not text.strip():
                        continue
                    probes += 1
                    texts.append(text)
                    if _FLASH_KV_RE.search(text):
                        flash_hits += 1
                    if _OOM_RE.search(text):
                        oom_hits += 1
        kinds: List[str] = []
        if probes and flash_hits == probes:
            kinds.append("kv_quantization_requires_flash_attention")
        if oom_hits:
            kinds.append("cuda_out_of_memory")
        out[action_id] = {
            "probe_count": probes,
            "flash_attention_hits": flash_hits,
            "oom_hits": oom_hits,
            "error_kinds": kinds,
            "child_run_ids": child_ids,
            "detail": texts[0][:1000] if texts else "",
            "quantized_kv_incompatible": bool(probes and flash_hits == probes),
        }
    return out


def _known_quantized_kv_incompatible(
    action: RepairAction, runs_dir: Path,
    runtime_identity: Optional[Dict[str, Any]] = None,
) -> bool:
    entry = _load_model_kv_compatibility(Path(runs_dir) / action.source_run_id, action.model)
    if not _kv_identity_matches(entry, action, runtime_identity):
        return False
    if entry.get("avoid_quantized_kv"):
        return True
    modes = entry.get("kv_modes") or {}
    return any(
        str((modes.get(kv) or {}).get("status")) == "incompatible"
        and "kv_quantization_requires_flash_attention" in ((modes.get(kv) or {}).get("error_kinds") or [])
        for kv in ("q8_0", "q4_0")
    )


def _kv_environment_check(
    kv_type: str,
    *,
    server_confirmed: bool = False,
    server_inspection: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    if kv_type == "current":
        return True, "current server KV type requested; no explicit KV assertion"

    inspection = server_inspection or inspect_ollama_kv_environment()
    shell_value = str(os.environ.get("OLLAMA_KV_CACHE_TYPE") or "").strip().lower()
    observed = str(inspection.get("effective_kv_type") or "").strip().lower()
    verified = bool(inspection.get("verified"))

    if shell_value != kv_type:
        return False, (
            f"requested KV type {kv_type!r}, but OLLAMA_KV_CACHE_TYPE in the repair process is "
            f"{shell_value or 'unset'!r}. Export the same value used by the Ollama service so the "
            "benchmark estimator and server configuration cannot silently diverge."
        )
    if verified and observed != kv_type:
        return False, (
            f"the running Ollama process reports {observed or 'unset'!r}, not requested {kv_type!r}; "
            "reconfigure/restart Ollama before applying the repair"
        )
    if verified and observed == kv_type:
        return True, f"running Ollama process environment verified as {kv_type}"
    if not server_confirmed:
        return False, (
            f"the repair shell requests {kv_type}, but the running Ollama process environment could "
            "not be verified. Reconfigure/restart the service, then repeat with --confirm-kv-server "
            "only after checking the live service."
        )
    return True, (
        f"operator confirmed the Ollama server was restarted with {kv_type}; the repair also exports "
        "the same value. Measured VRAM remains evidence of total memory behaviour, not proof of KV precision."
    )


def apply_plan(
    client: Any,
    cfg: Any,
    plan: RepairPlan,
    *,
    judge_mode: str = "off",
    judge_model: Optional[str] = None,
    rankings_dir: Optional[Path] = None,
    parent_repair_plan_id: Optional[str] = None,
    parent_repair_phase: Optional[str] = None,
    action_started_callback: Optional[Any] = None,
    child_started_callback: Optional[Any] = None,
    live_ui: str = "off",
    ranking_scope: str = "canonical",
) -> Dict[str, Any]:
    from . import report, runner, rankings
    from .ranking_controls import write_run_scope

    result: Dict[str, Any] = {
        "plan_id": plan.plan_id,
        "repair_policy_version": plan.repair_policy_version,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "actions_total": len(plan.actions),
        "completed": 0,
        "recovered": 0,
        "unresolved": 0,
        "terminal_failures": 0,
        "errors": 0,
        "timeouts": 0,
        "manual_items": len(plan.observations),
        "child_runs": [],
        "actions": [],
    }
    kv_ok, kv_note = _kv_environment_check(
        str(plan.options.get("kv_type") or "current"),
        server_confirmed=bool(plan.options.get("kv_server_configuration_confirmed")),
        server_inspection=plan.options.get("kv_server_inspection") or {},
    )

    # Judge each source run once even when several subjective rows are eligible.
    judge_run_ids = sorted({a.source_run_id for a in plan.actions if a.kind == "judge_existing_dump"})
    for source_run_id in judge_run_ids:
        source = _source_run_path(plan, source_run_id)
        source_actions = [
            action for action in plan.actions
            if action.kind == "judge_existing_dump" and action.source_run_id == source_run_id
        ]
        if judge_mode == "off" or not judge_model:
            judged = None
            status = "unresolved"
            reason = "judge mode/model not configured"
        else:
            judged = judge_run(client, source, judge_model=judge_model, judge_mode=judge_mode)
            status = "recovered" if judged.get("judged") else "unresolved"
            reason = None
        for action in source_actions:
            entry = {
                "action_id": action.action_id,
                "kind": "judge_existing_dump",
                "source_run_id": source_run_id,
                "tasks": action.tasks,
                "status": status,
                "reason": reason,
                "result": judged,
            }
            _record_action(plan, action, entry)
            result["actions"].append(entry)
            result["completed"] += 1
            result["recovered" if status == "recovered" else "unresolved"] += 1

    # Group bounded generation retries by model/config so each child run is compact.
    runnable = [a for a in plan.actions if a.kind != "judge_existing_dump"]
    functional_profile_cache: Dict[str, Dict[str, Any]] = {}
    gate_families_by_model: Dict[str, set[str]] = defaultdict(set)
    for pending_action in runnable:
        if pending_action.kind == "capability_gate" and pending_action.family:
            gate_families_by_model[pending_action.model].add(str(pending_action.family))
    for action in runnable:
        if action_started_callback:
            try:
                action_started_callback(action)
            except Exception:
                pass
        if not action.automatic:
            entry = {"action_id": action.action_id, "kind": action.kind,
                     "status": "manual_action_required", "reason": action.reason,
                     "details": action.details or {}}
            _record_action(plan, action, entry)
            result["actions"].append(entry)
            result["completed"] += 1
            result["unresolved"] += 1
            continue
        source = _source_run_path(plan, action.source_run_id)
        profile = (_read_json(source / "capability_report.json") or {}).get(action.model)
        if not profile:
            profile = interrogate_model(client, action.model, functional=False)

        tasks = list(action.tasks)
        gate_result = None
        if action.kind == "capability_gate":
            gate_profile = functional_profile_cache.get(action.model)
            if gate_profile is None:
                gate_profile = interrogate_model(
                    client, action.model, functional=True,
                    probe_families=sorted(gate_families_by_model.get(action.model) or {str(action.family)}),
                )
                functional_profile_cache[action.model] = gate_profile
            family_probe = (gate_profile.get("probes") or {}).get(str(action.family)) or {}
            probe_state = str((gate_profile.get("probe_states") or {}).get(str(action.family)) or "probe_failed")
            # A response with the wrong tiny-probe answer is a quality signal,
            # not proof that the lane is unavailable. Run the scored task.
            gate_ok = probe_state in {"confirmed_supported", "responded_contract_failed"}
            gate_result = {
                "profile": gate_profile, "family": action.family, "ok": gate_ok,
                "probe_state": probe_state, "responded": bool(family_probe.get("responded")),
            }
            if not gate_ok:
                definitive_unavailable = probe_state == "confirmed_unavailable"
                entry = {
                    "action_id": action.action_id,
                    "kind": action.kind,
                    "status": (
                        "capability_unavailable"
                        if definitive_unavailable
                        else "capability_probe_inconclusive"
                    ),
                    "gate": gate_result,
                }
                # Only definitive build/runtime evidence may exclude a capability
                # from rankings. Transient, ambiguous, or failed probes remain
                # unresolved evidence and must not poison capability_repair.json.
                if definitive_unavailable:
                    _record_unavailable_capability(
                        source, model=action.model, family=str(action.family),
                        action_id=action.action_id, gate_result=gate_result,
                    )
                _record_action(plan, action, entry)
                result["actions"].append(entry)
                result["completed"] += 1; result["unresolved"] += 1
                continue
            profile = gate_profile

        if action.kind == "retry_needle_guarded" and not kv_ok:
            entry = {"action_id": action.action_id, "kind": action.kind,
                     "status": "blocked_kv_configuration", "reason": kv_note}
            _record_action(plan, action, entry)
            result["actions"].append(entry)
            result["completed"] += 1; result["unresolved"] += 1
            continue

        base_overrides = action.overrides or {}
        retry_profiles = list(base_overrides.get("retry_profiles") or [{}])
        if action.kind != "retry_generation":
            retry_profiles = [base_overrides]
        action_recovered = False
        action_terminal_failure = False
        action_had_error = False
        action_timed_out = False
        attempt_entries: List[Dict[str, Any]] = []

        for attempt_number, attempt_overrides in enumerate(retry_profiles, start=1):
            child_cfg = copy.deepcopy(cfg)
            merged_overrides = dict(base_overrides)
            merged_overrides.pop("retry_profiles", None)
            merged_overrides.update(attempt_overrides or {})
            if merged_overrides.get("think"):
                child_cfg.think = str(merged_overrides["think"])
            if merged_overrides.get("num_predict"):
                child_cfg.num_predict_override = int(merged_overrides["num_predict"])
            if merged_overrides.get("vram_budget_gb"):
                child_cfg.vram_budget_gb = float(merged_overrides["vram_budget_gb"])
            child_cfg.dump_raw = True
            child_cfg.fingerprint = False

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            child_run_id = f"repair_{stamp}_{_slug(action.model)}_{action.action_id[:6]}_a{attempt_number}"
            child_dir = Path(plan.runs_dir) / child_run_id
            # Link the child to the repair campaign before runner.run() starts.
            # The watcher needs this during the long-running model call, not
            # only after apply_plan() returns.
            if rankings_dir is not None:
                write_run_scope(child_dir, scope=ranking_scope, rankings_dir=rankings_dir)
            if parent_repair_plan_id:
                _link_child_runs_to_parent(
                    Path(plan.runs_dir), [child_run_id],
                    plan_id=parent_repair_plan_id, action_id=action.action_id,
                    phase=str(parent_repair_phase or plan.options.get("phase") or "repair"),
                )
            if child_started_callback:
                try:
                    child_started_callback(action, child_run_id)
                except Exception:
                    pass
            metadata_by_task = {
                task_id: {
                    "repair_parent_run_id": action.source_run_id,
                    "repair_source_row_hash": (action.source_row_hashes or {}).get(task_id),
                    "repair_policy_version": plan.repair_policy_version,
                    "repair_plan_id": plan.plan_id,
                    "repair_action_id": action.action_id,
                    "repair_kind": action.kind,
                    "repair_attempt_number": attempt_number,
                    "repair_overrides": merged_overrides,
                    "repair_kv_note": kv_note if action.kind == "retry_needle_guarded" else None,
                }
                for task_id in tasks
            }
            try:
                client.flush_all()
                runner.run(
                    client, child_cfg,
                    level="full", out_dir=child_dir,
                    include=None, exclude=None, skip_offload=False,
                    categories=None, task_ids=tasks, task_regex=None,
                    family_base_only=False, context_aliases_only=False, context_only=False,
                    resume=False, judge_mode="off", dump_subjective=True, dump_raw=True,
                    status_interval=1.0, live_ui=live_ui, sample_mode="smart",
                    fingerprint_enabled=False, selected_models=[action.model],
                    capability_profiles={action.model: profile}, auto_probe=False,
                    row_metadata_by_task=metadata_by_task,
                )
                child_rows = _read_jsonl(child_dir / "raw_results.jsonl")
                needs_child_judge = any((_TASKS.get(task_id) or object()).scorer == "subjective" for task_id in tasks if _TASKS.get(task_id))
                if needs_child_judge and judge_mode != "off" and judge_model:
                    judge_run(client, child_dir, judge_model=judge_model, judge_mode=judge_mode)
                    child_rows = apply_judgements(child_dir, child_rows)
                report.build(child_dir, child_cfg)
                valid_tasks = {
                    str(row.get("task")) for row in child_rows
                    if _is_numeric(row.get("score"))
                    and not row.get("error_kind")
                    and (
                        str(row.get("task")) != "needle"
                        or float(row.get("needle_coverage") or 0.0) >= 1.0
                    )
                }
                # A task-equivalent capability probe that received a real
                # endpoint response followed by a scored insert task returning
                # an empty insertion is a measured zero-quality result, not an
                # unattempted cell. Transport/runtime errors remain unresolved.
                terminal_failure_tasks = {
                    str(row.get("task")) for row in child_rows
                    if action.kind == "capability_gate"
                    and str(action.family or "") == "insert"
                    and str((gate_result or {}).get("probe_state") or "")
                        in {"confirmed_supported", "responded_contract_failed"}
                    and _is_numeric(row.get("score"))
                    and float(row.get("score") or 0.0) <= 0.0
                    and str(row.get("error_kind") or "") == "empty_output"
                }
                resolved_tasks = valid_tasks | terminal_failure_tasks
                recovered_all = set(tasks).issubset(valid_tasks)
                terminally_resolved = set(tasks).issubset(resolved_tasks) and bool(terminal_failure_tasks)
                error_text = " ".join(_error_text(row) for row in child_rows)
                timed_out = bool(_TRANSIENT_RE.search(error_text) and re.search(r"timed?\s*out|timeout", error_text, re.I))
                status = (
                    "recovered" if recovered_all else
                    "measured_failure" if terminally_resolved else
                    "timeout" if timed_out else "unresolved"
                )
                attempt_entry = {
                    "attempt_number": attempt_number,
                    "status": status,
                    "child_run_id": child_run_id,
                    "overrides": merged_overrides,
                    "valid_tasks": sorted(valid_tasks),
                    "terminal_failure_tasks": sorted(terminal_failure_tasks),
                    "required_tasks": sorted(set(tasks)),
                    "rows": len(child_rows),
                }
                attempt_entries.append(attempt_entry)
                result["child_runs"].append(child_run_id)
                if recovered_all:
                    action_recovered = True
                    break
                if terminally_resolved:
                    action_terminal_failure = True
                    break
                if timed_out:
                    action_timed_out = True
            except Exception as exc:
                text = repr(exc)
                timed_out = bool(re.search(r"timed?\s*out|timeout", text, re.I))
                attempt_entries.append({
                    "attempt_number": attempt_number,
                    "status": "timeout" if timed_out else "error",
                    "error": text,
                    "child_run_id": child_run_id,
                    "overrides": merged_overrides,
                })
                action_had_error = True
                action_timed_out = action_timed_out or timed_out
            finally:
                try:
                    client.unload(action.model)
                except Exception:
                    pass

        status = (
            "recovered" if action_recovered else
            "measured_failure" if action_terminal_failure else
            "timeout" if action_timed_out else
            "error" if action_had_error else "unresolved"
        )
        entry = {
            "action_id": action.action_id, "kind": action.kind,
            "status": status, "tasks": tasks, "attempts": attempt_entries,
            "gate": gate_result,
            "kv_note": kv_note if action.kind == "retry_needle_guarded" else None,
            "next_required_kv_type": (
                "q4_0" if action.kind == "retry_needle_guarded"
                and str(plan.options.get("kv_type")) == "q8_0" and not action_recovered else None
            ),
        }
        _record_action(plan, action, entry)
        result["actions"].append(entry)
        result["completed"] += 1
        if action_recovered:
            result["recovered"] += 1
        elif action_terminal_failure:
            result["terminal_failures"] += 1
        elif action_timed_out:
            result["timeouts"] += 1
            result["unresolved"] += 1
        elif action_had_error:
            result["errors"] += 1
        else:
            result["unresolved"] += 1

    if rankings_dir is not None:
        try:
            rankings_result = rankings.write_rankings(
                Path(plan.runs_dir), rankings_dir, force_rescan=True,
                include_separate=(ranking_scope == "separate"),
                only_run_ids=(result.get("child_runs") if ranking_scope == "separate" else None),
            )
            result["rankings"] = rankings_result
        except Exception as exc:
            result["rankings_error"] = repr(exc)
    non_timeout_unresolved = max(0, int(result["unresolved"]) - int(result["timeouts"]))
    if result["timeouts"] and not result["recovered"] and not result["errors"] and not non_timeout_unresolved and not result["manual_items"]:
        outcome = "TIMEOUT"
    elif result["errors"] and not result["recovered"] and not result["unresolved"] and not result["manual_items"]:
        outcome = "FAILED"
    elif result["unresolved"] or result["errors"] or result["manual_items"]:
        outcome = "PARTIAL"
    else:
        outcome = "COMPLETE"
    result["outcome"] = outcome
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    return result


def _derived_plan(
    base: RepairPlan,
    actions: Sequence[RepairAction],
    *,
    phase: str,
    kv_type: str,
    server_inspection: Optional[Dict[str, Any]] = None,
) -> RepairPlan:
    """Create a phase-specific immutable plan without changing the base plan."""
    derived_actions: List[RepairAction] = []
    for action in actions:
        clone = copy.deepcopy(action)
        # Keep the deterministic source action ID across q8/q4 phases. The
        # append-only repair log records both phase plans and the latest status
        # then remains discoverable by the normal future-plan suppression logic.
        clone.overrides = dict(clone.overrides or {})
        clone.overrides["kv_type"] = kv_type
        clone.details = dict(clone.details or {})
        clone.details["service_phase"] = phase
        clone.details["parent_action_id"] = action.action_id
        derived_actions.append(clone)
    options = dict(base.options)
    options.update({
        "kv_type": kv_type,
        "kv_server_configuration_confirmed": True,
        "kv_server_inspection": dict(server_inspection or {}),
        "service_phase": phase,
        "parent_plan_id": base.plan_id,
    })
    seed = json.dumps({
        "parent": base.plan_id,
        "phase": phase,
        "kv_type": kv_type,
        "actions": [asdict(action) for action in derived_actions],
    }, sort_keys=True)
    return RepairPlan(
        schema_version=base.schema_version,
        repair_policy_version=base.repair_policy_version,
        plan_id=hashlib.sha256(seed.encode()).hexdigest()[:16],
        created_at=datetime.now(timezone.utc).isoformat(),
        runs_dir=base.runs_dir,
        selected_runs=list(base.selected_runs),
        actions=derived_actions,
        observations=[],
        counts=dict(Counter(action.kind for action in derived_actions)),
        options=options,
    )


def _unresolved_parent_action_ids(result: Dict[str, Any], phase_plan: RepairPlan) -> set[str]:
    parents = {action.action_id: str((action.details or {}).get("parent_action_id") or "") for action in phase_plan.actions}
    unresolved: set[str] = set()
    for entry in result.get("actions") or []:
        if entry.get("status") not in {"recovered", "measured_failure"}:
            parent = parents.get(str(entry.get("action_id") or ""))
            if parent:
                unresolved.add(parent)
    return unresolved


def _repair_campaign_run_id(plan_id: str) -> str:
    return f"repair_campaign_{plan_id}"


def _write_repair_status(runs_dir: Path, plan: RepairPlan, payload: Dict[str, Any]) -> Path:
    """Atomically update both legacy and discoverable repair status paths.

    The discoverable campaign directory lets ``llmb-watch --follow-queue`` see
    capability probes before a child benchmark directory exists. Child links
    remain useful when the operator explicitly watches a child run.
    """
    runs_dir = Path(runs_dir)
    status_path = runs_dir / f"repair_status_{plan.plan_id}.json"
    campaign_dir = runs_dir / _repair_campaign_run_id(plan.plan_id)
    payload = dict(payload)
    payload.setdefault("status_type", "repair")
    payload.setdefault("plan_id", plan.plan_id)
    payload.setdefault("run_id", campaign_dir.name)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        _atomic_write_json(status_path, payload)
        _atomic_write_json(campaign_dir / "status.json", payload)
    except Exception:
        pass  # status reporting must never fail the actual repair
    return status_path


def _link_child_runs_to_parent(
    runs_dir: Path, child_run_ids: Iterable[str], *, plan_id: str, action_id: str, phase: str
) -> None:
    """Write a small link file into each child run dir so the watcher can
    find its parent repair campaign without needing to change status.json's
    own well-tested schema."""
    for run_id in child_run_ids:
        run_path = Path(runs_dir) / run_id
        link_path = run_path / "repair_link.json"
        try:
            run_path.mkdir(parents=True, exist_ok=True)
            link_path.write_text(json.dumps({
                "parent_status_type": "repair",
                "repair_plan_id": plan_id,
                "repair_action_id": action_id,
                "repair_phase": phase,
            }, indent=2, sort_keys=True))
        except Exception:
            pass


def apply_plan_with_live_status(
    client: Any,
    cfg: Any,
    plan: RepairPlan,
    *,
    judge_mode: str = "off",
    judge_model: Optional[str] = None,
    rankings_dir: Optional[Path] = None,
    live_ui: str = "off",
    ranking_scope: str = "canonical",
) -> Dict[str, Any]:
    """Apply an ordinary repair while publishing a live parent campaign.

    This covers non-KV capability and generation repairs. RC9 only linked
    children inside the managed KV cascade, so ordinary repairs still rendered
    as generic one-model benchmark runs.
    """
    runs_dir = Path(plan.runs_dir)
    action_index_by_id = {action.action_id: index for index, action in enumerate(plan.actions, start=1)}

    def emit(phase: str, **fields: Any) -> None:
        payload = {
            "phase": phase,
            "actions_total": len(plan.actions),
            "actions_completed": fields.pop("actions_completed", 0),
            "recovered": fields.pop("recovered", 0),
            "terminal_failures": fields.pop("terminal_failures", 0),
            "unresolved": fields.pop("unresolved", 0),
            "errors": fields.pop("errors", 0),
            "current_action_id": fields.pop("current_action_id", None),
            "current_action_index": fields.pop("current_action_index", None),
            "current_model": fields.pop("current_model", None),
            "current_task": fields.pop("current_task", None),
            "current_family": fields.pop("current_family", None),
            "current_child_run": fields.pop("current_child_run", None),
        }
        payload.update(fields)
        _write_repair_status(runs_dir, plan, payload)

    emit("planning")

    def action_started(action: RepairAction) -> None:
        emit(
            "probing_capability" if action.kind == "capability_gate" else "running_action",
            current_action_id=action.action_id,
            current_action_index=action_index_by_id.get(action.action_id),
            current_model=action.model,
            current_task=",".join(action.tasks), current_family=action.family,
            probe_state=("running" if action.kind == "capability_gate" else None),
            probe_detail=(f"functional {action.family} capability gate" if action.kind == "capability_gate" else None),
        )

    def child_started(action: RepairAction, child_run_id: str) -> None:
        emit(
            "running_action", current_action_id=action.action_id,
            current_action_index=action_index_by_id.get(action.action_id),
            current_model=action.model, current_task=",".join(action.tasks),
            current_family=action.family, current_child_run=child_run_id,
        )

    try:
        result = apply_plan(
            client, cfg, plan, judge_mode=judge_mode, judge_model=judge_model,
            rankings_dir=rankings_dir, parent_repair_plan_id=plan.plan_id,
            parent_repair_phase="standard", action_started_callback=action_started,
            child_started_callback=child_started, live_ui=live_ui, ranking_scope=ranking_scope,
        )
    except Exception as exc:
        emit("failed", errors=1, error=repr(exc))
        raise

    emit(
        "complete" if result.get("outcome") == "COMPLETE" else "partial",
        actions_completed=int(result.get("completed") or 0),
        recovered=int(result.get("recovered") or 0),
        terminal_failures=int(result.get("terminal_failures") or 0),
        unresolved=int(result.get("unresolved") or 0),
        errors=int(result.get("errors") or 0),
        outcome=result.get("outcome"),
        child_runs=result.get("child_runs") or [],
    )
    return result


def apply_plan_with_managed_kv_cascade(
    client: Any,
    cfg: Any,
    plan: RepairPlan,
    controller: Optional[Any] = None,
    *,
    controller_factory: Optional[Any] = None,
    auto_confirm: Optional[bool] = None,
    judge_mode: str = "off",
    judge_model: Optional[str] = None,
    rankings_dir: Optional[Path] = None,
    keep_final_kv: bool = False,
    live_ui: str = "off",
    ranking_scope: str = "canonical",
) -> Dict[str, Any]:
    """Apply repair with a current-first, bounded KV fallback policy.

    Standard repairs and guarded needle actions are attempted under the current
    service configuration first. Temporary q8/q4 service mutations are used only
    for needle actions still unresolved and not already known incompatible with
    quantized KV on the same model/runtime build. The original service state is
    restored after any mutation unless ``keep_final_kv`` is explicit.
    """
    all_needle_actions = [action for action in plan.actions if action.kind == "retry_needle_guarded"]
    other_actions = [action for action in plan.actions if action.kind != "retry_needle_guarded"]
    controller_box: List[Optional[Any]] = [controller]

    def get_controller() -> Any:
        if controller_box[0] is None:
            if controller_factory is None:
                raise RuntimeError("managed KV fallback requires an Ollama service controller")
            controller_box[0] = controller_factory()
        return controller_box[0]

    def controller_unit() -> Optional[str]:
        return getattr(controller_box[0], "unit", None)

    unattended_requested = bool(
        auto_confirm if auto_confirm is not None
        else getattr(controller_box[0], "auto_confirm", False)
    )
    result: Dict[str, Any] = {
        "plan_id": plan.plan_id,
        "repair_policy_version": plan.repair_policy_version,
        "mode": ("current_first_unattended_kv_cascade" if unattended_requested else "current_first_supervised_kv_cascade"),
        "auto_confirm": unattended_requested,
        "sudo_mode": ("noninteractive_nopasswd" if unattended_requested else "interactive_sudo"),
        "service_unit": controller_unit(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "phases": [],
        "service_events": [],
        "restored_original_service_state": False,
        "cascade_policy": "current -> unresolved-only q8_0 -> unresolved-only q4_0 -> restore if mutated",
    }
    actions_total = len(plan.actions)
    runs_dir_path = Path(plan.runs_dir)
    unattended = unattended_requested
    privilege_note = (
        "privileged commands use sudo -n under the scoped NOPASSWD rule; no password prompt is permitted"
        if unattended else "sudo owns the password prompt"
    )

    def emit_status(phase: str, **overrides: Any) -> None:
        payload = {
            "phase": phase,
            "actions_total": actions_total,
            "actions_completed": overrides.pop("actions_completed", None) or 0,
            "recovered": overrides.pop("recovered", None) or 0,
            "unresolved": overrides.pop("unresolved", None) or 0,
            "errors": overrides.pop("errors", None) or 0,
            "service_unit": controller_unit(),
            "requested_kv_type": overrides.pop("requested_kv_type", None),
            "effective_kv_type": overrides.pop("effective_kv_type", None),
            "observed_kv_type": overrides.pop("observed_kv_type", None),
            "service_verified": overrides.pop("service_verified", None),
            "current_action_id": overrides.pop("current_action_id", None),
            "current_model": overrides.pop("current_model", None),
            "current_task": overrides.pop("current_task", None),
            "current_child_run": overrides.pop("current_child_run", None),
            "restored_original_service_state": result.get("restored_original_service_state", False),
            "cascade_policy": result.get("cascade_policy"),
            "auto_confirm": result.get("auto_confirm"),
            "sudo_mode": result.get("sudo_mode"),
        }
        payload.update(overrides)
        _write_repair_status(runs_dir_path, plan, payload)

    emit_status("planning")

    # Preserve ordinary benchmark comparability: do not run text/tool/vision
    # repairs under a temporary KV service setting intended only for needle.
    standard_outcome = "COMPLETE"
    if other_actions:
        emit_status("running_standard_actions")
        standard_plan = _derived_plan(plan, other_actions, phase="standard", kv_type="current")
        standard_result = apply_plan(
            client, cfg, standard_plan, judge_mode=judge_mode,
            judge_model=judge_model, rankings_dir=None,
            parent_repair_plan_id=plan.plan_id, parent_repair_phase="standard",
            child_started_callback=lambda action, child: emit_status(
                "running_standard_action", current_action_id=action.action_id,
                current_model=action.model, current_task=",".join(action.tasks),
                current_child_run=child,
            ),
            live_ui=live_ui,
        )
        standard_outcome = str(standard_result.get("outcome") or "PARTIAL")
        result["phases"].append({
            "phase": "standard", "plan_id": standard_plan.plan_id,
            "result": standard_result,
        })
        _link_child_runs_to_parent(
            runs_dir_path, standard_result.get("child_runs") or [],
            plan_id=plan.plan_id, action_id="standard", phase="standard",
        )

    # Current/default KV is always attempted first. q8/q4 are memory fallbacks,
    # not the default path. This avoids unnecessary service mutation and catches
    # builds such as DeepSeek-Coder-v2 that work at 65k under current KV but
    # cannot initialize quantized V cache without Flash Attention.
    current_unresolved: set[str] = set()
    if all_needle_actions:
        emit_status("running_current_kv_actions", requested_kv_type="current")
        current_plan = _derived_plan(
            plan, all_needle_actions, phase="current", kv_type="current",
            server_inspection={"effective_kv_type": None, "verified": False,
                               "effective_source": "current/default service configuration"},
        )
        current_result = apply_plan(
            client, cfg, current_plan, judge_mode=judge_mode, judge_model=judge_model,
            parent_repair_plan_id=plan.plan_id, parent_repair_phase="current",
            child_started_callback=lambda action, child: emit_status(
                "running_current_kv_action", requested_kv_type="current",
                current_action_id=action.action_id, current_model=action.model,
                current_task=",".join(action.tasks), current_child_run=child,
            ),
            live_ui=live_ui,
        )
        result["phases"].append({"phase": "current", "plan_id": current_plan.plan_id, "result": current_result})
        current_unresolved = _unresolved_parent_action_ids(current_result, current_plan)
        current_recovered = {a.action_id for a in all_needle_actions} - set(current_unresolved)
        for action in all_needle_actions:
            source_run = Path(plan.runs_dir) / action.source_run_id
            _record_kv_compatibility(
                source_run, model=action.model, kv_type="current",
                status=("supported" if action.action_id in current_recovered else "inconclusive"),
                error_kinds=[], child_run_ids=list(current_result.get("child_runs") or []),
                detail=("current/default KV recovered the guarded needle action"
                        if action.action_id in current_recovered
                        else "current/default KV did not complete all required needle depths"),
                model_digest=action.model_digest,
                runtime_identity=_kv_runtime_identity(client, controller_box[0]),
            )
        emit_status(
            "current_kv_complete", requested_kv_type="current",
            actions_completed=int(current_result.get("completed") or 0),
            recovered=int(current_result.get("recovered") or 0),
            unresolved=int(current_result.get("unresolved") or 0),
            errors=int(current_result.get("errors") or 0),
        )

    # Do not ask for sudo, discover a service, or mutate systemd unless the
    # current/default KV phase genuinely left needle work unresolved.
    if not current_unresolved:
        if rankings_dir is not None:
            from . import rankings
            result["rankings"] = rankings.write_rankings(
                Path(plan.runs_dir), rankings_dir, force_rescan=True,
                include_separate=(ranking_scope == "separate"),
                only_run_ids=(result.get("child_runs") if ranking_scope == "separate" else None),
            )
        result["service_events"] = list(getattr(controller_box[0], "events", []) or [])
        result["unresolved_needle_parent_actions"] = []
        result["outcome"] = "COMPLETE" if standard_outcome == "COMPLETE" else "PARTIAL"
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        emit_status("complete", actions_completed=actions_total)
        return result

    controller = get_controller()
    result["service_unit"] = controller.unit
    runtime_identity = _kv_runtime_identity(client, controller)
    known_quantized_incompatible_ids = {
        action.action_id for action in all_needle_actions
        if action.action_id in current_unresolved
        and _known_quantized_kv_incompatible(action, Path(plan.runs_dir), runtime_identity)
    }
    blocked_current_unresolved = set(current_unresolved) & set(known_quantized_incompatible_ids)
    needle_actions = [
        action for action in all_needle_actions
        if action.action_id in current_unresolved and action.action_id not in known_quantized_incompatible_ids
    ]
    if not needle_actions:
        if rankings_dir is not None:
            from . import rankings
            result["rankings"] = rankings.write_rankings(
                Path(plan.runs_dir), rankings_dir, force_rescan=True,
                include_separate=(ranking_scope == "separate"),
                only_run_ids=(result.get("child_runs") if ranking_scope == "separate" else None),
            )
        result["service_events"] = list(controller.events)
        result["unresolved_needle_parent_actions"] = sorted(blocked_current_unresolved)
        result["outcome"] = (
            "COMPLETE" if standard_outcome == "COMPLETE" and not blocked_current_unresolved else "PARTIAL"
        )
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        emit_status("complete", actions_completed=actions_total)
        return result

    controller.require_supervised_tty()
    original_env = os.environ.get("OLLAMA_KV_CACHE_TYPE")
    service_changed = False
    unresolved_parent_ids = set(blocked_current_unresolved) | {action.action_id for action in needle_actions}
    try:
        emit_status("discovering_service", requested_kv_type="q8_0")
        controller.confirm(
            "q8_0",
            f"LLM ModelBench will install its dedicated temporary drop-in for {controller.unit}, "
            "restart Ollama, verify the live process, and run only guarded needle repairs. "
            + privilege_note + ".",
        )
        emit_status("waiting_for_q8_confirmation", requested_kv_type="q8_0")
        controller.authorise_sudo()
        emit_status("restarting_q8", requested_kv_type="q8_0")
        q8_service = controller.set_kv_type("q8_0", phase="q8_0")
        service_changed = True
        os.environ["OLLAMA_KV_CACHE_TYPE"] = "q8_0"
        q8_inspection = {
            "effective_kv_type": q8_service.observed_kv_type,
            "effective_source": f"supervised systemd restart:{controller.unit}",
            "verified": q8_service.verified,
            "systemd_unit": controller.unit,
        }
        emit_status(
            "verifying_q8", requested_kv_type="q8_0",
            effective_kv_type=q8_service.kv_type, observed_kv_type=q8_service.observed_kv_type,
            service_verified=q8_service.verified,
        )
        q8_plan = _derived_plan(plan, needle_actions, phase="q8_0", kv_type="q8_0", server_inspection=q8_inspection)
        emit_status(
            "running_q8_action", requested_kv_type="q8_0",
            effective_kv_type=q8_service.kv_type, observed_kv_type=q8_service.observed_kv_type,
            service_verified=q8_service.verified,
            current_action_id=(needle_actions[0].action_id if needle_actions else None),
            current_model=(needle_actions[0].model if needle_actions else None),
        )
        q8_result = apply_plan(
            client, cfg, q8_plan, judge_mode=judge_mode, judge_model=judge_model,
            parent_repair_plan_id=plan.plan_id, parent_repair_phase="q8_0",
            child_started_callback=lambda action, child: emit_status(
                "running_q8_action", requested_kv_type="q8_0",
                effective_kv_type=q8_service.kv_type, observed_kv_type=q8_service.observed_kv_type,
                service_verified=q8_service.verified, current_action_id=action.action_id,
                current_model=action.model, current_task=",".join(action.tasks),
                current_child_run=child,
            ),
            live_ui=live_ui,
        )
        result["phases"].append({"phase": "q8_0", "plan_id": q8_plan.plan_id, "result": q8_result})
        _link_child_runs_to_parent(
            runs_dir_path, q8_result.get("child_runs") or [],
            plan_id=plan.plan_id, action_id="q8_0", phase="q8_0",
        )
        q8_unresolved = _unresolved_parent_action_ids(q8_result, q8_plan)
        q8_evidence = _phase_action_error_evidence(q8_result, runs_dir_path)
        flash_incompatible_parent_ids: set[str] = set()
        for action in q8_plan.actions:
            evidence = q8_evidence.get(action.action_id) or {}
            if not evidence.get("quantized_kv_incompatible"):
                continue
            parent_id = str((action.details or {}).get("parent_action_id") or action.action_id)
            flash_incompatible_parent_ids.add(parent_id)
            source_run = Path(plan.runs_dir) / action.source_run_id
            _record_kv_compatibility(
                source_run, model=action.model, kv_type="q8_0", status="incompatible",
                error_kinds=list(evidence.get("error_kinds") or []),
                child_run_ids=list(evidence.get("child_run_ids") or []),
                detail=str(evidence.get("detail") or ""),
                model_digest=action.model_digest,
                runtime_identity=runtime_identity,
            )
        unresolved_parent_ids = set(blocked_current_unresolved) | set(q8_unresolved)
        emit_status(
            "q8_complete", requested_kv_type="q8_0",
            effective_kv_type=q8_service.kv_type, observed_kv_type=q8_service.observed_kv_type,
            service_verified=q8_service.verified,
            actions_completed=int(q8_result.get("completed") or 0),
            recovered=int(q8_result.get("recovered") or 0),
            unresolved=int(q8_result.get("unresolved") or 0),
            errors=int(q8_result.get("errors") or 0),
        )

        q4_candidate_ids = set(unresolved_parent_ids) - set(blocked_current_unresolved) - set(flash_incompatible_parent_ids)
        if q4_candidate_ids:
            q4_actions = [action for action in needle_actions if action.action_id in q4_candidate_ids]
            emit_status(
                "waiting_for_q4_confirmation", requested_kv_type="q4_0",
                unresolved=len(unresolved_parent_ids),
            )
            controller.confirm(
                "q4_0",
                f"{len(q4_actions)} guarded needle repair action(s) remain unresolved after q8_0. "
                f"LLM ModelBench will change only its temporary drop-in for {controller.unit}, restart "
                "Ollama, verify q4_0, and retry only those unresolved actions. " + privilege_note + ".",
            )
            controller.authorise_sudo()
            q4_service = controller.set_kv_type("q4_0", phase="q4_0")
            os.environ["OLLAMA_KV_CACHE_TYPE"] = "q4_0"
            q4_inspection = {
                "effective_kv_type": q4_service.observed_kv_type,
                "effective_source": f"supervised systemd restart:{controller.unit}",
                "verified": q4_service.verified,
                "systemd_unit": controller.unit,
            }
            emit_status(
                "running_q4_action", requested_kv_type="q4_0",
                effective_kv_type=q4_service.kv_type, observed_kv_type=q4_service.observed_kv_type,
                service_verified=q4_service.verified,
                current_action_id=(q4_actions[0].action_id if q4_actions else None),
                current_model=(q4_actions[0].model if q4_actions else None),
            )
            q4_plan = _derived_plan(plan, q4_actions, phase="q4_0", kv_type="q4_0", server_inspection=q4_inspection)
            q4_result = apply_plan(
                client, cfg, q4_plan, judge_mode=judge_mode, judge_model=judge_model,
                parent_repair_plan_id=plan.plan_id, parent_repair_phase="q4_0",
                child_started_callback=lambda action, child: emit_status(
                    "running_q4_action", requested_kv_type="q4_0",
                    effective_kv_type=q4_service.kv_type, observed_kv_type=q4_service.observed_kv_type,
                    service_verified=q4_service.verified, current_action_id=action.action_id,
                    current_model=action.model, current_task=",".join(action.tasks),
                    current_child_run=child,
                ),
                live_ui=live_ui,
            )
            result["phases"].append({"phase": "q4_0", "plan_id": q4_plan.plan_id, "result": q4_result})
            _link_child_runs_to_parent(
                runs_dir_path, q4_result.get("child_runs") or [],
                plan_id=plan.plan_id, action_id="q4_0", phase="q4_0",
            )
            unresolved_parent_ids = (
                set(blocked_current_unresolved) | set(flash_incompatible_parent_ids)
                | _unresolved_parent_action_ids(q4_result, q4_plan)
            )
            emit_status(
                "q4_complete", requested_kv_type="q4_0",
                effective_kv_type=q4_service.kv_type, observed_kv_type=q4_service.observed_kv_type,
                service_verified=q4_service.verified,
                actions_completed=int(q4_result.get("completed") or 0),
                recovered=int(q4_result.get("recovered") or 0),
                unresolved=int(q4_result.get("unresolved") or 0),
                errors=int(q4_result.get("errors") or 0),
            )

        if keep_final_kv:
            result["restored_original_service_state"] = False
            result["service_left_at"] = os.environ.get("OLLAMA_KV_CACHE_TYPE")
        else:
            emit_status("waiting_for_restore_confirmation")
            controller.confirm(
                "restore",
                f"Needle cascade finished. LLM ModelBench will restore the pre-existing {controller.unit} "
                "drop-in state and restart Ollama once more. " + privilege_note + ".",
            )
            controller.authorise_sudo()
            emit_status("restoring")
            controller.restore()
            service_changed = False
            result["restored_original_service_state"] = True
    except Exception:
        # A failed phase must not silently leave a benchmark-owned service
        # override behind. Use the supervised sudo path for best-effort
        # restoration, then re-raise the original error.
        if (service_changed or bool(getattr(controller, "mutation_started", False))) and not keep_final_kv:
            try:
                emit_status("restoring_after_error")
                controller.authorise_sudo()
                controller.restore(phase="restore_after_error")
                result["restored_original_service_state"] = True
                service_changed = False
            except Exception as restore_exc:
                result["restore_error"] = repr(restore_exc)
        emit_status("failed", restored_original_service_state=result.get("restored_original_service_state", False))
        raise
    finally:
        if original_env is None:
            os.environ.pop("OLLAMA_KV_CACHE_TYPE", None)
        else:
            os.environ["OLLAMA_KV_CACHE_TYPE"] = original_env
        result["service_events"] = list(controller.events)

    if rankings_dir is not None:
        emit_status("refreshing_rankings")
        from . import rankings
        result["rankings"] = rankings.write_rankings(
            Path(plan.runs_dir), rankings_dir, force_rescan=True,
            include_separate=(ranking_scope == "separate"),
            only_run_ids=(result.get("child_runs") if ranking_scope == "separate" else None),
        )

    result["unresolved_needle_parent_actions"] = sorted(unresolved_parent_ids)
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    result["outcome"] = (
        "COMPLETE" if standard_outcome == "COMPLETE" and not unresolved_parent_ids
        else "PARTIAL"
    )
    emit_status(
        "complete",
        actions_completed=actions_total - len(unresolved_parent_ids),
        unresolved=len(unresolved_parent_ids),
        restored_original_service_state=result.get("restored_original_service_state", False),
    )
    return result
