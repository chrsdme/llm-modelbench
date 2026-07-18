"""Campaign workspace: manifest, path resolver, and state machine.

This is the RC19.1 foundation. One campaign = one directory,
``campaigns/<campaign_id>/``, containing everything that campaign ever
produces -- plan, primary evidence, recovery evidence, judge results,
candidate rankings, reports, logs, and the final review package. Nothing a
campaign produces should ever land outside its own directory.

Legacy ``runs/<run_id>/`` layouts are still readable (see
``is_legacy_run_dir``/``legacy_run_dir``) so existing evidence is not
orphaned by this change, but every *new* campaign uses this layout only.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import hashlib
import zipfile
import tempfile
import socket
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

CAMPAIGNS_ROOT = Path("campaigns")
MANIFEST_SCHEMA_VERSION = 1

_CAMPAIGN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_RESUMABLE_STATES = ("generating", "recovering", "judging")


class CampaignError(RuntimeError):
    """Raised for invalid campaign IDs, manifests, or state transitions.

    These errors are never silently swallowed. Callers must decide whether
    the campaign should stop, be resumed, or be classified as failed.
    """


def _inside(root: Path, candidate: Path) -> Path:
    """Return *candidate* only when it cannot escape the campaign root."""
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve()
    if not candidate_resolved.is_relative_to(root_resolved):
        raise CampaignError(f"campaign-managed path escapes campaign root: {candidate}")
    return candidate


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

STATES = (
    "created",
    "planned",
    "generating",
    "recovering",
    "judging",
    "packaged",
    "interrupted",
    "failed",
    "accepted",
    "rejected",
    "archived_diagnostic",
)

# These are publication/lifecycle terminal states. ``failed`` is deliberately
# not included: failed evidence must still be explicitly rejected or archived
# before cleanup or publication workflows may treat the campaign as closed.
TERMINAL_STATES = ("accepted", "rejected", "archived_diagnostic")

_ALLOWED_TRANSITIONS: Dict[str, tuple[str, ...]] = {
    "created": ("planned", "failed"),
    "planned": ("generating", "failed"),
    "generating": (
        "recovering",
        "judging",
        "packaged",
        "interrupted",
        "failed",
    ),
    "recovering": (
        "judging",
        "packaged",
        "interrupted",
        "failed",
    ),
    "judging": (
        "packaged",
        "interrupted",
        "failed",
    ),
    "packaged": ("accepted", "rejected", "archived_diagnostic"),
    # The general table lists all resumable phases. ``transition`` applies the
    # stronger rule that an interrupted campaign may resume only the exact
    # phase recorded in ``resume_state``.
    "interrupted": ("generating", "recovering", "judging", "failed"),
    "failed": ("rejected", "archived_diagnostic"),
    "accepted": (),
    "rejected": (),
    "archived_diagnostic": (),
}


def is_valid_transition(current: str, target: str) -> bool:
    if current not in STATES:
        raise CampaignError(f"unknown campaign state: {current!r}")
    if target not in STATES:
        raise CampaignError(f"unknown campaign state: {target!r}")
    return target in _ALLOWED_TRANSITIONS.get(current, ())


def is_terminal(state: str) -> bool:
    return state in TERMINAL_STATES


# ---------------------------------------------------------------------------
# Campaign ID validation
# ---------------------------------------------------------------------------

def validate_campaign_id(campaign_id: str) -> str:
    """Validate an identifier that will become a directory name."""
    if not campaign_id or not _CAMPAIGN_ID_RE.fullmatch(campaign_id):
        raise CampaignError(
            f"invalid campaign id {campaign_id!r}: only letters, digits, '.', '_', '-' are allowed"
        )
    if ".." in campaign_id:
        raise CampaignError(f"invalid campaign id {campaign_id!r}: must not contain '..'")
    return campaign_id


# ---------------------------------------------------------------------------
# Path resolver
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CampaignPaths:
    """Every path a campaign writes to, resolved once in one place."""

    campaign_id: str
    root: Path

    # Manifest is the single source of truth for state and state history.
    manifest: Path

    plan_dir: Path
    plan_json: Path
    inventory_json: Path
    capabilities_json: Path

    evidence_dir: Path
    primary_dir: Path
    primary_raw_results: Path
    primary_run_validity: Path
    primary_dumps_dir: Path

    recovery_dir: Path
    recovery_plan: Path
    recovery_result: Path
    recovery_attempts: Path
    recovery_children_dir: Path

    judge_dir: Path
    judge_results: Path
    judge_summary: Path

    rankings_dir: Path
    candidate_rankings_dir: Path

    reports_dir: Path
    model_cards_dir: Path

    logs_dir: Path
    campaign_log: Path

    packages_dir: Path
    checksums_dir: Path
    checksums_json: Path
    readiness_dir: Path
    readiness_json: Path
    adoption_dir: Path
    adoption_record: Path
    lock_file: Path


def campaign_root(
    campaign_id: str,
    *,
    campaigns_root: Path = CAMPAIGNS_ROOT,
) -> Path:
    validate_campaign_id(campaign_id)
    return campaigns_root / campaign_id


def resolve_paths(
    campaign_id: str,
    *,
    campaigns_root: Path = CAMPAIGNS_ROOT,
) -> CampaignPaths:
    root = campaign_root(campaign_id, campaigns_root=campaigns_root)
    plan_dir = root / "plan"
    evidence_dir = root / "evidence"
    primary_dir = evidence_dir / "primary"
    recovery_dir = evidence_dir / "recovery"
    judge_dir = evidence_dir / "judge"
    rankings_dir = root / "rankings"
    reports_dir = root / "reports"
    logs_dir = root / "logs"
    packages_dir = root / "packages"
    checksums_dir = root / "checksums"
    readiness_dir = root / "readiness"
    adoption_dir = root / "adoption"

    paths = CampaignPaths(
        campaign_id=campaign_id,
        root=root,
        manifest=root / "manifest.json",
        plan_dir=plan_dir,
        plan_json=plan_dir / "plan.json",
        inventory_json=plan_dir / "inventory.json",
        capabilities_json=plan_dir / "capabilities.json",
        evidence_dir=evidence_dir,
        primary_dir=primary_dir,
        primary_raw_results=primary_dir / "raw_results.jsonl",
        primary_run_validity=primary_dir / "run_validity.json",
        primary_dumps_dir=primary_dir / "dumps",
        recovery_dir=recovery_dir,
        recovery_plan=recovery_dir / "recovery_plan.json",
        recovery_result=recovery_dir / "recovery_result.json",
        recovery_attempts=recovery_dir / "recovery_attempts.jsonl",
        recovery_children_dir=recovery_dir / "children",
        judge_dir=judge_dir,
        judge_results=judge_dir / "judge_results.jsonl",
        judge_summary=judge_dir / "judge_summary.json",
        rankings_dir=rankings_dir,
        candidate_rankings_dir=rankings_dir / "candidate",
        reports_dir=reports_dir,
        model_cards_dir=reports_dir / "model_cards",
        logs_dir=logs_dir,
        campaign_log=logs_dir / "campaign.log",
        packages_dir=packages_dir,
        checksums_dir=checksums_dir,
        checksums_json=checksums_dir / "sha256.json",
        readiness_dir=readiness_dir,
        readiness_json=readiness_dir / "summary.json",
        adoption_dir=adoption_dir,
        adoption_record=adoption_dir / "adoption.json",
        lock_file=root / ".campaign.lock",
    )
    for value in paths.__dict__.values():
        if isinstance(value, Path):
            _inside(root, value)
    return paths


def create_campaign_dirs(paths: CampaignPaths) -> None:
    """Create the complete campaign directory tree up front."""
    for directory in (
        paths.root,
        paths.plan_dir,
        paths.evidence_dir,
        paths.primary_dir,
        paths.primary_dumps_dir,
        paths.recovery_dir,
        paths.recovery_children_dir,
        paths.judge_dir,
        paths.rankings_dir,
        paths.candidate_rankings_dir,
        paths.reports_dir,
        paths.model_cards_dir,
        paths.logs_dir,
        paths.packages_dir,
        paths.checksums_dir,
        paths.readiness_dir,
        paths.adoption_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Legacy runs/<run_id>/ compatibility
# ---------------------------------------------------------------------------

def is_legacy_run_dir(run_dir: Path) -> bool:
    """Return whether *run_dir* uses the pre-campaign layout."""
    return (
        (run_dir / "raw_results.jsonl").exists()
        and not (run_dir / "manifest.json").exists()
    )


def legacy_run_dir(
    run_id: str,
    *,
    runs_dir: Path = Path("runs"),
) -> Path:
    return runs_dir / run_id


def owning_campaign_path(path: Path, *, campaigns_root: Path = CAMPAIGNS_ROOT) -> Optional[CampaignPaths]:
    """Resolve a nested campaign path to its owner, refusing malformed roots."""
    target = path.resolve()
    root = campaigns_root.resolve()
    if not target.is_relative_to(root):
        return None
    relative = target.relative_to(root)
    if not relative.parts:
        return None
    campaign_id = relative.parts[0]
    paths = resolve_paths(campaign_id, campaigns_root=campaigns_root)
    if not paths.root.exists() or not paths.manifest.exists():
        return None
    _inside(paths.root, target)
    load_manifest(paths)
    return paths


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class CampaignManifest:
    campaign_id: str
    created_at: str
    version: str
    schema_version: int = MANIFEST_SCHEMA_VERSION
    models: List[str] = field(default_factory=list)
    level: str = "full"
    judge_model: Optional[str] = None
    state: str = "created"
    resume_state: Optional[str] = None
    state_history: List[Dict[str, str]] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def new(
        cls,
        campaign_id: str,
        *,
        models: List[str],
        level: str = "full",
        version: str = "",
    ) -> "CampaignManifest":
        validate_campaign_id(campaign_id)
        now = datetime.now(timezone.utc).isoformat()
        manifest = cls(
            campaign_id=campaign_id,
            created_at=now,
            version=version,
            models=list(models),
            level=level,
        )
        manifest.state_history.append({"state": "created", "at": now})
        return manifest


def _atomic_write_text(path: Path, text: str) -> None:
    """Write, flush, fsync, and atomically replace *path*.

    A failed write or replace leaves the previous manifest intact and removes
    the temporary file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_manifest(
    paths: CampaignPaths,
    manifest: CampaignManifest,
) -> None:
    _atomic_write_text(
        paths.manifest,
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
    )


def write_campaign_plan(paths: CampaignPaths, plan: Dict[str, Any], *, inventory: List[Dict[str, Any]], capabilities: Dict[str, Any], configuration: Dict[str, Any]) -> Dict[str, Any]:
    """Persist the accepted pre-generation contract atomically."""
    from .runner import _task_hash
    from .tasks import TASKS
    task_map = {task.id: _task_hash(task) for task in TASKS}
    accepted = dict(plan)
    accepted.update({
        "campaign_id": paths.campaign_id,
        "generation_judge_mode": "off",
        "task_hashes": {task_id: task_map[task_id] for model in plan.get("active_models", []) for task_id in model.get("tasks", []) if task_id in task_map},
        "configuration": configuration,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recovery_policy_version": RECOVERY_POLICY_VERSION if "RECOVERY_POLICY_VERSION" in globals() else "pending",
    })
    _atomic_write_text(paths.plan_json, json.dumps(accepted, indent=2, sort_keys=True))
    _atomic_write_text(paths.inventory_json, json.dumps(inventory, indent=2, sort_keys=True))
    _atomic_write_text(paths.capabilities_json, json.dumps(capabilities, indent=2, sort_keys=True))
    return accepted


def load_manifest(paths: CampaignPaths) -> CampaignManifest:
    if not paths.manifest.exists():
        raise CampaignError(
            f"no manifest at {paths.manifest}; "
            f"is {paths.campaign_id!r} a real campaign?"
        )
    try:
        data = json.loads(paths.manifest.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("manifest must be a JSON object")
        known = set(CampaignManifest.__dataclass_fields__)
        manifest = CampaignManifest(**{key: value for key, value in data.items() if key in known})
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CampaignError(
            f"invalid campaign manifest at {paths.manifest}: {exc}"
        ) from exc

    validate_campaign_id(manifest.campaign_id)
    if not isinstance(manifest.schema_version, int) or manifest.schema_version < 1:
        raise CampaignError("manifest contains an invalid schema_version")
    if not isinstance(manifest.models, list) or not all(isinstance(model, str) for model in manifest.models):
        raise CampaignError("manifest models must be a list of strings")
    if not isinstance(manifest.state_history, list) or not all(isinstance(item, dict) for item in manifest.state_history):
        raise CampaignError("manifest state_history must be a list of objects")
    if manifest.campaign_id != paths.campaign_id:
        raise CampaignError(
            f"manifest campaign id {manifest.campaign_id!r} does not match "
            f"resolved campaign {paths.campaign_id!r}"
        )
    if manifest.state not in STATES:
        raise CampaignError(
            f"manifest contains unknown campaign state: {manifest.state!r}"
        )
    if manifest.state == "interrupted":
        if manifest.resume_state not in _RESUMABLE_STATES:
            raise CampaignError(
                "interrupted campaign manifest must record a valid resume_state"
            )
    elif manifest.resume_state is not None:
        raise CampaignError(
            f"campaign state {manifest.state!r} must not retain resume_state "
            f"{manifest.resume_state!r}"
        )
    return manifest


# ---------------------------------------------------------------------------
# Campaign lock
# ---------------------------------------------------------------------------

def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_lock(paths: CampaignPaths) -> Optional[Dict[str, Any]]:
    if not paths.lock_file.exists():
        return None
    try:
        value = json.loads(paths.lock_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CampaignError(f"invalid campaign lock at {paths.lock_file}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("pid"), int) or not isinstance(value.get("hostname"), str):
        raise CampaignError(f"invalid campaign lock at {paths.lock_file}")
    return value


def lock_is_stale(lock: Dict[str, Any]) -> bool:
    """Prove stale only for a dead PID on this host; remote locks stay active."""
    return lock.get("hostname") == socket.gethostname() and not _pid_is_running(lock["pid"])


def acquire_lock(paths: CampaignPaths, *, operation: str, phase: str = "") -> Dict[str, Any]:
    """Atomically acquire a lock without deleting an active or ambiguous lock."""
    existing = read_lock(paths)
    if existing is not None:
        if not lock_is_stale(existing):
            raise CampaignError(f"campaign is locked by pid {existing['pid']} on {existing['hostname']}")
        # A same-host dead PID is proof enough to replace a stale task lock.
        paths.lock_file.unlink()
    record = {
        "pid": os.getpid(), "hostname": socket.gethostname(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": operation, "phase": phase,
    }
    try:
        fd = os.open(paths.lock_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise CampaignError("campaign lock was acquired concurrently") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    return record


def release_lock(paths: CampaignPaths, lock: Dict[str, Any]) -> None:
    current = read_lock(paths)
    if current is None:
        return
    if current != lock:
        raise CampaignError("refusing to release a lock owned by another operation")
    paths.lock_file.unlink()


def transition(
    paths: CampaignPaths,
    manifest: CampaignManifest,
    target: str,
) -> CampaignManifest:
    """Return and persist a validated new manifest state.

    The input manifest is not mutated. Therefore a failed atomic write leaves
    both the persisted manifest and the caller's in-memory object unchanged.
    """
    current = manifest.state
    if not is_valid_transition(current, target):
        raise CampaignError(
            f"illegal campaign state transition: {current!r} -> {target!r} "
            f"(allowed from {current!r}: "
            f"{_ALLOWED_TRANSITIONS.get(current, ())})"
        )

    if current == "interrupted" and target != "failed":
        if manifest.resume_state not in _RESUMABLE_STATES:
            raise CampaignError(
                "interrupted campaign has no valid recorded resume_state"
            )
        if target != manifest.resume_state:
            raise CampaignError(
                f"interrupted campaign must resume {manifest.resume_state!r}, "
                f"not {target!r}"
            )

    now = datetime.now(timezone.utc).isoformat()
    next_resume_state = manifest.resume_state
    history_entry: Dict[str, str] = {"state": target, "at": now}

    if target == "interrupted":
        if current not in _RESUMABLE_STATES:
            raise CampaignError(
                f"campaign state {current!r} cannot be interrupted"
            )
        next_resume_state = current
        history_entry["resume_state"] = current
    elif current == "interrupted":
        next_resume_state = None

    updated = replace(
        manifest,
        state=target,
        resume_state=next_resume_state,
        state_history=[*manifest.state_history, history_entry],
    )
    write_manifest(paths, updated)
    return updated


def create_campaign(
    campaign_id: str,
    *,
    models: List[str],
    level: str = "full",
    version: str = "",
    campaigns_root: Path = CAMPAIGNS_ROOT,
) -> tuple[CampaignPaths, CampaignManifest]:
    """Create a new campaign and refuse non-empty ID reuse."""
    paths = resolve_paths(campaign_id, campaigns_root=campaigns_root)
    if paths.root.exists() and any(paths.root.iterdir()):
        raise CampaignError(
            f"campaign directory already exists and is not empty: "
            f"{paths.root}; use a new campaign id to preserve prior evidence"
        )
    create_campaign_dirs(paths)
    manifest = CampaignManifest.new(
        campaign_id,
        models=models,
        level=level,
        version=version,
    )
    write_manifest(paths, manifest)
    return paths, manifest


def sync_primary_reports(paths: CampaignPaths) -> List[Path]:
    """Publish report views under the campaign report root without moving evidence."""
    copied: List[Path] = []
    for source in paths.primary_dir.iterdir():
        if source.is_file() and source.name in {
            "report.html", "scorecard.md", "scorecard.csv", "routing.md",
            "prune.md", "clones.md", "regression.md", "summary.json",
        }:
            target = paths.reports_dir / source.name
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_campaign(paths: CampaignPaths, *, allow_active_lock: bool = False) -> Path:
    """Create one self-contained review zip and a verifiable source inventory."""
    manifest = load_manifest(paths)
    if manifest.state not in (*TERMINAL_STATES, "packaged"):
        raise CampaignError("only packaged or terminal campaigns may be packaged")
    if read_lock(paths) is not None and not allow_active_lock:
        raise CampaignError("refusing to package an actively locked campaign")
    package = paths.packages_dir / f"{paths.campaign_id}-review.zip"
    inventory: Dict[str, str] = {}
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(paths.root.rglob("*")):
            if not source.is_file() or source == package or paths.packages_dir in source.parents or source == paths.lock_file:
                continue
            relative = source.relative_to(paths.root).as_posix()
            archive.write(source, relative)
            inventory[relative] = _sha256(source)
    _atomic_write_text(paths.checksums_json, json.dumps({"files": inventory, "package": _sha256(package)}, indent=2, sort_keys=True))
    return package


def verify_package(paths: CampaignPaths) -> bool:
    if not paths.checksums_json.exists():
        return False
    data = json.loads(paths.checksums_json.read_text(encoding="utf-8"))
    package = paths.packages_dir / f"{paths.campaign_id}-review.zip"
    return package.exists() and data.get("package") == _sha256(package) and zipfile.is_zipfile(package)


def cleanup_campaign(paths: CampaignPaths, *, apply: bool = False) -> List[Path]:
    """Retention cleanup is deliberately conservative: only disposable dumps qualify."""
    manifest = load_manifest(paths)
    if manifest.state not in TERMINAL_STATES:
        raise CampaignError("cleanup requires an accepted, rejected, or archived campaign")
    if read_lock(paths) is not None or not verify_package(paths):
        raise CampaignError("cleanup requires an unlocked campaign with verified package")
    candidates = [paths.primary_dumps_dir]
    if apply:
        for target in candidates:
            if target.exists():
                shutil.rmtree(target)
    return candidates


def migrate_legacy_run(run_id: str, campaign_id: str, *, runs_dir: Path = Path("runs"), campaigns_root: Path = CAMPAIGNS_ROOT, apply: bool = False) -> CampaignPaths:
    """Copy a legacy run into isolated primary evidence; source evidence is untouched."""
    source = legacy_run_dir(run_id, runs_dir=runs_dir)
    if not is_legacy_run_dir(source):
        raise CampaignError(f"legacy source is not a readable run: {source}")
    paths = resolve_paths(campaign_id, campaigns_root=campaigns_root)
    if paths.root.exists() and any(paths.root.iterdir()):
        raise CampaignError(f"campaign migration target already exists: {paths.root}")
    if not apply:
        return paths
    paths, manifest = create_campaign(campaign_id, models=[], version="legacy-migration", campaigns_root=campaigns_root)
    shutil.copytree(source, paths.primary_dir, dirs_exist_ok=True)
    manifest.notes["legacy_source"] = str(source)
    write_manifest(paths, manifest)
    return paths


# ---------------------------------------------------------------------------
# Unattended methodology policy (RC19.2)
# ---------------------------------------------------------------------------

RECOVERY_POLICY_VERSION = "rc19.2.1"
TERMINAL_DISPOSITIONS = {
    "scored", "judged", "confirmed_capability_unavailable",
    "capability_measured_failure", "environment_limited", "operator_excluded",
    "terminal_model_failure", "harness_failure", "awaiting_external_judge",
}


def visible_answer(row: Dict[str, Any]) -> bool:
    """A scored row, including score zero, is immutable evidence not a retry cue."""
    return row.get("score") is not None and not row.get("error_kind")


def recovery_profiles(default_budget: int, *, allow_extended: bool = False) -> List[Dict[str, Any]]:
    budget = max(1, int(default_budget or 2048))
    profiles = [
        {"attempt": 1, "num_predict": budget, "think": "off"},
        {"attempt": 2, "num_predict": max(budget * 2, 4096), "think": "off"},
    ]
    if allow_extended:
        profiles.append({"attempt": 3, "num_predict": max(budget * 4, 8192), "think": "off"})
    return profiles


def classify_recovery_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a primary cell without score fishing or host remediation."""
    if visible_answer(row):
        return {"disposition": "scored", "retry": False, "reason": "visible scorable answer"}
    kind = str(row.get("error_kind") or "")
    text = str(row.get("error") or row.get("reason") or "").lower()
    if kind == "thinking_only":
        return {"disposition": "thinking_only_pending_retry", "retry": True, "reason": kind}
    if kind == "empty_output":
        return {"disposition": "empty_output_pending_retry", "retry": True, "reason": kind}
    if "timeout" in text or " 5" in text or "http" in text and "5" in text:
        return {"disposition": "transient_retry_pending", "retry": True, "reason": "transient transport failure"}
    if kind == "harness_error":
        return {"disposition": "harness_failure", "retry": False, "reason": kind}
    if kind:
        return {"disposition": "terminal_model_failure", "retry": False, "reason": kind}
    return {"disposition": "harness_failure", "retry": False, "reason": "unscorable row without error classification"}


def write_readiness(paths: CampaignPaths, rows: List[Dict[str, Any]], *, judge_available: bool = True) -> Dict[str, Any]:
    dispositions = [str(row.get("disposition") or classify_recovery_row(row)["disposition"]) for row in rows]
    pending = [item for item in dispositions if item not in TERMINAL_DISPOSITIONS]
    if pending:
        state = "not_ready_manual_items"
    elif "harness_failure" in dispositions:
        state = "not_ready_harness_failure"
    elif "awaiting_external_judge" in dispositions or not judge_available:
        state = "not_ready_external_judge"
    else:
        state = "ready_for_adoption"
    summary = {"campaign_id": paths.campaign_id, "readiness": state, "rows": len(rows),
               "terminal_rows": len(rows) - len(pending), "pending_dispositions": pending,
               "policy_version": RECOVERY_POLICY_VERSION}
    _atomic_write_text(paths.readiness_json, json.dumps(summary, indent=2, sort_keys=True))
    return summary


CAPABILITY_STATES = {
    "confirmed_supported", "confirmed_unavailable", "responded_contract_failed",
    "capability_measured_failure", "probe_unresolved_transient", "environment_limited",
    "operator_excluded", "conflicting_evidence/manual_review",
}


def classify_capability_probe(probes: List[Dict[str, Any]]) -> str:
    """Resolve bounded functional probes without mistaking transport failure for absence."""
    if not probes:
        return "probe_unresolved_transient"
    states = {str(item.get("state") or "") for item in probes}
    if "operator_excluded" in states:
        return "operator_excluded"
    if "environment_limited" in states:
        return "environment_limited"
    if "supported" in states and "unavailable" in states:
        return "conflicting_evidence/manual_review"
    if "supported" in states:
        return "confirmed_supported"
    if "unavailable" in states:
        return "confirmed_unavailable"
    if "responded_contract_failed" in states:
        return "responded_contract_failed"
    if "measured_failure" in states:
        return "capability_measured_failure"
    return "probe_unresolved_transient"


def select_campaign_judge(inventory: List[Dict[str, Any]], cohort: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Select an installed text judge outside the tested cohort deterministically."""
    cohort_names = {str(item.get("name") or item.get("model") or "") for item in cohort}
    cohort_digests = {str(item.get("digest") or item.get("model_digest_resolved") or "") for item in cohort}
    cohort_families = [str(item.get("architecture_family") or "") for item in cohort]
    majority = max(set(cohort_families), key=cohort_families.count) if cohort_families else ""
    candidates = []
    for item in inventory:
        name, digest = str(item.get("name") or ""), str(item.get("digest") or "")
        families = item.get("supported_families") or item.get("families") or []
        if name in cohort_names or (digest and digest in cohort_digests) or "text" not in families:
            continue
        if int(item.get("context") or item.get("context_length") or 0) and int(item.get("context") or item.get("context_length")) < 1024:
            continue
        candidates.append(item)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (
        not bool(item.get("calibrated")),
        str(item.get("architecture_family") or "") == majority,
        -int(item.get("priority") or 0), str(item.get("name") or ""),
    ))[0]


def adopt_campaign(paths: CampaignPaths, *, rankings_dir: Path, dry_run: bool = True) -> Dict[str, Any]:
    """Transactionally overlay validated candidate rows into canonical rankings."""
    manifest = load_manifest(paths)
    readiness = json.loads(paths.readiness_json.read_text()) if paths.readiness_json.exists() else {}
    if readiness.get("readiness") != "ready_for_adoption":
        raise CampaignError("campaign is not ready_for_adoption")
    if not verify_package(paths):
        raise CampaignError("campaign package checksums do not verify")
    candidate_raw = paths.candidate_rankings_dir / "master_raw.jsonl"
    if not candidate_raw.exists():
        raise CampaignError("candidate rankings evidence is missing")
    def read_rows(path: Path) -> List[Dict[str, Any]]:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    current = read_rows(rankings_dir / "master_raw.jsonl")
    incoming = read_rows(candidate_raw)
    for row in incoming:
        row["ranking_scope"] = "canonical"
        row["canonical_rankings"] = True
        row["campaign_id"] = manifest.campaign_id
    index = {(str(row.get("run_id")), str(row.get("_source_signature"))): row for row in current}
    added = replaced = 0
    for row in incoming:
        key = (str(row.get("run_id")), str(row.get("_source_signature")))
        previous = index.get(key)
        if previous == row:
            continue
        if previous is not None:
            current.remove(previous); replaced += 1
        current.append(row); index[key] = row; added += 1
    preview = {"campaign_id": manifest.campaign_id, "rows_incoming": len(incoming), "rows_added_or_updated": added,
               "rows_replaced": replaced, "dry_run": bool(dry_run)}
    if dry_run:
        return preview
    parent = rankings_dir.parent
    temp = Path(tempfile.mkdtemp(prefix=".campaign-adopt-", dir=str(parent)))
    try:
        if rankings_dir.exists():
            shutil.copytree(rankings_dir, temp, dirs_exist_ok=True)
        raw = temp / "master_raw.jsonl"
        _atomic_write_text(raw, "".join(json.dumps(row, sort_keys=True) + "\n" for row in current))
        from . import rankings
        rankings.write_rankings(temp / "no-runs", temp, force_rescan=True)
        backup = parent / f".rankings-backup-{paths.campaign_id}"
        if backup.exists():
            raise CampaignError(f"adoption backup path already exists: {backup}")
        if rankings_dir.exists():
            os.replace(rankings_dir, backup)
        os.replace(temp, rankings_dir)
        if backup.exists():
            shutil.rmtree(backup)
        _atomic_write_text(paths.adoption_record, json.dumps({**preview, "adopted_at": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True))
        transition(paths, manifest, "accepted")
        return preview
    except BaseException:
        if not rankings_dir.exists() and 'backup' in locals() and backup.exists():
            os.replace(backup, rankings_dir)
        raise
    finally:
        if temp.exists():
            shutil.rmtree(temp)
