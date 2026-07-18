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
    effective_rows: Path

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
        effective_rows=evidence_dir / "effective_rows.jsonl",
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


def _campaign_plan_payload(paths: CampaignPaths, plan: Dict[str, Any], *, configuration: Dict[str, Any], created_at: Optional[str] = None) -> Dict[str, Any]:
    """Build the persisted pre-generation contract without writing it."""
    from .runner import _task_hash
    from .tasks import TASKS
    task_map = {task.id: _task_hash(task) for task in TASKS}
    accepted = dict(plan)
    accepted.update({
        "campaign_id": paths.campaign_id,
        "generation_judge_mode": "off",
        "task_hashes": {task_id: task_map[task_id] for model in plan.get("active_models", []) for task_id in model.get("tasks", []) if task_id in task_map},
        "configuration": configuration,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "recovery_policy_version": RECOVERY_POLICY_VERSION if "RECOVERY_POLICY_VERSION" in globals() else "pending",
    })
    return accepted


def write_campaign_plan(paths: CampaignPaths, plan: Dict[str, Any], *, inventory: List[Dict[str, Any]], capabilities: Dict[str, Any], configuration: Dict[str, Any]) -> Dict[str, Any]:
    """Persist the accepted pre-generation contract atomically."""
    accepted = _campaign_plan_payload(paths, plan, configuration=configuration)
    _atomic_write_text(paths.plan_json, json.dumps(accepted, indent=2, sort_keys=True))
    _atomic_write_text(paths.inventory_json, json.dumps(inventory, indent=2, sort_keys=True))
    _atomic_write_text(paths.capabilities_json, json.dumps(capabilities, indent=2, sort_keys=True))
    return accepted


def campaign_plan_equivalent(existing: Dict[str, Any], proposed: Dict[str, Any]) -> bool:
    """Compare campaign plan contracts while ignoring volatile write time."""
    left = dict(existing)
    right = dict(proposed)
    left.pop("created_at", None)
    right.pop("created_at", None)
    return left == right


def campaign_replan_refusal(manifest: CampaignManifest) -> str:
    """Human-facing fail-closed message for unsafe in-place replanning."""
    if manifest.state == "interrupted":
        allowed = f"campaign resume {manifest.campaign_id}"
    elif manifest.state == "planned":
        allowed = f"campaign run --campaign-id {manifest.campaign_id}"
    else:
        allowed = f"campaign status {manifest.campaign_id}"
    return (
        f"campaign plan refused for campaign {manifest.campaign_id!r}: "
        f"current state is {manifest.state!r}; allowed next action: {allowed}; "
        "create a new campaign ID when plan settings must change."
    )


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
    fd, temp_name = tempfile.mkstemp(prefix=f".{paths.campaign_id}-", suffix=".zip.tmp", dir=paths.packages_dir)
    os.close(fd)
    temp_package = Path(temp_name)
    inventory: List[Dict[str, Any]] = []
    duplicate_primary_reports = {"report.html", "scorecard.md", "scorecard.csv", "routing.md", "prune.md", "clones.md", "regression.md", "summary.json"}
    try:
        payloads: List[tuple[Path, str]] = []
        for source in sorted(paths.root.rglob("*")):
            if not source.is_file() or source in {package, temp_package, paths.lock_file} or paths.packages_dir in source.parents or paths.checksums_dir in source.parents or paths.readiness_dir in source.parents:
                continue
            relative_parts = source.relative_to(paths.root).parts
            if len(relative_parts) >= 2 and relative_parts[0] == "evidence" and str(relative_parts[1]).startswith("repair_"):
                continue
            # Per-response dumps are disposable intermediates.  Immutable raw
            # results and any recovery child evidence are packaged separately;
            # including dumps would make a conservative retention cleanup stale
            # the otherwise verified review package.
            if source == paths.primary_dumps_dir or paths.primary_dumps_dir in source.parents:
                continue
            if source.parent == paths.primary_dir and source.name in duplicate_primary_reports:
                continue
            relative = source.relative_to(paths.root).as_posix()
            payloads.append((source, relative))
            inventory.append({"path": relative, "sha256": _sha256(source), "size": source.stat().st_size, "role": relative.split('/', 1)[0]})
        inventory_bytes = json.dumps({"campaign_id": paths.campaign_id, "files": inventory}, indent=2, sort_keys=True).encode()
        checksums = [*inventory, {"path": "package/inventory.json", "sha256": hashlib.sha256(inventory_bytes).hexdigest(), "size": len(inventory_bytes), "role": "package"}]
        with zipfile.ZipFile(temp_package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source, relative in payloads:
                archive.write(source, relative)
            archive.writestr("package/inventory.json", inventory_bytes)
            archive.writestr("package/sha256.json", json.dumps({"files": checksums}, indent=2, sort_keys=True))
        os.replace(temp_package, package)
    except BaseException:
        temp_package.unlink(missing_ok=True)
        raise
    _atomic_write_text(paths.checksums_json, json.dumps({"package": _sha256(package), "size": package.stat().st_size,
        "source_files": {entry["path"]: {"sha256": entry["sha256"], "size": entry["size"]} for entry in inventory}}, indent=2, sort_keys=True))
    return package


def verify_package_details(paths: CampaignPaths) -> Dict[str, Any]:
    package = paths.packages_dir / f"{paths.campaign_id}-review.zip"
    result: Dict[str, Any] = {"campaign_id": paths.campaign_id, "package_path": str(package), "package_digest": None,
        "verified_at": datetime.now(timezone.utc).isoformat(), "valid": False, "file_count": 0,
        "verified_checksum_count": 0, "required_files_valid": False, "recovery_references_valid": False,
        "judge_references_valid": False, "terminal_ledger_valid": False, "candidate_rankings_valid": False,
        "unexpected_file_count": 0, "errors": [], "warnings": []}
    if not paths.checksums_json.exists():
        result["errors"].append("missing external package checksum"); return result
    try:
        data = json.loads(paths.checksums_json.read_text(encoding="utf-8"))
    except Exception:
        result["errors"].append("invalid external package checksum"); return result
    if not package.exists() or not zipfile.is_zipfile(package):
        result["errors"].append("missing or invalid zip package"); return result
    result["package_digest"] = _sha256(package)
    if data.get("package") != result["package_digest"] or data.get("size") != package.stat().st_size:
        result["errors"].append("stale package digest or size"); return result
    for relative, expected in (data.get("source_files") or {}).items():
        source = paths.root / relative
        if not source.is_file() or source.stat().st_size != expected.get("size") or _sha256(source) != expected.get("sha256"):
            result["errors"].append(f"stale source evidence: {relative}")
    with zipfile.ZipFile(package) as archive:
        infos = archive.infolist(); names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            result["errors"].append("duplicate archive member")
        for info in infos:
            pure = Path(info.filename)
            if pure.is_absolute() or ".." in pure.parts:
                result["errors"].append(f"unsafe archive path: {info.filename}")
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                result["errors"].append(f"unsupported symlink: {info.filename}")
        if result["errors"]:
            return result
        with tempfile.TemporaryDirectory(prefix="llmb-package-verify-") as extraction:
            archive.extractall(extraction)
        try:
            internal = json.loads(archive.read("package/sha256.json"))
            inventory = json.loads(archive.read("package/inventory.json"))
        except (KeyError, ValueError):
            result["errors"].append("missing or malformed internal metadata"); return result
        if not isinstance(inventory, dict):
            result["errors"].append("invalid package inventory format")
            return result
        entries = internal.get("files")
        if not isinstance(entries, list) or not all(isinstance(x, dict) for x in entries):
            result["errors"].append("invalid internal checksum format"); return result
        listed = {x.get("path") for x in entries}
        unexpected = set(names) - listed - {"package/sha256.json"}
        result["unexpected_file_count"] = len(unexpected)
        if unexpected: result["errors"].append(f"unexpected unlisted files: {sorted(unexpected)}")
        for entry in entries:
            name = entry.get("path")
            if name not in names: result["errors"].append(f"missing listed file: {name}"); continue
            content = archive.read(name)
            if len(content) != entry.get("size"): result["errors"].append(f"size mismatch: {name}")
            if hashlib.sha256(content).hexdigest() != entry.get("sha256"): result["errors"].append(f"checksum mismatch: {name}")
            else: result["verified_checksum_count"] += 1
        required = {"manifest.json","plan/plan.json","plan/inventory.json","plan/capabilities.json","evidence/primary/raw_results.jsonl","evidence/primary/run_validity.json","evidence/primary/model_identities.json","evidence/effective_rows.jsonl","reports/readiness.json","reports/readiness.md","rankings/candidate/master_raw.jsonl","rankings/candidate/master_summary.json","package/inventory.json","package/sha256.json"}
        missing = required - set(names)
        if missing: result["errors"].append(f"missing required files: {sorted(missing)}")
        result["required_files_valid"] = not missing
        result["terminal_ledger_valid"] = "evidence/effective_rows.jsonl" in names
        result["candidate_rankings_valid"] = {"rankings/candidate/master_raw.jsonl","rankings/candidate/master_summary.json"} <= set(names)
        result["recovery_references_valid"] = True
        result["judge_references_valid"] = True
        if "evidence/effective_rows.jsonl" in names:
            try:
                effective = [json.loads(line) for line in archive.read("evidence/effective_rows.jsonl").decode().splitlines() if line]
            except Exception:
                effective = []; result["errors"].append("invalid effective terminal ledger")
            recovered = [row for row in effective if row.get("result_origin") == "recovered" or row.get("recovery_attempt_number")]
            judged = [row for row in effective if row.get("result_origin") == "judged" or row.get("judge_row_hash")]
            if recovered:
                needed = {"evidence/recovery/recovery_plan.json","evidence/recovery/recovery_result.json","evidence/recovery/recovery_attempts.jsonl"}
                if not needed <= set(names):
                    result["recovery_references_valid"] = False; result["errors"].append("missing referenced recovery evidence")
                for row in recovered:
                    child = row.get("recovery_child_id")
                    if child and f"evidence/recovery/children/{child}/attempt.json" not in names:
                        result["recovery_references_valid"] = False; result["errors"].append(f"missing referenced recovery child: {child}")
            if judged:
                needed = {"evidence/judge/judge_selection.json","evidence/judge/judge_results.jsonl","evidence/judge/judge_summary.json"}
                if not needed <= set(names):
                    result["judge_references_valid"] = False; result["errors"].append("missing referenced judge evidence")
        result["file_count"] = len(names)
    result["valid"] = not result["errors"]
    return result


def verify_package(paths: CampaignPaths) -> bool:
    details = verify_package_details(paths)
    if not details["valid"]:
        for readiness_path in (paths.readiness_json, paths.reports_dir / "readiness.json"):
            if readiness_path.exists():
                try:
                    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
                    readiness["readiness"] = "not_ready_manual_items"
                    readiness["package_verification"] = details
                    readiness["blockers"] = sorted(set(readiness.get("blockers") or []) | {"package_verification_failed"})
                    _atomic_write_text(readiness_path, json.dumps(readiness, indent=2, sort_keys=True))
                except Exception:
                    pass
    return bool(details["valid"])


def _json_object(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _cleanup_pending_blockers(paths: CampaignPaths) -> List[str]:
    blockers: List[str] = []
    recovery = _json_object(paths.recovery_result)
    if str(recovery.get("status") or "").lower() in {"pending", "running", "interrupted", "in_progress"}:
        blockers.append("pending_recovery")
    for action in recovery.get("actions") or []:
        if str(action.get("status") or "").lower() in {"pending", "running", "retry_pending", "in_progress"}:
            blockers.append("pending_recovery")
            break
    judge = _json_object(paths.judge_summary)
    if int(judge.get("pending") or 0) > 0 or str(judge.get("status") or "").lower() in {"pending", "running", "interrupted", "in_progress"}:
        blockers.append("pending_judging")
    if paths.effective_rows.exists():
        try:
            rows = [json.loads(line) for line in paths.effective_rows.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError, json.JSONDecodeError):
            blockers.append("invalid_effective_terminal_ledger")
        else:
            dispositions = [str(row.get("terminal_disposition") or "") for row in rows]
            if any("pending" in value for value in dispositions):
                blockers.append("pending_terminal_disposition")
            if any("manual" in value or "conflicting" in value for value in dispositions):
                blockers.append("unresolved_manual_item")
    return sorted(set(blockers))


def cleanup_campaign(paths: CampaignPaths, *, apply: bool = False) -> Dict[str, Any]:
    """Plan or apply a conservative, evidence-preserving campaign cleanup."""
    manifest = load_manifest(paths)
    blockers: List[str] = []
    warnings: List[str] = []
    if manifest.state not in TERMINAL_STATES:
        blockers.append(f"ineligible_state:{manifest.state}")
    lock = read_lock(paths)
    stale_lock = bool(lock and lock_is_stale(lock))
    if lock and not stale_lock:
        blockers.append("active_or_ambiguous_lock")
    elif stale_lock:
        warnings.append("proven_stale_same_host_lock")
    packages = sorted(paths.packages_dir.glob("*.zip")) if paths.packages_dir.exists() else []
    if len(packages) != 1:
        blockers.append("missing_final_package" if not packages else "multiple_final_packages")
    verification = verify_package_details(paths)
    if not verification.get("valid"):
        blockers.append("package_verification_failed")
    blockers.extend(_cleanup_pending_blockers(paths))
    for required, label in ((paths.effective_rows, "missing_effective_terminal_ledger"),
                            (paths.reports_dir / "readiness.md", "missing_readiness_markdown")):
        if not required.is_file():
            blockers.append(label)
    if not paths.readiness_json.is_file() and not (paths.reports_dir / "readiness.json").is_file():
        blockers.append("missing_readiness")

    candidates: List[Path] = []
    if paths.primary_dumps_dir.exists():
        try:
            _inside(paths.root, paths.primary_dumps_dir)
        except CampaignError:
            blockers.append("unsafe_cleanup_target")
        else:
            if paths.primary_dumps_dir.is_symlink():
                blockers.append("symlink_cleanup_target")
            elif paths.primary_dumps_dir.is_dir():
                candidates.append(paths.primary_dumps_dir)
            else:
                blockers.append("unexpected_cleanup_target_type")
    blockers = sorted(set(blockers))
    removable_files = [file for target in candidates for file in target.rglob("*") if file.is_file() and not file.is_symlink()]
    bytes_proposed = sum(file.stat().st_size for file in removable_files)
    retained = [file for file in sorted(paths.root.rglob("*"))
                if file.is_file() and not any(file == target or target in file.parents for target in candidates)]
    result: Dict[str, Any] = {
        "campaign_id": paths.campaign_id,
        "dry_run": not apply,
        "applied": False,
        "eligible": not blockers,
        "blockers": blockers,
        "files_proposed_for_removal": [str(file.relative_to(paths.root)) for file in removable_files],
        "targets_proposed_for_removal": [str(target.relative_to(paths.root)) for target in candidates],
        "files_removed": [],
        "bytes_proposed": bytes_proposed,
        "bytes_removed": 0,
        "files_retained": [str(file.relative_to(paths.root)) for file in retained],
        "verification": verification,
        "policy_version": CLEANUP_POLICY_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "errors": [],
        "warnings": warnings,
    }
    if apply:
        if blockers:
            raise CampaignError("cleanup refused: " + ", ".join(blockers))
        if stale_lock:
            # Only same-host dead-PID locks reach this branch.
            paths.lock_file.unlink()
        try:
            for target in candidates:
                _inside(paths.root, target)
                shutil.rmtree(target)
        except BaseException as exc:
            result["errors"].append(str(exc))
            raise CampaignError(f"cleanup failed while removing an enumerated redundant target: {exc}") from exc
        result["applied"] = True
        result["files_removed"] = result["files_proposed_for_removal"]
        result["bytes_removed"] = bytes_proposed
        audit = paths.root / "cleanup" / "cleanup_record.json"
        _inside(paths.root, audit)
        _atomic_write_text(audit, json.dumps(result, indent=2, sort_keys=True))
    return result


def cleanup_all_campaigns(*, campaigns_root: Path = CAMPAIGNS_ROOT, apply: bool = False) -> Dict[str, Any]:
    """Process eligible campaigns and report, rather than aborting on, unsafe ones."""
    results: List[Dict[str, Any]] = []
    if campaigns_root.exists():
        for root in sorted(item for item in campaigns_root.iterdir() if item.is_dir() and not item.is_symlink()):
            try:
                paths = resolve_paths(root.name, campaigns_root=campaigns_root)
                preview = cleanup_campaign(paths, apply=False)
                results.append(cleanup_campaign(paths, apply=True) if apply and preview["eligible"] else preview)
            except CampaignError as exc:
                results.append({"campaign_id": root.name, "eligible": False, "applied": False,
                                "blockers": [str(exc)], "errors": [str(exc)]})
    return {"dry_run": not apply, "applied": bool(apply), "campaigns": results,
            "eligible_count": sum(bool(item.get("eligible")) for item in results),
            "processed_count": sum(bool(item.get("applied")) for item in results),
            "policy_version": CLEANUP_POLICY_VERSION}


def _legacy_source_files(source: Path) -> List[Path]:
    if source.is_symlink():
        raise CampaignError("legacy source may not be a symlink")
    files: List[Path] = []
    for item in sorted(source.rglob("*")):
        if item.is_symlink():
            raise CampaignError(f"legacy source contains unsupported symlink: {item}")
        if item.is_file():
            resolved = item.resolve()
            if not resolved.is_relative_to(source.resolve()):
                raise CampaignError(f"legacy source file escapes source root: {item}")
            files.append(item)
    return files


def _legacy_rows(path: Path) -> List[Dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CampaignError(f"malformed legacy raw evidence: {exc}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise CampaignError("malformed legacy raw evidence: expected JSON object rows")
    return rows


def migrate_legacy_run(run_id: str, campaign_id: str, *, runs_dir: Path = Path("runs"), campaigns_root: Path = CAMPAIGNS_ROOT, apply: bool = False) -> Dict[str, Any]:
    """Plan or perform a copy-only legacy import with forensic provenance."""
    validate_campaign_id(run_id)
    source = legacy_run_dir(run_id, runs_dir=runs_dir)
    runs_root = runs_dir.resolve()
    if not source.resolve().is_relative_to(runs_root):
        raise CampaignError("legacy source escapes runs root")
    if not is_legacy_run_dir(source):
        raise CampaignError(f"legacy source is not a readable run: {source}")
    files = _legacy_source_files(source)
    rows = _legacy_rows(source / "raw_results.jsonl")
    source_checksums = {file.relative_to(source).as_posix(): {"sha256": _sha256(file), "size": file.stat().st_size} for file in files}
    paths = resolve_paths(campaign_id, campaigns_root=campaigns_root)
    if paths.root.exists() and any(paths.root.iterdir()):
        raise CampaignError(f"campaign migration target already exists: {paths.root}")
    recovery_names = {"repair_plan.json", "repair_result.json", "repair_results.jsonl", "repair_attempts.jsonl"}
    judge_names = {"judge_selection.json", "judge_results.jsonl", "judge_summary.json"}
    report_names = {"report.html", "scorecard.md", "scorecard.csv", "routing.md", "prune.md", "clones.md", "regression.md", "summary.json"}
    sidecars = {file.relative_to(source).as_posix():
                ("recovery" if file.name in recovery_names else "judge" if file.name in judge_names else "reports" if file.name in report_names else "primary")
                for file in files}
    unavailable = ["original_model_digest", "original_task_hash", "historical_capability_probe"]
    result: Dict[str, Any] = {
        "source_run_id": run_id, "source_path": str(source.resolve()),
        "destination_campaign_id": campaign_id, "destination_path": str(paths.root),
        "dry_run": not apply, "applied": False,
        "files_to_copy": sorted(source_checksums), "files_copied": [],
        "source_checksums": source_checksums, "sidecars_detected": sidecars,
        "sidecars_mapped": {}, "unavailable_historical_fields": unavailable,
        "source_immutability_verified": False, "destination_valid": False,
        "policy_version": MIGRATION_POLICY_VERSION, "errors": [], "warnings": [],
    }
    if not apply:
        return result
    created = False
    try:
        paths, manifest = create_campaign(campaign_id, models=sorted({str(row.get("model") or "legacy_model_unavailable") for row in rows}),
                                          version="legacy-migration", campaigns_root=campaigns_root)
        created = True
        for file in files:
            relative = file.relative_to(source)
            lane = sidecars[relative.as_posix()]
            if lane == "recovery":
                target = paths.recovery_dir / ("recovery_" + file.name.removeprefix("repair_"))
            elif lane == "judge":
                target = paths.judge_dir / file.name
            elif lane == "reports":
                target = paths.reports_dir / file.name
            else:
                target = paths.primary_dir / relative
            _inside(paths.root, target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
            result["files_copied"].append(relative.as_posix())
            result["sidecars_mapped"][relative.as_posix()] = target.relative_to(paths.root).as_posix()
        if not paths.primary_run_validity.exists():
            _atomic_write_text(paths.primary_run_validity, json.dumps({"status": "valid", "source": "legacy_import", "historical_value": "historical_value_unavailable"}, indent=2))
        models = sorted({str(row.get("model") or "legacy_model_unavailable") for row in rows})
        identities = {model: {"digest": "legacy_digest_unavailable", "source": "historical_value_unavailable"} for model in models}
        _atomic_write_text(paths.primary_dir / "model_identities.json", json.dumps(identities, indent=2, sort_keys=True))
        tasks = sorted({str(row.get("task") or "legacy_task_unavailable") for row in rows})
        plan = {"campaign_id": campaign_id, "legacy_source_run_id": run_id, "generation_judge_mode": "historical_value_unavailable",
                "task_hashes": {task: "legacy_task_hash_unavailable" for task in tasks}, "models": models,
                "historical_fields": {name: "historical_value_unavailable" for name in unavailable}}
        _atomic_write_text(paths.plan_json, json.dumps(plan, indent=2, sort_keys=True))
        _atomic_write_text(paths.inventory_json, json.dumps([{"name": model, "digest": "legacy_digest_unavailable"} for model in models], indent=2, sort_keys=True))
        _atomic_write_text(paths.capabilities_json, json.dumps({model: {"state": "historical_value_unavailable"} for model in models}, indent=2, sort_keys=True))
        effective: List[Dict[str, Any]] = []
        candidate: List[Dict[str, Any]] = []
        for index, row in enumerate(rows):
            visible = row.get("score") is not None
            disposition = "scored" if visible else "terminal_model_failure"
            effective.append({"model": row.get("model"), "model_digest_resolved": "legacy_digest_unavailable", "task": row.get("task"),
                              "task_hash": "legacy_task_hash_unavailable", "primary_row_index": index,
                              "primary_row_hash": hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest(),
                              "effective_score": row.get("score"), "effective_reason": row.get("reason"),
                              "result_origin": "primary", "terminal_disposition": disposition})
            imported = dict(row); imported.update({"run_id": row.get("run_id") or run_id, "campaign_id": campaign_id,
                                                  "ranking_scope": "separate", "model_digest_resolved": "legacy_digest_unavailable",
                                                  "task_hash": "legacy_task_hash_unavailable", "terminal_disposition": disposition})
            candidate.append(imported)
        _atomic_write_text(paths.effective_rows, "".join(json.dumps(row, sort_keys=True) + "\n" for row in effective))
        _atomic_write_text(paths.candidate_rankings_dir / "master_raw.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidate))
        _atomic_write_text(paths.candidate_rankings_dir / "master_summary.json", json.dumps({"source": "legacy_import", "rows": len(candidate)}, indent=2))
        readiness = {"campaign_id": campaign_id, "readiness": "diagnostic_only", "rows": len(rows), "terminal_rows": len(effective),
                     "blockers": ["historical_identity_or_task_hash_unavailable"], "historical_limitations": unavailable}
        _atomic_write_text(paths.readiness_json, json.dumps(readiness, indent=2, sort_keys=True))
        _atomic_write_text(paths.reports_dir / "readiness.json", json.dumps(readiness, indent=2, sort_keys=True))
        _atomic_write_text(paths.reports_dir / "readiness.md", "# Legacy campaign readiness\n\n- readiness: diagnostic_only\n- historical identities/task hashes: unavailable\n")
        after = {file.relative_to(source).as_posix(): {"sha256": _sha256(file), "size": file.stat().st_size} for file in _legacy_source_files(source)}
        if after != source_checksums:
            raise CampaignError("legacy source changed during migration")
        result["source_immutability_verified"] = True
        provenance = {"migration_id": f"legacy-{campaign_id}", "migrated_at": datetime.now(timezone.utc).isoformat(),
                      "source_run_id": run_id, "source_path": str(source.resolve()), "source_checksums": source_checksums,
                      "destination_campaign_id": campaign_id, "detected_source_layout_version": "legacy-run-v1",
                      "imported_files": result["files_copied"], "skipped_files": [], "unavailable_historical_fields": unavailable,
                      "inferred_fields": {"models_and_tasks": "read from raw_results.jsonl"}, "original_run_version": "historical_value_unavailable",
                      "original_model_identities": identities, "task_ids_and_hashes": plan["task_hashes"],
                      "sidecar_mappings": result["sidecars_mapped"], "migration_policy_version": MIGRATION_POLICY_VERSION,
                      "source_immutability_verified": True}
        _atomic_write_text(paths.plan_dir / "migration_provenance.json", json.dumps(provenance, indent=2, sort_keys=True))
        manifest.notes.update({"legacy_source": str(source), "migration_provenance": "plan/migration_provenance.json"})
        write_manifest(paths, manifest)
        for state in ("planned", "generating", "packaged", "archived_diagnostic"):
            manifest = transition(paths, manifest, state)
        package_campaign(paths)
        result["destination_valid"] = bool(verify_package_details(paths)["valid"])
        if not result["destination_valid"]:
            raise CampaignError("migrated campaign package did not verify")
        result["applied"] = True
        result["dry_run"] = False
        return result
    except BaseException:
        if created and paths.root.exists():
            # The destination was created by this invocation and never became a
            # successfully returned migration.  Source evidence is untouched.
            shutil.rmtree(paths.root)
        raise


# ---------------------------------------------------------------------------
# Unattended methodology policy (RC19.2)
# ---------------------------------------------------------------------------

RECOVERY_POLICY_VERSION = "rc19.2.1"
CLEANUP_POLICY_VERSION = "rc20-retention-1"
MIGRATION_POLICY_VERSION = "rc20-legacy-copy-1"
TERMINAL_DISPOSITIONS = {
    "scored", "judged", "confirmed_capability_unavailable",
    "capability_measured_failure", "environment_limited", "operator_excluded",
    "terminal_model_failure", "terminal_thinking_only", "terminal_empty",
    "terminal_transient", "harness_failure", "awaiting_external_judge",
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
    if kind in {"capability_unavailable", "confirmed_capability_unavailable"}:
        return {"disposition": "confirmed_capability_unavailable", "retry": False, "reason": kind}
    if kind in {"capability_measured_failure", "measured_failure"}:
        return {"disposition": "capability_measured_failure", "retry": False, "reason": kind}
    if kind == "environment_limited":
        return {"disposition": "environment_limited", "retry": False, "reason": kind}
    if kind == "operator_excluded":
        return {"disposition": "operator_excluded", "retry": False, "reason": kind}
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
    if str(row.get("reason") or "").startswith("raw only, judge off"):
        return {"disposition": "awaiting_external_judge", "retry": False, "reason": "subjective output awaiting post-hoc judge"}
    return {"disposition": "harness_failure", "retry": False, "reason": "unscorable row without error classification"}


def execute_recovery_phase(paths: CampaignPaths, client: Any, cfg: Any, *, budget: int = 2048,
                           build_plan_fn: Any = None, apply_plan_fn: Any = None) -> Dict[str, Any]:
    """Run bounded recovery while proving primary evidence remains immutable."""
    from . import repair
    build_plan_fn = build_plan_fn or repair.build_plan
    apply_plan_fn = apply_plan_fn or repair.apply_plan
    primary_before = paths.primary_raw_results.read_bytes()
    manifest = load_manifest(paths)
    if manifest.state == "generating":
        transition(paths, manifest, "recovering")
    elif manifest.state == "interrupted" and manifest.resume_state == "recovering":
        transition(paths, manifest, "recovering")
    elif manifest.state != "recovering":
        raise CampaignError(f"recovery cannot start from {manifest.state!r}")
    plan = build_plan_fn(paths.evidence_dir, run_id="primary", think_retry_num_predict=int(budget),
                         judge_mode="off", include_missing=False)
    _atomic_write_text(paths.recovery_plan, json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    result = apply_plan_fn(client, cfg, plan, rankings_dir=paths.candidate_rankings_dir, ranking_scope="separate")
    records = []
    for index, action in enumerate(result.get("actions", []), 1):
        attempts = list(action.get("attempts") or []) or [{"attempt_number": index}]
        for attempt in attempts:
            child_id = str(attempt.get("child_run_id") or action.get("child_run_id") or f"recovery-{index:04d}")
            child_dir = paths.recovery_children_dir / child_id
            child_dir.mkdir(parents=True, exist_ok=True)
            source_child = paths.evidence_dir / child_id
            if source_child.is_dir():
                for source_file in sorted(source_child.rglob("*")):
                    if source_file.is_file():
                        relative_child = source_file.relative_to(source_child)
                        if relative_child.parts and relative_child.parts[0] not in {
                            "raw", "subjective", "raw_results.jsonl", "run_validity.json",
                            "model_identities.json", "config.json", "filters.json",
                            "capability_report.json", "ranking_scope.json", "status.json",
                            "summary_meta.json", "skipped_models.json",
                        }:
                            continue
                        target = child_dir / source_file.relative_to(source_child)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_file, target)
            record = {"campaign_id": paths.campaign_id, "parent_row_hash": action.get("parent_row_hash"),
                      "parent_run_id": "primary", "child_run_id": child_id,
                      "action_id": action.get("action_id"), "kind": action.get("kind"),
                      "model": action.get("model"), "model_digest": action.get("model_digest"),
                      "task": action.get("task") or (action.get("tasks") or [None])[0],
                      "task_hash": action.get("task_hash"),
                      "tasks": action.get("tasks") or attempt.get("required_tasks") or [],
                      "attempt_number": int(attempt.get("attempt_number") or index),
                      "output_budget": action.get("output_budget"), "context": action.get("context"),
                      "think_mode": action.get("think_mode", "off"),
                      "configuration": attempt.get("overrides") or {},
                      "started_at": action.get("started_at"), "ended_at": action.get("ended_at"),
                      "wall_time_seconds": action.get("wall_time_seconds", 0),
                      "raw_response_reference": f"children/{child_id}/raw_results.jsonl",
                      "output_classification": attempt.get("status") or action.get("status"),
                      "visible_answer": bool(action.get("visible_answer") or attempt.get("valid_tasks")),
                      "score": action.get("score", attempt.get("score")),
                      "reason": action.get("reason"),
                      "error_classification": action.get("status"),
                      "stop_reason": action.get("stop_reason") or action.get("reason") or attempt.get("status") or action.get("status"),
                      "policy_version": RECOVERY_POLICY_VERSION}
            _atomic_write_text(child_dir / "attempt.json", json.dumps(record, indent=2, sort_keys=True))
            records.append(record)
    _atomic_write_text(paths.recovery_attempts, "".join(json.dumps(item, sort_keys=True) + "\n" for item in records))
    _atomic_write_text(paths.recovery_result, json.dumps(result, indent=2, sort_keys=True))
    if paths.primary_raw_results.read_bytes() != primary_before:
        raise CampaignError("recovery mutated immutable primary evidence")
    return result


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _primary_row_hash(row: Dict[str, Any]) -> str:
    from .repair import _row_hash
    return _row_hash(row)


def _json_compact_hash(row: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def _child_rows_by_repair_hash(paths: CampaignPaths) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for child_dir in sorted(paths.recovery_children_dir.iterdir() if paths.recovery_children_dir.exists() else []):
        raw = child_dir / "raw_results.jsonl"
        for row in _read_jsonl(raw):
            source = str(row.get("repair_source_row_hash") or "")
            task = str(row.get("task") or "")
            if source and task:
                candidate = dict(row)
                candidate["_recovery_child_id"] = child_dir.name
                out[f"{source}:{task}"] = candidate
    return out


def _recovery_actions_by_source(paths: CampaignPaths) -> Dict[str, Dict[str, Any]]:
    plan = _json_object(paths.recovery_plan)
    result = _json_object(paths.recovery_result)
    planned = {str(action.get("action_id") or ""): action for action in plan.get("actions") or []}
    out: Dict[str, Dict[str, Any]] = {}
    for action in result.get("actions") or []:
        action_id = str(action.get("action_id") or "")
        source_hashes = (planned.get(action_id) or {}).get("source_row_hashes") or {}
        for task in action.get("tasks") or []:
            source = str(source_hashes.get(str(task)) or "")
            if source:
                out[f"{source}:{task}"] = action
    return out


def _terminal_after_recovery(primary: Dict[str, Any], action: Optional[Dict[str, Any]], child: Optional[Dict[str, Any]]) -> str:
    if child and child.get("score") is not None and not child.get("error_kind"):
        return "scored"
    status = str((action or {}).get("status") or "")
    kind = str(primary.get("error_kind") or "")
    if status == "measured_failure":
        return "capability_measured_failure"
    if status in {"timeout"}:
        return "terminal_transient"
    if kind == "thinking_only":
        return "terminal_thinking_only"
    if kind == "empty_output":
        return "terminal_empty"
    text = str(primary.get("reason") or primary.get("error") or "").lower()
    if "timeout" in text or " 5" in text or ("http" in text and "5" in text):
        return "terminal_transient"
    return classify_recovery_row(primary)["disposition"]


def _candidate_from_effective(row: Dict[str, Any], manifest: CampaignManifest) -> Dict[str, Any]:
    candidate = {
        "run_id": "primary",
        "campaign_id": manifest.campaign_id,
        "model": row.get("model"),
        "model_digest_resolved": row.get("model_digest_resolved"),
        "task": row.get("task"),
        "task_hash": row.get("task_hash"),
        "score": row.get("effective_score"),
        "reason": row.get("effective_reason"),
        "terminal_disposition": row.get("terminal_disposition"),
        "result_origin": row.get("result_origin"),
        "ranking_scope": "separate",
        "canonical_rankings": False,
        "_source_signature": row.get("effective_row_hash"),
    }
    return candidate


def write_readiness(paths: CampaignPaths, rows: List[Dict[str, Any]], *, judge_available: bool = True) -> Dict[str, Any]:
    manifest = load_manifest(paths)
    raw_primary_rows = _read_jsonl(paths.primary_raw_results)
    child_by_source = _child_rows_by_repair_hash(paths)
    action_by_source = _recovery_actions_by_source(paths)
    judge_rows = _read_jsonl(paths.judge_results)
    judge_by_source = {str(row.get("source_row_hash") or ""): row for row in judge_rows if row.get("status") == "judged"}
    effective: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        raw_primary = raw_primary_rows[index] if index < len(raw_primary_rows) else row
        primary_hash = _primary_row_hash(raw_primary)
        task = str(row.get("task") or "")
        key = f"{primary_hash}:{task}"
        child = child_by_source.get(key)
        action = action_by_source.get(key)
        classification = classify_recovery_row(row)
        source = row
        origin = "primary"
        disposition = str(row.get("disposition") or classification["disposition"])
        recovery_attempt_number = None
        recovery_child_id = None
        if child or action:
            source = child or row
            origin = "recovered" if child and child.get("score") is not None and not child.get("error_kind") else "recovery_terminal"
            disposition = _terminal_after_recovery(row, action, child)
            recovery_child_id = (child or {}).get("_recovery_child_id")
            if child:
                recovery_attempt_number = child.get("repair_attempt_number")
        judge_source_hash = str(row.get("judge_source_row_hash") or "")
        judge_row = judge_by_source.get(judge_source_hash)
        if row.get("posthoc_judged") or judge_row:
            source = row
            origin = "judged"
            disposition = "judged"
        item = {
            "model": row.get("model"),
            "model_digest_resolved": row.get("model_digest_resolved") or source.get("model_digest_resolved") or source.get("model_digest"),
            "task": row.get("task"),
            "task_hash": row.get("task_hash") or source.get("task_hash"),
            "primary_row_index": index,
            "primary_row_hash": primary_hash,
            "recovery_row_hash": _primary_row_hash(child) if child else None,
            "recovery_child_id": recovery_child_id,
            "recovery_attempt_number": recovery_attempt_number,
            "judge_source_row_hash": judge_source_hash or None,
            "judge_row_hash": _json_compact_hash(judge_row) if judge_row else None,
            "result_origin": origin,
            "effective_score": source.get("score"),
            "effective_reason": source.get("reason"),
            "terminal_disposition": disposition,
            "capability_status": disposition if disposition in {"confirmed_capability_unavailable", "capability_measured_failure"} else None,
            "environment_status": disposition if disposition == "environment_limited" else None,
            "harness_status": disposition if disposition == "harness_failure" else None,
            "provenance": {
                "campaign_id": paths.campaign_id,
                "primary_run_id": "primary",
                "recovery_action_id": (action or {}).get("action_id"),
                "recovery_status": (action or {}).get("status"),
                "judge_model": row.get("judge_model") or (judge_row or {}).get("judge_model"),
                "judge_mode": row.get("judge_mode") or (judge_row or {}).get("judge_mode"),
            },
        }
        item["correctness"] = (
            "correct" if item.get("effective_score") == 100 else
            "visible_wrong" if item.get("effective_score") is not None else
            "non_scorable"
        )
        item["effective_row_hash"] = _json_compact_hash(item)
        effective.append(item)
    _atomic_write_text(paths.effective_rows, "".join(json.dumps(row, sort_keys=True) + "\n" for row in effective))

    dispositions = [str(row["terminal_disposition"]) for row in effective]
    pending = [item for item in dispositions if item not in TERMINAL_DISPOSITIONS]
    blockers = list(pending)
    if "harness_failure" in dispositions:
        blockers.append("harness_failure")
    if "awaiting_external_judge" in dispositions or not judge_available:
        blockers.append("awaiting_external_judge")
    state = "ready_for_adoption" if not blockers else (
        "not_ready_harness_failure" if "harness_failure" in blockers else
        "not_ready_external_judge" if "awaiting_external_judge" in blockers else
        "not_ready_manual_items"
    )
    summary = {
        "campaign_id": paths.campaign_id,
        "readiness": state,
        "total_applicable_cells": len(rows),
        "rows": len(rows),
        "terminal_rows": len(rows) - len(pending),
        "pending_dispositions": pending,
        "primary_correct": sum(row["result_origin"] == "primary" and row["correctness"] == "correct" for row in effective),
        "primary_visible_wrong": sum(row["result_origin"] == "primary" and row["correctness"] == "visible_wrong" for row in effective),
        "primary_partial": sum(row["result_origin"] == "primary" and isinstance(row.get("effective_score"), (int, float)) and 0 < float(row["effective_score"]) < 100 for row in effective),
        "recovered_to_correct": sum(row["result_origin"] == "recovered" and row["correctness"] == "correct" for row in effective),
        "recovered_to_visible_wrong": sum(row["result_origin"] == "recovered" and row["correctness"] == "visible_wrong" for row in effective),
        "recovery_exhausted": sum(row["result_origin"] == "recovery_terminal" for row in effective),
        "terminal_thinking_only": dispositions.count("terminal_thinking_only"),
        "terminal_empty": dispositions.count("terminal_empty"),
        "terminal_transient": dispositions.count("terminal_transient"),
        "capability_unavailable": dispositions.count("confirmed_capability_unavailable"),
        "capability_measured_failure": dispositions.count("capability_measured_failure"),
        "environment_limited": dispositions.count("environment_limited"),
        "operator_excluded": dispositions.count("operator_excluded"),
        "subjective_eligible": sum(bool(row.get("judge_source_row_hash") or row.get("judge_model")) for row in effective),
        "judged": dispositions.count("judged"),
        "awaiting_external_judge": dispositions.count("awaiting_external_judge"),
        "harness_failure": dispositions.count("harness_failure"),
        "manual_conflicting_items": sum(d in {"conflicting_evidence/manual_review"} for d in dispositions),
        "blockers": sorted(set(blockers)),
        "policy_version": RECOVERY_POLICY_VERSION,
    }
    candidate_rows = [_candidate_from_effective(row, manifest) for row in effective]
    _atomic_write_text(paths.candidate_rankings_dir / "master_raw.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidate_rows))
    _atomic_write_text(paths.candidate_rankings_dir / "master_summary.json", json.dumps({"campaign_id": paths.campaign_id, "rows": len(candidate_rows)}, indent=2, sort_keys=True))
    _atomic_write_text(paths.readiness_json, json.dumps(summary, indent=2, sort_keys=True))
    _atomic_write_text(paths.reports_dir / "readiness.json", json.dumps(summary, indent=2, sort_keys=True))
    _atomic_write_text(paths.reports_dir / "readiness.md", "# Campaign readiness\n\n" + "\n".join(f"- {key}: {value}" for key, value in summary.items()) + "\n")
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
    candidate_raw = paths.candidate_rankings_dir / "master_raw.jsonl"
    def read_rows(path: Path) -> List[Dict[str, Any]]:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    if manifest.state == "accepted" and paths.adoption_record.exists():
        incoming_noop, current_noop = read_rows(candidate_raw), read_rows(rankings_dir / "master_raw.jsonl")
        current_by_cell = {(str(row.get("run_id")),str(row.get("model")),str(row.get("task"))): row for row in current_noop}
        if all(current_by_cell.get((str(row.get("run_id")),str(row.get("model")),str(row.get("task"))),{}).get("_source_signature") == row.get("_source_signature") for row in incoming_noop):
            return {"campaign_id": manifest.campaign_id, "campaign_version": manifest.version, "readiness": "ready_for_adoption",
                    "package_verified": True, "rows_incoming": len(incoming_noop), "rows_added_or_updated": 0,
                    "rows_replaced": 0, "rows_unchanged": len(incoming_noop), "changes": [], "blockers": [],
                    "warnings": ["campaign already adopted with identical signatures"], "would_be_noop": True, "dry_run": bool(dry_run)}
    readiness = json.loads(paths.readiness_json.read_text()) if paths.readiness_json.exists() else {}
    if readiness.get("readiness") != "ready_for_adoption":
        raise CampaignError("campaign is not ready_for_adoption")
    package_verification = verify_package_details(paths)
    if not package_verification["valid"]:
        raise CampaignError("campaign package checksums do not verify")
    if not paths.effective_rows.exists():
        raise CampaignError("effective terminal evidence is missing")
    effective = read_jsonl = [json.loads(line) for line in paths.effective_rows.read_text().splitlines() if line.strip()]
    plan = json.loads(paths.plan_json.read_text())
    planned_hashes = plan.get("task_hashes") or {}
    for row in effective:
        task = str(row.get("task") or "")
        if task and planned_hashes.get(task) and row.get("task_hash") != planned_hashes[task]:
            raise CampaignError(f"task hash mismatch for {task}")
        if row.get("terminal_disposition") == "harness_failure":
            raise CampaignError("unresolved harness failure in effective evidence")
    selection_path = paths.judge_dir / "judge_selection.json"
    judge_selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    judge = judge_selection.get("judge") or {}
    cohort_digests = {str(item.get("digest") or "") for item in judge_selection.get("cohort") or []}
    if judge.get("digest") and str(judge["digest"]) in cohort_digests:
        raise CampaignError("judge digest conflicts with tested cohort")
    if not candidate_raw.exists():
        raise CampaignError("candidate rankings evidence is missing")
    current = read_rows(rankings_dir / "master_raw.jsonl")
    incoming = read_rows(candidate_raw)
    for row in incoming:
        row["ranking_scope"] = "canonical"
        row["canonical_rankings"] = True
        row["campaign_id"] = manifest.campaign_id
    index = {(str(row.get("run_id")), str(row.get("model")), str(row.get("task"))): row for row in current}
    added = replaced = unchanged = 0
    changes: List[Dict[str, Any]] = []
    for row in incoming:
        key = (str(row.get("run_id")), str(row.get("model")), str(row.get("task")))
        previous = index.get(key)
        if previous is not None and previous.get("_source_signature") == row.get("_source_signature"):
            unchanged += 1
            continue
        changes.append({"key": {"run_id": key[0], "incoming_signature": row.get("_source_signature"), "existing_signature": previous.get("_source_signature") if previous else None, "model": row.get("model"), "model_digest": row.get("model_digest_resolved"), "task": row.get("task"), "task_hash": row.get("task_hash")}, "operation": "replace" if previous is not None else "add", "old": {"score": previous.get("score"), "reason": previous.get("reason"), "disposition": previous.get("terminal_disposition")} if previous else None, "new": {"score": row.get("score"), "reason": row.get("reason"), "disposition": row.get("terminal_disposition")}, "scope_conversion": "separate_to_canonical"})
        if previous is not None:
            current.remove(previous); replaced += 1
        current.append(row); index[key] = row; added += 1
    def aggregate(rows):
        by_model: Dict[str, List[float]] = {}
        for item in rows:
            if isinstance(item.get("score"), (int, float)): by_model.setdefault(str(item.get("model")), []).append(float(item["score"]))
        return {model: sum(scores)/len(scores) for model, scores in by_model.items()}
    old_aggregate, new_aggregate = aggregate(read_rows(rankings_dir / "master_raw.jsonl")), aggregate(current)
    preview = {"campaign_id": manifest.campaign_id, "campaign_version": manifest.version,
               "manifest_schema_version": manifest.schema_version, "readiness": readiness.get("readiness"),
               "package_path": package_verification["package_path"], "package_digest": package_verification["package_digest"],
               "package_verification": package_verification, "package_verified": True,
               "rows_incoming": len(incoming), "rows_added_or_updated": added, "rows_replaced": replaced,
               "rows_unchanged": unchanged, "rows_excluded": 0, "changes": changes,
               "old_coverage": len(read_rows(rankings_dir / "master_raw.jsonl")), "new_coverage": len(current),
               "old_model_aggregates": old_aggregate, "new_model_aggregates": new_aggregate,
               "judge": judge, "judge_validation": "valid", "tested_cohort_digests": sorted(cohort_digests),
               "canonical_scope_conversion": True, "blockers": [], "warnings": [],
               "would_be_noop": added == 0, "dry_run": bool(dry_run)}
    if dry_run:
        return preview
    if added == 0 and manifest.state == "accepted":
        return preview
    parent = rankings_dir.parent
    transaction_id = hashlib.sha256(f"{manifest.campaign_id}:{datetime.now(timezone.utc).isoformat()}".encode()).hexdigest()[:16]
    temp = Path(tempfile.mkdtemp(prefix=".campaign-adopt-", dir=str(parent)))
    try:
        if rankings_dir.exists():
            shutil.copytree(rankings_dir, temp, dirs_exist_ok=True)
        raw = temp / "master_raw.jsonl"
        _atomic_write_text(raw, "".join(json.dumps(row, sort_keys=True) + "\n" for row in current))
        from . import rankings
        rankings.write_rankings(temp / "no-runs", temp, force_rescan=True)
        required_artifacts = [raw, temp / "master_summary.json", temp / "master_report_data.json"]
        if not all(item.is_file() for item in required_artifacts):
            raise CampaignError("strict rankings rebuild produced incomplete artifacts")
        adopted = read_rows(raw)
        if any(row.get("ranking_scope") != "canonical" or not row.get("canonical_rankings") for row in adopted if row.get("campaign_id") == manifest.campaign_id):
            raise CampaignError("adopted rows are not visible in canonical scope")
        backup = parent / f".rankings-backup-{paths.campaign_id}-{transaction_id}"
        if backup.exists():
            raise CampaignError(f"adoption backup path already exists: {backup}")
        if rankings_dir.exists():
            os.replace(rankings_dir, backup)
        os.replace(temp, rankings_dir)
        before_digest = _sha256(backup / "master_raw.jsonl") if (backup / "master_raw.jsonl").exists() else None
        after_digest = _sha256(rankings_dir / "master_raw.jsonl")
        record = {**preview, "transaction_id": transaction_id, "adopted_at": datetime.now(timezone.utc).isoformat(),
                  "confirmation_type": "typed_campaign_id", "manifest_digest": _sha256(paths.manifest),
                  "canonical_source_before_digest": before_digest, "canonical_source_after_digest": after_digest,
                  "final_validation": "passed", "final_campaign_state": "accepted"}
        _atomic_write_text(paths.adoption_record, json.dumps(record, indent=2, sort_keys=True))
        transition(paths, manifest, "accepted")
        if backup.exists():
            shutil.rmtree(backup)
        return preview
    except BaseException:
        if 'backup' in locals() and backup.exists():
            if rankings_dir.exists(): shutil.rmtree(rankings_dir)
            os.replace(backup, rankings_dir)
        if paths.adoption_record.exists() and manifest.state != "accepted":
            paths.adoption_record.unlink()
        raise
    finally:
        if temp.exists():
            shutil.rmtree(temp)
