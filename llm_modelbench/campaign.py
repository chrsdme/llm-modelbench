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
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

CAMPAIGNS_ROOT = Path("campaigns")

_CAMPAIGN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_RESUMABLE_STATES = ("generating", "recovering", "judging")


class CampaignError(RuntimeError):
    """Raised for invalid campaign IDs, manifests, or state transitions.

    These errors are never silently swallowed. Callers must decide whether
    the campaign should stop, be resumed, or be classified as failed.
    """


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

    return CampaignPaths(
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
    )


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


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class CampaignManifest:
    campaign_id: str
    created_at: str
    version: str
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


def load_manifest(paths: CampaignPaths) -> CampaignManifest:
    if not paths.manifest.exists():
        raise CampaignError(
            f"no manifest at {paths.manifest}; "
            f"is {paths.campaign_id!r} a real campaign?"
        )
    try:
        data = json.loads(paths.manifest.read_text(encoding="utf-8"))
        manifest = CampaignManifest(**data)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CampaignError(
            f"invalid campaign manifest at {paths.manifest}: {exc}"
        ) from exc

    validate_campaign_id(manifest.campaign_id)
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
