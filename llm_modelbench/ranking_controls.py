"""Ranking-scope and non-destructive exclusion controls.

This module deliberately avoids storing account names, real user names, or other
personal identifiers. Exclusion entries are public-release safe and contain only
model/run identifiers, reason text, timestamps, and action names.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

SCOPE_CANONICAL = "canonical"
SCOPE_SEPARATE = "separate"
SCOPE_EXCLUDED = "excluded"
SCOPE_ARCHIVED = "archived"
EXCLUSIONS_FILE = "exclusions.json"
AUDIT_FILE = "audit_log.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def default_exclusions() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "excluded_runs": {},
        "excluded_models": {},
        "archived_runs": {},
    }


def load_exclusions(rankings_dir: Path) -> Dict[str, Any]:
    data = default_exclusions()
    stored = _read_json(Path(rankings_dir) / EXCLUSIONS_FILE)
    for key in ("excluded_runs", "excluded_models", "archived_runs"):
        if isinstance(stored.get(key), dict):
            data[key] = stored[key]
    data["schema_version"] = int(stored.get("schema_version") or 1)
    return data


def save_exclusions(rankings_dir: Path, data: Dict[str, Any]) -> None:
    payload = default_exclusions()
    payload.update({k: v for k, v in data.items() if k in payload or k == "schema_version"})
    _atomic_json(Path(rankings_dir) / EXCLUSIONS_FILE, payload)


def append_audit(rankings_dir: Path, action: str, target_type: str, target: str, *, reason: Optional[str] = None) -> None:
    entry = {
        "schema_version": 1,
        "event": action,
        "target_type": target_type,
        "target": str(target),
        "reason": str(reason or ""),
        "recorded_at": utc_now(),
    }
    path = Path(rankings_dir) / AUDIT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _entry(reason: Optional[str]) -> Dict[str, Any]:
    return {
        "reason": str(reason or "manual ranking exclusion"),
        "updated_at": utc_now(),
    }


def set_model_excluded(rankings_dir: Path, model: str, excluded: bool, *, reason: Optional[str] = None) -> Dict[str, Any]:
    data = load_exclusions(rankings_dir)
    key = str(model)
    if excluded:
        data.setdefault("excluded_models", {})[key] = _entry(reason)
        action = "exclude_model"
    else:
        data.setdefault("excluded_models", {}).pop(key, None)
        action = "include_model"
    save_exclusions(rankings_dir, data)
    append_audit(rankings_dir, action, "model", key, reason=reason)
    return data


def set_run_excluded(rankings_dir: Path, run_id: str, excluded: bool, *, reason: Optional[str] = None) -> Dict[str, Any]:
    data = load_exclusions(rankings_dir)
    key = str(run_id)
    if excluded:
        data.setdefault("excluded_runs", {})[key] = _entry(reason)
        data.setdefault("archived_runs", {}).pop(key, None)
        action = "exclude_run"
    else:
        data.setdefault("excluded_runs", {}).pop(key, None)
        action = "include_run"
    save_exclusions(rankings_dir, data)
    append_audit(rankings_dir, action, "run", key, reason=reason)
    return data


def set_run_archived(rankings_dir: Path, run_id: str, archived: bool, *, reason: Optional[str] = None) -> Dict[str, Any]:
    data = load_exclusions(rankings_dir)
    key = str(run_id)
    if archived:
        data.setdefault("archived_runs", {})[key] = _entry(reason or "manual archive")
        data.setdefault("excluded_runs", {}).pop(key, None)
        action = "archive_run"
    else:
        data.setdefault("archived_runs", {}).pop(key, None)
        action = "unarchive_run"
    save_exclusions(rankings_dir, data)
    append_audit(rankings_dir, action, "run", key, reason=reason)
    return data


def write_run_scope(run_dir: Path, *, scope: str, ranking_update: str = "auto", rankings_dir: Optional[Path] = None) -> Path:
    payload = {
        "schema_version": 1,
        "ranking_scope": scope,
        "ranking_update": ranking_update,
        "canonical_rankings": scope == SCOPE_CANONICAL,
        "created_at": utc_now(),
    }
    if rankings_dir is not None:
        payload["rankings_dir"] = str(rankings_dir)
    path = Path(run_dir) / "ranking_scope.json"
    _atomic_json(path, payload)
    return path


def read_run_scope(run_dir: Path) -> Dict[str, Any]:
    data = _read_json(Path(run_dir) / "ranking_scope.json")
    scope = str(data.get("ranking_scope") or SCOPE_CANONICAL)
    if scope not in {SCOPE_CANONICAL, SCOPE_SEPARATE, SCOPE_EXCLUDED, SCOPE_ARCHIVED}:
        scope = SCOPE_CANONICAL
    return {
        "ranking_scope": scope,
        "ranking_update": str(data.get("ranking_update") or "auto"),
        "canonical_rankings": bool(data.get("canonical_rankings", scope == SCOPE_CANONICAL)),
        "rankings_dir": data.get("rankings_dir"),
    }


def excluded_run_ids(exclusions: Dict[str, Any]) -> Set[str]:
    return set(str(k) for k in (exclusions.get("excluded_runs") or {}).keys()) | set(
        str(k) for k in (exclusions.get("archived_runs") or {}).keys()
    )


def excluded_model_keys(exclusions: Dict[str, Any]) -> Set[str]:
    return set(str(k).casefold() for k in (exclusions.get("excluded_models") or {}).keys())


def model_matches(row_or_model: Dict[str, Any], excluded_keys: Set[str]) -> bool:
    if not excluded_keys:
        return False
    candidates = {
        str(row_or_model.get("model") or ""),
        str(row_or_model.get("display_name") or ""),
        str(row_or_model.get("digest") or ""),
        str(row_or_model.get("model_digest_resolved") or ""),
    }
    for name in row_or_model.get("names_seen") or []:
        candidates.add(str(name))
    identity = row_or_model.get("model_identity") or {}
    if isinstance(identity, dict):
        candidates.add(str(identity.get("digest") or ""))
    return any(candidate.casefold() in excluded_keys for candidate in candidates if candidate)


def summarize_exclusions(exclusions: Dict[str, Any]) -> Dict[str, int]:
    return {
        "excluded_runs": len(exclusions.get("excluded_runs") or {}),
        "archived_runs": len(exclusions.get("archived_runs") or {}),
        "excluded_models": len(exclusions.get("excluded_models") or {}),
    }
