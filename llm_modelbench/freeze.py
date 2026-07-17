"""Create and verify a compact, reproducible pre-release regression freeze."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from . import __version__
from .rankings import _CURRENT_HASHES


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _source_files(repo_root: Path) -> List[Path]:
    files: List[Path] = []
    for root in (repo_root / "llm_modelbench", repo_root / "tests"):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".html", ".json"} and "__pycache__" not in path.parts:
                files.append(path)
    for name in ("pyproject.toml", "CHANGELOG.md", "README.md"):
        path = repo_root / name
        if path.exists():
            files.append(path)
    return sorted(set(files))


def _snapshot_files(out_dir: Path, *, include_checksums: bool = False) -> List[Path]:
    excluded = {"SHA256SUMS.txt", "SHA256SUMS.local.txt"}
    return sorted(
        path for path in out_dir.rglob("*")
        if path.is_file() and (include_checksums or path.name not in excluded)
    )


def verify_freeze(out_dir: Path) -> Dict[str, Any]:
    """Verify a snapshot independently of the caller's current directory."""
    out_dir = Path(out_dir)
    checksum_path = out_dir / "SHA256SUMS.local.txt"
    if not checksum_path.exists():
        raise ValueError(f"missing portable checksum manifest: {checksum_path}")
    checked = 0
    failures = []
    for line in checksum_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            expected, rel = line.split(None, 1)
            rel = rel.strip().lstrip("*")
        except ValueError:
            failures.append({"path": None, "reason": f"invalid checksum line: {line}"})
            continue
        path = out_dir / rel
        if not path.exists():
            failures.append({"path": rel, "reason": "missing"})
            continue
        actual = _sha256(path)
        checked += 1
        if actual != expected:
            failures.append({"path": rel, "reason": "checksum_mismatch", "expected": expected, "actual": actual})
    return {
        "snapshot": str(out_dir),
        "checked": checked,
        "passed": not failures,
        "failures": failures,
        "checksum_path": str(checksum_path),
    }


def create_freeze(
    repo_root: Path,
    runs_dir: Path,
    rankings_dir: Path,
    out_dir: Path,
    *,
    label: str = "pre-rankings-v3",
    include_rankings: bool = True,
) -> Dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    runs_dir = Path(runs_dir)
    rankings_dir = Path(rankings_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _json(rankings_dir / "master_summary.json")
    raw_path = rankings_dir / "master_raw.jsonl"
    payload_path = rankings_dir / "master_report_data.json"
    if not isinstance(summary, list):
        raise ValueError(f"cannot read rankings summary: {rankings_dir / 'master_summary.json'}")

    status_counts = Counter(str(row.get("quality_status") or "unknown") for row in summary)
    source_hashes = {
        str(path.relative_to(repo_root)): _sha256(path)
        for path in _source_files(repo_root)
    }
    model_expectations = []
    for row in sorted(summary, key=lambda item: str(item.get("display_name") or "")):
        profile = row.get("long_context_profile") or {}
        model_expectations.append({
            "model": row.get("display_name"),
            "digest": row.get("digest"),
            "quality_status": row.get("quality_status"),
            "overall_mean_score": row.get("overall_mean_score"),
            "coverage_ratio": row.get("coverage_ratio"),
            "capability_limited": bool(row.get("capability_limited")),
            "capability_measured_failure": bool(row.get("capability_measured_failure")),
            "recovery_limited": bool(row.get("recovery_limited")),
            "max_verified_ctx": profile.get("max_verified_ctx"),
            "target_status": profile.get("target_status"),
        })

    manifest = {
        "schema_version": 2,
        "label": label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "llm_modelbench_version": __version__,
        "repo_root": str(repo_root),
        "runs_dir": str(runs_dir),
        "rankings_dir": str(rankings_dir),
        "rankings": {
            "models": len(summary),
            "status_counts": dict(sorted(status_counts.items())),
            "raw_rows": sum(1 for line in raw_path.read_text().splitlines() if line.strip()) if raw_path.exists() else None,
            "master_summary_sha256": _sha256(rankings_dir / "master_summary.json"),
            "master_raw_sha256": _sha256(raw_path) if raw_path.exists() else None,
            "master_report_data_sha256": _sha256(payload_path) if payload_path.exists() else None,
            "master_report_v3_data_sha256": _sha256(rankings_dir / "master_report_v3_data.json") if (rankings_dir / "master_report_v3_data.json").exists() else None,
            "master_report_v3_html_sha256": _sha256(rankings_dir / "master_report_v3.html") if (rankings_dir / "master_report_v3.html").exists() else None,
            "master_report_v3_1_data_sha256": _sha256(rankings_dir / "master_report_v3_1_data.json") if (rankings_dir / "master_report_v3_1_data.json").exists() else None,
            "master_report_v3_1_html_sha256": _sha256(rankings_dir / "master_report_v3_1.html") if (rankings_dir / "master_report_v3_1.html").exists() else None,
            "exclusions_sha256": _sha256(rankings_dir / "exclusions.json") if (rankings_dir / "exclusions.json").exists() else None,
            "audit_log_sha256": _sha256(rankings_dir / "audit_log.jsonl") if (rankings_dir / "audit_log.jsonl").exists() else None,
        },
        "task_contract_hashes": dict(sorted(_CURRENT_HASHES.items())),
        "source_hashes": source_hashes,
        "model_expectations": model_expectations,
    }
    manifest_path = out_dir / "freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (out_dir / "task_contracts.json").write_text(json.dumps(manifest["task_contract_hashes"], indent=2, sort_keys=True))
    (out_dir / "ranking_expectations.json").write_text(json.dumps({
        "label": label,
        "version": __version__,
        "models": len(summary),
        "status_counts": dict(sorted(status_counts.items())),
        "rows": manifest["rankings"]["raw_rows"],
        "model_expectations": model_expectations,
    }, indent=2, sort_keys=True))

    copied: List[str] = []
    if include_rankings:
        snapshot_dir = out_dir / "rankings"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "master_summary.json",
            "master_report_data.json",
            "master_report.html",
            "master_report_v3_data.json",
            "master_report_v3.html",
            "master_report_v3_1_data.json",
            "master_report_v3_1.html",
            "exclusions.json",
            "audit_log.jsonl",
        ):
            src = rankings_dir / name
            if src.exists():
                shutil.copy2(src, snapshot_dir / name)
                copied.append(str(snapshot_dir / name))
        split_src = rankings_dir / "v3_1"
        if split_src.exists():
            split_dst = snapshot_dir / "v3_1"
            if split_dst.exists():
                shutil.rmtree(split_dst)
            shutil.copytree(split_src, split_dst)
            for copied_file in sorted(path for path in split_dst.rglob("*") if path.is_file()):
                copied.append(str(copied_file))

    readme = [
        f"# LLM ModelBench regression freeze: {label}",
        "",
        f"- Version: `{__version__}`",
        f"- Models: `{len(summary)}`",
        f"- Status counts: `{dict(sorted(status_counts.items()))}`",
        f"- Raw ranking rows: `{manifest['rankings']['raw_rows']}`",
        "",
        "This snapshot freezes task hashes, source hashes, ranking expectations, and selected ranking artifacts before Rankings V3.",
        "It contains no model weights, prompts from private documents, or system credentials.",
        "",
        "Verify from any directory with:",
        "",
        "```bash",
        "./VERIFY.sh",
        "```",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n")
    verify_script = out_dir / "VERIFY.sh"
    verify_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"${BASH_SOURCE[0]}\")\"\nsha256sum -c SHA256SUMS.local.txt\n"
    )
    verify_script.chmod(0o755)

    local_lines = [
        f"{_sha256(path)}  {path.relative_to(out_dir)}"
        for path in _snapshot_files(out_dir)
    ]
    local_path = out_dir / "SHA256SUMS.local.txt"
    local_path.write_text("\n".join(local_lines) + "\n")

    # The main checksum file is intentionally valid from the repository root,
    # matching the command printed by the CLI and documentation. A portable
    # snapshot-local manifest is retained beside it.
    try:
        out_prefix = out_dir.resolve().relative_to(repo_root)
        root_lines = [
            f"{_sha256(path)}  {out_prefix / path.relative_to(out_dir)}"
            for path in _snapshot_files(out_dir, include_checksums=True)
            if path.name != "SHA256SUMS.txt"
        ]
        verify_command = f"sha256sum -c {out_prefix / 'SHA256SUMS.txt'}"
    except ValueError:
        root_lines = [
            f"{_sha256(path)}  {path.resolve()}"
            for path in _snapshot_files(out_dir, include_checksums=True)
            if path.name != "SHA256SUMS.txt"
        ]
        verify_command = f"sha256sum -c {out_dir.resolve() / 'SHA256SUMS.txt'}"
    checksum_path = out_dir / "SHA256SUMS.txt"
    checksum_path.write_text("\n".join(root_lines) + "\n")

    verification = verify_freeze(out_dir)
    if not verification["passed"]:
        raise RuntimeError(f"new freeze failed self-verification: {verification['failures']}")

    return {
        "label": label,
        "out_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "checksums_path": str(checksum_path),
        "portable_checksums_path": str(local_path),
        "verify_command": verify_command,
        "verification": verification,
        "models": len(summary),
        "status_counts": dict(sorted(status_counts.items())),
        "raw_rows": manifest["rankings"]["raw_rows"],
        "copied_rankings": copied,
    }
