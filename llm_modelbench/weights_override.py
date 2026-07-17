"""Strict parsing for report-time category-weight overrides."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Optional


def parse_weight_overrides(spec: Optional[str], defaults: Dict[str, float]) -> Dict[str, float]:
    merged = dict(defaults)
    if not spec:
        return merged
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"malformed weight override {part!r}, expected category=value")
        category, value = (item.strip() for item in part.split("=", 1))
        if not category or category not in defaults:
            raise ValueError(f"unknown category {category!r}. Known: {sorted(defaults)}")
        try:
            merged[category] = float(value)
        except ValueError as exc:
            raise ValueError(f"malformed weight value for {category!r}: {value!r}") from exc
    return merged


def copy_run_for_override(source: Path, destination: Path) -> Path:
    """Copy a completed run before writing an override report, preserving the source."""
    if destination.exists():
        raise ValueError(f"override report output already exists: {destination}")
    shutil.copytree(source, destination)
    return destination
