"""Model/task filtering helpers for fast staged benchmark runs.

These filters are intentionally conservative and dependency-free. They exist to avoid
wasting deep runs on context aliases that only differ by context/KV-cache settings.
Quality ranking should be done on base weights; context aliases should be tested only
for long-context needle/offload behaviour.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .tasks import Task

# Name markers that usually indicate a context/window alias rather than new weights.
# ``exp`` is only matched as a separated token, to avoid false positives inside words.
CONTEXT_ALIAS_RE = re.compile(
    r"(?i)(?:"
    r"(?:^|[-_:./])(?:64k|128k|32k|16k)(?:$|[-_:./])|"
    r"(?:^|[-_:./])ctx(?:$|[-_:./])|"
    r"context|long[-_ ]?context|"
    r"(?:^|[-_:./])exp(?:$|[-_:./])"
    r")"
)

CONTEXT_TASK_IDS = {"needle"}
CONTEXT_TASK_CATEGORIES = {"long_context"}


@dataclass(frozen=True)
class Skip:
    model: str
    reason: str


def is_context_alias(name: str) -> bool:
    """Return True when a model name appears to be a context/window alias.

    This deliberately uses name markers rather than fingerprinting, because it must work
    before a benchmark run. It catches aliases such as ``hermes3-8b-64k`` and
    ``qwen25-coder-14b-64k-exp`` while avoiding broad matches like ``experimental``.
    """
    return bool(CONTEXT_ALIAS_RE.search(name or ""))


def filter_models(
    models: Sequence[str],
    *,
    family_base_only: bool = False,
    context_aliases_only: bool = False,
) -> Tuple[List[str], List[Skip]]:
    """Apply context-alias model filters and return ``(kept, skipped)``.

    ``family_base_only`` and ``context_aliases_only`` are mutually exclusive by CLI
    contract, but this function still handles accidental double-use by preferring the
    narrower ``context_aliases_only`` interpretation.
    """
    kept: List[str] = []
    skipped: List[Skip] = []
    for model in models:
        alias = is_context_alias(model)
        if context_aliases_only:
            if alias:
                kept.append(model)
            else:
                skipped.append(Skip(model, "not_context_alias"))
        elif family_base_only:
            if alias:
                skipped.append(Skip(model, "context_alias_skipped"))
            else:
                kept.append(model)
        else:
            kept.append(model)
    return kept, skipped


def parse_task_ids(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    ids = [x.strip() for x in raw.split(",") if x.strip()]
    return ids or None


def filter_tasks(
    tasks: Sequence[Task],
    *,
    task_ids: Optional[Sequence[str]] = None,
    task_regex: Optional[str] = None,
    context_only: bool = False,
) -> List[Task]:
    """Filter task list by explicit IDs, regex, and/or context-only shortcut."""
    out = list(tasks)
    if context_only:
        out = [t for t in out if t.id in CONTEXT_TASK_IDS or t.category in CONTEXT_TASK_CATEGORIES or t.scorer == "needle"]
    if task_ids:
        wanted = {t.strip() for t in task_ids if str(t).strip()}
        out = [t for t in out if t.id in wanted]
    if task_regex:
        rx = re.compile(task_regex, re.I)
        out = [t for t in out if rx.search(t.id) or rx.search(t.category) or rx.search(t.scorer)]
    return out


def validate_task_ids(task_ids: Optional[Iterable[str]], available: Iterable[str]) -> List[str]:
    """Return task IDs that do not exist, preserving user spelling."""
    if not task_ids:
        return []
    known = set(available)
    return [tid for tid in task_ids if tid not in known]


def describe_filters(
    *,
    task_ids: Optional[Sequence[str]] = None,
    task_regex: Optional[str] = None,
    family_base_only: bool = False,
    context_aliases_only: bool = False,
    context_only: bool = False,
) -> List[str]:
    parts: List[str] = []
    if task_ids:
        parts.append("tasks=" + ",".join(task_ids))
    if task_regex:
        parts.append(f"task_regex={task_regex}")
    if family_base_only:
        parts.append("family_base_only")
    if context_aliases_only:
        parts.append("context_aliases_only")
    if context_only:
        parts.append("context_only")
    return parts
