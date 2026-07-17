"""Interactive benchmark planning wizard.

Unlike ``run --select`` (models only), the wizard edits both the model set and
the test scope before returning an accepted plan.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

from .capabilities import interrogate_models
from .planner import build_plan, render_plan
from .selection import select_models
from .tasks import TASKS


def _require_tty() -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError("wizard requires an interactive terminal; use run --models/--all with --yes in scripts")


def _parse_csv(value: str) -> Optional[List[str]]:
    values = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    return values or None


def interactive_plan(
    client: Any,
    cfg: Any,
    *,
    initial_level: str = "short",
    judge_mode: str = "off",
    initial_categories: Optional[List[str]] = None,
    initial_task_ids: Optional[List[str]] = None,
    plan_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return an accepted plan and the options used to construct it."""
    _require_tty()
    installed = [row.get("name") for row in client.tags() if row.get("name")]
    if not installed:
        raise RuntimeError("no installed Ollama models were found")
    selected = select_models(installed)
    print("\nInterrogating selected models. This uses small functional probe requests...")
    profiles = interrogate_models(client, selected, functional=True)

    level = initial_level
    categories: Optional[List[str]] = list(initial_categories) if initial_categories else None
    task_ids: Optional[List[str]] = list(initial_task_ids) if initial_task_ids else None
    current_judge = judge_mode
    fixed = dict(plan_kwargs or {})

    while True:
        plan = build_plan(
            client,
            cfg,
            **fixed,
            level=level,
            categories=categories,
            task_ids=task_ids,
            judge_mode=current_judge,
            selected_models=selected,
            auto_probe=True,
            capability_profiles=profiles,
        )
        print("\n" + render_plan(plan))
        print("\nWizard actions:")
        print("  a  accept this plan and start")
        print("  m  reselect models")
        print("  l  change level (smoke/short/full)")
        print("  c  set categories (comma-separated; blank = capability-routed defaults)")
        print("  t  set exact task IDs (comma/semicolon-separated; blank = all routed tasks)")
        print("  j  change judge mode (off/single/panel)")
        print("  q  cancel")
        action = input("Choice: ").strip().lower()
        if action in {"a", "accept"}:
            return plan, {
                "selected_models": selected,
                "capability_profiles": profiles,
                "level": level,
                "categories": categories,
                "task_ids": task_ids,
                "judge_mode": current_judge,
            }
        if action in {"q", "quit"}:
            raise SystemExit("wizard cancelled")
        if action == "m":
            selected = select_models(installed, preselected=selected)
            print("\nRe-interrogating the changed model selection...")
            profiles = interrogate_models(client, selected, functional=True)
        elif action == "l":
            value = input("Level [smoke/short/full]: ").strip().lower()
            if value in {"smoke", "short", "full"}:
                level = value
            else:
                print("Invalid level; unchanged.")
        elif action == "c":
            known = sorted({task.category for task in TASKS})
            print("Known categories: " + ", ".join(known))
            categories = _parse_csv(input("Categories (blank = automatic): ").strip())
            unknown = sorted(set(categories or []) - set(known))
            if unknown:
                print("Unknown categories: " + ", ".join(unknown) + "; reverting to automatic.")
                categories = None
        elif action == "t":
            task_ids = _parse_csv(input("Exact task IDs (blank = all routed): ").strip())
        elif action == "j":
            value = input("Judge [off/single/panel]: ").strip().lower()
            if value in {"off", "single", "panel"}:
                current_judge = value
            else:
                print("Invalid judge mode; unchanged.")
        else:
            print("Unknown action.")
