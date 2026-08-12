"""Central unattended decision policy (Anvil Stage 1.5, ANVIL_MASTER_PLAN.md v2.2).

``--unattended`` is a decision *policy*, not a synonym for "yes to
everything." It authorizes specific, enumerated automatic decisions;
anything not explicitly permitted must return a typed blocker instead of
prompting on stdin or silently proceeding. Deeper code is expected to ask
``policy.permits(Action.X)`` rather than inspect CLI flags directly, so that
later stages (3B/4/6B) can add new gated actions without touching every
call site that currently does ``if args.unattended: ...``.

Deliberately excluded from this policy on purpose: ``--yes``,
``--auto-confirm``, and ``--force`` remain their own explicit flags with
their own existing semantics (see ``cli.py``'s ``_confirm_profile_change``/
``_confirm_destructive_compute`` and ``repair.py``'s sudo/NOPASSWD-gated
auto-confirm path, documented in ``docs/auto_confirm_sudoers.md``).
``unattended=True`` must never imply privileged or destructive authority --
a user asking for an unattended benchmark run does not thereby authorize
host mutation or repair. Stage 8 owns eventual public CLI consolidation of
these flags; this module only normalizes them into one internal decision
surface, it does not rename or remove any of them.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    """A specific automatic decision some caller wants policy authorization for."""

    BACKEND_AUTO_SELECT = "backend_auto_select"


@dataclass(frozen=True)
class DecisionPolicy:
    """What an unattended run is authorized to decide automatically.

    ``unattended=False`` (the default) permits nothing -- every gated
    action returns "not permitted," matching today's behavior of never
    silently resolving ambiguity. Even under ``unattended=True``, each
    action must be separately, explicitly permitted; there is no blanket
    grant.
    """

    unattended: bool = False
    allow_backend_auto_selection: bool = False

    def permits(self, action: Action) -> bool:
        if not self.unattended:
            return False
        if action is Action.BACKEND_AUTO_SELECT:
            return self.allow_backend_auto_selection
        return False


DEFAULT_POLICY = DecisionPolicy()
