"""Composed operational preflight (Anvil Stage 1.3, ANVIL_MASTER_PLAN.md v2.2).

One composed detection sequence -- GPU inventory -> backend discovery ->
backend selection -> VRAM/topology budget -- resolved exactly once per
real (non-read-only) CLI invocation and threaded down explicitly as an
immutable :class:`PreflightResult`, rather than re-detected independently
at each call site. Prior to this module, GPU detection alone was triggered
from six-plus separate places (``Config.load()``, ``cli._client()`` via
``discover_runtimes()``, ``cmd_run``'s own explicit ``detect_gpus()`` call,
the campaign runtime-identity path, ``placement.topology_for_config()``,
and ``doctor.py``), each recomputing the same information independently --
see ``local_only/anvil/ANVIL_PROGRESS.md``'s Stage 1.3 research for the
full inventory. This module composes the first three of those into one
call; the remaining call sites (``runner.py``'s per-task needle KV
estimate, ``cmd_campaign``, ``cmd_inventory``) are deliberately deferred to
a follow-up slice rather than folded in here, to keep this change bounded
and independently verifiable.

This module discovers, validates, and resolves. It does not launch, load,
or switch models (Stage 3B/4's job), and it must not run functional
capability probes (Stage 2's sole capability-authority pipeline) --
``RuntimeBackend.capability_hints()`` stays metadata/hints only wherever
this module's result is consumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, List, Optional, Tuple

from .decision_policy import DecisionPolicy
from .hardware import GPUDevice, detect_gpus
from .placement import topology_for_config
from .runtime_profiles import RuntimeCandidate, RuntimeSelectionError, discover_runtimes, select_runtime
from .topology_budget import TopologyBudget


class OperationMode(Enum):
    """Structural classification of a CLI invocation's mutation posture.

    A property to check, not a per-command allowlist to keep in sync by
    hand (``if command not in {"report", "diff", ...}``) -- read-only
    commands (``report``, ``diff``, ``simulate``, ``serve``,
    ``repeat-report``) must never discover-and-save profiles, run probes,
    load models, launch runtimes, or write evidence.
    """

    READ_ONLY = "read_only"
    OPERATIONAL = "operational"
    MUTATING_ADMIN = "mutating_admin"


@dataclass(frozen=True)
class PreflightBlocker:
    """A typed reason preflight could not resolve a usable runtime.

    Preferred over letting the underlying ``RuntimeSelectionError`` escape
    directly, so a blocked preflight is a first-class result callers can
    inspect (and later integrate with readiness/campaign-status reporting)
    rather than only an exception to catch.
    """

    reason: str
    detail: str


@dataclass(frozen=True)
class PreflightResult:
    """Immutable, resolved-once snapshot of the pre-execution environment.

    Downstream code must consume this result rather than re-detecting GPUs
    or re-selecting a backend for the same invocation -- a backend
    appearing or disappearing between planning and execution would
    otherwise silently change the experiment mid-run.
    """

    gpu_inventory: Tuple[GPUDevice, ...]
    candidates: Tuple[RuntimeCandidate, ...]
    selected_candidate: Optional[RuntimeCandidate]
    topology: TopologyBudget
    blocker: Optional[PreflightBlocker] = None

    @property
    def blocked(self) -> bool:
        return self.blocker is not None


def resolve_operational_preflight(
    cfg: Any,
    *,
    explicit_profile: Optional[str] = None,
    default_profile: Optional[str] = None,
    interactive: bool = False,
    policy: Optional[DecisionPolicy] = None,
    store_path: Any = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    gpu_inventory: Optional[Iterable[GPUDevice]] = None,
    inventory_fn: Callable[[], Iterable[GPUDevice]] = detect_gpus,
    discover_fn: Callable[..., List[RuntimeCandidate]] = discover_runtimes,
    select_fn: Callable[..., RuntimeCandidate] = select_runtime,
    topology_fn: Callable[..., TopologyBudget] = topology_for_config,
) -> PreflightResult:
    """Run the composed GPU -> backend discovery -> selection -> topology sequence once.

    ``discover_fn``/``select_fn``/``inventory_fn``/``topology_fn`` are
    injectable (matching this codebase's existing pattern, e.g.
    ``discover_runtimes(..., http_probe=...)``) so a caller module's own
    monkeypatched references -- not just the ones imported here -- are
    exactly what gets exercised; this keeps existing test fixtures that
    patch a caller's local name for these functions valid without change.
    """
    inventory = tuple(inventory_fn() if gpu_inventory is None else gpu_inventory)
    candidates = tuple(discover_fn(cfg, store_path=store_path, gpu_devices=inventory))
    topology = topology_fn(cfg, inventory=inventory)
    try:
        selected = select_fn(
            candidates, explicit_profile=explicit_profile, default_profile=default_profile,
            interactive=interactive, policy=policy, input_fn=input_fn, output_fn=output_fn,
        )
    except RuntimeSelectionError as exc:
        return PreflightResult(inventory, candidates, None, topology, PreflightBlocker(exc.reason, str(exc)))
    return PreflightResult(inventory, candidates, selected, topology, None)
