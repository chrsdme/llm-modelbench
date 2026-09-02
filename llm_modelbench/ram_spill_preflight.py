"""Anvil Stage 3.2C-2b -- explicit RAM-spill execution policy + host-RAM preflight.

RAM spill is an *explicit operator override* (amendment §6): default OFF, never
self-enabling.  This module composes -- it does not replace -- the single GPU
placement authority in :mod:`~llm_modelbench.topology_budget`.

Two small, pure things live here:

1. ``placement_label_for`` -- projects a
   :class:`~llm_modelbench.topology_budget.WorkloadFit` classification onto the
   small public execution vocabulary (§14): ``full_gpu`` / ``multi_gpu`` /
   ``ram_spill``.  This is a *projection*: ``FIT_LABELS`` in ``topology_budget``
   is untouched, so the 3.2B/3.2C-1 change-attribution fixtures do not move.

2. ``resolve_spill_preflight`` -- given the GPU fit result, the *scalar* safe
   capacity of the selected GPU pool (computed by the caller from topology,
   never re-derived here), an ``allow_ram_spill`` permission, and a host-memory
   snapshot, decides whether an otherwise GPU-infeasible workload may attempt
   GPU + physical host RAM execution.

Frozen invariants this module keeps:

* Only ``cpu_spill_required`` enters RAM preflight.  ``confirmed_no_fit`` is a
  proven complete no-fit and stays ``environment_infeasible`` regardless of the
  flag -- RAM preflight never reinterprets it.
* Swap is 0 bytes of capacity (§8/§12).  ``MemAvailable`` only; ``SwapFree`` is
  recorded as telemetry, never added.
* Unknown workload requirement or unknown host memory => fail closed
  (``environment_unknown``), never spill (§11).
* A ``ram_spill`` result retains its selected GPU UUIDs (§14) -- the resident
  portion occupies the configured GPUs, only the overflow spills.
* This is *permission-to-attempt* logic, not a prediction of exact backend
  allocation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from .topology_budget import WorkloadFit

__all__ = [
    "SAFE_RAM_FRACTION",
    "PLACEMENT_LABELS",
    "SpillPreflight",
    "placement_label_for",
    "resolve_spill_preflight",
]

_MIB = 1024 * 1024

# Single fixed conservative host-RAM safety fraction (amendment §9 max), the
# RAM analogue of ``topology_budget.SAFE_VRAM_FRACTION``.  Never adaptive,
# workload-specific, learned, or tuned.
SAFE_RAM_FRACTION = 0.85

# The complete public execution-placement vocabulary (§14).  Deliberately small
# and disjoint from ``topology_budget.FIT_LABELS``.
PLACEMENT_LABELS = ("full_gpu", "multi_gpu", "ram_spill")

# Feasibility verdicts this preflight can return.
_INFEASIBLE = "environment_infeasible"
_UNKNOWN = "environment_unknown"


@dataclass(frozen=True)
class SpillPreflight:
    """Typed result of the host-RAM spill preflight.

    ``feasible`` is ``True`` only for a resolved placement (``full_gpu`` /
    ``multi_gpu`` / ``ram_spill``); ``False`` for ``environment_infeasible``;
    ``None`` for ``environment_unknown`` (fail-closed, insufficient evidence).
    """

    feasible: Optional[bool]
    # One of PLACEMENT_LABELS when feasible is True; otherwise _INFEASIBLE / _UNKNOWN.
    resolution: str
    reason: str
    selected_gpu_uuids: Tuple[str, ...]
    required_bytes: Optional[int]
    safe_selected_gpu_capacity_bytes: Optional[int]
    estimated_ram_spill_bytes: Optional[int]
    safe_host_ram_bytes: Optional[int]
    mem_available_bytes: Optional[int]
    swap_free_bytes: Optional[int]
    swap_in_use: bool
    unknown_components: Tuple[str, ...]

    @property
    def is_ram_spill(self) -> bool:
        return self.feasible is True and self.resolution == "ram_spill"

    def to_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "resolution": self.resolution,
            "reason": self.reason,
            "selected_gpu_uuids": list(self.selected_gpu_uuids),
            "required_bytes": self.required_bytes,
            "safe_selected_gpu_capacity_bytes": self.safe_selected_gpu_capacity_bytes,
            "estimated_ram_spill_bytes": self.estimated_ram_spill_bytes,
            "safe_host_ram_bytes": self.safe_host_ram_bytes,
            "mem_available_bytes": self.mem_available_bytes,
            "swap_free_bytes": self.swap_free_bytes,
            "swap_in_use": self.swap_in_use,
            "unknown_components": list(self.unknown_components),
        }


def placement_label_for(fit: WorkloadFit) -> Optional[str]:
    """Project a GPU-resident ``WorkloadFit`` onto the public placement label.

    Returns ``full_gpu`` / ``multi_gpu`` for a GPU-resident fit, ``None`` for
    any classification that is not by itself a resolved GPU-resident placement
    (``cpu_spill_required`` -- resolution needs the RAM preflight;
    ``confirmed_no_fit`` / ``unknown`` -- not a placement at all).
    """
    if fit.classification in ("single_gpu_fit", "candidate_single_gpu_fit"):
        return "full_gpu"
    if fit.classification == "multi_gpu_conditional_fit":
        return "multi_gpu"
    return None


def _mem_bytes(snapshot: Mapping[str, object], key: str) -> Optional[int]:
    """Read one MB-valued field from a ``_read_proc_meminfo``-shaped snapshot as bytes."""
    value = snapshot.get(key)
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0:
        return None
    return int(float(value) * _MIB)


def resolve_spill_preflight(
    fit: WorkloadFit,
    *,
    safe_selected_gpu_capacity_bytes: Optional[int],
    allow_ram_spill: bool,
    host_meminfo: Mapping[str, object],
    known_workload_bytes: Optional[int] = None,
) -> SpillPreflight:
    """Compose the GPU fit result with a conservative host-RAM preflight.

    ``safe_selected_gpu_capacity_bytes`` is the §4-discounted capacity of the
    *selected* GPU pool, computed by the caller from the authoritative topology
    (never re-derived here).  ``known_workload_bytes`` is the sum of the
    *known* workload components (the same lower bound 3.2C-2a's spill branch
    fires on) -- ``WorkloadFit.required_bytes`` is ``None`` whenever any
    component is unknown, so the caller supplies this lower bound for the
    overflow arithmetic; ``fit.required_bytes`` is used when it is complete and
    this is omitted.  If neither is available the preflight fails closed (§11).
    ``host_meminfo`` is a ``_read_proc_meminfo``-shaped mapping (MB-valued
    ``ram_available_mb`` / ``swap_free_mb`` / ...); pass a fixture in tests so
    the decision never depends on the real host.
    """
    classification = fit.classification
    selected = tuple(fit.selected_gpu_uuids)
    unknown_components = tuple(fit.unknown_components)

    mem_available_bytes = _mem_bytes(host_meminfo, "ram_available_mb")
    swap_free_bytes = _mem_bytes(host_meminfo, "swap_free_mb")
    swap_used_bytes = _mem_bytes(host_meminfo, "swap_used_mb")
    swap_in_use = bool(swap_used_bytes)
    safe_host_ram_bytes = (
        int(mem_available_bytes * SAFE_RAM_FRACTION) if mem_available_bytes is not None else None
    )

    def result(feasible, resolution, reason, *, spill=None):
        return SpillPreflight(
            feasible=feasible,
            resolution=resolution,
            reason=reason,
            selected_gpu_uuids=selected if resolution in PLACEMENT_LABELS else (),
            required_bytes=fit.required_bytes,
            safe_selected_gpu_capacity_bytes=safe_selected_gpu_capacity_bytes,
            estimated_ram_spill_bytes=spill,
            safe_host_ram_bytes=safe_host_ram_bytes,
            mem_available_bytes=mem_available_bytes,
            swap_free_bytes=swap_free_bytes,
            swap_in_use=swap_in_use,
            unknown_components=unknown_components,
        )

    # --- GPU-resident placements: the flag must not change these (§5) --------
    label = placement_label_for(fit)
    if label is not None:
        return result(True, label, "gpu_resident_placement")

    # --- classifications that are not a placement ---------------------------
    if classification == "confirmed_no_fit":
        # A proven *complete* no-fit.  RAM preflight never reinterprets this.
        return result(False, _INFEASIBLE, "confirmed_no_fit_not_spill_eligible")
    if classification != "cpu_spill_required":
        # "unknown" (or any future non-placement label): fail closed.
        return result(None, _UNKNOWN, f"topology_fit_{classification}")

    # --- classification == cpu_spill_required: the only RAM-preflight entry --
    if not allow_ram_spill:
        # 3.2C-2a already proves cpu_spill_required is only returned when the
        # caller passed allow_cpu_spill=True; belt-and-braces for direct callers.
        return result(False, _INFEASIBLE, "ram_spill_not_permitted")

    # §11: without a defensible workload lower bound or selected-pool capacity
    # there is no overflow arithmetic -- never spill from an unbounded guess.
    workload_bytes = fit.required_bytes if known_workload_bytes is None else known_workload_bytes
    if workload_bytes is None or safe_selected_gpu_capacity_bytes is None:
        return result(None, _UNKNOWN, "spill_overflow_inputs_unknown")

    # The workload lower bound (weights + requested KV) is compared directly to
    # the safe selected-pool capacity: whatever the caller counts in
    # ``known_workload_bytes`` is what the GPU pool must hold before spill.
    estimated_ram_spill = max(0, workload_bytes - safe_selected_gpu_capacity_bytes)

    if estimated_ram_spill == 0:
        # The selected pool actually holds the workload once the discount is
        # applied; no host residency expected.  Treat as a GPU-resident multi
        # placement (cpu_spill_required implies the whole eligible pool).
        return result(True, "multi_gpu" if len(selected) > 1 else "full_gpu",
                      "gpu_pool_holds_workload_after_discount", spill=0)

    if mem_available_bytes is None or safe_host_ram_bytes is None:
        return result(None, _UNKNOWN, "host_ram_evidence_unavailable", spill=estimated_ram_spill)

    # §8/§12: swap contributes 0.  The comparison is against safe *physical*
    # available RAM only, no matter how large SwapFree is.
    if estimated_ram_spill <= safe_host_ram_bytes:
        return result(True, "ram_spill", "spill_within_safe_host_ram", spill=estimated_ram_spill)
    return result(False, _INFEASIBLE, "spill_exceeds_safe_host_ram", spill=estimated_ram_spill)
