"""Shared runner/planner topology-aware placement decision."""
from __future__ import annotations

from typing import Any, Iterable, Optional

from .hardware import GPUDevice, detect_gpus
from .topology_budget import TopologyBudget, WorkloadFit, evaluate_workload_fit, topology_from_inventory


def topology_for_config(cfg: Any, inventory: Optional[Iterable[GPUDevice]] = None) -> TopologyBudget:
    """Apply explicit operator caps without converting them into host totals."""
    devices = tuple(detect_gpus() if inventory is None else inventory)
    policy = dict(getattr(cfg, "gpu_policy_ceilings_mib", {}) or {})
    legacy_gb = float(getattr(cfg, "vram_budget_gb", 0) or 0)
    if legacy_gb > 0:
        legacy_mib = legacy_gb * 1024
        for device in devices:
            if device.uuid:
                policy[device.uuid] = min(policy.get(device.uuid, legacy_mib), legacy_mib)
    return topology_from_inventory(devices, policy_ceilings_mib=policy,
                                   aggregate_policy_cap_mib=getattr(cfg, "aggregate_policy_ceiling_mib", None))


def model_placement_fit(model_row: dict, cfg: Any, *, inventory: Optional[Iterable[GPUDevice]] = None,
                        selected_gpu_uuid: Optional[str] = None, allow_cpu_spill: bool = False) -> WorkloadFit:
    """A conservative shared decision: model weight is known; runtime/KV are not."""
    size = model_row.get("size")
    weight = int(size) if isinstance(size, (int, float)) and not isinstance(size, bool) and size >= 0 else None
    return evaluate_workload_fit(topology_for_config(cfg, inventory), weight_bytes=weight,
                                 selected_gpu_uuid=selected_gpu_uuid, allow_cpu_spill=allow_cpu_spill)


def skip_offload_allowed(model_row: dict, cfg: Any, *, inventory: Optional[Iterable[GPUDevice]] = None) -> bool:
    """Skip only a definite topology no-fit; unknown overhead must not be guessed."""
    return model_placement_fit(model_row, cfg, inventory=inventory).classification != "confirmed_no_fit"
