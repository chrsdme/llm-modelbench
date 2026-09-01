"""Authoritative, topology-aware GPU memory budgets and placement advice.

The model is deliberately pure: callers supply inventory and any live samples.
CUDA ordinals are never used as identity; a physical GPU UUID is the key.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Tuple

from .hardware import GPUDevice


MIB = 1024 * 1024
FIT_LABELS = {"single_gpu_fit", "candidate_single_gpu_fit", "multi_gpu_conditional_fit", "cpu_spill_required", "confirmed_no_fit", "unknown"}


def _bytes_from_mib(value: Optional[float]) -> Optional[int]:
    return None if value is None else max(0, int(float(value) * MIB))


@dataclass(frozen=True)
class GPUMemoryBudget:
    uuid: str
    pci_bus_id: Optional[str]
    installed_capacity_bytes: Optional[int]
    live_used_bytes: Optional[int] = None
    live_free_bytes: Optional[int] = None
    policy_ceiling_bytes: Optional[int] = None
    runtime_reclaimable_bytes: int = 0
    unrelated_nonreclaimable_bytes: int = 0
    backend_usable_limit_bytes: Optional[int] = None
    display_active: Optional[bool] = None
    # NVIDIA host-level ordinal (nvidia-smi index).  Execution convenience only:
    # the UUID above remains the durable identity.  Carried so placement can
    # honour "primary GPU / GPU0 first" deterministically without treating the
    # ordinal as identity.  Absent evidence sorts after every known ordinal.
    physical_index: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.uuid:
            raise ValueError("GPU budget requires a stable physical UUID")
        if self.physical_index is not None and (isinstance(self.physical_index, bool) or not isinstance(self.physical_index, int) or self.physical_index < 0):
            raise ValueError("physical_index must be a non-negative host ordinal")
        for field in ("installed_capacity_bytes", "live_used_bytes", "live_free_bytes", "policy_ceiling_bytes", "runtime_reclaimable_bytes", "unrelated_nonreclaimable_bytes", "backend_usable_limit_bytes"):
            value = getattr(self, field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{field} must be a non-negative byte count")

    @property
    def effective_now_bytes(self) -> Optional[int]:
        """Immediately allocatable capacity; display activity is evidence, not a tax."""
        candidates = [value for value in (self.policy_ceiling_bytes, self.live_free_bytes, self.backend_usable_limit_bytes) if value is not None]
        if not candidates:
            return self.installed_capacity_bytes
        return min(candidates)

    @property
    def effective_after_reclaim_bytes(self) -> Optional[int]:
        now = self.effective_now_bytes
        if now is None:
            return None
        # Only explicitly selected-runtime residency is reclaimable.  In
        # particular, unrelated CUDA processes never enter this calculation.
        ceiling = min(value for value in (self.policy_ceiling_bytes, self.backend_usable_limit_bytes) if value is not None) if any(value is not None for value in (self.policy_ceiling_bytes, self.backend_usable_limit_bytes)) else None
        value = now + self.runtime_reclaimable_bytes
        return min(value, ceiling) if ceiling is not None else value

    def to_dict(self) -> dict:
        return {"uuid": self.uuid, "pci_bus_id": self.pci_bus_id, "installed_capacity_bytes": self.installed_capacity_bytes,
                "live_used_bytes": self.live_used_bytes, "live_free_bytes": self.live_free_bytes,
                "policy_ceiling_bytes": self.policy_ceiling_bytes, "runtime_reclaimable_bytes": self.runtime_reclaimable_bytes,
                "unrelated_nonreclaimable_bytes": self.unrelated_nonreclaimable_bytes,
                "backend_usable_limit_bytes": self.backend_usable_limit_bytes, "effective_now_bytes": self.effective_now_bytes,
                "effective_after_reclaim_bytes": self.effective_after_reclaim_bytes, "display_active": self.display_active,
                "physical_index": self.physical_index}


@dataclass(frozen=True)
class TopologyBudget:
    devices: Tuple[GPUMemoryBudget, ...]
    aggregate_policy_cap_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        devices = tuple(sorted(self.devices, key=lambda item: item.uuid))
        if len({item.uuid for item in devices}) != len(devices):
            raise ValueError("GPU budget UUIDs must be unique")
        object.__setattr__(self, "devices", devices)

    @property
    def placement_order(self) -> Tuple[GPUMemoryBudget, ...]:
        """Devices in deterministic placement priority: primary GPU / GPU0 first.

        Keyed on the host ordinal (``physical_index``), then PCI bus id, then
        UUID -- every tie-breaker is a stable physical property, never live
        capacity.  A device without a known ordinal sorts after every device
        that has one.  ``devices`` itself stays UUID-sorted for identity-stable
        enumeration; this is the order the placement decision walks.
        """
        _NO_ORDINAL = (1, 0)

        def key(item: GPUMemoryBudget):
            ordinal = _NO_ORDINAL if item.physical_index is None else (0, item.physical_index)
            return (ordinal, item.pci_bus_id or "", item.uuid)

        return tuple(sorted(self.devices, key=key))

    @property
    def max_single_effective_bytes(self) -> Optional[int]:
        values = [item.effective_now_bytes for item in self.devices if item.effective_now_bytes is not None]
        return max(values) if values else None

    @property
    def aggregate_effective_bytes(self) -> Optional[int]:
        values = [item.effective_now_bytes for item in self.devices]
        if not values or any(value is None for value in values):
            return None
        total = sum(values)
        return min(total, self.aggregate_policy_cap_bytes) if self.aggregate_policy_cap_bytes is not None else total

    def to_dict(self) -> dict:
        return {"devices": [item.to_dict() for item in self.devices], "max_single_effective_bytes": self.max_single_effective_bytes,
                "aggregate_effective_bytes": self.aggregate_effective_bytes, "aggregate_policy_cap_bytes": self.aggregate_policy_cap_bytes}


def topology_from_inventory(inventory: Iterable[GPUDevice], *, live_by_uuid_mib: Optional[Mapping[str, Mapping[str, float]]] = None,
                            policy_ceilings_mib: Optional[Mapping[str, float]] = None,
                            backend_usable_limits_mib: Optional[Mapping[str, float]] = None,
                            runtime_reclaimable_mib: Optional[Mapping[str, float]] = None,
                            unrelated_mib: Optional[Mapping[str, float]] = None,
                            aggregate_policy_cap_mib: Optional[float] = None) -> TopologyBudget:
    """Build budget records from physical inventory and optional bounded evidence."""
    live_by_uuid, policy_ceilings_mib = live_by_uuid_mib or {}, policy_ceilings_mib or {}
    backend_usable_limits_mib, runtime_reclaimable_mib, unrelated_mib = backend_usable_limits_mib or {}, runtime_reclaimable_mib or {}, unrelated_mib or {}
    records = []
    for device in inventory:
        if not device.uuid:
            continue
        live = live_by_uuid.get(device.uuid, {})
        raw_index = getattr(device, "physical_index", None)
        physical_index = raw_index if isinstance(raw_index, int) and not isinstance(raw_index, bool) and raw_index >= 0 else None
        records.append(GPUMemoryBudget(device.uuid, device.pci_bus_id, _bytes_from_mib(device.total_vram_mb),
            _bytes_from_mib(live.get("used")), _bytes_from_mib(live.get("free")), _bytes_from_mib(policy_ceilings_mib.get(device.uuid)),
            _bytes_from_mib(runtime_reclaimable_mib.get(device.uuid)) or 0, _bytes_from_mib(unrelated_mib.get(device.uuid)) or 0,
            _bytes_from_mib(backend_usable_limits_mib.get(device.uuid)), live.get("display_active"), physical_index))
    return TopologyBudget(tuple(records), _bytes_from_mib(aggregate_policy_cap_mib))


@dataclass(frozen=True)
class WorkloadFit:
    classification: str
    selected_gpu_uuids: Tuple[str, ...]
    required_bytes: Optional[int]
    unknown_components: Tuple[str, ...]
    detail: str

    def __post_init__(self) -> None:
        if self.classification not in FIT_LABELS:
            raise ValueError("invalid topology fit classification")


def evaluate_workload_fit(topology: TopologyBudget, *, weight_bytes: Optional[int], kv_cache_bytes: Optional[int] = None,
                          runtime_overhead_bytes: Optional[int] = None, device_overhead_bytes: Optional[int] = None,
                          selected_gpu_uuid: Optional[str] = None, allow_cpu_spill: bool = False) -> WorkloadFit:
    """Prefer one suitable physical card; propose layer split only when needed."""
    components = {"weights": weight_bytes, "requested_context_kv": kv_cache_bytes, "runtime_overhead": runtime_overhead_bytes, "device_overhead": device_overhead_bytes}
    unknown = tuple(name for name, value in components.items() if value is None)
    known_total = sum(value for value in components.values() if value is not None)
    required = None if unknown else known_total
    # Walk devices in placement priority (primary GPU / GPU0 first), not by
    # capacity.  An explicit operator selection narrows the pool to that one
    # physical card and is always honoured first.
    eligible = [item for item in topology.placement_order if selected_gpu_uuid is None or item.uuid == selected_gpu_uuid]
    if selected_gpu_uuid and not eligible:
        return WorkloadFit("unknown", (), required, unknown, "selected physical GPU UUID is absent from topology")
    singles = [item for item in eligible if item.effective_now_bytes is not None and known_total <= item.effective_now_bytes]
    if singles:
        # The first fitting card in placement priority -- the primary GPU when it
        # fits, even if a later card is larger.  Placement is about which card,
        # not the fastest card.
        uuid = singles[0].uuid
        return WorkloadFit("single_gpu_fit" if not unknown else "candidate_single_gpu_fit", (uuid,), required, unknown, "complete workload fits one physical GPU" if not unknown else "known workload lower bound fits one physical GPU; unknown components remain")
    aggregate = sum(item.effective_now_bytes or 0 for item in eligible)
    if topology.aggregate_policy_cap_bytes is not None:
        aggregate = min(aggregate, topology.aggregate_policy_cap_bytes)
    if len(eligible) > 1 and known_total <= aggregate:
        # Multi-GPU is a capacity fallback, never an optimisation: take the
        # minimum number of cards that together hold the workload, accumulating
        # in placement priority.  Membership and order are fully deterministic.
        subset: list = []
        running = 0
        for item in eligible:
            subset.append(item)
            running += item.effective_now_bytes or 0
            if known_total <= running:
                break
        ordered = tuple(item.uuid for item in subset)
        return WorkloadFit("multi_gpu_conditional_fit", ordered, required, unknown or ("per_device_overhead_distribution",), "no single GPU fits; layer split is conditional")
    if allow_cpu_spill:
        return WorkloadFit("cpu_spill_required", (), required, unknown, "GPU topology cannot safely fit the workload without CPU/RAM spill")
    if eligible and known_total > aggregate:
        return WorkloadFit("confirmed_no_fit", (), required, (), "known complete workload exceeds eligible GPU capacity")
    return WorkloadFit("unknown", (), required, unknown or ("no_eligible_gpu_capacity",), "topology capacity is unavailable or workload is incomplete")
