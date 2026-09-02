"""Conservative, backend-neutral RC21 Stage 7 runtime-fit profiling.

The pure evaluator intentionally does not know how to start a runtime, load a
model, or translate CUDA ordinals.  Collection is opt-in and uses only the
existing bounded inventory/live-telemetry and backend metadata interfaces.

**Advisory boundary (Anvil Stage 3.2C-3).**  ``runtime-fit`` is an *advisory
diagnostic lens*.  It is **not** the authoritative execution placement or
environment-feasibility path for scored benchmarks.  That authority is the
Stage 3.2 topology/preflight path
(:mod:`~llm_modelbench.topology_budget` /
:mod:`~llm_modelbench.placement` / :mod:`~llm_modelbench.ram_spill_preflight`),
consumed by :mod:`~llm_modelbench.runner`.  Nothing here feeds ``runner.py``,
``campaign.py``, the planner, or scoring: ``runtime_fit.json`` is written only
by the operator-invoked ``runtime-fit`` subcommand and reporting treats it as
advisory metadata, never as a score input.  In particular this module does
**not** evaluate host physical-RAM capacity, so it cannot and does not decide
RAM-spill feasibility -- see ``evaluate_runtime_fit``'s spill branch.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable, Iterable, Mapping, Optional, Tuple

from .hardware import GPUDevice, detect_gpus
from .telemetry import GPUCollectionResult, TelemetryCollectionError, collect_nvidia_gpu_samples, utc_now


RUNTIME_FIT_SCHEMA_VERSION = 1
DEFAULT_RESERVE_MIB = 512
_UUID = re.compile(r"^GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$")
_DECISIONS = {"confirmed_fit", "candidate_fit", "conditional_fit", "confirmed_no_fit", "unknown"}
_STRATEGIES = {None, "layer_split", "tensor_split"}
_MAX_DETAIL = 512


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return " ".join(value.split())[:_MAX_DETAIL]


def _uuid(value: object) -> str:
    text = _text(value, "physical GPU UUID")
    if not _UUID.fullmatch(text):
        raise ValueError("physical GPU UUID must be canonical NVIDIA GPU-UUID")
    return text


def _bytes(value: object, field: str, *, allow_none: bool = True) -> Optional[int]:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer byte count")
    return value


def _mib(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{field} must be a finite non-negative MiB value")
    return int(float(value) * 1024 * 1024)


def _ordered(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted({_text(value, "reason") for value in values}))


@dataclass(frozen=True)
class RuntimeFitModel:
    name: str
    weight_bytes: Optional[int]
    weight_provenance: str
    requested_context: Optional[int] = None
    model_max_context: Optional[int] = None
    runtime_overhead_bytes: Optional[int] = None
    architecture: Optional[Mapping[str, object]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "model name"))
        object.__setattr__(self, "weight_bytes", _bytes(self.weight_bytes, "weight_bytes"))
        object.__setattr__(self, "runtime_overhead_bytes", _bytes(self.runtime_overhead_bytes, "runtime_overhead_bytes"))
        object.__setattr__(self, "weight_provenance", _text(self.weight_provenance, "weight provenance"))
        for field in ("requested_context", "model_max_context"):
            value = getattr(self, field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{field} must be a positive integer or None")
        if self.architecture is not None:
            object.__setattr__(self, "architecture", dict(self.architecture))

    def to_dict(self) -> dict:
        return {"name": self.name, "weight_bytes": self.weight_bytes, "weight_provenance": self.weight_provenance,
                "requested_context": self.requested_context, "model_max_context": self.model_max_context,
                "runtime_overhead_bytes": self.runtime_overhead_bytes, "architecture": dict(self.architecture or {})}


@dataclass(frozen=True)
class RuntimeFitProfile:
    name: str
    backend: str
    physical_gpu_uuids: Tuple[str, ...] = ()
    strategy: Optional[str] = None
    allocation_weights: object = ()
    allow_cpu_spill: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "runtime profile name"))
        if self.backend not in {"ollama", "llama_cpp"}:
            raise ValueError("runtime profile backend must be ollama or llama_cpp")
        uuids = tuple(_uuid(value) for value in self.physical_gpu_uuids)
        if len(uuids) != len(set(uuids)):
            raise ValueError("runtime profile GPU UUIDs must be unique")
        object.__setattr__(self, "physical_gpu_uuids", tuple(sorted(uuids)))
        if self.strategy not in _STRATEGIES:
            raise ValueError("unsupported runtime fit strategy")
        if self.strategy is not None and len(uuids) < 2:
            raise ValueError("multi-device strategy requires at least two physical UUIDs")
        raw_weights = self.allocation_weights
        if isinstance(raw_weights, Mapping):
            pairs = tuple(raw_weights.items())
        else:
            pairs = tuple(raw_weights)
        normalized_weights = tuple(sorted(
            (_uuid(key), float(value)) for key, value in pairs
            if not isinstance(value, bool) and isinstance(value, (int, float))
        ))
        if len(normalized_weights) != len(pairs):
            raise ValueError("allocation weights must be finite positive numbers")
        if any(not math.isfinite(value) or value <= 0 for _, value in normalized_weights):
            raise ValueError("allocation weights must be finite positive numbers")
        if len({key for key, _ in normalized_weights}) != len(normalized_weights):
            raise ValueError("allocation weights must not duplicate physical GPU UUIDs")
        if normalized_weights and {key for key, _ in normalized_weights} != set(uuids):
            raise ValueError("allocation weights must cover exactly the declared GPU UUIDs")
        if normalized_weights and self.strategy is None:
            raise ValueError("allocation weights require an explicit strategy")
        object.__setattr__(self, "allocation_weights", normalized_weights)
        if self.allow_cpu_spill is not None and not isinstance(self.allow_cpu_spill, bool):
            raise ValueError("allow_cpu_spill must be bool or None")

    def to_dict(self) -> dict:
        return {"name": self.name, "backend": self.backend, "physical_gpu_uuids": list(self.physical_gpu_uuids),
                "strategy": self.strategy, "allocation_weights": {key: value for key, value in self.allocation_weights}, "allow_cpu_spill": self.allow_cpu_spill}


@dataclass(frozen=True)
class DeviceFitAssessment:
    gpu_uuid: str
    installed_capacity_bytes: Optional[int]
    live_free_capacity_bytes: Optional[int]
    reserve_bytes: int
    allocated_weight_bytes: Optional[int]
    decision: str
    reasons: Tuple[str, ...]
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "gpu_uuid", _uuid(self.gpu_uuid))
        for field in ("installed_capacity_bytes", "live_free_capacity_bytes", "reserve_bytes", "allocated_weight_bytes"):
            object.__setattr__(self, field, _bytes(getattr(self, field), field))
        if self.decision not in _DECISIONS:
            raise ValueError("invalid fit decision")
        object.__setattr__(self, "reasons", _ordered(self.reasons))
        object.__setattr__(self, "detail", _text(self.detail, "fit detail"))
        if self.decision == "confirmed_no_fit" and "lower_bound_exceeds_capacity" not in self.reasons:
            raise ValueError("confirmed_no_fit requires a reliable exceeding lower bound")

    def to_dict(self) -> dict:
        return {"gpu_uuid": self.gpu_uuid, "installed_capacity_bytes": self.installed_capacity_bytes,
                "live_free_capacity_bytes": self.live_free_capacity_bytes, "reserve_bytes": self.reserve_bytes,
                "allocated_weight_bytes": self.allocated_weight_bytes, "decision": self.decision,
                "reasons": list(self.reasons), "detail": self.detail}


@dataclass(frozen=True)
class RuntimeFitResult:
    """Advisory diagnostic estimate (Anvil Stage 3.2C-3).

    A ``conditional_fit`` with reason ``cpu_spill_permitted_host_capacity_unchecked``
    means spill is operator-permitted for this *estimate*, but host physical-RAM
    capacity is not evaluated here -- actual RAM-spill feasibility is determined
    by the benchmark execution preflight, not by this result.
    """

    schema_version: int
    timestamp_utc: str
    model: RuntimeFitModel
    profile: RuntimeFitProfile
    reserve_bytes: int
    kv_cache_bytes: Optional[int]
    kv_provenance: str
    device_assessments: Tuple[DeviceFitAssessment, ...]
    decision: str
    reasons: Tuple[str, ...]
    errors: Tuple[TelemetryCollectionError, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_FIT_SCHEMA_VERSION:
            raise ValueError("unsupported runtime fit schema")
        if self.decision not in _DECISIONS:
            raise ValueError("invalid fit decision")
        object.__setattr__(self, "reserve_bytes", _bytes(self.reserve_bytes, "reserve_bytes", allow_none=False))
        object.__setattr__(self, "kv_cache_bytes", _bytes(self.kv_cache_bytes, "kv_cache_bytes"))
        object.__setattr__(self, "kv_provenance", _text(self.kv_provenance, "KV provenance"))
        assessments = tuple(sorted(self.device_assessments, key=lambda item: item.gpu_uuid))
        if len({item.gpu_uuid for item in assessments}) != len(assessments):
            raise ValueError("duplicate device assessments")
        object.__setattr__(self, "device_assessments", assessments)
        object.__setattr__(self, "reasons", _ordered(self.reasons))
        if self.decision == "confirmed_fit" and (self.model.weight_bytes is None or self.model.runtime_overhead_bytes is None or self.kv_cache_bytes is None):
            raise ValueError("confirmed_fit requires all mandatory requirements")
        if len(self.profile.physical_gpu_uuids) > 1 and self.profile.strategy is None and self.decision == "confirmed_fit":
            raise ValueError("multi-GPU confirmed fit requires explicit strategy")

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version, "timestamp_utc": self.timestamp_utc, "model": self.model.to_dict(),
                "profile": self.profile.to_dict(), "reserve_bytes": self.reserve_bytes, "kv_cache_bytes": self.kv_cache_bytes,
                "kv_provenance": self.kv_provenance, "device_assessments": [item.to_dict() for item in self.device_assessments],
                "decision": self.decision, "reasons": list(self.reasons), "errors": [item.to_dict() for item in self.errors]}


def calculate_kv_cache_bytes(model: RuntimeFitModel) -> Tuple[Optional[int], str]:
    """Return a derived KV lower-bound only for explicit, reliable metadata."""
    if model.requested_context is None:
        return None, "unknown_requested_context"
    if model.model_max_context is not None and model.requested_context > model.model_max_context:
        return None, "requested_context_exceeds_model_maximum"
    arch = model.architecture or {}
    required = ("layer_count", "kv_head_count", "head_dimension", "kv_dtype_bytes")
    if any(key not in arch for key in required):
        return None, "unknown_architecture"
    try:
        layers, heads, dimension, dtype = (int(arch[key]) for key in required)
        sequences = int(arch.get("parallel_sequences", 1))
    except (TypeError, ValueError):
        return None, "invalid_architecture"
    if min(layers, heads, dimension, dtype, sequences) <= 0:
        return None, "invalid_architecture"
    # keys + values × layers × KV heads × head dimension × context × sequences.
    return 2 * layers * heads * dimension * model.requested_context * dtype * sequences, "derived_kv_cache_bytes"


def _capacity_decision(weight: Optional[int], total: Optional[int], free: Optional[int], reserve: int,
                       *, complete_requirements: bool) -> Tuple[str, Tuple[str, ...], str]:
    capacity = free if free is not None else total
    basis = "live free" if free is not None else "installed"
    if capacity is None:
        return "unknown", ("capacity_unavailable",), "physical capacity is unavailable"
    available = max(0, capacity - reserve)
    if weight is None:
        return "unknown", ("weight_requirement_unknown",), f"{basis} capacity is known but model-weight requirement is unknown"
    if weight > available:
        return "confirmed_no_fit", ("lower_bound_exceeds_capacity", f"{basis}_capacity"), f"model weight lower bound exceeds reserved {basis} capacity"
    if complete_requirements:
        return "confirmed_fit", (f"{basis}_capacity",), f"all modeled requirements fit reserved {basis} capacity"
    return "candidate_fit", ("unknown_runtime_overhead", f"{basis}_capacity"), f"model-weight lower bound fits reserved {basis} capacity; overhead remains unknown"


def evaluate_runtime_fit(*, inventory: Iterable[GPUDevice], live_telemetry: Optional[GPUCollectionResult],
                         model: RuntimeFitModel, profile: RuntimeFitProfile, reserve_mib: float = DEFAULT_RESERVE_MIB,
                         timestamp_utc: Optional[str] = None) -> RuntimeFitResult:
    """Pure evaluator.  It never reads procfs, invokes a process, or contacts a runtime."""
    reserve = _mib(reserve_mib, "reserve_mib")
    devices = tuple(inventory)
    by_uuid = {}
    for device in devices:
        if device.uuid is None:
            continue
        uuid = _uuid(device.uuid)
        if uuid in by_uuid:
            raise ValueError("duplicate physical inventory UUID")
        by_uuid[uuid] = device
    if profile.physical_gpu_uuids and by_uuid and not set(profile.physical_gpu_uuids).issubset(by_uuid):
        raise ValueError("runtime profile declares GPU UUID absent from authoritative inventory")
    live_by_uuid = {}
    errors = []
    if live_telemetry is not None:
        for sample in live_telemetry.samples:
            live_by_uuid[_uuid(sample.identity.uuid)] = sample.memory_free_mib
        errors.extend(live_telemetry.errors)
    kv, kv_provenance = calculate_kv_cache_bytes(model)
    complete = model.weight_bytes is not None and model.runtime_overhead_bytes is not None and kv is not None
    targets = profile.physical_gpu_uuids or tuple(sorted(by_uuid))
    if not targets:
        return RuntimeFitResult(RUNTIME_FIT_SCHEMA_VERSION, timestamp_utc or utc_now(), model, profile, reserve, kv, kv_provenance,
                                (), "unknown", ("no_physical_gpu_inventory",), tuple(errors))
    multi_without_strategy = len(targets) > 1 and profile.strategy is None
    allocation_unknown = len(targets) > 1 and profile.strategy is not None and not profile.allocation_weights
    weight_map = dict(profile.allocation_weights)
    weights = tuple(weight_map[uuid] for uuid in targets) if weight_map else tuple(1.0 for _ in targets)
    divisor = sum(weights)
    assessments = []
    for uuid, fraction in zip(targets, weights):
        device = by_uuid.get(uuid)
        installed = _mib(device.total_vram_mb, "total_vram_mb") if device and device.total_vram_mb is not None else None
        free = _mib(live_by_uuid[uuid], "memory_free_mib") if live_by_uuid.get(uuid) is not None else None
        allocated = None if model.weight_bytes is None else (
            model.weight_bytes if (multi_without_strategy or allocation_unknown) else int(model.weight_bytes * fraction / divisor)
        )
        decision, reasons, detail = _capacity_decision(allocated, installed, free, reserve, complete_requirements=complete and len(targets) == 1)
        if len(targets) > 1 and decision == "candidate_fit":
            decision, reasons, detail = "conditional_fit", tuple(reasons) + ("per_device_overhead_unknown",), "allocated model-weight lower bound fits; per-device overhead distribution is unknown"
        assessments.append(DeviceFitAssessment(uuid, installed, free, reserve, allocated, decision, reasons, detail))
    decisions = {item.decision for item in assessments}
    if multi_without_strategy:
        overall, reasons = "unknown", ("explicit_multi_device_strategy_required",)
    elif allocation_unknown:
        overall, reasons = "unknown", ("per_device_allocation_unknown",)
    elif "confirmed_no_fit" in decisions:
        overall, reasons = "confirmed_no_fit", ("lower_bound_exceeds_capacity",)
    elif profile.allow_cpu_spill:
        # Diagnostic only: spill is operator-permitted for this estimate, but
        # this advisory path does not sample host physical RAM, so it cannot
        # judge whether the overflow actually fits.  Actual RAM-spill
        # feasibility is decided by the benchmark execution preflight
        # (``ram_spill_preflight``), never here (Anvil Stage 3.2C-3).
        overall, reasons = "conditional_fit", ("cpu_spill_permitted_host_capacity_unchecked",)
    elif len(targets) > 1:
        overall, reasons = ("conditional_fit", ("explicit_layer_split", "per_device_overhead_unknown")) if profile.strategy == "layer_split" else ("unknown", ("unsupported_tensor_split_memory_model",))
    elif "unknown" in decisions:
        overall, reasons = "unknown", assessments[0].reasons
    else:
        overall, reasons = assessments[0].decision, assessments[0].reasons
    return RuntimeFitResult(RUNTIME_FIT_SCHEMA_VERSION, timestamp_utc or utc_now(), model, profile, reserve, kv, kv_provenance,
                            tuple(assessments), overall, tuple(reasons), tuple(errors))


def collect_runtime_fit(*, client: object, model_name: str, profile: RuntimeFitProfile,
                        requested_context: Optional[int] = None, reserve_mib: float = DEFAULT_RESERVE_MIB,
                        inventory_collector: Callable[[], Iterable[GPUDevice]] = detect_gpus,
                        live_collector: Callable[[], GPUCollectionResult] = collect_nvidia_gpu_samples) -> RuntimeFitResult:
    """Bounded read-only evidence wrapper; no inference or runtime mutation."""
    size = getattr(client, "model_size_bytes")(model_name)
    info = getattr(client, "model_info")(model_name)
    maximum = getattr(client, "context_length")(model_name)
    model = RuntimeFitModel(model_name, size, "backend_declared_model_size_bytes" if size is not None else "unknown",
                            requested_context=requested_context, model_max_context=maximum, architecture=info)
    try:
        inventory = tuple(inventory_collector())
    except Exception:
        inventory = ()
    try:
        live = live_collector()
    except Exception:
        live = None
    return evaluate_runtime_fit(inventory=inventory, live_telemetry=live, model=model, profile=profile, reserve_mib=reserve_mib)
