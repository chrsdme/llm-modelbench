"""Additive RC21 Stage 6D runtime telemetry snapshot assembly.

This module composes the Stage 6B/6C public APIs.  It deliberately has no
import-time collection and no dependency on runner, reports, or lifecycle code.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Callable, Iterable, Optional, Tuple
from urllib.parse import urlsplit

from .hardware import GPUDevice, detect_gpus
from .process_telemetry import (
    GPUProcessCollectionResult, ProcessDiscoveryResult, RuntimeAttributionResult,
    attribute_runtime_gpus, collect_nvidia_gpu_processes, discover_runtime_processes,
    reconcile_gpu_process_stable_identities,
    nvidia_process_command,
)
from .telemetry import (
    GPUCollectionResult, TelemetryCollectionError, _normalise_timestamp,
    _ordered_errors, collect_nvidia_gpu_samples, join_inventory_samples,
)

RUNTIME_TELEMETRY_SCHEMA_VERSION = 1
_PHYSICAL_UUID = re.compile(r"^GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$")
_COMPLETENESS = {"socket_evidence_complete", "live_gpu_telemetry_available", "gpu_process_query_available", "physical_inventory_available"}


def _physical_uuid(value: object) -> str:
    text = str(value or "").strip()
    if not _PHYSICAL_UUID.fullmatch(text):
        raise ValueError("physical GPU UUID must be canonical NVIDIA GPU-UUID")
    return text


def _endpoint_port(endpoint: Optional[str]) -> Optional[int]:
    if not endpoint:
        return None
    try:
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("endpoint must be an http(s) URL without credentials")
        return parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None


def _error(operation: str, detail: object) -> TelemetryCollectionError:
    return TelemetryCollectionError(operation, "failed", str(detail))


def _snapshot_errors(errors: Iterable[TelemetryCollectionError]) -> Tuple[TelemetryCollectionError, ...]:
    """Keep nested evidence intact while making the aggregate summary concise."""
    unique = {(item.operation, item.state, item.detail, item.query_tier): item for item in errors}
    return _ordered_errors(unique.values())


@dataclass(frozen=True)
class RuntimeTelemetrySnapshot:
    """A deterministic, partial-evidence-safe view of one external runtime."""

    schema_version: int
    timestamp_utc: str
    backend: str
    endpoint: Optional[str]
    runtime_profile: Optional[str]
    endpoint_available: bool
    process_discovery: ProcessDiscoveryResult
    physical_inventory: Tuple[GPUDevice, ...]
    live_gpu_telemetry: GPUCollectionResult
    gpu_processes: GPUProcessCollectionResult
    attribution: RuntimeAttributionResult
    declared_gpu_uuids: Tuple[str, ...]
    observed_gpu_uuids: Tuple[str, ...]
    errors: Tuple[TelemetryCollectionError, ...]
    completeness: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_TELEMETRY_SCHEMA_VERSION:
            raise ValueError("unsupported runtime telemetry snapshot schema")
        if self.backend not in {"ollama", "llama_cpp"}:
            raise ValueError("runtime telemetry backend must be ollama or llama_cpp")
        object.__setattr__(self, "timestamp_utc", _normalise_timestamp(self.timestamp_utc))
        declared = tuple(sorted({_physical_uuid(value) for value in self.declared_gpu_uuids}))
        observed = tuple(sorted({_physical_uuid(value) for value in self.observed_gpu_uuids}))
        if observed != tuple(self.attribution.observed_gpu_uuids):
            raise ValueError("observed GPU UUIDs must derive from runtime attribution")
        object.__setattr__(self, "declared_gpu_uuids", declared)
        object.__setattr__(self, "observed_gpu_uuids", observed)
        inventory = tuple(self.physical_inventory)
        inventory_uuids = [item.uuid for item in inventory if item.uuid is not None]
        if any(not _PHYSICAL_UUID.fullmatch(value) for value in inventory_uuids):
            raise ValueError("physical inventory contains an invalid GPU UUID")
        if len(inventory_uuids) != len(set(inventory_uuids)):
            raise ValueError("physical inventory UUIDs must be unique")
        if inventory_uuids and not set(observed).issubset(set(inventory_uuids)):
            raise ValueError("observed GPU UUIDs must exist in available physical inventory")
        object.__setattr__(self, "physical_inventory", tuple(sorted(inventory, key=lambda item: (item.uuid or "", item.physical_index))))
        object.__setattr__(self, "errors", _snapshot_errors(self.errors))
        completeness = tuple(sorted({str(item).strip() for item in self.completeness if str(item).strip()}))
        if any(item not in _COMPLETENESS for item in completeness):
            raise ValueError("unknown runtime telemetry completeness indicator")
        object.__setattr__(self, "completeness", completeness)

    @property
    def declared_only_gpu_uuids(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.declared_gpu_uuids) - set(self.observed_gpu_uuids)))

    @property
    def observed_undeclared_gpu_uuids(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.observed_gpu_uuids) - set(self.declared_gpu_uuids)))

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "timestamp_utc": self.timestamp_utc,
            "backend": self.backend,
            "runtime_profile": self.runtime_profile,
            "endpoint": self.endpoint,
            "endpoint_available": self.endpoint_available,
            "process_discovery": self.process_discovery.to_dict(),
            "physical_inventory": [item.__dict__ for item in self.physical_inventory],
            "live_gpu_telemetry": self.live_gpu_telemetry.to_dict(),
            "gpu_processes": self.gpu_processes.to_dict(),
            "attribution": self.attribution.to_dict(),
            "declared_gpu_uuids": list(self.declared_gpu_uuids),
            "observed_gpu_uuids": list(self.observed_gpu_uuids),
            "declared_only_gpu_uuids": list(self.declared_only_gpu_uuids),
            "observed_undeclared_gpu_uuids": list(self.observed_undeclared_gpu_uuids),
            "errors": [item.to_dict() for item in self.errors],
            "completeness": list(self.completeness),
        }


def collect_runtime_telemetry(*, backend: str, endpoint: Optional[str], runtime_profile: Optional[str] = None,
                              declared_gpu_uuids: Iterable[str] = (), timestamp_utc: Optional[object] = None,
                              inventory_collector: Callable[[], Iterable[GPUDevice]] = detect_gpus,
                              live_collector: Callable[[], GPUCollectionResult] = collect_nvidia_gpu_samples,
                              process_collector: Callable[..., ProcessDiscoveryResult] = discover_runtime_processes,
                              gpu_process_collector: Callable[[], GPUProcessCollectionResult] = collect_nvidia_gpu_processes) -> RuntimeTelemetrySnapshot:
    """Collect one bounded before/query/after envelope; callers opt in explicitly."""
    if backend not in {"ollama", "llama_cpp"}:
        raise ValueError("backend must be ollama or llama_cpp")
    timestamp = _normalise_timestamp(datetime.now(timezone.utc) if timestamp_utc is None else timestamp_utc)
    declared = tuple(declared_gpu_uuids)
    port = _endpoint_port(endpoint)
    errors = []
    try:
        before = process_collector(backend=backend, endpoint_port=port)
    except Exception as exc:
        before = ProcessDiscoveryResult((), (_error("runtime_process_before", exc),), timestamp,
                                        inspected_pid_count=0, socket_evidence_complete=False)
    try:
        inventory = tuple(inventory_collector())
    except Exception as exc:
        inventory = (); errors.append(_error("physical_inventory", exc))
    try:
        live = live_collector()
    except Exception as exc:
        live = GPUCollectionResult((), (_error("live_gpu_telemetry", exc),), (), None, False, timestamp, "runtime-telemetry")
    try:
        gpu_processes = gpu_process_collector()
    except Exception as exc:
        gpu_processes = GPUProcessCollectionResult((), (_error("gpu_processes", exc),), timestamp,
                                                   nvidia_process_command(), False)
    worker_hints = tuple(sorted({sample.pid for sample in gpu_processes.samples}))
    try:
        worker_before = process_collector(backend=backend, endpoint_port=port, pid_hints=worker_hints,
                                          retain_pid_hints=True)
    except Exception as exc:
        worker_before = ProcessDiscoveryResult((), (_error("runtime_worker_before", exc),), timestamp,
                                                inspected_pid_count=0, socket_evidence_complete=False)
    try:
        after = process_collector(backend=backend, endpoint_port=port, pid_hints=worker_hints,
                                  retain_pid_hints=True)
    except Exception as exc:
        after = ProcessDiscoveryResult((), (_error("runtime_process_after", exc),), timestamp,
                                       inspected_pid_count=0, socket_evidence_complete=False)
    reconciled, reconcile_errors = reconcile_gpu_process_stable_identities(
        gpu_processes.samples, before=tuple(before.processes) + tuple(worker_before.processes), after=after.processes
    )
    attribution = attribute_runtime_gpus(
        backend=backend, endpoint_port=port, processes=after.processes,
        gpu_process_samples=reconciled, declared_gpu_uuids=declared,
        timestamp_utc=timestamp, errors=tuple(reconcile_errors),
        socket_evidence_complete=before.socket_evidence_complete and after.socket_evidence_complete,
    )
    # Stage 6B's UUID join remains its own evidence; do not synthesize inventory rows.
    try:
        joined = join_inventory_samples(inventory, live.samples)
        errors.extend(joined.errors)
    except Exception as exc:
        errors.append(_error("inventory_live_join", exc))
    completeness = []
    if before.socket_evidence_complete and after.socket_evidence_complete:
        completeness.append("socket_evidence_complete")
    if live.successful_tier is not None:
        completeness.append("live_gpu_telemetry_available")
    if gpu_processes.successful:
        completeness.append("gpu_process_query_available")
    if inventory:
        completeness.append("physical_inventory_available")
    endpoint_available = bool(port is not None and any(port in item.listening_ports for item in after.processes))
    errors.extend(before.errors); errors.extend(worker_before.errors); errors.extend(after.errors); errors.extend(live.errors); errors.extend(gpu_processes.errors); errors.extend(attribution.errors)
    return RuntimeTelemetrySnapshot(RUNTIME_TELEMETRY_SCHEMA_VERSION, timestamp, backend, endpoint, runtime_profile,
                                    endpoint_available, after, inventory, live, gpu_processes, attribution,
                                    declared, attribution.observed_gpu_uuids, tuple(errors), tuple(completeness))
