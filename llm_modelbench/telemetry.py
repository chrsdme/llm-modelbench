"""Pure, bounded NVIDIA physical-GPU sampling for RC21 Stage 6B.

This module is deliberately not wired into legacy scalar telemetry.  It has no
import-time collection side effects and exposes injectable subprocess seams for
fixture-driven tests.
"""
from __future__ import annotations

import csv
import io
import math
import subprocess
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Iterable, List, Optional, Tuple

from .hardware import GPUDevice


NVIDIA_SMI_TIMEOUT_SECONDS = 5.0
MAX_NVIDIA_SMI_STDOUT_BYTES = 1024 * 1024
MAX_NVIDIA_SMI_STDERR_BYTES = 64 * 1024
MAX_TELEMETRY_ERRORS = 64
MAX_TELEMETRY_DETAIL_CHARS = 512

BASELINE_QUERY_FIELDS = (
    "index", "uuid", "pci.bus_id", "name", "memory.total", "memory.used",
    "memory.free", "utilization.gpu", "temperature.gpu", "power.draw",
)
EXTENDED_QUERY_FIELDS = BASELINE_QUERY_FIELDS + (
    "utilization.memory", "power.limit", "clocks.current.graphics",
    "clocks.current.memory", "pstate",
)
QUERY_FIELDS = {"baseline": BASELINE_QUERY_FIELDS, "extended": EXTENDED_QUERY_FIELDS}

_METRIC_FIELDS = (
    "memory_total_mib", "memory_used_mib", "memory_free_mib",
    "utilization_gpu_pct", "utilization_memory_pct", "temperature_gpu_c",
    "power_draw_w", "power_limit_w", "graphics_clock_mhz", "memory_clock_mhz", "pstate",
)
_FIELD_ORDER = {field: index for index, field in enumerate(_METRIC_FIELDS)}
_FIELD_TO_QUERY = {
    "memory_total_mib": "memory.total", "memory_used_mib": "memory.used",
    "memory_free_mib": "memory.free", "utilization_gpu_pct": "utilization.gpu",
    "utilization_memory_pct": "utilization.memory", "temperature_gpu_c": "temperature.gpu",
    "power_draw_w": "power.draw", "power_limit_w": "power.limit",
    "graphics_clock_mhz": "clocks.current.graphics", "memory_clock_mhz": "clocks.current.memory",
    "pstate": "pstate",
}
_FIELD_LABELS = {value: key for key, value in _FIELD_TO_QUERY.items()}
_FIELD_STATES = {"observed", "unsupported", "unavailable", "failed", "malformed", "not_queried"}
_ERROR_STATES = {"unsupported", "unavailable", "failed", "malformed"}
_NOT_SUPPORTED = {"[not supported]", "not supported"}
_PERMISSION = {"[insufficient permissions]", "insufficient permissions"}
_UNAVAILABLE = {"", "n/a", "na", "[n/a]", "unknown"}


def _bounded_detail(value: object) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).replace("\x00", " ").split())
    return text[:MAX_TELEMETRY_DETAIL_CHARS] or None


def _normalise_timestamp(value: object) -> str:
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp_utc must be an ISO-8601 timestamp") from exc
    else:
        raise TypeError("timestamp_utc must be a datetime or ISO-8601 string")
    if timestamp.tzinfo is None:
        raise ValueError("timestamp_utc must include a timezone")
    return timestamp.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def utc_now() -> str:
    return _normalise_timestamp(datetime.now(timezone.utc))


@dataclass(frozen=True)
class TelemetryFieldState:
    field: str
    state: str
    detail: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise ValueError("telemetry field state requires a non-empty field")
        if self.state not in _FIELD_STATES:
            raise ValueError(f"invalid telemetry field state: {self.state}")
        detail = _bounded_detail(self.detail)
        if self.detail is not None and detail != self.detail:
            object.__setattr__(self, "detail", detail)
        if self.state == "observed" and detail is not None:
            raise ValueError("observed telemetry fields must not carry failure detail")

    def to_dict(self) -> dict:
        return {"field": self.field, "state": self.state, "detail": self.detail}


@dataclass(frozen=True)
class PhysicalGPUIdentity:
    uuid: str
    pci_bus_id: Optional[str] = None
    observed_index: Optional[int] = None
    name: Optional[str] = None
    driver_version: Optional[str] = None
    compute_capability: Optional[str] = None

    def __post_init__(self) -> None:
        uuid = str(self.uuid or "").strip()
        if not uuid or _marker_state(uuid):
            raise ValueError("physical GPU identity requires a usable UUID")
        object.__setattr__(self, "uuid", uuid)
        if self.observed_index is not None:
            if not isinstance(self.observed_index, int) or isinstance(self.observed_index, bool) or self.observed_index < 0:
                raise ValueError("observed GPU index must be a non-negative integer")
        for field in ("pci_bus_id", "name", "driver_version", "compute_capability"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _normalise_optional_identity(value))

    def to_dict(self) -> dict:
        return {
            "uuid": self.uuid, "pci_bus_id": self.pci_bus_id,
            "observed_index": self.observed_index, "name": self.name,
            "driver_version": self.driver_version,
            "compute_capability": self.compute_capability,
        }


def _finite_nonnegative(value: Optional[float], field: str, *, percent: bool = False) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric or None")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    if percent and float(value) > 100:
        raise ValueError(f"{field} must be between 0 and 100")


@dataclass(frozen=True)
class GPUSample:
    identity: PhysicalGPUIdentity
    timestamp_utc: str
    collector_source: str
    query_tier: str
    memory_total_mib: Optional[float]
    memory_used_mib: Optional[float]
    memory_free_mib: Optional[float]
    utilization_gpu_pct: Optional[float]
    utilization_memory_pct: Optional[float]
    temperature_gpu_c: Optional[float]
    power_draw_w: Optional[float]
    power_limit_w: Optional[float]
    graphics_clock_mhz: Optional[float]
    memory_clock_mhz: Optional[float]
    pstate: Optional[str]
    field_states: Tuple[TelemetryFieldState, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, PhysicalGPUIdentity):
            raise TypeError("GPU sample identity must be PhysicalGPUIdentity")
        object.__setattr__(self, "timestamp_utc", _normalise_timestamp(self.timestamp_utc))
        if not isinstance(self.collector_source, str) or not self.collector_source.strip():
            raise ValueError("GPU sample requires a collector source")
        if self.query_tier not in QUERY_FIELDS:
            raise ValueError("GPU sample query tier must be baseline or extended")
        for field in _METRIC_FIELDS[:-1]:
            _finite_nonnegative(getattr(self, field), field, percent=field.startswith("utilization_"))
        if self.pstate is not None and (not isinstance(self.pstate, str) or not self.pstate.strip()):
            raise ValueError("pstate must be a non-empty string or None")
        states = tuple(self.field_states)
        by_field = {state.field: state for state in states}
        if len(by_field) != len(states):
            raise ValueError("duplicate telemetry field state records are invalid")
        if set(by_field) != set(_METRIC_FIELDS):
            raise ValueError("GPU sample requires exactly one state for every metric field")
        for field in _METRIC_FIELDS:
            value = getattr(self, field)
            state = by_field[field]
            if value is None and state.state == "observed":
                raise ValueError(f"{field} is None but marked observed")
            if value is not None and state.state != "observed":
                raise ValueError(f"{field} has a value but is not marked observed")
        object.__setattr__(self, "field_states", tuple(sorted(states, key=lambda state: _FIELD_ORDER[state.field])))

    def to_dict(self) -> dict:
        return {
            "identity": self.identity.to_dict(), "timestamp_utc": self.timestamp_utc,
            "collector_source": self.collector_source, "query_tier": self.query_tier,
            **{field: getattr(self, field) for field in _METRIC_FIELDS},
            "field_states": [state.to_dict() for state in self.field_states],
        }


@dataclass(frozen=True)
class TelemetryCollectionError:
    operation: str
    state: str
    detail: str
    query_tier: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError("telemetry collection error requires an operation")
        if self.state not in _ERROR_STATES:
            raise ValueError(f"invalid collection error state: {self.state}")
        detail = _bounded_detail(self.detail)
        if not detail:
            raise ValueError("telemetry collection error requires detail")
        object.__setattr__(self, "detail", detail)
        if self.query_tier is not None and self.query_tier not in QUERY_FIELDS:
            raise ValueError("invalid telemetry query tier")

    def to_dict(self) -> dict:
        return {"operation": self.operation, "state": self.state,
                "detail": self.detail, "query_tier": self.query_tier}


def _ordered_errors(errors: Iterable[TelemetryCollectionError]) -> Tuple[TelemetryCollectionError, ...]:
    values = sorted(errors, key=lambda error: (error.operation, error.query_tier or "", error.state, error.detail))
    if len(values) <= MAX_TELEMETRY_ERRORS:
        return tuple(values)
    truncated = values[:MAX_TELEMETRY_ERRORS - 1]
    truncated.append(TelemetryCollectionError("collection", "failed", "telemetry errors truncated"))
    return tuple(truncated)


@dataclass(frozen=True)
class GPUCollectionResult:
    samples: Tuple[GPUSample, ...]
    errors: Tuple[TelemetryCollectionError, ...]
    attempted_tiers: Tuple[str, ...]
    successful_tier: Optional[str]
    fallback_used: bool
    timestamp_utc: str
    source: str

    def __post_init__(self) -> None:
        attempted_tiers = tuple(self.attempted_tiers)
        if len(attempted_tiers) != len(set(attempted_tiers)):
            raise ValueError("attempted tiers must not contain duplicates")
        for tier in attempted_tiers:
            if tier not in QUERY_FIELDS:
                raise ValueError("attempted tier is invalid")
        if self.successful_tier is not None and self.successful_tier not in QUERY_FIELDS:
            raise ValueError("successful tier is invalid")
        if self.successful_tier is not None and self.successful_tier not in attempted_tiers:
            raise ValueError("successful tier must be attempted")
        samples = tuple(self.samples)
        if samples and self.successful_tier is None:
            raise ValueError("samples require a successful tier")
        if not isinstance(self.fallback_used, bool):
            raise ValueError("fallback_used must be boolean")
        if self.fallback_used and attempted_tiers != ("extended", "baseline"):
            raise ValueError("fallback requires exactly extended then baseline attempts")
        if not self.fallback_used and attempted_tiers == ("extended", "baseline"):
            raise ValueError("extended-to-baseline attempts require fallback_used")
        if len(attempted_tiers) == 2 and not self.fallback_used:
            raise ValueError("two collector attempts require fallback_used")
        object.__setattr__(self, "timestamp_utc", _normalise_timestamp(self.timestamp_utc))
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("collection result requires a source")
        uuids = [sample.identity.uuid for sample in samples]
        if len(uuids) != len(set(uuids)):
            raise ValueError("collection result samples require unique UUIDs")
        object.__setattr__(self, "attempted_tiers", attempted_tiers)
        object.__setattr__(self, "samples", tuple(sorted(samples, key=lambda sample: sample.identity.uuid)))
        object.__setattr__(self, "errors", _ordered_errors(self.errors))

    def to_dict(self) -> dict:
        return {
            "samples": [sample.to_dict() for sample in self.samples],
            "errors": [error.to_dict() for error in self.errors],
            "attempted_tiers": list(self.attempted_tiers), "successful_tier": self.successful_tier,
            "fallback_used": self.fallback_used, "timestamp_utc": self.timestamp_utc,
            "source": self.source,
        }


@dataclass(frozen=True)
class CommandResult:
    returncode: Optional[int]
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    stdout_overflow: bool = False
    stderr_overflow: bool = False
    executable_missing: bool = False
    runner_failure: Optional[str] = None


CommandRunner = Callable[[Tuple[str, ...], float, int, int], CommandResult]


def command_for_tier(tier: str) -> Tuple[str, ...]:
    try:
        fields = QUERY_FIELDS[tier]
    except KeyError as exc:
        raise ValueError("query tier must be baseline or extended") from exc
    return ("nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits")


def _bounded_subprocess(command: Tuple[str, ...], timeout: float, max_stdout: int, max_stderr: int) -> CommandResult:
    """Run without a shell and drain pipes into bounded in-memory buffers."""
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, shell=False)
    except FileNotFoundError:
        return CommandResult(None, executable_missing=True)
    except OSError as exc:
        return CommandResult(None, runner_failure=_bounded_detail(exc))

    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    overflow = {"stdout": False, "stderr": False}

    def drain(name: str, stream: object, limit: int) -> None:
        reader = stream  # typing aid for the injected-free standard-library path
        while True:
            chunk = reader.read(8192)  # type: ignore[union-attr]
            if not chunk:
                return
            remaining = limit - len(buffers[name])
            if remaining > 0:
                buffers[name].extend(chunk[:remaining])
            if len(chunk) > remaining:
                overflow[name] = True

    assert process.stdout is not None and process.stderr is not None
    threads = [
        threading.Thread(target=drain, args=("stdout", process.stdout, max_stdout), daemon=True),
        threading.Thread(target=drain, args=("stderr", process.stderr, max_stderr), daemon=True),
    ]
    for thread in threads:
        thread.start()
    timed_out = False
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        returncode = process.wait()
    for thread in threads:
        thread.join(timeout=1.0)
    return CommandResult(
        returncode, buffers["stdout"].decode("utf-8", "replace"),
        buffers["stderr"].decode("utf-8", "replace"), timed_out,
        overflow["stdout"], overflow["stderr"],
    )


def _marker_state(value: str) -> Optional[str]:
    marker = value.strip().lower()
    if marker in _NOT_SUPPORTED:
        return "unsupported"
    if marker in _PERMISSION:
        return "unavailable"
    if marker in _UNAVAILABLE:
        return "unavailable"
    return None


def _normalise_optional_identity(value: object) -> Optional[str]:
    """Keep identity text unless nvidia-smi emitted a recognised missing marker."""
    text = str(value).strip()
    return None if not text or _marker_state(text) else text


def _numeric(value: str, field: str, *, percent: bool = False) -> Tuple[Optional[float], TelemetryFieldState]:
    marker = _marker_state(value)
    if marker:
        return None, TelemetryFieldState(field, marker, value.strip() or None)
    try:
        parsed = float(value.strip())
    except (TypeError, ValueError):
        return None, TelemetryFieldState(field, "malformed", "invalid numeric value")
    if not math.isfinite(parsed) or parsed < 0 or (percent and parsed > 100):
        return None, TelemetryFieldState(field, "malformed", "numeric value outside allowed range")
    return parsed, TelemetryFieldState(field, "observed")


def _pstate(value: str) -> Tuple[Optional[str], TelemetryFieldState]:
    marker = _marker_state(value)
    if marker:
        return None, TelemetryFieldState("pstate", marker, value.strip() or None)
    text = value.strip()
    if not text:
        return None, TelemetryFieldState("pstate", "unavailable")
    return text, TelemetryFieldState("pstate", "observed")


def _not_queried_states(tier: str) -> List[TelemetryFieldState]:
    queried = set(QUERY_FIELDS[tier])
    return [TelemetryFieldState(field, "not_queried") for field, query in _FIELD_TO_QUERY.items() if query not in queried]


def parse_nvidia_gpu_csv(raw_text: str, *, query_tier: str, timestamp_utc: object,
                         source: str = "nvidia-smi") -> GPUCollectionResult:
    """Parse one bounded fixed-tier CSV response without executing a command."""
    if query_tier not in QUERY_FIELDS:
        raise ValueError("query tier must be baseline or extended")
    timestamp = _normalise_timestamp(timestamp_utc)
    fields = QUERY_FIELDS[query_tier]
    errors: List[TelemetryCollectionError] = []
    try:
        rows = list(csv.reader(io.StringIO(raw_text or "", newline=""), strict=True))
    except csv.Error as exc:
        error = TelemetryCollectionError("csv", "malformed", f"invalid CSV: {exc}", query_tier)
        return GPUCollectionResult((), (error,), (query_tier,), None, False, timestamp, source)
    data_rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not data_rows:
        errors.append(TelemetryCollectionError("csv", "unavailable", "nvidia-smi returned no GPU rows", query_tier))
    schema_errors = []
    for row_number, row in enumerate(data_rows, 1):
        if len(row) != len(fields):
            direction = "missing" if len(row) < len(fields) else "extra"
            schema_errors.append(TelemetryCollectionError(
                "csv_schema", "malformed", f"row {row_number} has {direction} columns", query_tier))
    if schema_errors:
        return GPUCollectionResult((), tuple(errors) + tuple(schema_errors), (query_tier,), None,
                                   False, timestamp, source)

    row_values = [(row_number, dict(zip(fields, row))) for row_number, row in enumerate(data_rows, 1)]
    uuid_counts = {}
    for _, values in row_values:
        uuid = values["uuid"].strip()
        if uuid and not _marker_state(uuid):
            uuid_counts[uuid] = uuid_counts.get(uuid, 0) + 1
    duplicate_uuids = {uuid for uuid, count in uuid_counts.items() if count > 1}
    for uuid in sorted(duplicate_uuids):
        errors.append(TelemetryCollectionError("csv_uuid", "malformed", f"duplicate GPU UUID: {uuid}", query_tier))

    samples: List[GPUSample] = []
    for row_number, values in row_values:
        uuid = values["uuid"].strip()
        if _marker_state(uuid) or not uuid:
            errors.append(TelemetryCollectionError("csv_uuid", "malformed", f"row {row_number} has no usable UUID", query_tier))
            continue
        if uuid in duplicate_uuids:
            continue
        index_value = values["index"].strip()
        observed_index: Optional[int]
        try:
            observed_index = int(index_value)
            if observed_index < 0:
                raise ValueError
        except (TypeError, ValueError):
            observed_index = None
            errors.append(TelemetryCollectionError("csv_index", "malformed", f"row {row_number} has invalid GPU index", query_tier))
        pci = _normalise_optional_identity(values["pci.bus_id"])
        name = _normalise_optional_identity(values["name"])
        identity = PhysicalGPUIdentity(uuid, pci_bus_id=pci, observed_index=observed_index, name=name)
        metrics = {field: None for field in _METRIC_FIELDS}
        states: List[TelemetryFieldState] = []
        for query_name, field in _FIELD_LABELS.items():
            if query_name not in values:
                continue
            value, state = (_pstate(values[query_name]) if field == "pstate" else _numeric(
                values[query_name], field, percent=field.startswith("utilization_")))
            metrics[field] = value
            states.append(state)
        states.extend(_not_queried_states(query_tier))
        samples.append(GPUSample(identity=identity, timestamp_utc=timestamp, collector_source=source,
                                 query_tier=query_tier, field_states=tuple(states), **metrics))
    return GPUCollectionResult(tuple(samples), tuple(errors), (query_tier,), query_tier,
                               False, timestamp, source)


def _command_error(result: CommandResult, tier: str) -> Optional[TelemetryCollectionError]:
    if result.executable_missing:
        return TelemetryCollectionError("nvidia-smi", "unavailable", "nvidia-smi executable not found", tier)
    if result.timed_out:
        return TelemetryCollectionError("nvidia-smi", "failed", "nvidia-smi command timed out", tier)
    if result.stdout_overflow:
        return TelemetryCollectionError("nvidia-smi", "failed", "nvidia-smi stdout exceeded bounded limit", tier)
    if result.stderr_overflow:
        return TelemetryCollectionError("nvidia-smi", "failed", "nvidia-smi stderr exceeded bounded limit", tier)
    if result.runner_failure:
        return TelemetryCollectionError("nvidia-smi", "failed", result.runner_failure, tier)
    if result.returncode != 0:
        detail = _bounded_detail(result.stderr) or "nvidia-smi returned non-zero exit"
        return TelemetryCollectionError("nvidia-smi", "failed", detail, tier)
    return None


def _invoke_runner(runner: CommandRunner, tier: str) -> CommandResult:
    try:
        result = runner(command_for_tier(tier), NVIDIA_SMI_TIMEOUT_SECONDS,
                        MAX_NVIDIA_SMI_STDOUT_BYTES, MAX_NVIDIA_SMI_STDERR_BYTES)
        stdout_overflow = result.stdout_overflow or len(result.stdout.encode("utf-8", "replace")) > MAX_NVIDIA_SMI_STDOUT_BYTES
        stderr_overflow = result.stderr_overflow or len(result.stderr.encode("utf-8", "replace")) > MAX_NVIDIA_SMI_STDERR_BYTES
        if stdout_overflow or stderr_overflow:
            return replace(result, stdout_overflow=stdout_overflow, stderr_overflow=stderr_overflow)
        return result
    except Exception as exc:
        return CommandResult(None, runner_failure=_bounded_detail(exc))


def collect_nvidia_gpu_samples(*, runner: CommandRunner = _bounded_subprocess,
                               timestamp_utc: Optional[object] = None,
                               source: str = "nvidia-smi") -> GPUCollectionResult:
    """Run the fixed extended tier and, only where safe, one baseline fallback."""
    timestamp = _normalise_timestamp(datetime.now(timezone.utc) if timestamp_utc is None else timestamp_utc)
    attempted: List[str] = []
    errors: List[TelemetryCollectionError] = []
    extended = _invoke_runner(runner, "extended")
    attempted.append("extended")
    command_error = _command_error(extended, "extended")
    if command_error is None:
        parsed = parse_nvidia_gpu_csv(extended.stdout, query_tier="extended", timestamp_utc=timestamp, source=source)
        schema_failure = any(error.operation == "csv_schema" for error in parsed.errors)
        if parsed.successful_tier == "extended":
            return GPUCollectionResult(parsed.samples, parsed.errors, tuple(attempted), "extended", False, timestamp, source)
        if not schema_failure:
            return GPUCollectionResult((), parsed.errors, tuple(attempted), None, False, timestamp, source)
        errors.extend(parsed.errors)
        fallback_allowed = True
    else:
        errors.append(command_error)
        fallback_allowed = extended.returncode is not None and not any((
            extended.executable_missing, extended.timed_out, extended.stdout_overflow,
            extended.stderr_overflow, bool(extended.runner_failure),
        ))
    if not fallback_allowed:
        return GPUCollectionResult((), tuple(errors), tuple(attempted), None, False, timestamp, source)

    baseline = _invoke_runner(runner, "baseline")
    attempted.append("baseline")
    command_error = _command_error(baseline, "baseline")
    if command_error is not None:
        errors.append(command_error)
        return GPUCollectionResult((), tuple(errors), tuple(attempted), None, True, timestamp, source)
    parsed = parse_nvidia_gpu_csv(baseline.stdout, query_tier="baseline", timestamp_utc=timestamp, source=source)
    successful_tier = "baseline" if parsed.successful_tier == "baseline" else None
    samples = parsed.samples if successful_tier else ()
    return GPUCollectionResult(samples, tuple(errors) + parsed.errors, tuple(attempted), successful_tier,
                               True, timestamp, source)


@dataclass(frozen=True)
class GPUInventoryJoinResult:
    samples: Tuple[GPUSample, ...]
    errors: Tuple[TelemetryCollectionError, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", tuple(sorted(self.samples, key=lambda sample: sample.identity.uuid)))
        object.__setattr__(self, "errors", _ordered_errors(self.errors))

    def to_dict(self) -> dict:
        return {"samples": [sample.to_dict() for sample in self.samples],
                "errors": [error.to_dict() for error in self.errors]}


def join_inventory_samples(inventory: Iterable[GPUDevice], samples: Iterable[GPUSample]) -> GPUInventoryJoinResult:
    """Purely enrich live samples from existing inventory, joining only by UUID."""
    errors: List[TelemetryCollectionError] = []
    inventory_groups = {}
    for device in inventory:
        uuid = str(device.uuid or "").strip()
        if uuid and not _marker_state(uuid):
            inventory_groups.setdefault(uuid, []).append(device)
    ambiguous_uuids = {uuid for uuid, devices in inventory_groups.items() if len(devices) > 1}
    for uuid in sorted(ambiguous_uuids):
        errors.append(TelemetryCollectionError("inventory_uuid", "malformed", f"duplicate inventory GPU UUID: {uuid}"))
    by_uuid = {uuid: devices[0] for uuid, devices in inventory_groups.items() if uuid not in ambiguous_uuids}
    sample_values = tuple(samples)
    joined = []
    sample_uuids = {sample.identity.uuid for sample in sample_values}
    for sample in sample_values:
        if sample.identity.uuid in ambiguous_uuids:
            joined.append(sample)
            continue
        device = by_uuid.get(sample.identity.uuid)
        if device is None:
            errors.append(TelemetryCollectionError("inventory_join", "unavailable", f"live sample UUID not present in inventory: {sample.identity.uuid}"))
            joined.append(sample)
            continue
        if device.pci_bus_id and sample.identity.pci_bus_id and device.pci_bus_id != sample.identity.pci_bus_id:
            errors.append(TelemetryCollectionError("inventory_pci", "malformed", f"PCI bus ID differs for GPU UUID: {sample.identity.uuid}"))
        identity = replace(sample.identity, driver_version=device.driver_version,
                           compute_capability=device.compute_capability)
        joined.append(replace(sample, identity=identity))
    for uuid in sorted(set(by_uuid) - sample_uuids):
        errors.append(TelemetryCollectionError("inventory_join", "unavailable", f"inventory UUID not present in live samples: {uuid}"))
    return GPUInventoryJoinResult(tuple(joined), tuple(errors))
