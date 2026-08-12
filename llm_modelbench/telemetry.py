"""RC21 telemetry layer, merged in Anvil Stage 1.1 (ANVIL_MASTER_PLAN.md v2.2)
from three previously separate modules that formed a strict linear import
chain (``telemetry`` <- ``process_telemetry`` <- ``runtime_telemetry``):

- Physical-GPU sampling (originally Stage 6B): pure, bounded ``nvidia-smi``
  CSV collection and parsing, with no import-time collection side effects
  and injectable subprocess seams for fixture-driven tests.
- Process-to-GPU attribution evidence (originally Stage 6C): pure, bounded
  procfs process discovery and ``nvidia-smi --query-compute-apps`` sampling,
  reconciled into runtime-GPU attributions. All OS/command seams remain
  injectable for fixture tests.
- Runtime telemetry snapshot assembly (originally Stage 6D): composes the
  two sections above into one deterministic, partial-evidence-safe
  snapshot of an external runtime. Still has no import-time collection and
  no dependency on runner, reports, or lifecycle code.

The merge is additive-only: every public name each former module exported
is preserved under this one module. The two nearly-identical bounded
subprocess runners the split had accumulated (one per former module) are
consolidated into a single ``_bounded_subprocess``; the process-attribution
section's command-error classifier keeps its own name
(``_process_command_error``) since its bounded-stdout/stderr check is
inlined differently than the GPU-sample section's (which relies on
``_invoke_runner`` pre-correcting overflow flags) -- forcing those two into
one function would risk quietly changing behavior neither call site asked
for.
"""
from __future__ import annotations

import csv
import io
import math
import os
import re
import subprocess
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from .hardware import GPUDevice, detect_gpus


# ---------------------------------------------------------------------------
# Physical-GPU sampling (formerly telemetry.py, Stage 6B)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Process-to-GPU attribution evidence (formerly process_telemetry.py, Stage 6C)
# ---------------------------------------------------------------------------

PROC_ROOT = Path("/proc")
MAX_PROCESS_PIDS = 256
MAX_PROC_FILE_BYTES = 64 * 1024
MAX_PROCESS_CMDLINE_BYTES = 8 * 1024
MAX_PROCESS_CMDLINE_ARGS = 64
MAX_PROCESS_FDS = 256
MAX_PROCESS_SOCKETS = 128
MAX_OLLAMA_WORKER_ANCESTOR_DEPTH = 8
NVIDIA_PROCESS_TIMEOUT_SECONDS = 5.0
MAX_NVIDIA_PROCESS_STDOUT_BYTES = 1024 * 1024
MAX_NVIDIA_PROCESS_STDERR_BYTES = 64 * 1024
NVIDIA_PROCESS_FIELDS = ("pid", "process_name", "gpu_uuid", "used_gpu_memory")
_CONFIDENCE = {"confirmed", "probable", "profile-declared only", "unavailable"}


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _bounded_text(value: object) -> Optional[str]:
    return _bounded_detail(value)


def _normalise_sources(values: Iterable[object]) -> Tuple[str, ...]:
    return tuple(sorted({text for value in values if (text := _bounded_text(value))}))


def _substantive_process_identity(item: "RuntimeProcessIdentity") -> Tuple[object, ...]:
    """Identity evidence excludes the time at which that evidence was observed."""
    return (item.pid, item.start_time_ticks, item.parent_pid, item.executable,
            item.command_name, item.command_line, item.listening_ports,
            item.backend_hint, item.discovery_sources)


def _normalise_processes(values: Iterable["RuntimeProcessIdentity"], *, reject_ambiguous: bool) -> Tuple[Tuple["RuntimeProcessIdentity", ...], Tuple[TelemetryCollectionError, ...]]:
    groups, errors = {}, []
    for item in values:
        groups.setdefault(item.stable_identity, []).append(item)
    normalized = []
    for identity, group in sorted(groups.items(), key=lambda pair: (pair[0][0], -1 if pair[0][1] is None else pair[0][1])):
        if len({_substantive_process_identity(item) for item in group}) != 1:
            if reject_ambiguous: raise ValueError("conflicting duplicate stable process identity")
            errors.append(TelemetryCollectionError("runtime_identity", "malformed", f"ambiguous stable runtime identity: {identity[0]}")); continue
        normalized.append(max(group, key=lambda item: item.timestamp_utc))
    by_pid = {}
    for item in normalized: by_pid.setdefault(item.pid, []).append(item)
    for pid, group in sorted(by_pid.items()):
        if len(group) > 1:
            if reject_ambiguous: raise ValueError("multiple stable process identities for PID")
            errors.append(TelemetryCollectionError("runtime_pid", "malformed", f"ambiguous runtime PID identities: {pid}"))
    return tuple(sorted(normalized, key=lambda p: (p.pid, -1 if p.start_time_ticks is None else p.start_time_ticks))), _ordered_errors(errors)


@dataclass(frozen=True)
class RuntimeProcessIdentity:
    pid: int
    start_time_ticks: Optional[int]
    parent_pid: Optional[int]
    executable: Optional[str]
    command_name: Optional[str]
    command_line: Tuple[str, ...]
    listening_ports: Tuple[int, ...]
    backend_hint: Optional[str]
    discovery_sources: Tuple[str, ...]
    timestamp_utc: str

    def __post_init__(self) -> None:
        _positive_int(self.pid, "PID")
        if self.start_time_ticks is not None:
            _positive_int(self.start_time_ticks, "start_time_ticks")
        if self.parent_pid is not None and (isinstance(self.parent_pid, bool) or not isinstance(self.parent_pid, int) or self.parent_pid < 0):
            raise ValueError("parent_pid must be a non-negative integer or None")
        if self.backend_hint is not None and self.backend_hint not in {"ollama", "llama_cpp"}:
            raise ValueError("backend_hint must be ollama, llama_cpp, or None")
        object.__setattr__(self, "timestamp_utc", _normalise_timestamp(self.timestamp_utc))
        for name in ("executable", "command_name"):
            value = getattr(self, name)
            object.__setattr__(self, name, _bounded_text(value))
        raw_args = tuple(self.command_line)
        if len(raw_args) > MAX_PROCESS_CMDLINE_ARGS or any(not isinstance(arg, str) for arg in raw_args):
            raise ValueError("command line exceeds argument bound or contains non-string argument")
        if sum(len(arg.encode("utf-8")) + 1 for arg in raw_args) > MAX_PROCESS_CMDLINE_BYTES:
            raise ValueError("command line exceeds byte bound")
        object.__setattr__(self, "command_line", raw_args)
        ports = tuple(sorted(set(self.listening_ports)))
        if any(isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535 for port in ports):
            raise ValueError("listening ports must be integers from 1 through 65535")
        object.__setattr__(self, "listening_ports", ports)
        object.__setattr__(self, "discovery_sources", _normalise_sources(self.discovery_sources))

    @property
    def stable_identity(self) -> Tuple[int, Optional[int]]:
        return self.pid, self.start_time_ticks

    def to_dict(self) -> dict:
        return {"pid": self.pid, "start_time_ticks": self.start_time_ticks, "parent_pid": self.parent_pid,
                "executable": self.executable, "command_name": self.command_name,
                "command_line": list(self.command_line), "listening_ports": list(self.listening_ports),
                "backend_hint": self.backend_hint, "discovery_sources": list(self.discovery_sources),
                "timestamp_utc": self.timestamp_utc}


@dataclass(frozen=True)
class GPUProcessSample:
    gpu_uuid: str
    pid: int
    process_name: Optional[str]
    used_gpu_memory_mib: Optional[float]
    timestamp_utc: str
    collector_source: str
    field_states: Tuple[TelemetryFieldState, ...]
    process_start_time_ticks: Optional[int] = None

    def __post_init__(self) -> None:
        uuid = str(self.gpu_uuid or "").strip()
        if not uuid or _marker_state(uuid):
            raise ValueError("GPU process sample requires a usable GPU UUID")
        _positive_int(self.pid, "PID")
        if self.process_start_time_ticks is not None:
            _positive_int(self.process_start_time_ticks, "process_start_time_ticks")
        object.__setattr__(self, "gpu_uuid", uuid)
        object.__setattr__(self, "timestamp_utc", _normalise_timestamp(self.timestamp_utc))
        if not isinstance(self.collector_source, str) or not (source := _bounded_text(self.collector_source)):
            raise ValueError("GPU process sample requires a collector source")
        object.__setattr__(self, "collector_source", source)
        object.__setattr__(self, "process_name", None if _marker_state(str(self.process_name or "")) else _bounded_text(self.process_name))
        memory = self.used_gpu_memory_mib
        if memory is not None and (isinstance(memory, bool) or not isinstance(memory, (int, float)) or
                                   not math.isfinite(float(memory)) or float(memory) < 0):
            raise ValueError("used_gpu_memory_mib must be finite, non-negative, or None")
        states = tuple(self.field_states)
        if len(states) != 1 or states[0].field != "used_gpu_memory_mib":
            raise ValueError("GPU process sample requires one used_gpu_memory_mib field state")
        if (memory is None) == (states[0].state == "observed"):
            raise ValueError("GPU process memory value and field state disagree")
        object.__setattr__(self, "field_states", states)

    def to_dict(self) -> dict:
        return {"gpu_uuid": self.gpu_uuid, "pid": self.pid, "process_name": self.process_name,
                "used_gpu_memory_mib": self.used_gpu_memory_mib, "timestamp_utc": self.timestamp_utc,
                "collector_source": self.collector_source,
                "field_states": [state.to_dict() for state in self.field_states],
                "process_start_time_ticks": self.process_start_time_ticks}


@dataclass(frozen=True)
class RuntimeGPUAttribution:
    runtime_pid: Optional[int]
    runtime_start_time_ticks: Optional[int]
    gpu_uuid: Optional[str]
    process_sample: Optional[GPUProcessSample]
    confidence: str
    evidence_sources: Tuple[str, ...]
    detail: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("runtime_pid", "runtime_start_time_ticks"):
            value = getattr(self, name)
            if value is not None:
                _positive_int(value, name)
        if self.gpu_uuid is not None and (not str(self.gpu_uuid).strip() or _marker_state(str(self.gpu_uuid))):
            raise ValueError("GPU attribution UUID must be usable or None")
        if self.gpu_uuid is not None:
            object.__setattr__(self, "gpu_uuid", str(self.gpu_uuid).strip())
        if self.process_sample is not None and self.gpu_uuid != self.process_sample.gpu_uuid:
            raise ValueError("process sample GPU UUID must match attribution")
        if self.runtime_start_time_ticks is not None and self.runtime_pid is None:
            raise ValueError("runtime start time requires runtime PID")
        if self.confidence not in _CONFIDENCE:
            raise ValueError("invalid attribution confidence")
        if self.confidence == "profile-declared only" and (self.gpu_uuid is None or self.process_sample is not None):
            raise ValueError("profile-declared only attribution requires an unobserved UUID")
        object.__setattr__(self, "evidence_sources", _normalise_sources(self.evidence_sources))
        object.__setattr__(self, "detail", _bounded_text(self.detail))
        if self.confidence in {"confirmed", "probable"}:
            if self.runtime_pid is None or self.gpu_uuid is None or self.process_sample is None:
                raise ValueError(f"{self.confidence} attribution requires runtime PID, UUID, and process sample")
            if self.process_sample.pid != self.runtime_pid or not self.evidence_sources:
                raise ValueError(f"{self.confidence} attribution process evidence is contradictory")
            if self.runtime_start_time_ticks is not None and self.process_sample.process_start_time_ticks is not None and self.runtime_start_time_ticks != self.process_sample.process_start_time_ticks:
                raise ValueError("attribution has explicit mismatched stable process evidence")
        if self.confidence == "confirmed" and (self.runtime_start_time_ticks is None or
                                                self.process_sample.process_start_time_ticks != self.runtime_start_time_ticks):
            raise ValueError("confirmed attribution requires matching stable process evidence")

    def to_dict(self) -> dict:
        return {"runtime_pid": self.runtime_pid, "runtime_start_time_ticks": self.runtime_start_time_ticks,
                "gpu_uuid": self.gpu_uuid, "process_sample": self.process_sample.to_dict() if self.process_sample else None,
                "confidence": self.confidence, "evidence_sources": list(self.evidence_sources), "detail": self.detail}


@dataclass(frozen=True)
class ProcessDiscoveryResult:
    processes: Tuple[RuntimeProcessIdentity, ...]
    errors: Tuple[TelemetryCollectionError, ...]
    timestamp_utc: str
    source: str = "procfs"
    inspected_pid_count: int = 0
    socket_evidence_complete: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.inspected_pid_count, bool) or not isinstance(self.inspected_pid_count, int) or self.inspected_pid_count < 0:
            raise ValueError("inspected_pid_count must be non-negative")
        if not isinstance(self.socket_evidence_complete, bool):
            raise ValueError("socket_evidence_complete must be boolean")
        source = _bounded_text(self.source)
        if not source: raise ValueError("process discovery source is required")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "timestamp_utc", _normalise_timestamp(self.timestamp_utc))
        processes, _ = _normalise_processes(self.processes, reject_ambiguous=True)
        object.__setattr__(self, "processes", processes)
        object.__setattr__(self, "errors", _ordered_errors(self.errors))

    def to_dict(self) -> dict:
        return {"processes": [item.to_dict() for item in self.processes], "errors": [item.to_dict() for item in self.errors],
                "timestamp_utc": self.timestamp_utc, "source": self.source, "inspected_pid_count": self.inspected_pid_count,
                "socket_evidence_complete": self.socket_evidence_complete}


@dataclass(frozen=True)
class GPUProcessCollectionResult:
    samples: Tuple[GPUProcessSample, ...]
    errors: Tuple[TelemetryCollectionError, ...]
    timestamp_utc: str
    attempted_query: Tuple[str, ...]
    successful: bool

    def __post_init__(self) -> None:
        if not isinstance(self.successful, bool):
            raise ValueError("successful must be boolean")
        if not isinstance(self.attempted_query, tuple) or any(not isinstance(arg, str) or not _bounded_text(arg) for arg in self.attempted_query):
            raise ValueError("attempted query must be bounded non-empty string tuple")
        if self.attempted_query != nvidia_process_command():
            raise ValueError("GPU process query must use the fixed NVIDIA command")
        object.__setattr__(self, "timestamp_utc", _normalise_timestamp(self.timestamp_utc))
        samples = tuple(sorted(self.samples, key=lambda item: (item.gpu_uuid, item.pid)))
        if len({(item.gpu_uuid, item.pid) for item in samples}) != len(samples):
            raise ValueError("GPU process samples require unique GPU UUID/PID pairs")
        if samples and not self.successful:
            raise ValueError("GPU process samples require a successful query")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "errors", _ordered_errors(self.errors))

    def to_dict(self) -> dict:
        return {"samples": [item.to_dict() for item in self.samples], "errors": [item.to_dict() for item in self.errors],
                "timestamp_utc": self.timestamp_utc, "attempted_query": list(self.attempted_query), "successful": self.successful}


@dataclass(frozen=True)
class RuntimeAttributionResult:
    processes: Tuple[RuntimeProcessIdentity, ...]
    attributions: Tuple[RuntimeGPUAttribution, ...]
    declared_gpu_uuids: Tuple[str, ...]
    observed_gpu_uuids: Tuple[str, ...]
    errors: Tuple[TelemetryCollectionError, ...]
    timestamp_utc: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_utc", _normalise_timestamp(self.timestamp_utc))
        processes, _ = _normalise_processes(self.processes, reject_ambiguous=True)
        object.__setattr__(self, "processes", processes)
        for name in ("declared_gpu_uuids",):
            values = tuple(sorted({str(value).strip() for value in getattr(self, name) if str(value).strip() and not _marker_state(str(value))}))
            object.__setattr__(self, name, values)
        object.__setattr__(self, "attributions", tuple(sorted(self.attributions, key=lambda a: (a.gpu_uuid or "", a.runtime_pid or -1, a.confidence))))
        keys = [(item.runtime_pid, item.runtime_start_time_ticks, item.gpu_uuid) for item in self.attributions]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate runtime GPU attribution relationships are invalid")
        observed = tuple(sorted({item.gpu_uuid for item in self.attributions if item.confidence in {"confirmed", "probable"} and item.gpu_uuid}))
        supplied = tuple(sorted({str(value).strip() for value in self.observed_gpu_uuids if str(value).strip() and not _marker_state(str(value))}))
        if supplied and supplied != observed:
            raise ValueError("observed GPU UUIDs must match runtime attribution evidence")
        object.__setattr__(self, "observed_gpu_uuids", observed)
        object.__setattr__(self, "errors", _ordered_errors(self.errors))

    def to_dict(self) -> dict:
        return {"processes": [item.to_dict() for item in self.processes], "attributions": [item.to_dict() for item in self.attributions],
                "declared_gpu_uuids": list(self.declared_gpu_uuids), "observed_gpu_uuids": list(self.observed_gpu_uuids),
                "errors": [item.to_dict() for item in self.errors], "timestamp_utc": self.timestamp_utc}


def parse_proc_stat(text: str) -> Tuple[int, int, int]:
    """Return PID, parent PID, and start ticks from one Linux proc stat record."""
    match = re.match(r"^(\d+)\s+\((.*)\)\s+\S\s+(.*)$", str(text or "").strip())
    if not match:
        raise ValueError("malformed proc stat")
    pid, tail = int(match.group(1)), match.group(3).split()
    if len(tail) <= 18:
        raise ValueError("malformed proc stat")
    ppid, start = int(tail[0]), int(tail[18])
    _positive_int(pid, "PID")
    if ppid < 0 or start <= 0:
        raise ValueError("malformed proc stat")
    return pid, ppid, start


def parse_proc_net_tcp(text: str, *, with_errors: bool = False):
    """Return LISTEN socket records, optionally with bounded malformed-row evidence."""
    found, errors = set(), []
    for number, line in enumerate(str(text or "").splitlines()[1:], 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 10:
            errors.append(TelemetryCollectionError("proc_net", "malformed", f"malformed proc-net row {number}")); continue
        if not re.fullmatch(r"[0-9A-Fa-f]{2}", fields[3]):
            errors.append(TelemetryCollectionError("proc_net", "malformed", f"invalid proc-net state row {number}")); continue
        if fields[3].upper() != "0A": continue
        try:
            local = fields[1]
            if not re.fullmatch(r"[0-9A-Fa-f]+:[0-9A-Fa-f]+", local): raise ValueError
            address, port_text = local.split(":")
            if not address: raise ValueError
            port = int(port_text, 16)
            inode = fields[9]
            if not 1 <= port <= 65535 or not inode.isdigit(): raise ValueError
        except (IndexError, ValueError):
            errors.append(TelemetryCollectionError("proc_net", "malformed", f"malformed proc-net LISTEN row {number}"))
            continue
        found.add((port, inode))
    result = tuple(sorted(found))
    return (result, _ordered_errors(errors)) if with_errors else result


def _read_bounded(path: Path, limit: int = MAX_PROC_FILE_BYTES) -> bytes:
    with path.open("rb") as handle:
        return handle.read(limit + 1)[:limit]


def _read_bounded_with_overflow(path: Path, limit: int = MAX_PROC_FILE_BYTES) -> Tuple[bytes, bool]:
    with path.open("rb") as handle:
        value = handle.read(limit + 1)
    return value[:limit], len(value) > limit


def _classify(backend: Optional[str], executable: Optional[str], command_name: Optional[str], command_line: Sequence[str]) -> Optional[str]:
    argv0 = os.path.basename(command_line[0]).lower() if command_line else ""
    bases = {os.path.basename(executable or "").lower(), (command_name or "").lower(), argv0}
    ollama = bool(bases & {"ollama", "ollama runner", "ollama serve"})
    llama = bool(bases & {"llama-server", "llama_server"})
    detected = "ollama" if ollama else "llama_cpp" if llama else None
    return detected if backend is None or detected == backend else None


def discover_runtime_processes(*, proc_root: Path = PROC_ROOT, backend: Optional[str] = None,
                               endpoint_port: Optional[int] = None, timestamp_utc: Optional[object] = None,
                               pid_hints: Iterable[int] = (), retain_pid_hints: bool = False) -> ProcessDiscoveryResult:
    """Read bounded procfs evidence only; expected access failures are structured."""
    if backend is not None and backend not in {"ollama", "llama_cpp"}:
        raise ValueError("backend must be ollama, llama_cpp, or None")
    if endpoint_port is not None and (isinstance(endpoint_port, bool) or not isinstance(endpoint_port, int) or not 1 <= endpoint_port <= 65535):
        raise ValueError("endpoint_port must be from 1 through 65535")
    timestamp = _normalise_timestamp(datetime.now(timezone.utc) if timestamp_utc is None else timestamp_utc)
    errors, socket_ports, socket_complete = [], {}, True
    for name in ("tcp", "tcp6"):
        try:
            raw, overflowed = _read_bounded_with_overflow(proc_root / "net" / name)
            if overflowed:
                errors.append(TelemetryCollectionError("proc_net", "failed", f"/proc/net/{name} exceeded bounded limit")); socket_complete = False; continue
            records, parse_errors = parse_proc_net_tcp(raw.decode("utf-8", "replace"), with_errors=True)
            errors.extend(parse_errors)
            if parse_errors: socket_complete = False
            for port, inode in records:
                socket_ports.setdefault(inode, set()).add(port)
        except OSError as exc:
            errors.append(TelemetryCollectionError("proc_net", "unavailable", f"cannot read /proc/net/{name}: {exc}")); socket_complete = False
    try:
        all_pid_dirs = sorted((entry for entry in proc_root.iterdir() if entry.name.isdigit()), key=lambda item: int(item.name))
    except OSError as exc:
        error = TelemetryCollectionError("procfs", "unavailable", f"cannot list procfs: {exc}")
        return ProcessDiscoveryResult((), tuple(errors) + (error,), timestamp, inspected_pid_count=0, socket_evidence_complete=False)
    hints = sorted({_positive_int(value, "PID hint") for value in pid_hints})
    by_pid = {int(entry.name): entry for entry in all_pid_dirs}
    selected = [by_pid[pid] for pid in hints if pid in by_pid]
    selected.extend(entry for entry in all_pid_dirs if entry not in selected)
    if len(selected) > MAX_PROCESS_PIDS:
        errors.append(TelemetryCollectionError("procfs", "unavailable", "PID inspection truncated at bounded limit"))
        socket_complete = False
        selected = selected[:MAX_PROCESS_PIDS]
    processes = []
    for entry in selected:
        pid = int(entry.name)
        try:
            raw_stat, stat_overflow = _read_bounded_with_overflow(entry / "stat")
            if stat_overflow: raise ValueError("stat exceeded bounded limit")
            stat_pid, ppid, start = parse_proc_stat(raw_stat.decode("utf-8", "replace"))
            if stat_pid != pid:
                raise ValueError("PID changed during inspection")
        except (OSError, ValueError) as exc:
            errors.append(TelemetryCollectionError("proc_stat", "unavailable", f"cannot inspect PID {pid}: {exc}"))
            continue
        sources = ["proc_stat"]
        try:
            raw_comm, comm_overflow = _read_bounded_with_overflow(entry / "comm", 1024)
            command_name = None if comm_overflow else raw_comm.decode("utf-8", "strict").strip() or None
            if comm_overflow: errors.append(TelemetryCollectionError("proc_comm", "unavailable", f"PID {pid} command name exceeded bounded limit"))
            if command_name: sources.append("proc_comm")
        except (OSError, UnicodeDecodeError):
            errors.append(TelemetryCollectionError("proc_comm", "unavailable", f"cannot use PID {pid} command name"))
            command_name = None
        try:
            raw_cmdline, cmdline_overflow = _read_bounded_with_overflow(entry / "cmdline", MAX_PROCESS_CMDLINE_BYTES)
            empty_cmdline = raw_cmdline == b""
            parts = raw_cmdline.split(b"\0")[:-1] if raw_cmdline.endswith(b"\0") else ()
            bad_cmdline = not empty_cmdline and (cmdline_overflow or not raw_cmdline.endswith(b"\0") or len(parts) > MAX_PROCESS_CMDLINE_ARGS)
            command_line = () if bad_cmdline else tuple(part.decode("utf-8", "strict") for part in parts)
            if bad_cmdline: errors.append(TelemetryCollectionError("proc_cmdline", "unavailable", f"PID {pid} command line is truncated or malformed"))
            if command_line: sources.append("proc_cmdline")
        except (OSError, UnicodeDecodeError):
            errors.append(TelemetryCollectionError("proc_cmdline", "unavailable", f"cannot use PID {pid} command line"))
            command_line = ()
        try:
            executable = os.readlink(entry / "exe")
            sources.append("proc_exe")
        except OSError:
            executable = None
        ports = set()
        try:
            fd_values = sorted((entry / "fd").iterdir(), key=lambda item: item.name)
            if len(fd_values) > MAX_PROCESS_FDS: errors.append(TelemetryCollectionError("proc_fd", "unavailable", f"PID {pid} file descriptors exceeded bounded limit")); socket_complete = False
            for fd in fd_values[:MAX_PROCESS_FDS]:
                try:
                    target = os.readlink(fd)
                except OSError:
                    socket_complete = False
                    continue
                match = re.fullmatch(r"socket:\[(\d+)\]", target)
                if match:
                    ports.update(socket_ports.get(match.group(1), set()))
            if ports: sources.append("proc_fd_socket")
        except OSError as exc:
            errors.append(TelemetryCollectionError("proc_fd", "unavailable", f"cannot inspect PID {pid} file descriptors: {exc}")); socket_complete = False
        if len(ports) > MAX_PROCESS_SOCKETS: errors.append(TelemetryCollectionError("proc_fd", "unavailable", f"PID {pid} listening ports exceeded bounded limit")); socket_complete = False
        try:
            final_stat, final_overflow = _read_bounded_with_overflow(entry / "stat")
            if final_overflow: raise ValueError("final stat exceeded bounded limit")
            final_pid, _, final_start = parse_proc_stat(final_stat.decode("utf-8", "replace"))
            if (final_pid, final_start) != (stat_pid, start): raise ValueError("PID identity changed during inspection")
        except (OSError, ValueError) as exc:
            errors.append(TelemetryCollectionError("proc_pid_reuse", "unavailable", f"cannot revalidate PID {pid}: {exc}")); continue
        hint = _classify(backend, executable, command_name, command_line)
        if hint is None and not (endpoint_port is not None and endpoint_port in ports) and not (retain_pid_hints and pid in hints):
            continue
        processes.append(RuntimeProcessIdentity(pid, start, ppid, executable, command_name, command_line,
                                                tuple(sorted(ports))[:MAX_PROCESS_SOCKETS], hint, tuple(sources), timestamp))
    return ProcessDiscoveryResult(tuple(processes), tuple(errors), timestamp, inspected_pid_count=len(selected), socket_evidence_complete=socket_complete)


def _ollama_worker_lineage(worker: RuntimeProcessIdentity, owners: Iterable[RuntimeProcessIdentity],
                           processes: Iterable[RuntimeProcessIdentity]) -> Tuple[bool, Optional[str]]:
    """Prove bounded parentage to an Ollama endpoint owner; never use name alone."""
    owner_ids = {item.stable_identity for item in owners}
    by_pid = {item.pid: item for item in processes}
    current, seen = worker, set()
    for _ in range(MAX_OLLAMA_WORKER_ANCESTOR_DEPTH):
        if current.stable_identity in owner_ids:
            return True, None
        if current.pid in seen:
            return False, "Ollama worker ancestry loop"
        seen.add(current.pid)
        if current.parent_pid is None or current.parent_pid <= 0:
            return False, "Ollama worker ancestry unavailable"
        parent = by_pid.get(current.parent_pid)
        if parent is None:
            return False, "Ollama worker ancestry unavailable"
        current = parent
    return False, "Ollama worker ancestry exceeded bounded depth"


def nvidia_process_command() -> Tuple[str, ...]:
    return ("nvidia-smi", f"--query-compute-apps={','.join(NVIDIA_PROCESS_FIELDS)}", "--format=csv,noheader,nounits")


def _process_command_error(result: CommandResult) -> Optional[TelemetryCollectionError]:
    if result.executable_missing: return TelemetryCollectionError("nvidia-smi", "unavailable", "nvidia-smi executable not found")
    if result.timed_out: return TelemetryCollectionError("nvidia-smi", "failed", "nvidia-smi command timed out")
    if result.stdout_overflow or len(result.stdout.encode("utf-8", "replace")) > MAX_NVIDIA_PROCESS_STDOUT_BYTES: return TelemetryCollectionError("nvidia-smi", "failed", "nvidia-smi stdout exceeded bounded limit")
    if result.stderr_overflow or len(result.stderr.encode("utf-8", "replace")) > MAX_NVIDIA_PROCESS_STDERR_BYTES: return TelemetryCollectionError("nvidia-smi", "failed", "nvidia-smi stderr exceeded bounded limit")
    if result.runner_failure: return TelemetryCollectionError("nvidia-smi", "failed", result.runner_failure)
    if result.returncode != 0: return TelemetryCollectionError("nvidia-smi", "failed", _bounded_detail(result.stderr) or "nvidia-smi returned non-zero exit")
    return None


def parse_nvidia_gpu_process_csv(raw_text: str, *, timestamp_utc: object, source: str = "nvidia-smi") -> GPUProcessCollectionResult:
    timestamp = _normalise_timestamp(timestamp_utc)
    try:
        rows = list(csv.reader(io.StringIO(raw_text or "", newline=""), strict=True))
    except csv.Error as exc:
        return GPUProcessCollectionResult((), (TelemetryCollectionError("csv", "malformed", f"invalid CSV: {exc}"),), timestamp, nvidia_process_command(), False)
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    schema = [TelemetryCollectionError("csv_schema", "malformed", f"row {number} has {'missing' if len(row) < 4 else 'extra'} columns") for number, row in enumerate(rows, 1) if len(row) != 4]
    if schema:
        return GPUProcessCollectionResult((), tuple(schema), timestamp, nvidia_process_command(), False)
    values = [dict(zip(NVIDIA_PROCESS_FIELDS, row)) for row in rows]
    pairs = {}
    errors = []
    for value in values:
        uuid, pid_text = value["gpu_uuid"].strip(), value["pid"].strip()
        if uuid and not _marker_state(uuid):
            try:
                pid = int(pid_text)
                if pid > 0: pairs[(uuid, pid)] = pairs.get((uuid, pid), 0) + 1
            except ValueError: pass
    duplicates = {pair for pair, count in pairs.items() if count > 1}
    for uuid, pid in sorted(duplicates): errors.append(TelemetryCollectionError("csv_pair", "malformed", f"duplicate GPU UUID/PID: {uuid}/{pid}"))
    samples = []
    for number, value in enumerate(values, 1):
        uuid = value["gpu_uuid"].strip()
        if not uuid or _marker_state(uuid):
            errors.append(TelemetryCollectionError("csv_uuid", "malformed", f"row {number} has no usable GPU UUID")); continue
        try:
            pid = int(value["pid"].strip())
            _positive_int(pid, "PID")
        except (ValueError, TypeError):
            errors.append(TelemetryCollectionError("csv_pid", "malformed", f"row {number} has invalid PID")); continue
        if (uuid, pid) in duplicates: continue
        memory_text = value["used_gpu_memory"].strip()
        marker = _marker_state(memory_text)
        if marker:
            memory, state = None, TelemetryFieldState("used_gpu_memory_mib", marker, memory_text or None)
        else:
            try: memory = float(memory_text)
            except ValueError: memory = None
            if memory is None or not math.isfinite(memory) or memory < 0:
                memory, state = None, TelemetryFieldState("used_gpu_memory_mib", "malformed", "invalid GPU memory value")
            else: state = TelemetryFieldState("used_gpu_memory_mib", "observed")
        process_name = value["process_name"].strip()
        samples.append(GPUProcessSample(uuid, pid, None if _marker_state(process_name) else process_name or None, memory, timestamp, source, (state,)))
    return GPUProcessCollectionResult(tuple(samples), tuple(errors), timestamp, nvidia_process_command(), True)


def collect_nvidia_gpu_processes(*, runner: CommandRunner = _bounded_subprocess, timestamp_utc: Optional[object] = None, source: str = "nvidia-smi") -> GPUProcessCollectionResult:
    timestamp = _normalise_timestamp(datetime.now(timezone.utc) if timestamp_utc is None else timestamp_utc)
    command = nvidia_process_command()
    try: result = runner(command, NVIDIA_PROCESS_TIMEOUT_SECONDS, MAX_NVIDIA_PROCESS_STDOUT_BYTES, MAX_NVIDIA_PROCESS_STDERR_BYTES)
    except Exception as exc: result = CommandResult(None, runner_failure=_bounded_detail(exc))
    error = _process_command_error(result)
    if error: return GPUProcessCollectionResult((), (error,), timestamp, command, False)
    parsed = parse_nvidia_gpu_process_csv(result.stdout, timestamp_utc=timestamp, source=source)
    return GPUProcessCollectionResult(parsed.samples, parsed.errors, timestamp, command, parsed.successful)


def reconcile_gpu_process_stable_identities(samples: Iterable[GPUProcessSample], *, before: Iterable[RuntimeProcessIdentity],
                                            after: Iterable[RuntimeProcessIdentity]) -> Tuple[Tuple[GPUProcessSample, ...], Tuple[TelemetryCollectionError, ...]]:
    """Purely bind start ticks only when one unchanged stable identity surrounds a GPU query."""
    before_values, before_errors = _normalise_processes(before, reject_ambiguous=False)
    after_values, after_errors = _normalise_processes(after, reject_ambiguous=False)
    errors, bound = list(before_errors) + list(after_errors), []
    def unique_by_pid(values):
        grouped = {}
        for item in values: grouped.setdefault(item.pid, []).append(item)
        return {pid: group[0] for pid, group in grouped.items() if len(group) == 1 and group[0].start_time_ticks is not None}
    before_by_pid, after_by_pid = unique_by_pid(before_values), unique_by_pid(after_values)
    for sample in sorted(samples, key=lambda item: (item.gpu_uuid, item.pid)):
        before_item, after_item = before_by_pid.get(sample.pid), after_by_pid.get(sample.pid)
        if before_item is not None and before_item == after_item:
            bound.append(GPUProcessSample(sample.gpu_uuid, sample.pid, sample.process_name, sample.used_gpu_memory_mib,
                                          sample.timestamp_utc, sample.collector_source, sample.field_states, before_item.start_time_ticks))
        else:
            if before_item is not None or after_item is not None:
                errors.append(TelemetryCollectionError("runtime_pid_reuse", "unavailable", f"cannot reconcile GPU process PID: {sample.pid}"))
            bound.append(GPUProcessSample(sample.gpu_uuid, sample.pid, sample.process_name, sample.used_gpu_memory_mib,
                                          sample.timestamp_utc, sample.collector_source, sample.field_states, None))
    return tuple(bound), _ordered_errors(errors)


def attribute_runtime_gpus(*, backend: str, endpoint_port: Optional[int], processes: Iterable[RuntimeProcessIdentity],
                           gpu_process_samples: Iterable[GPUProcessSample], declared_gpu_uuids: Iterable[str] = (),
                           timestamp_utc: object, errors: Iterable[TelemetryCollectionError] = (),
                           socket_evidence_complete: bool = True) -> RuntimeAttributionResult:
    if backend not in {"ollama", "llama_cpp"}: raise ValueError("backend must be ollama or llama_cpp")
    identity_groups = {}
    for process in processes: identity_groups.setdefault(process.stable_identity, []).append(process)
    process_values, diagnostic_errors = [], list(errors)
    for identity, group in sorted(identity_groups.items(), key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1])):
        rendered = {_substantive_process_identity(item) for item in group}
        if len(rendered) != 1:
            diagnostic_errors.append(TelemetryCollectionError("runtime_identity", "malformed", f"ambiguous stable runtime identity: {identity[0]}")); continue
        process_values.append(max(group, key=lambda item: item.timestamp_utc))
    process_values = tuple(process_values)
    sample_values = tuple(gpu_process_samples)
    declared = {str(value).strip() for value in declared_gpu_uuids if str(value).strip() and not _marker_state(str(value))}
    pid_groups = {}
    for process in process_values:
        pid_groups.setdefault(process.pid, []).append(process)
    ambiguous_pids = {pid for pid, group in pid_groups.items() if len(group) > 1}
    for pid in sorted(ambiguous_pids):
        diagnostic_errors.append(TelemetryCollectionError("runtime_pid", "malformed", f"ambiguous runtime PID identities: {pid}"))
    result_processes = tuple(process for process in process_values if process.pid not in ambiguous_pids)
    compatible = {pid: group[0] for pid, group in pid_groups.items() if len(group) == 1 and pid not in ambiguous_pids and group[0].backend_hint == backend}
    endpoint_owners = [p for p in process_values if endpoint_port is not None and endpoint_port in p.listening_ports]
    compatible_owners = [p for p in endpoint_owners if p.pid in compatible]
    owners_unambiguous = len(compatible_owners) == 1 and len(endpoint_owners) == 1
    endpoint_unique = socket_evidence_complete and owners_unambiguous
    if endpoint_owners and not owners_unambiguous:
        diagnostic_errors.append(TelemetryCollectionError("endpoint_owner", "malformed", "endpoint ownership is ambiguous or incompatible"))
    matched = []
    for sample in sample_values:
        process = compatible.get(sample.pid)
        worker = False
        if process is None:
            candidate = next((item for item in process_values if item.pid == sample.pid), None)
            if backend != "ollama" or candidate is None:
                diagnostic_errors.append(TelemetryCollectionError("runtime_match", "unavailable", f"GPU process PID is not a compatible runtime: {sample.pid}")); continue
            executable = os.path.basename(candidate.executable or "").replace(" (deleted)", "").lower()
            command = os.path.basename(candidate.command_line[0]).lower() if candidate.command_line else ""
            if executable not in {"llama-server", "llama_server"} and command not in {"llama-server", "llama_server"}:
                diagnostic_errors.append(TelemetryCollectionError("runtime_match", "unavailable", f"GPU process PID is not an Ollama worker form: {sample.pid}")); continue
            proven, detail = _ollama_worker_lineage(candidate, compatible_owners, process_values)
            if not proven:
                diagnostic_errors.append(TelemetryCollectionError("runtime_lineage", "unavailable", detail or f"cannot prove Ollama worker lineage: {sample.pid}")); continue
            process, worker = candidate, True
        starts_present = process.start_time_ticks is not None and sample.process_start_time_ticks is not None
        if starts_present and sample.process_start_time_ticks != process.start_time_ticks:
            diagnostic_errors.append(TelemetryCollectionError("runtime_pid_reuse", "malformed", f"GPU process PID start time mismatch: {sample.pid}")); continue
        stable = starts_present
        endpoint_owned = endpoint_unique and compatible_owners[0].pid == process.pid
        confidence = "confirmed" if endpoint_owned and stable else "probable"
        sources = ["nvidia_compute_apps", *process.discovery_sources]
        if worker:
            sources.append("ollama_worker_lineage")
        detail = None if confidence == "confirmed" else "unique endpoint ownership or verified matching process start time is unavailable"
        matched.append(RuntimeGPUAttribution(process.pid, process.start_time_ticks, sample.gpu_uuid, sample, confidence, sources, detail))
    matched_uuids = {item.gpu_uuid for item in matched if item.confidence in {"confirmed", "probable"}}
    anchors = list(compatible.values())
    anchor = anchors[0] if len(anchors) == 1 and socket_evidence_complete and (not endpoint_owners or endpoint_unique) else None
    anchor_detail = "declared GPU UUID has no matching observed process allocation" if anchor else "declared GPU UUID has no unambiguous compatible runtime process"
    for uuid in sorted(declared - matched_uuids):
        matched.append(RuntimeGPUAttribution(anchor.pid if anchor else None, anchor.start_time_ticks if anchor else None, uuid, None,
                                             "profile-declared only", ("runtime_profile",), anchor_detail))
    return RuntimeAttributionResult(result_processes, tuple(matched), tuple(declared), (), tuple(diagnostic_errors), timestamp_utc)


# ---------------------------------------------------------------------------
# Runtime telemetry snapshot assembly (formerly runtime_telemetry.py, Stage 6D)
# ---------------------------------------------------------------------------

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
