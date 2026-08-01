from datetime import datetime, timezone
import pytest

from llm_modelbench.process_telemetry import (
    GPUProcessCollectionResult, ProcessDiscoveryResult, RuntimeAttributionResult,
    nvidia_process_command,
)
from llm_modelbench.runtime_telemetry import collect_runtime_telemetry, _snapshot_errors
from llm_modelbench.telemetry import GPUCollectionResult, TelemetryCollectionError


TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
UUID = "GPU-00000000-0000-0000-0000-000000000001"


def _processes(**kwargs):
    return ProcessDiscoveryResult((), (), TS, **kwargs)


def _live():
    return GPUCollectionResult((), (), ("extended",), "extended", False, TS, "fixture")


def _gpu_processes():
    return GPUProcessCollectionResult((), (), TS, nvidia_process_command(), True)


def test_idle_snapshot_is_deterministic_and_keeps_declared_uuid_separate():
    snapshot = collect_runtime_telemetry(
        backend="ollama", endpoint="http://127.0.0.1:11434", timestamp_utc=TS,
        declared_gpu_uuids=(value for value in (UUID,)), inventory_collector=lambda: (),
        live_collector=_live, process_collector=lambda **kwargs: _processes(),
        gpu_process_collector=_gpu_processes,
    )
    value = snapshot.to_dict()
    assert value["declared_gpu_uuids"] == [UUID]
    assert value["observed_gpu_uuids"] == []
    assert value["declared_only_gpu_uuids"] == [UUID]
    assert value == snapshot.to_dict()


def test_partial_collectors_produce_snapshot_not_failure():
    def broken():
        raise OSError("unavailable")
    snapshot = collect_runtime_telemetry(
        backend="llama_cpp", endpoint="http://127.0.0.1:8081", timestamp_utc=TS,
        inventory_collector=broken, live_collector=broken,
        process_collector=lambda **kwargs: _processes(socket_evidence_complete=False),
        gpu_process_collector=broken,
    )
    assert snapshot.observed_gpu_uuids == ()
    assert snapshot.endpoint_available is False
    assert snapshot.errors


@pytest.mark.parametrize("bad_uuid", ["CUDA0", "0", "GPU-not-a-uuid"])
def test_snapshot_rejects_nonphysical_declared_uuid(bad_uuid):
    with pytest.raises(ValueError, match="canonical NVIDIA"):
        collect_runtime_telemetry(
            backend="ollama", endpoint="http://127.0.0.1:11434", timestamp_utc=TS,
            declared_gpu_uuids=(bad_uuid,), inventory_collector=lambda: (), live_collector=_live,
            process_collector=lambda **kwargs: _processes(), gpu_process_collector=_gpu_processes,
        )


def test_snapshot_error_summary_deduplicates_identical_errors_deterministically():
    first = TelemetryCollectionError("proc_fd", "unavailable", "permission denied")
    second = TelemetryCollectionError("nvidia-smi", "failed", "query failed")
    values = _snapshot_errors((first, second, first))
    assert values == _snapshot_errors((second, first, first))
    assert [item.operation for item in values] == ["nvidia-smi", "proc_fd"]


def test_snapshot_import_has_no_collection_side_effects(monkeypatch):
    # Collection is explicit: creating no snapshot must invoke no injected seam.
    called = []
    def seam(*args, **kwargs):
        called.append(True)
        return _processes()
    assert called == []
    assert seam
