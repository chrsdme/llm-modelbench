import io
from dataclasses import replace

import pytest

from llm_modelbench.hardware import GPUDevice
from llm_modelbench import telemetry


TIMESTAMP = "2026-07-31T12:00:00Z"


def _line(index=0, uuid="GPU-a", pci="00000000:01:00.0", name="NVIDIA Test", *, extended=False,
          total="16384", used="0", free="16384", util="0", temp="35", power="0"):
    values = [index, uuid, pci, name, total, used, free, util]
    if extended:
        values += ["0", temp, power, "250", "1800", "9000", "P0"]
    else:
        values += [temp, power]
    import csv
    import io
    output = io.StringIO()
    csv.writer(output).writerow(values)
    return output.getvalue()


def _parsed(*, extended=False):
    return telemetry.parse_nvidia_gpu_csv(
        _line(extended=extended), query_tier="extended" if extended else "baseline",
        timestamp_utc=TIMESTAMP,
    )


def test_module_import_and_parser_have_no_hardware_side_effect(monkeypatch):
    monkeypatch.setattr("llm_modelbench.hardware.detect_gpus", lambda: (_ for _ in ()).throw(AssertionError("not called")))
    assert _parsed().samples[0].identity.uuid == "GPU-a"


def test_model_field_state_rules_and_zero_are_explicit():
    sample = _parsed().samples[0]
    assert sample.memory_used_mib == 0.0
    assert next(state for state in sample.field_states if state.field == "memory_used_mib").state == "observed"
    with pytest.raises(ValueError, match="None but marked observed"):
        replace(sample, memory_total_mib=None)
    with pytest.raises(ValueError, match="not marked observed"):
        replace(sample, memory_total_mib=1.0, field_states=tuple(
            telemetry.TelemetryFieldState(state.field, "unavailable") if state.field == "memory_total_mib" else state
            for state in sample.field_states
        ))
    with pytest.raises(ValueError, match="duplicate"):
        replace(sample, field_states=sample.field_states + (sample.field_states[0],))
    with pytest.raises(ValueError, match="invalid telemetry field state"):
        telemetry.TelemetryFieldState("memory_used_mib", "missing")


@pytest.mark.parametrize("index", [True, False, -1])
def test_identity_rejects_boolean_or_negative_index(index):
    with pytest.raises(ValueError, match="index"):
        telemetry.PhysicalGPUIdentity("GPU-a", observed_index=index)


@pytest.mark.parametrize("field,value", [
    ("memory_total_mib", float("nan")), ("power_draw_w", float("inf")),
    ("utilization_gpu_pct", 101.0), ("temperature_gpu_c", -1.0),
])
def test_sample_rejects_nonfinite_or_out_of_range_metrics(field, value):
    with pytest.raises(ValueError):
        replace(_parsed().samples[0], **{field: value})


def test_serialization_is_deterministic_and_ordered():
    second = telemetry.parse_nvidia_gpu_csv(_line(uuid="GPU-z") + _line(uuid="GPU-a"), query_tier="baseline", timestamp_utc=TIMESTAMP)
    first = telemetry.parse_nvidia_gpu_csv(_line(uuid="GPU-a") + _line(uuid="GPU-z"), query_tier="baseline", timestamp_utc=TIMESTAMP)
    assert first.to_dict() == second.to_dict()
    assert [item["identity"]["uuid"] for item in first.to_dict()["samples"]] == ["GPU-a", "GPU-z"]


def test_parser_handles_zero_rows_quoted_names_and_arbitrary_n():
    empty = telemetry.parse_nvidia_gpu_csv("\n\n", query_tier="baseline", timestamp_utc=TIMESTAMP)
    assert empty.samples == () and empty.errors[0].state == "unavailable"
    raw = _line(index=4, uuid="GPU-b", name="NVIDIA, B") + _line(index=1, uuid="GPU-a", name="NVIDIA A")
    result = telemetry.parse_nvidia_gpu_csv(raw, query_tier="baseline", timestamp_utc=TIMESTAMP)
    assert [sample.identity.uuid for sample in result.samples] == ["GPU-a", "GPU-b"]
    assert result.samples[1].identity.name == "NVIDIA, B"


def test_parser_returns_structured_error_for_invalid_csv():
    result = telemetry.parse_nvidia_gpu_csv('0,GPU-a,"unterminated\n', query_tier="baseline", timestamp_utc=TIMESTAMP)
    assert result.samples == () and result.errors[0].state == "malformed"


@pytest.mark.parametrize("raw,operation", [
    (_line(uuid="GPU-a") + _line(uuid="GPU-a"), "csv_uuid"),
    (_line(uuid="N/A"), "csv_uuid"),
    ("0,GPU-a\n", "csv_schema"),
    (_line(total="bad"), None),
    (_line(index="not-index"), "csv_index"),
])
def test_parser_reports_bad_rows_without_coercion(raw, operation):
    result = telemetry.parse_nvidia_gpu_csv(raw, query_tier="baseline", timestamp_utc=TIMESTAMP)
    if operation:
        assert any(error.operation == operation for error in result.errors)
    else:
        state = next(state for state in result.samples[0].field_states if state.field == "memory_total_mib")
        assert result.samples[0].memory_total_mib is None and state.state == "malformed"


def test_parser_rejects_extra_columns_and_maps_markers_and_bounds():
    extra = telemetry.parse_nvidia_gpu_csv(
        _line().rstrip("\r\n") + ",extra\n", query_tier="baseline", timestamp_utc=TIMESTAMP,
    )
    assert extra.samples == () and extra.successful_tier is None
    assert extra.errors[0].operation == "csv_schema"

    result = telemetry.parse_nvidia_gpu_csv(
        _line(total="[Not Supported]", used="N/A", free="[Insufficient Permissions]", util="101"),
        query_tier="baseline", timestamp_utc=TIMESTAMP,
    )
    sample = result.samples[0]
    states = {state.field: state.state for state in sample.field_states}
    assert states["memory_total_mib"] == "unsupported"
    assert states["memory_used_mib"] == "unavailable"
    assert states["memory_free_mib"] == "unavailable"
    assert states["utilization_gpu_pct"] == "malformed"


def test_duplicate_live_uuid_groups_are_excluded_deterministically():
    duplicate = _line(uuid="GPU-duplicate", name="first") + _line(uuid="GPU-duplicate", name="second")
    raw = duplicate + _line(uuid="GPU-valid")
    result = telemetry.parse_nvidia_gpu_csv(raw, query_tier="baseline", timestamp_utc=TIMESTAMP)
    reversed_result = telemetry.parse_nvidia_gpu_csv(
        _line(uuid="GPU-valid") + _line(uuid="GPU-duplicate", name="second") + _line(uuid="GPU-duplicate", name="first"),
        query_tier="baseline", timestamp_utc=TIMESTAMP,
    )
    assert [sample.identity.uuid for sample in result.samples] == ["GPU-valid"]
    assert result.to_dict() == reversed_result.to_dict()
    assert [error.detail for error in result.errors] == ["duplicate GPU UUID: GPU-duplicate"]


def test_three_duplicate_live_rows_emit_one_bounded_group_error():
    result = telemetry.parse_nvidia_gpu_csv(
        "".join(_line(uuid="GPU-duplicate", name=str(index)) for index in range(3)),
        query_tier="baseline", timestamp_utc=TIMESTAMP,
    )
    assert result.samples == ()
    assert [error.to_dict() for error in result.errors] == [{
        "operation": "csv_uuid", "state": "malformed", "detail": "duplicate GPU UUID: GPU-duplicate", "query_tier": "baseline",
    }]


def test_parser_requires_exact_schema_but_keeps_quoted_names():
    missing = telemetry.parse_nvidia_gpu_csv("0,GPU-a\n", query_tier="baseline", timestamp_utc=TIMESTAMP)
    assert missing.samples == () and missing.successful_tier is None
    assert missing.errors[0].detail == "row 1 has missing columns"
    quoted = telemetry.parse_nvidia_gpu_csv(_line(name="NVIDIA, quoted"), query_tier="baseline", timestamp_utc=TIMESTAMP)
    assert quoted.successful_tier == "baseline"
    assert quoted.samples[0].identity.name == "NVIDIA, quoted"


@pytest.mark.parametrize("marker", ["N/A", "NA", "[N/A]", "unknown", "[Not Supported]", "Not Supported", "[Insufficient Permissions]", "Insufficient Permissions"])
def test_optional_identity_markers_normalize_to_none_and_uuid_markers_remain_invalid(marker):
    result = telemetry.parse_nvidia_gpu_csv(_line(pci=marker, name=marker), query_tier="baseline", timestamp_utc=TIMESTAMP)
    assert result.samples[0].identity.pci_bus_id is None
    assert result.samples[0].identity.name is None
    invalid = telemetry.parse_nvidia_gpu_csv(_line(uuid=marker), query_tier="baseline", timestamp_utc=TIMESTAMP)
    assert invalid.samples == ()
    assert invalid.errors[0].operation == "csv_uuid"
    with pytest.raises(ValueError, match="usable UUID"):
        telemetry.PhysicalGPUIdentity(marker)


def test_extended_parser_preserves_extended_fields():
    sample = _parsed(extended=True).samples[0]
    assert sample.utilization_memory_pct == 0.0
    assert sample.power_limit_w == 250.0
    assert sample.graphics_clock_mhz == 1800.0
    assert sample.pstate == "P0"


class _Runner:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, command, timeout, max_stdout, max_stderr):
        self.calls.append((command, timeout, max_stdout, max_stderr))
        return self.results.pop(0)


def test_commands_and_extended_success_use_exactly_one_attempt():
    runner = _Runner([telemetry.CommandResult(0, stdout=_line(extended=True))])
    result = telemetry.collect_nvidia_gpu_samples(runner=runner, timestamp_utc=TIMESTAMP)
    assert result.successful_tier == "extended" and not result.fallback_used
    assert result.attempted_tiers == ("extended",)
    assert runner.calls[0][0] == (
        "nvidia-smi", f"--query-gpu={','.join(telemetry.EXTENDED_QUERY_FIELDS)}", "--format=csv,noheader,nounits",
    )
    assert all(isinstance(part, str) for part in runner.calls[0][0])


def test_default_subprocess_runner_uses_argument_vector_without_shell(monkeypatch):
    captured = {}

    class Process:
        stdout = io.BytesIO(b"ok")
        stderr = io.BytesIO(b"")

        def wait(self, timeout=None):
            return 0

        def kill(self):
            raise AssertionError("unexpected kill")

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(telemetry.subprocess, "Popen", fake_popen)
    result = telemetry._bounded_subprocess(("nvidia-smi", "--query-gpu=index"), 1, 16, 16)
    assert result.returncode == 0
    assert captured["command"] == ("nvidia-smi", "--query-gpu=index")
    assert captured["kwargs"]["shell"] is False


def test_extended_nonzero_has_one_fixed_baseline_fallback_without_stderr_guessing():
    runner = _Runner([
        telemetry.CommandResult(1, stderr="driver rejected mysterious optional thing"),
        telemetry.CommandResult(0, stdout=_line()),
    ])
    result = telemetry.collect_nvidia_gpu_samples(runner=runner, timestamp_utc=TIMESTAMP)
    assert result.successful_tier == "baseline" and result.fallback_used
    assert result.attempted_tiers == ("extended", "baseline")
    assert runner.calls[1][0] == telemetry.command_for_tier("baseline")
    assert not any("optional thing" in state.detail for sample in result.samples for state in sample.field_states if state.detail)


def test_extended_schema_failure_has_one_baseline_fallback():
    runner = _Runner([
        telemetry.CommandResult(0, stdout="0,GPU-a\n"),
        telemetry.CommandResult(0, stdout=_line()),
    ])
    result = telemetry.collect_nvidia_gpu_samples(runner=runner, timestamp_utc=TIMESTAMP)
    assert result.successful_tier == "baseline"
    assert any(error.operation == "csv_schema" for error in result.errors)


def test_extended_extra_column_failure_has_one_baseline_fallback():
    runner = _Runner([
        telemetry.CommandResult(0, stdout=_line(extended=True).rstrip("\r\n") + ",extra\n"),
        telemetry.CommandResult(0, stdout=_line()),
    ])
    result = telemetry.collect_nvidia_gpu_samples(runner=runner, timestamp_utc=TIMESTAMP)
    assert result.successful_tier == "baseline"
    assert result.attempted_tiers == ("extended", "baseline")
    assert len(runner.calls) == 2


def test_invalid_csv_does_not_report_a_successful_tier():
    runner = _Runner([telemetry.CommandResult(0, stdout='0,GPU-a,"unterminated\n')])
    result = telemetry.collect_nvidia_gpu_samples(runner=runner, timestamp_utc=TIMESTAMP)
    assert result.successful_tier is None
    assert result.samples == ()
    assert result.attempted_tiers == ("extended",)


def test_baseline_schema_failure_is_not_successful_and_has_no_samples():
    runner = _Runner([
        telemetry.CommandResult(1, stderr="extended unavailable"),
        telemetry.CommandResult(0, stdout=_line().rstrip("\r\n") + ",extra\n"),
    ])
    result = telemetry.collect_nvidia_gpu_samples(runner=runner, timestamp_utc=TIMESTAMP)
    assert result.successful_tier is None
    assert result.samples == ()
    assert result.attempted_tiers == ("extended", "baseline")


def test_zero_gpu_output_is_a_successful_fixed_schema_collection():
    runner = _Runner([telemetry.CommandResult(0, stdout="\n")])
    result = telemetry.collect_nvidia_gpu_samples(runner=runner, timestamp_utc=TIMESTAMP)
    assert result.samples == ()
    assert result.successful_tier == "extended"
    assert result.errors[0].operation == "csv"


@pytest.mark.parametrize("command_result", [
    telemetry.CommandResult(None, executable_missing=True),
    telemetry.CommandResult(9, timed_out=True),
    telemetry.CommandResult(0, stdout_overflow=True),
    telemetry.CommandResult(0, stderr_overflow=True),
    telemetry.CommandResult(None, runner_failure="internal runner failure"),
])
def test_collection_environment_failures_are_structured_without_fallback(command_result):
    runner = _Runner([command_result])
    result = telemetry.collect_nvidia_gpu_samples(runner=runner, timestamp_utc=TIMESTAMP)
    assert result.samples == () and result.successful_tier is None
    assert len(runner.calls) == 1
    assert result.errors[0].state in {"failed", "unavailable"}


def test_collection_bounded_stderr_and_both_tiers_failure():
    runner = _Runner([
        telemetry.CommandResult(1, stderr="x" * 900),
        telemetry.CommandResult(1, stderr="baseline bad"),
    ])
    result = telemetry.collect_nvidia_gpu_samples(runner=runner, timestamp_utc=TIMESTAMP)
    assert result.successful_tier is None and result.attempted_tiers == ("extended", "baseline")
    assert all(len(error.detail) <= telemetry.MAX_TELEMETRY_DETAIL_CHARS for error in result.errors)


def test_injected_runner_oversized_stdout_is_not_parsed():
    runner = _Runner([telemetry.CommandResult(0, stdout="x" * (telemetry.MAX_NVIDIA_SMI_STDOUT_BYTES + 1))])
    result = telemetry.collect_nvidia_gpu_samples(runner=runner, timestamp_utc=TIMESTAMP)
    assert result.samples == () and result.successful_tier is None
    assert result.errors[0].detail == "nvidia-smi stdout exceeded bounded limit"


def test_error_limit_is_bounded():
    raw = "".join(_line(uuid="N/A") for _ in range(100))
    result = telemetry.parse_nvidia_gpu_csv(raw, query_tier="baseline", timestamp_utc=TIMESTAMP)
    assert len(result.errors) == telemetry.MAX_TELEMETRY_ERRORS
    assert result.errors[-1].detail == "telemetry errors truncated"


def test_collection_result_invariants_reject_contradictory_states():
    sample = _parsed().samples[0]
    valid = dict(samples=(), errors=(), attempted_tiers=("baseline",), successful_tier="baseline",
                 fallback_used=False, timestamp_utc=TIMESTAMP, source="test")
    assert telemetry.GPUCollectionResult(**valid).successful_tier == "baseline"
    assert telemetry.GPUCollectionResult(samples=(), errors=(), attempted_tiers=("extended",), successful_tier="extended",
                                         fallback_used=False, timestamp_utc=TIMESTAMP, source="test").successful_tier == "extended"
    assert telemetry.GPUCollectionResult(samples=(sample,), errors=(), attempted_tiers=("extended", "baseline"),
                                         successful_tier="baseline", fallback_used=True, timestamp_utc=TIMESTAMP,
                                         source="test").successful_tier == "baseline"
    with pytest.raises(ValueError, match="must be attempted"):
        telemetry.GPUCollectionResult(**{**valid, "attempted_tiers": ("extended",)})
    with pytest.raises(ValueError, match="samples require"):
        telemetry.GPUCollectionResult(**{**valid, "samples": (sample,), "successful_tier": None})
    with pytest.raises(ValueError, match="duplicates"):
        telemetry.GPUCollectionResult(**{**valid, "attempted_tiers": ("baseline", "baseline")})
    with pytest.raises(ValueError, match="fallback"):
        telemetry.GPUCollectionResult(**{**valid, "fallback_used": True})
    with pytest.raises(ValueError, match="require fallback"):
        telemetry.GPUCollectionResult(**{**valid, "attempted_tiers": ("extended", "baseline")})


def _device(uuid, index, pci, driver="575.57", capability="12.0", name="same"):
    return GPUDevice(index, uuid, pci, name, 1024.0, driver, capability)


def test_join_uses_only_uuid_and_enriches_inventory_metadata():
    samples = telemetry.parse_nvidia_gpu_csv(_line(index=9, uuid="GPU-a", pci="0000:01:00.0"), query_tier="baseline", timestamp_utc=TIMESTAMP).samples
    joined = telemetry.join_inventory_samples([_device("GPU-a", 1, "0000:01:00.0")], samples)
    identity = joined.samples[0].identity
    assert identity.observed_index == 9 and identity.driver_version == "575.57" and identity.compute_capability == "12.0"
    assert not joined.errors


def test_join_does_not_match_by_index_name_or_pci_and_reports_mismatches():
    samples = telemetry.parse_nvidia_gpu_csv(_line(uuid="GPU-sample", pci="0000:01:00.0", name="same"), query_tier="baseline", timestamp_utc=TIMESTAMP).samples
    joined = telemetry.join_inventory_samples([_device("GPU-inventory", 0, "0000:01:00.0", name="same")], samples)
    assert joined.samples[0].identity.driver_version is None
    assert {error.operation for error in joined.errors} == {"inventory_join"}


def test_join_reordered_many_inventory_only_sample_only_pci_and_duplicate_cases():
    samples = telemetry.parse_nvidia_gpu_csv(
        _line(index=7, uuid="GPU-b", pci="0000:07:00.0") + _line(index=2, uuid="GPU-a", pci="0000:02:00.0"),
        query_tier="baseline", timestamp_utc=TIMESTAMP,
    ).samples
    inventory = [
        _device("GPU-a", 99, "0000:99:00.0"), _device("GPU-b", 1, "0000:07:00.0"),
        _device("GPU-c", 3, "0000:03:00.0"), _device("GPU-b", 4, "0000:04:00.0"),
    ]
    joined = telemetry.join_inventory_samples(inventory, samples)
    assert [sample.identity.uuid for sample in joined.samples] == ["GPU-a", "GPU-b"]
    assert {error.operation for error in joined.errors} == {"inventory_pci", "inventory_join", "inventory_uuid"}
    assert [error.to_dict() for error in joined.errors] == [error.to_dict() for error in telemetry.join_inventory_samples(reversed(inventory), reversed(samples)).errors]


def test_duplicate_inventory_uuid_is_never_used_for_enrichment():
    samples = telemetry.parse_nvidia_gpu_csv(
        _line(uuid="GPU-ambiguous") + _line(uuid="GPU-unique", pci="0000:03:00.0"), query_tier="baseline", timestamp_utc=TIMESTAMP,
    ).samples
    inventory = [
        _device("GPU-ambiguous", 0, "0000:01:00.0", driver="first", capability="1.0"),
        _device("GPU-ambiguous", 1, "0000:02:00.0", driver="second", capability="2.0"),
        _device("GPU-unique", 2, "0000:03:00.0", driver="unique", capability="3.0"),
    ]
    joined = telemetry.join_inventory_samples(inventory, samples)
    identities = {sample.identity.uuid: sample.identity for sample in joined.samples}
    assert identities["GPU-ambiguous"].driver_version is None
    assert identities["GPU-ambiguous"].compute_capability is None
    assert identities["GPU-unique"].driver_version == "unique"
    assert [error.detail for error in joined.errors] == ["duplicate inventory GPU UUID: GPU-ambiguous"]
    changed_reversed = [
        _device("GPU-unique", 2, "0000:03:00.0", driver="unique", capability="3.0"),
        _device("GPU-ambiguous", 9, "0000:09:00.0", driver="changed", capability="9.0"),
        _device("GPU-ambiguous", 8, "0000:08:00.0", driver="other", capability="8.0"),
    ]
    assert joined.to_dict() == telemetry.join_inventory_samples(changed_reversed, reversed(samples)).to_dict()
