import json
from types import SimpleNamespace

import pytest

from llm_modelbench.hardware import GPUDevice
from llm_modelbench.runtime_fit import (
    RuntimeFitModel, RuntimeFitProfile, RuntimeFitResult, calculate_kv_cache_bytes,
    evaluate_runtime_fit,
)


TS = "2026-08-01T00:00:00Z"
U1 = "GPU-00000000-0000-0000-0000-000000000001"
U2 = "GPU-00000000-0000-0000-0000-000000000002"
U3 = "GPU-00000000-0000-0000-0000-000000000003"


def _devices(*rows):
    return tuple(GPUDevice(index, uuid, None, "fixture", mib, None, None) for index, uuid, mib in rows)


def _model(weight=4 * 1024**3, *, overhead=None, context=None, architecture=None, maximum=None):
    return RuntimeFitModel("fixture", weight, "declared_disk_bytes", context, maximum, overhead, architecture)


def _profile(*uuids, strategy=None, weights=(), spill=None):
    return RuntimeFitProfile("fixture", "llama_cpp", uuids, strategy, weights, spill)


def test_single_gpu_candidate_and_confirmed_no_fit_are_conservative():
    inventory = _devices((0, U1, 8192))
    candidate = evaluate_runtime_fit(inventory=inventory, live_telemetry=None, model=_model(), profile=_profile(U1), timestamp_utc=TS)
    assert candidate.decision == "candidate_fit"
    no_fit = evaluate_runtime_fit(inventory=inventory, live_telemetry=None, model=_model(9 * 1024**3), profile=_profile(U1), timestamp_utc=TS)
    assert no_fit.decision == "confirmed_no_fit"
    assert "lower_bound_exceeds_capacity" in no_fit.reasons


def test_explicit_layer_split_is_per_device_not_aggregate_capacity():
    inventory = _devices((0, U1, 4096), (1, U2, 12288), (2, U3, 12288))
    model = _model(10 * 1024**3)
    no_strategy = evaluate_runtime_fit(inventory=inventory, live_telemetry=None, model=model, profile=_profile(U1, U2), timestamp_utc=TS)
    assert no_strategy.decision == "unknown"
    assert [item.gpu_uuid for item in no_strategy.device_assessments] == [U1, U2]
    split = evaluate_runtime_fit(inventory=inventory, live_telemetry=None, model=model, profile=_profile(U1, U2, strategy="layer_split", weights={U1: 1, U2: 3}), timestamp_utc=TS)
    assert split.decision == "conditional_fit"
    assert [item.gpu_uuid for item in split.device_assessments] == [U1, U2]
    failed = evaluate_runtime_fit(inventory=inventory, live_telemetry=None, model=model, profile=_profile(U1, U2, strategy="layer_split", weights={U1: 3, U2: 1}), timestamp_utc=TS)
    assert failed.decision == "confirmed_no_fit"


def test_identity_validation_and_deterministic_serialization():
    inventory = _devices((2, U3, 8192), (0, U1, 8192), (1, U2, 8192))
    result = evaluate_runtime_fit(inventory=inventory, live_telemetry=None, model=_model(),
                                  profile=_profile(U3, U1, U2, strategy="layer_split", weights={U1: 1, U2: 1, U3: 1}), timestamp_utc=TS)
    value = result.to_dict()
    assert [item["gpu_uuid"] for item in value["device_assessments"]] == [U1, U2, U3]
    assert json.dumps(value, sort_keys=True, allow_nan=False) == json.dumps(result.to_dict(), sort_keys=True, allow_nan=False)
    with pytest.raises(ValueError, match="duplicate physical inventory"):
        evaluate_runtime_fit(inventory=_devices((0, U1, 1), (1, U1, 1)), live_telemetry=None, model=_model(), profile=_profile(), timestamp_utc=TS)
    with pytest.raises(ValueError, match="canonical"):
        _profile("CUDA0")


def test_kv_requires_complete_architecture_and_never_defaults_zero():
    unknown, reason = calculate_kv_cache_bytes(_model(context=4096))
    assert unknown is None and reason == "unknown_architecture"
    derived, reason = calculate_kv_cache_bytes(_model(context=4096, maximum=8192, architecture={"layer_count": 2, "kv_head_count": 4, "head_dimension": 8, "kv_dtype_bytes": 2}))
    assert derived == 2 * 2 * 4 * 8 * 4096 * 2
    assert reason == "derived_kv_cache_bytes"
    exceeded, reason = calculate_kv_cache_bytes(_model(context=8193, maximum=8192, architecture={"layer_count": 2, "kv_head_count": 4, "head_dimension": 8, "kv_dtype_bytes": 2}))
    assert exceeded is None and "exceeds" in reason


def test_spill_is_conditional_and_result_invariants_reject_invalid_confirmation():
    inventory = _devices((0, U1, 8192))
    spill = evaluate_runtime_fit(inventory=inventory, live_telemetry=None, model=_model(), profile=_profile(U1, spill=True), timestamp_utc=TS)
    assert spill.decision == "conditional_fit"
    with pytest.raises(ValueError, match="confirmed_fit"):
        RuntimeFitResult(1, TS, _model(), _profile(U1), 1, None, "unknown", (), "confirmed_fit", ())


def test_diagnostic_spill_reason_is_honest_about_not_checking_host_ram():
    """Anvil Stage 3.2C-3: runtime-fit does not sample host RAM, so its
    operator-permitted spill branch must not claim spill feasibility."""
    inventory = _devices((0, U1, 8192))
    spill = evaluate_runtime_fit(inventory=inventory, live_telemetry=None, model=_model(),
                                 profile=_profile(U1, spill=True), timestamp_utc=TS)
    assert spill.reasons == ("cpu_spill_permitted_host_capacity_unchecked",)
    # the previous, misleadingly-optimistic reason is gone
    assert "cpu_spill_required_or_permitted" not in spill.reasons


def test_runtime_fit_is_advisory_and_not_consumed_by_execution_or_scoring():
    """The advisory boundary is a real contract: no execution/scoring module
    imports the runtime_fit evaluator, and reporting treats its artifact as
    advisory-only (asserted in test_report_offline)."""
    import ast
    import pathlib

    pkg = pathlib.Path(__file__).resolve().parent.parent / "llm_modelbench"
    consumers = []
    for name in ("runner.py", "campaign.py", "planner.py", "scoring.py", "aggregate.py"):
        tree = ast.parse((pkg / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "runtime_fit" in node.module:
                consumers.append(name)
            if isinstance(node, ast.Import) and any("runtime_fit" in a.name for a in node.names):
                consumers.append(name)
    assert consumers == [], f"runtime_fit must stay advisory; imported by {consumers}"


def test_invalid_reserve_and_profile_inventory_mismatch_rejected():
    with pytest.raises(ValueError, match="reserve_mib"):
        evaluate_runtime_fit(inventory=(), live_telemetry=None, model=_model(), profile=_profile(), reserve_mib=float("nan"), timestamp_utc=TS)
    with pytest.raises(ValueError, match="absent"):
        evaluate_runtime_fit(inventory=_devices((0, U1, 1)), live_telemetry=None, model=_model(), profile=_profile(U2), timestamp_utc=TS)


def test_live_identity_and_allocation_weights_must_be_canonical_numeric_evidence():
    with pytest.raises(ValueError, match="allocation weights"):
        _profile(U1, U2, strategy="layer_split", weights={U1: True, U2: 1})
    with pytest.raises(ValueError, match="allocation weights"):
        _profile(U1, U2, strategy="layer_split", weights={U1: "1", U2: 1})
    with pytest.raises(ValueError, match="must not duplicate"):
        _profile(U1, U2, strategy="layer_split", weights=((U1, 1), (U1, 1), (U2, 1)))
    live = SimpleNamespace(
        samples=(SimpleNamespace(identity=SimpleNamespace(uuid="CUDA0"), memory_free_mib=1),), errors=(),
    )
    with pytest.raises(ValueError, match="canonical"):
        evaluate_runtime_fit(inventory=_devices((0, U1, 8192)), live_telemetry=live,
                             model=_model(), profile=_profile(U1), timestamp_utc=TS)
