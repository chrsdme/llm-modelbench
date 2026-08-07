from llm_modelbench.config import Config
from llm_modelbench.hardware import GPUDevice
from llm_modelbench.placement import model_placement_fit, topology_for_config
from llm_modelbench.topology_budget import MIB, evaluate_workload_fit, topology_from_inventory


U5060 = "GPU-11111111-1111-1111-1111-111111111111"
U3060 = "GPU-22222222-2222-2222-2222-222222222222"


def _inventory():
    return (
        GPUDevice(7, U5060, "00000000:01:00.0", "fixture-5060-class", 16311, None, None),
        GPUDevice(3, U3060, "00000000:09:00.0", "fixture-3060-class", 12288, None, None),
    )


def _topology(*, free_5060=16311, free_3060=12288, aggregate=None):
    return topology_from_inventory(_inventory(), live_by_uuid_mib={U5060: {"free": free_5060}, U3060: {"free": free_3060}},
                                   policy_ceilings_mib={U5060: 15872, U3060: 11776}, aggregate_policy_cap_mib=aggregate)


def test_unequal_physical_inventory_and_policy_aggregate_are_uuid_keyed():
    topology = _topology()
    assert [item.uuid for item in topology.devices] == [U5060, U3060]
    assert [item.installed_capacity_bytes // MIB for item in topology.devices] == [16311, 12288]
    assert [item.effective_now_bytes // MIB for item in topology.devices] == [15872, 11776]
    assert topology.aggregate_effective_bytes // MIB == 27648


def test_policy_and_live_free_clamp_without_a_display_penalty():
    topology = _topology(free_5060=16000, free_3060=11000)
    first, second = topology.devices
    assert first.effective_now_bytes // MIB == 15872
    assert second.effective_now_bytes // MIB == 11000
    display = topology_from_inventory(_inventory(), live_by_uuid_mib={U5060: {"free": 15000, "display_active": True}})
    assert display.devices[0].effective_now_bytes // MIB == 15000


def test_selected_runtime_can_reclaim_but_unrelated_cuda_memory_never_can():
    topology = topology_from_inventory(_inventory(), live_by_uuid_mib={U5060: {"free": 12000}}, policy_ceilings_mib={U5060: 15872}, runtime_reclaimable_mib={U5060: 2000}, unrelated_mib={U5060: 3000})
    record = next(item for item in topology.devices if item.uuid == U5060)
    assert record.effective_now_bytes // MIB == 12000
    assert record.effective_after_reclaim_bytes // MIB == 14000
    assert record.unrelated_nonreclaimable_bytes // MIB == 3000


def test_single_gpu_first_explicit_selection_and_layer_split_are_conservative():
    topology = _topology()
    fit = evaluate_workload_fit(topology, weight_bytes=12 * 1024**3, kv_cache_bytes=1024**3, runtime_overhead_bytes=512 * MIB, device_overhead_bytes=256 * MIB)
    assert fit.classification == "single_gpu_fit" and fit.selected_gpu_uuids == (U5060,)
    smaller = evaluate_workload_fit(topology, weight_bytes=10 * 1024**3, selected_gpu_uuid=U3060)
    assert smaller.classification == "candidate_single_gpu_fit" and smaller.selected_gpu_uuids == (U3060,)
    split = evaluate_workload_fit(topology, weight_bytes=20 * 1024**3)
    assert split.classification == "multi_gpu_conditional_fit"
    assert split.selected_gpu_uuids[0] == U5060


def test_aggregate_override_and_legacy_cap_are_operator_only():
    topology = _topology(aggregate=26 * 1024)
    assert evaluate_workload_fit(topology, weight_bytes=14 * 1024**3).selected_gpu_uuids == (U5060,)
    assert evaluate_workload_fit(topology, weight_bytes=27 * 1024**3).classification == "confirmed_no_fit"
    cfg = Config(vram_budget_gb=0, gpu_policy_ceilings_mib={U5060: 15872, U3060: 11776})
    assert model_placement_fit({"size": int(13 * 1024**3)}, cfg, inventory=_inventory()).classification == "candidate_single_gpu_fit"
    assert topology_for_config(Config(vram_budget_gb=12.0), _inventory()).max_single_effective_bytes == int(12 * 1024**3)
