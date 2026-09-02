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
    # effective_now is clamped by the fixed §4 safety ceiling (SAFE_VRAM_FRACTION
    # of installed) -- here it is below both the policy ceiling and live-free,
    # so it dominates: 16311*0.88 and 12288*0.88.
    assert [item.effective_now_bytes // MIB for item in topology.devices] == [14353, 10813]
    assert topology.aggregate_effective_bytes // MIB == 25167


def test_policy_and_live_free_clamp_without_a_display_penalty():
    topology = _topology(free_5060=16000, free_3060=11000)
    first, second = topology.devices
    # U5060: min(policy 15872, live-free 16000, safe 14353) -> safe wins.
    assert first.effective_now_bytes // MIB == 14353
    # U3060: min(policy 11776, live-free 11000, safe 10813) -> safe wins.
    assert second.effective_now_bytes // MIB == 10813
    # Display activity is still not a tax: a live-free reading below the safety
    # ceiling is honoured as-is, whether or not the display is active.
    display = topology_from_inventory(_inventory(), live_by_uuid_mib={U5060: {"free": 12000, "display_active": True}})
    assert display.devices[0].effective_now_bytes // MIB == 12000


def test_safe_vram_fraction_is_a_single_fixed_constant_in_the_frozen_window():
    from llm_modelbench.topology_budget import GPUMemoryBudget, SAFE_VRAM_FRACTION

    assert 0.85 <= SAFE_VRAM_FRACTION <= 0.90
    # No other evidence: effective_now is exactly physical * fraction, never the
    # full installed capacity.
    bare = GPUMemoryBudget(U5060, "00000000:01:00.0", 16311 * MIB)
    assert bare.safe_capacity_bytes == int(16311 * MIB * SAFE_VRAM_FRACTION)
    assert bare.effective_now_bytes == bare.safe_capacity_bytes
    # A live-free reading below the safety ceiling still wins (min of the two).
    tighter = GPUMemoryBudget(U5060, "00000000:01:00.0", 16311 * MIB, live_free_bytes=8000 * MIB)
    assert tighter.effective_now_bytes == 8000 * MIB
    # Unknown installed capacity -> no safety ceiling to apply.
    assert GPUMemoryBudget(U5060, "00000000:01:00.0", None).safe_capacity_bytes is None


def test_selected_runtime_can_reclaim_but_unrelated_cuda_memory_never_can():
    topology = topology_from_inventory(_inventory(), live_by_uuid_mib={U5060: {"free": 12000}}, policy_ceilings_mib={U5060: 15872}, runtime_reclaimable_mib={U5060: 2000}, unrelated_mib={U5060: 3000})
    record = next(item for item in topology.devices if item.uuid == U5060)
    assert record.effective_now_bytes // MIB == 12000
    assert record.effective_after_reclaim_bytes // MIB == 14000
    assert record.unrelated_nonreclaimable_bytes // MIB == 3000


def test_single_gpu_first_explicit_selection_and_layer_split_are_conservative():
    topology = _topology()
    # The primary card (U3060, host ordinal 3) cannot hold ~13.75 GiB, so the
    # decision falls through to the only single card that fits.  Selection keys
    # on placement priority, not on "largest is fastest".
    fit = evaluate_workload_fit(topology, weight_bytes=12 * 1024**3, kv_cache_bytes=1024**3, runtime_overhead_bytes=512 * MIB, device_overhead_bytes=256 * MIB)
    assert fit.classification == "single_gpu_fit" and fit.selected_gpu_uuids == (U5060,)
    smaller = evaluate_workload_fit(topology, weight_bytes=10 * 1024**3, selected_gpu_uuid=U3060)
    assert smaller.classification == "candidate_single_gpu_fit" and smaller.selected_gpu_uuids == (U3060,)
    split = evaluate_workload_fit(topology, weight_bytes=20 * 1024**3)
    assert split.classification == "multi_gpu_conditional_fit"
    # Multi-GPU membership accumulates in placement priority (primary GPU first),
    # not largest-first: the primary card leads the split.
    assert split.selected_gpu_uuids == (U3060, U5060)


def test_primary_gpu_is_used_even_when_a_later_card_is_larger():
    # Both cards can hold the workload.  The primary GPU (U3060, host ordinal 3)
    # wins over the larger U5060 (ordinal 7): placement is which card, not
    # fastest card.
    topology = _topology()
    fit = evaluate_workload_fit(topology, weight_bytes=8 * 1024**3, kv_cache_bytes=512 * MIB, runtime_overhead_bytes=512 * MIB, device_overhead_bytes=256 * MIB)
    assert fit.classification == "single_gpu_fit"
    assert fit.selected_gpu_uuids == (U3060,)


def test_multi_gpu_uses_the_minimum_number_of_cards_not_every_eligible_card():
    # Three cards, ~26 GiB workload: the first two in placement priority already
    # hold it, so the third must not appear in the split.
    U_A = "GPU-33333333-3333-3333-3333-333333333333"
    inventory = (
        GPUDevice(0, U_A, "00000000:02:00.0", "fixture-a", 16000, None, None),
        GPUDevice(1, U5060, "00000000:01:00.0", "fixture-5060-class", 16311, None, None),
        GPUDevice(2, U3060, "00000000:09:00.0", "fixture-3060-class", 12288, None, None),
    )
    topology = topology_from_inventory(inventory)
    fit = evaluate_workload_fit(topology, weight_bytes=26 * 1024**3)
    assert fit.classification == "multi_gpu_conditional_fit"
    assert fit.selected_gpu_uuids == (U_A, U5060)


def test_placement_is_deterministic_across_inventory_orderings():
    forward = topology_from_inventory(_inventory())
    reversed_inv = topology_from_inventory(tuple(reversed(_inventory())))
    args = dict(weight_bytes=8 * 1024**3)
    assert evaluate_workload_fit(forward, **args).selected_gpu_uuids == evaluate_workload_fit(reversed_inv, **args).selected_gpu_uuids == (U3060,)


def test_ordinal_is_execution_order_only_uuid_remains_identity():
    topology = _topology()
    # The persisted per-device identity stays UUID-keyed and UUID-sorted; only
    # the placement walk reorders by host ordinal.
    assert [item.uuid for item in topology.devices] == [U5060, U3060]
    assert [item.uuid for item in topology.placement_order] == [U3060, U5060]
    assert evaluate_workload_fit(topology, weight_bytes=8 * 1024**3).selected_gpu_uuids == (U3060,)


def test_missing_ordinal_falls_back_to_stable_pci_then_uuid_order():
    from llm_modelbench.topology_budget import GPUMemoryBudget, TopologyBudget

    topology = TopologyBudget((
        GPUMemoryBudget(U5060, "00000000:09:00.0", 16311 * MIB, physical_index=None),
        GPUMemoryBudget(U3060, "00000000:01:00.0", 12288 * MIB, physical_index=None),
    ))
    # Neither device carries a host ordinal; fall through to the lower PCI bus
    # id, deterministically -- never to live capacity.
    assert [item.uuid for item in topology.placement_order] == [U3060, U5060]


def test_infeasible_pool_is_no_fit_not_a_silent_ram_spill():
    topology = _topology()
    fit = evaluate_workload_fit(topology, weight_bytes=40 * 1024**3)
    assert fit.classification == "confirmed_no_fit"
    assert fit.selected_gpu_uuids == ()


def test_ram_spill_selects_the_whole_gpu_pool_never_leaves_capacity_idle():
    # Workload exceeds the combined pool; with spill explicitly permitted the
    # resident portion still occupies every configured GPU (§11), the overflow
    # is what spills.
    topology = _topology()
    fit = evaluate_workload_fit(topology, weight_bytes=40 * 1024**3, allow_cpu_spill=True)
    assert fit.classification == "cpu_spill_required"
    assert set(fit.selected_gpu_uuids) == {U5060, U3060}


def test_ram_spill_never_fires_on_unknown_gpu_capacity():
    from llm_modelbench.topology_budget import GPUMemoryBudget, TopologyBudget

    # No installed capacity, no live-free: capacity is simply unknown.  Spill
    # must not silently enable itself (§6) -- the answer is "unknown", not
    # "cpu_spill_required".
    blind = TopologyBudget((
        GPUMemoryBudget(U5060, "00000000:01:00.0", None, physical_index=0),
        GPUMemoryBudget(U3060, "00000000:09:00.0", None, physical_index=1),
    ))
    fit = evaluate_workload_fit(blind, weight_bytes=40 * 1024**3, allow_cpu_spill=True)
    assert fit.classification == "unknown"
    assert fit.selected_gpu_uuids == ()


def test_ram_spill_off_by_default_even_when_the_pool_cannot_fit():
    topology = _topology()
    assert evaluate_workload_fit(topology, weight_bytes=40 * 1024**3).classification == "confirmed_no_fit"


def test_aggregate_override_and_legacy_cap_are_operator_only():
    topology = _topology(aggregate=26 * 1024)
    assert evaluate_workload_fit(topology, weight_bytes=14 * 1024**3).selected_gpu_uuids == (U5060,)
    assert evaluate_workload_fit(topology, weight_bytes=27 * 1024**3).classification == "confirmed_no_fit"
    cfg = Config(vram_budget_gb=0, gpu_policy_ceilings_mib={U5060: 15872, U3060: 11776})
    assert model_placement_fit({"size": int(13 * 1024**3)}, cfg, inventory=_inventory()).classification == "candidate_single_gpu_fit"
    assert topology_for_config(Config(vram_budget_gb=12.0), _inventory()).max_single_effective_bytes == int(12 * 1024**3)
