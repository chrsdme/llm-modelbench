"""Anvil Stage 3.2C-2b -- explicit RAM-spill execution policy + host-RAM preflight.

Covers: the placement-label projection, the conservative host-RAM preflight and
its boundaries, swap exclusion, fail-closed on unknown inputs, the CLI flag
surface on `run` / `campaign run`, config-file rejection of a spill key, the
runtime-identity permission representation, and the runner regression that a
spill permission must not silently un-skip a needle depth.
"""
import json

import pytest

from llm_modelbench.config import Config
from llm_modelbench.hardware import GPUDevice, _read_proc_meminfo
from llm_modelbench.topology_budget import MIB, WorkloadFit, evaluate_workload_fit, topology_from_inventory
from llm_modelbench.ram_spill_preflight import (
    PLACEMENT_LABELS,
    SAFE_RAM_FRACTION,
    placement_label_for,
    resolve_spill_preflight,
)

GB = 1024 ** 3
U_A = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
U_B = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def _meminfo(*, available_mb, swap_free_mb=0.0, swap_used_mb=0.0):
    return {
        "ram_total_mb": (available_mb or 0) + 4096.0,
        "ram_available_mb": available_mb,
        "swap_total_mb": (swap_free_mb or 0) + (swap_used_mb or 0),
        "swap_free_mb": swap_free_mb,
        "swap_used_mb": swap_used_mb,
    }


def _fit(classification, *, selected=(U_A,), required=None, unknown=()):
    return WorkloadFit(classification, tuple(selected), required, tuple(unknown), "fixture")


# --------------------------------------------------------------------------- #
# constants / vocabulary                                                       #
# --------------------------------------------------------------------------- #

def test_safe_ram_fraction_is_a_single_fixed_conservative_constant():
    assert SAFE_RAM_FRACTION == 0.85  # amendment §9 approved maximum
    assert 0.0 < SAFE_RAM_FRACTION <= 0.85


def test_placement_vocabulary_is_small_and_disjoint_from_fit_labels():
    from llm_modelbench.topology_budget import FIT_LABELS

    assert PLACEMENT_LABELS == ("full_gpu", "multi_gpu", "ram_spill")
    assert set(PLACEMENT_LABELS).isdisjoint(FIT_LABELS)


# --------------------------------------------------------------------------- #
# label projection                                                             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("classification,expected", [
    ("single_gpu_fit", "full_gpu"),
    ("candidate_single_gpu_fit", "full_gpu"),
    ("multi_gpu_conditional_fit", "multi_gpu"),
    ("cpu_spill_required", None),
    ("confirmed_no_fit", None),
    ("unknown", None),
])
def test_placement_label_projection(classification, expected):
    assert placement_label_for(_fit(classification)) == expected


# --------------------------------------------------------------------------- #
# GPU-resident placements: the flag must not change them (§5)                   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("allow", [False, True])
@pytest.mark.parametrize("classification,label", [
    ("single_gpu_fit", "full_gpu"),
    ("multi_gpu_conditional_fit", "multi_gpu"),
])
def test_gpu_resident_placement_is_identical_with_and_without_the_flag(allow, classification, label):
    fit = _fit(classification, selected=(U_A, U_B) if label == "multi_gpu" else (U_A,), required=8 * GB)
    result = resolve_spill_preflight(
        fit, safe_selected_gpu_capacity_bytes=40 * GB, allow_ram_spill=allow,
        host_meminfo=_meminfo(available_mb=64000),
    )
    assert result.feasible is True
    assert result.resolution == label
    assert result.resolution != "ram_spill"
    assert result.selected_gpu_uuids == fit.selected_gpu_uuids


# --------------------------------------------------------------------------- #
# confirmed_no_fit is never reinterpreted by RAM preflight                      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("allow", [False, True])
def test_confirmed_no_fit_stays_infeasible_regardless_of_the_flag(allow):
    result = resolve_spill_preflight(
        _fit("confirmed_no_fit", selected=(), required=80 * GB),
        safe_selected_gpu_capacity_bytes=None, allow_ram_spill=allow,
        host_meminfo=_meminfo(available_mb=512000),
    )
    assert result.feasible is False
    assert result.resolution == "environment_infeasible"
    assert result.reason == "confirmed_no_fit_not_spill_eligible"


# --------------------------------------------------------------------------- #
# cpu_spill_required: the only RAM-preflight entry                             #
# --------------------------------------------------------------------------- #

def test_spill_forbidden_without_the_flag():
    result = resolve_spill_preflight(
        _fit("cpu_spill_required", selected=(U_A, U_B), required=40 * GB),
        safe_selected_gpu_capacity_bytes=30 * GB, allow_ram_spill=False,
        host_meminfo=_meminfo(available_mb=64000),
    )
    assert result.feasible is False
    assert result.resolution == "environment_infeasible"
    assert result.reason == "ram_spill_not_permitted"


def test_spill_permitted_when_estimated_overflow_fits_safe_physical_ram():
    # overflow = 40 - 34 = 6 GB; safe RAM = 8 * 0.85 = 6.8 GB -> permitted.
    result = resolve_spill_preflight(
        _fit("cpu_spill_required", selected=(U_A, U_B), required=40 * GB),
        safe_selected_gpu_capacity_bytes=34 * GB, allow_ram_spill=True,
        host_meminfo=_meminfo(available_mb=8 * 1024),
    )
    assert result.feasible is True
    assert result.resolution == "ram_spill"
    assert result.selected_gpu_uuids == (U_A, U_B)  # §14: retains the GPU pool
    assert result.estimated_ram_spill_bytes == 6 * GB


def test_spill_denied_when_estimated_overflow_exceeds_safe_physical_ram():
    # overflow = 6 GB; safe RAM = 5 * 0.85 = 4.25 GB -> infeasible.
    result = resolve_spill_preflight(
        _fit("cpu_spill_required", selected=(U_A,), required=40 * GB),
        safe_selected_gpu_capacity_bytes=34 * GB, allow_ram_spill=True,
        host_meminfo=_meminfo(available_mb=5 * 1024),
    )
    assert result.feasible is False
    assert result.resolution == "environment_infeasible"
    assert result.reason == "spill_exceeds_safe_host_ram"


def test_swap_cannot_rescue_an_otherwise_infeasible_spill():
    # Same as the denial case, but with a huge SwapFree. Swap contributes 0 (§8/§12).
    result = resolve_spill_preflight(
        _fit("cpu_spill_required", selected=(U_A,), required=40 * GB),
        safe_selected_gpu_capacity_bytes=34 * GB, allow_ram_spill=True,
        host_meminfo=_meminfo(available_mb=5 * 1024, swap_free_mb=100 * 1024),
    )
    assert result.feasible is False
    assert result.resolution == "environment_infeasible"
    # swap is still recorded as telemetry, just never counted
    assert result.swap_free_bytes == 100 * 1024 * MIB


def test_existing_swap_usage_is_recorded_as_a_degraded_signal_not_capacity():
    result = resolve_spill_preflight(
        _fit("cpu_spill_required", selected=(U_A,), required=40 * GB),
        safe_selected_gpu_capacity_bytes=34 * GB, allow_ram_spill=True,
        host_meminfo=_meminfo(available_mb=8 * 1024, swap_used_mb=2048.0),
    )
    assert result.swap_in_use is True
    assert result.feasible is True  # decision still made on physical RAM alone


# --------------------------------------------------------------------------- #
# fail closed on unknown inputs (§11)                                          #
# --------------------------------------------------------------------------- #

def test_unknown_workload_requirement_never_authorizes_spill():
    result = resolve_spill_preflight(
        _fit("cpu_spill_required", selected=(U_A,), required=None, unknown=("weights",)),
        safe_selected_gpu_capacity_bytes=34 * GB, allow_ram_spill=True,
        host_meminfo=_meminfo(available_mb=512000),
    )
    assert result.feasible is None
    assert result.resolution == "environment_unknown"
    assert result.reason == "spill_overflow_inputs_unknown"


def test_unknown_selected_gpu_capacity_never_authorizes_spill():
    result = resolve_spill_preflight(
        _fit("cpu_spill_required", selected=(U_A,), required=40 * GB),
        safe_selected_gpu_capacity_bytes=None, allow_ram_spill=True,
        host_meminfo=_meminfo(available_mb=512000),
    )
    assert result.feasible is None
    assert result.resolution == "environment_unknown"


def test_unknown_host_memory_never_authorizes_spill():
    result = resolve_spill_preflight(
        _fit("cpu_spill_required", selected=(U_A,), required=40 * GB),
        safe_selected_gpu_capacity_bytes=34 * GB, allow_ram_spill=True,
        host_meminfo={},  # no MemAvailable
    )
    assert result.feasible is None
    assert result.resolution == "environment_unknown"
    assert result.reason == "host_ram_evidence_unavailable"


def test_topology_unknown_fit_fails_closed():
    result = resolve_spill_preflight(
        _fit("unknown", selected=(), required=None),
        safe_selected_gpu_capacity_bytes=None, allow_ram_spill=True,
        host_meminfo=_meminfo(available_mb=512000),
    )
    assert result.feasible is None
    assert result.resolution == "environment_unknown"


def test_spill_branch_with_zero_overflow_resolves_to_a_gpu_resident_placement():
    # cpu_spill_required but the discounted pool actually holds the workload.
    result = resolve_spill_preflight(
        _fit("cpu_spill_required", selected=(U_A, U_B)),
        safe_selected_gpu_capacity_bytes=30 * GB, allow_ram_spill=True,
        host_meminfo=_meminfo(available_mb=8 * 1024), known_workload_bytes=20 * GB,
    )
    assert result.feasible is True
    assert result.resolution == "multi_gpu"
    assert result.estimated_ram_spill_bytes == 0


# --------------------------------------------------------------------------- #
# §16 paired behaviour: the flag authorizes only the fallback                   #
# --------------------------------------------------------------------------- #

def test_paired_behaviour_flag_only_authorizes_the_fallback():
    """Same model/GPU/host evidence; only allow_ram_spill differs.

    Everything the GPU-placement decision consumes must be identical -- the
    assertion is against the selection inputs and the eligible pool, not the
    WorkloadFit object (which legitimately differs: proven no-fit returns
    confirmed_no_fit/() without permission and cpu_spill_required/full-pool
    with it -- that difference *is* the fallback being authorized).
    """
    def _inv():
        return (
            GPUDevice(0, U_A, "00000000:01:00.0", "fixture-a", 16000, None, None),
            GPUDevice(1, U_B, "00000000:09:00.0", "fixture-b", 16000, None, None),
        )

    topo_without = topology_from_inventory(_inv())
    topo_with = topology_from_inventory(_inv())
    weight = 40 * GB  # exceeds the whole discounted pool (~28 GB)

    # weights + KV fully specified so required_bytes is complete on both sides
    kv = 512 * 1024 * 1024
    without = evaluate_workload_fit(topo_without, weight_bytes=weight, kv_cache_bytes=kv,
                                    runtime_overhead_bytes=0, device_overhead_bytes=0, allow_cpu_spill=False)
    with_flag = evaluate_workload_fit(topo_with, weight_bytes=weight, kv_cache_bytes=kv,
                                      runtime_overhead_bytes=0, device_overhead_bytes=0, allow_cpu_spill=True)

    # identical selection inputs / eligible pool / requirement -- pinned to
    # literals so a future re-baseline of the pool capacity moves this test too.
    assert without.required_bytes == with_flag.required_bytes == weight + kv
    assert [d.uuid for d in topo_without.placement_order] == [d.uuid for d in topo_with.placement_order] == [U_A, U_B]
    # 16000 MiB * 0.88 safe fraction, per device, on both independent topologies
    expected_caps = [int(16000 * MIB * 0.88), int(16000 * MIB * 0.88)]
    assert [d.effective_now_bytes for d in topo_without.devices] == expected_caps
    assert [d.effective_now_bytes for d in topo_with.devices] == expected_caps

    # the only difference is the fallback
    assert without.classification == "confirmed_no_fit"
    assert without.selected_gpu_uuids == ()
    assert with_flag.classification == "cpu_spill_required"
    assert with_flag.selected_gpu_uuids == (U_A, U_B)

    pool_cap = sum(d.safe_capacity_bytes for d in topo_with.devices)
    r_without = resolve_spill_preflight(without, safe_selected_gpu_capacity_bytes=None,
                                        allow_ram_spill=False, host_meminfo=_meminfo(available_mb=64000))
    r_with = resolve_spill_preflight(with_flag, safe_selected_gpu_capacity_bytes=pool_cap,
                                     allow_ram_spill=True, host_meminfo=_meminfo(available_mb=64000))
    assert r_without.resolution == "environment_infeasible"
    assert r_with.resolution == "ram_spill"


# --------------------------------------------------------------------------- #
# _read_proc_meminfo now exposes SwapFree                                       #
# --------------------------------------------------------------------------- #

def test_read_proc_meminfo_exposes_swap_free(monkeypatch, tmp_path):
    fake = tmp_path / "meminfo"
    fake.write_text(
        "MemTotal:       32000000 kB\n"
        "MemAvailable:   16000000 kB\n"
        "SwapTotal:       8000000 kB\n"
        "SwapFree:        6000000 kB\n"
    )
    real_open = open
    monkeypatch.setattr("builtins.open", lambda p, *a, **k: real_open(fake if p == "/proc/meminfo" else p, *a, **k))
    snap = _read_proc_meminfo()
    assert snap["swap_free_mb"] == pytest.approx(6000000 / 1024.0, rel=1e-3)
    assert snap["swap_used_mb"] == pytest.approx(2000000 / 1024.0, rel=1e-3)


# --------------------------------------------------------------------------- #
# CLI surface (§18)                                                            #
# --------------------------------------------------------------------------- #

def _parse(argv):
    from llm_modelbench.cli import build_parser

    return build_parser().parse_args(argv)


def test_run_accepts_allow_ram_spill_default_false():
    assert _parse(["run", "--mock"]).allow_ram_spill is False
    assert _parse(["run", "--mock", "--allow-ram-spill"]).allow_ram_spill is True


def test_campaign_run_accepts_allow_ram_spill_default_false():
    base = ["campaign", "run", "--campaign-id", "c1"]
    assert _parse(base).allow_ram_spill is False
    assert _parse(base + ["--allow-ram-spill"]).allow_ram_spill is True


def test_allow_ram_spill_is_not_on_the_global_parser_or_read_only_subcommands():
    # The flag is added only to the two real execution surfaces (§3); it must be
    # entirely absent -- not merely defaulted -- everywhere else.
    for argv in (["inventory"], ["runtime-fit", "--model", "m"], ["plan"],
                 ["campaign", "plan", "--campaign-id", "c1"]):
        assert "allow_ram_spill" not in vars(_parse(argv)), argv


def test_runtime_fit_allow_cpu_spill_flag_is_untouched():
    ns = _parse(["runtime-fit", "--model", "m", "--allow-cpu-spill"])
    assert ns.allow_cpu_spill is True


@pytest.mark.parametrize("extra,expected", [([], False), (["--allow-ram-spill"], True)])
def test_cmd_run_propagates_the_flag_onto_cfg(monkeypatch, tmp_path, extra, expected):
    """The single §19 propagation point: cmd_run copies args.allow_ram_spill
    onto cfg (campaign run reuses this exact args namespace, so one path).

    The assignment (cli.py) is the first thing cmd_run does with the flag,
    ahead of GPU detection / client construction; we stop cmd_run right after
    it by making the next step raise, then inspect the mutated cfg.
    """
    from llm_modelbench import cli, hardware

    sentinel = RuntimeError("stop after propagation")
    def _boom(*a, **k):
        raise sentinel
    monkeypatch.setattr(hardware, "detect_gpus", _boom)
    args = cli.build_parser().parse_args(
        ["run", "--mock", "--tasks", "py_anagram", "--out", str(tmp_path), "--run-id", "r", "--yes"] + extra
    )
    cfg = Config()
    with pytest.raises(RuntimeError, match="stop after propagation"):
        cli.cmd_run(args, cfg)
    assert cfg.allow_ram_spill is expected


# --------------------------------------------------------------------------- #
# config-file rejection (§4/§26)                                               #
# --------------------------------------------------------------------------- #

def test_config_file_cannot_enable_ram_spill(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"allow_ram_spill": True}))
    with pytest.raises(SystemExit) as excinfo:
        Config.load(str(cfg_path))
    assert "allow_ram_spill" in str(excinfo.value)


def test_default_config_does_not_grant_spill():
    cfg = Config()
    assert getattr(cfg, "allow_ram_spill", False) is False


# --------------------------------------------------------------------------- #
# runtime identity: permission is the compatibility representation (§20)        #
# --------------------------------------------------------------------------- #

class _FakeProfile:
    backend = "ollama"
    endpoint = "http://127.0.0.1:11434"
    name = None
    provenance = "configured"
    physical_gpu_uuids = ()


class _FakeClient:
    def version(self):
        return "test"


def _identity(cfg):
    from llm_modelbench.runtime_identity import collect_runtime_identity

    return collect_runtime_identity(client=_FakeClient(), profile=_FakeProfile(), model_name="m",
                                    model_row={"digest": "sha256:abc"}, config=cfg, inventory=())


def test_absent_spill_permission_keeps_a_stable_identity_hash():
    plain = Config()
    flagged_false = Config()
    flagged_false.allow_ram_spill = False
    assert _identity(plain).identity_hash == _identity(flagged_false).identity_hash
    assert _identity(plain).execution.allow_cpu_spill is None


def test_granting_spill_permission_is_identity_bearing_and_flips_spill_policy_changed():
    from llm_modelbench.runtime_identity import compare_runtime_identities

    without = _identity(Config())
    granted = Config()
    granted.allow_ram_spill = True
    with_permission = _identity(granted)

    assert with_permission.execution.allow_cpu_spill is True
    assert without.identity_hash != with_permission.identity_hash
    codes = [m.code for m in compare_runtime_identities(without, with_permission).mismatches]
    assert codes == ["spill_policy_changed"]


# --------------------------------------------------------------------------- #
# runner regression: spill permission must not silently un-skip a needle depth  #
# --------------------------------------------------------------------------- #

class _SizedClient:
    """Minimal client exposing a model weight and metadata KV bytes-per-token."""

    def __init__(self, weight_bytes):
        self._weight = weight_bytes

    def model_size_bytes(self, _model):
        return self._weight


def _cfg_with_two_small_gpus():
    cfg = Config()
    return cfg


def test_proven_no_fit_still_skips_the_needle_even_with_spill_permitted(monkeypatch):
    """A proven GPU no-fit that spill *cannot* rescue keeps kv_exceeds_budget True.

    Before 3.2C-2b the gate read the raw classification; a spill permission
    flipped confirmed_no_fit -> cpu_spill_required and un-skipped the depth.
    Now the gate reads the resolved preflight feasibility (§5/§23).
    """
    from llm_modelbench import runner

    inv = (
        GPUDevice(0, U_A, "00000000:01:00.0", "fixture-a", 8000, None, None),
        GPUDevice(1, U_B, "00000000:09:00.0", "fixture-b", 8000, None, None),
    )
    monkeypatch.setattr(runner, "_kv_bytes_per_token", lambda *a, **k: (200_000, "metadata"))
    # host RAM far too small to hold the overflow of a ~40 GB workload
    monkeypatch.setattr(runner, "host_memory_snapshot", lambda: {"ram_available_mb": 512.0, "swap_free_mb": 0.0, "swap_used_mb": 0.0})

    client = _SizedClient(40 * GB)

    cfg_plain = Config()
    kv_plain = runner._needle_kv_estimate(client, cfg_plain, "m", 128_000, None, gpu_inventory=inv)

    cfg_spill = Config()
    cfg_spill.allow_ram_spill = True
    kv_spill = runner._needle_kv_estimate(client, cfg_spill, "m", 128_000, None, gpu_inventory=inv)

    assert kv_plain["kv_exceeds_budget"] is True
    assert kv_spill["kv_exceeds_budget"] is True  # spill cannot rescue -> still skipped
    assert kv_spill["placement_feasible"] is False


def test_spill_permission_that_actually_fits_ram_unskips_the_depth(monkeypatch):
    from llm_modelbench import runner

    inv = (
        GPUDevice(0, U_A, "00000000:01:00.0", "fixture-a", 8000, None, None),
        GPUDevice(1, U_B, "00000000:09:00.0", "fixture-b", 8000, None, None),
    )
    monkeypatch.setattr(runner, "_kv_bytes_per_token", lambda *a, **k: (1000, "metadata"))
    monkeypatch.setattr(runner, "host_memory_snapshot",
                        lambda: {"ram_available_mb": 256000.0, "swap_free_mb": 0.0, "swap_used_mb": 0.0})

    client = _SizedClient(15 * GB)  # small overflow past the ~14 GB discounted pool

    cfg_plain = Config()
    kv_plain = runner._needle_kv_estimate(client, cfg_plain, "m", 16_000, None, gpu_inventory=inv)
    assert kv_plain["kv_exceeds_budget"] is True  # infeasible without permission

    cfg_spill = Config()
    cfg_spill.allow_ram_spill = True
    kv_spill = runner._needle_kv_estimate(client, cfg_spill, "m", 16_000, None, gpu_inventory=inv)
    assert kv_spill["kv_exceeds_budget"] is False
    assert kv_spill["placement_resolution"] == "ram_spill"
