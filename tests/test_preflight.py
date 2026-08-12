from types import SimpleNamespace

from llm_modelbench.decision_policy import DecisionPolicy
from llm_modelbench.preflight import (
    OperationMode,
    PreflightBlocker,
    PreflightResult,
    resolve_operational_preflight,
)
from llm_modelbench.runtime_profiles import RuntimeCandidate, RuntimeProfile
from llm_modelbench.topology_budget import TopologyBudget


def _cfg():
    return SimpleNamespace(vram_budget_gb=0, gpu_policy_ceilings_mib={}, aggregate_policy_ceiling_mib=None)


def _candidate(name, backend, endpoint, recommended=False):
    return RuntimeCandidate(RuntimeProfile(name, backend, endpoint), "healthy", ("fixture",), "fixture", recommended)


def _topology_fn(cfg, inventory=None):
    return TopologyBudget(devices=())


def test_operation_mode_has_the_three_documented_members():
    assert {mode.value for mode in OperationMode} == {"read_only", "operational", "mutating_admin"}


def test_composes_inventory_into_both_discovery_and_topology():
    seen = {}

    def discover_fn(cfg, *, store_path, gpu_devices):
        seen["discover_gpu_devices"] = gpu_devices
        return [_candidate("only", "ollama", "http://127.0.0.1:11434")]

    def topology_fn(cfg, inventory=None):
        seen["topology_inventory"] = inventory
        return TopologyBudget(devices=())

    result = resolve_operational_preflight(
        _cfg(), gpu_inventory=("fake-device",), discover_fn=discover_fn, topology_fn=topology_fn,
    )
    assert not result.blocked
    assert result.selected_candidate.profile.name == "only"
    assert seen["discover_gpu_devices"] == ("fake-device",)
    assert seen["topology_inventory"] == ("fake-device",)
    assert result.gpu_inventory == ("fake-device",)


def test_explicit_gpu_inventory_bypasses_inventory_fn():
    def unexpected_detection():
        raise AssertionError("inventory_fn must not be called when gpu_inventory is supplied")

    def discover_fn(cfg, *, store_path, gpu_devices):
        return [_candidate("only", "ollama", "http://127.0.0.1:11434")]

    result = resolve_operational_preflight(
        _cfg(), gpu_inventory=(), inventory_fn=unexpected_detection,
        discover_fn=discover_fn, topology_fn=_topology_fn,
    )
    assert not result.blocked


def test_ambiguous_selection_returns_typed_blocker_not_an_exception():
    def discover_fn(cfg, *, store_path, gpu_devices):
        return [_candidate("ollama", "ollama", "http://127.0.0.1:11434"),
                _candidate("llama", "llama_cpp", "http://127.0.0.1:8081")]

    result = resolve_operational_preflight(
        _cfg(), gpu_inventory=(), discover_fn=discover_fn, topology_fn=_topology_fn,
    )
    assert result.blocked
    assert result.selected_candidate is None
    assert isinstance(result.blocker, PreflightBlocker)
    assert result.blocker.reason == "runtime_selection_ambiguous"
    assert len(result.candidates) == 2


def test_decisive_winner_under_permissive_unattended_policy_is_not_blocked():
    def discover_fn(cfg, *, store_path, gpu_devices):
        return [_candidate("ollama", "ollama", "http://127.0.0.1:11434"),
                _candidate("llama", "llama_cpp", "http://127.0.0.1:8081", recommended=True)]

    policy = DecisionPolicy(unattended=True, allow_backend_auto_selection=True)
    result = resolve_operational_preflight(
        _cfg(), gpu_inventory=(), discover_fn=discover_fn, topology_fn=_topology_fn, policy=policy,
    )
    assert not result.blocked
    assert result.selected_candidate.profile.name == "llama"


def test_preflight_result_blocked_property():
    unblocked = PreflightResult((), (), None, TopologyBudget(devices=()), None)
    assert unblocked.blocked is False
    blocked = PreflightResult((), (), None, TopologyBudget(devices=()), PreflightBlocker("x", "y"))
    assert blocked.blocked is True
