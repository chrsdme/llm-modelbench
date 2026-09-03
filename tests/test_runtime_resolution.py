"""Anvil Stage 3B.2 slices C-F -- deterministic runtime resolution +
fit/preflight integration. Policy/planning only: the resolver launches
nothing.

Covers the prompt §10 required-test matrix:
* repeated identical resolution gives identical result
* shuffled candidate order gives identical result
* deterministic equivalent-candidate tie-break
* _recommended() mutation has no authority over the result
* missing explicit backend authority does not silently guess
* compatible discovered runtime + fit => approved result
* runtime unavailable / incompatible => structured rejection
* insufficient authoritative evidence => fail closed
* one-GPU fit chooses one GPU
* minimum multi-GPU only when one GPU cannot safely fit
* RAM spill never activates without explicit permission
* explicit RAM spill still fails when safe physical RAM is insufficient
* swap is not counted as capacity
* resolver/preflight never launches a runtime
* candidate iteration order cannot affect outcome
"""
import subprocess


from llm_modelbench.capabilities import MeasuredCapabilityState
from llm_modelbench.hardware import GPUDevice
from llm_modelbench.runtime_profiles import RuntimeCandidate, RuntimeProfile
from llm_modelbench.runtime_resolution import (
    RequiredCapability,
    ResolvedRuntime,
    RuntimeResolutionStatus,
    resolve_runtime,
)
from llm_modelbench.topology_budget import topology_from_inventory

U_A = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
U_B = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
U_C = "GPU-cccccccc-cccc-cccc-cccc-cccccccccccc"

GB = 1024 * 1024 * 1024


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _candidate(*, name="ollama-local", backend="ollama", endpoint="http://127.0.0.1:11434",
               health="healthy", recommended=False, source=("saved_profile",),
               physical_gpu_uuids=()):
    return RuntimeCandidate(
        profile=RuntimeProfile(name=name, backend=backend, endpoint=endpoint,
                               provenance="configured",
                               physical_gpu_uuids=tuple(physical_gpu_uuids)),
        health=health, source=tuple(source), detail=f"{health} fixture", recommended=recommended,
    )


def _single_gpu_topology(*, installed_mb=24000, free_mb=None):
    live = {U_A: {"free": free_mb}} if free_mb is not None else None
    return topology_from_inventory(
        (GPUDevice(0, U_A, "0000:01:00.0", "fixture-A", installed_mb, None, None),),
        live_by_uuid_mib=live,
    )


def _dual_gpu_topology(*, a_mb=16000, b_mb=16000, a_free_mb=None, b_free_mb=None):
    live = {}
    if a_free_mb is not None:
        live[U_A] = {"free": a_free_mb}
    if b_free_mb is not None:
        live[U_B] = {"free": b_free_mb}
    return topology_from_inventory(
        (
            GPUDevice(0, U_A, "0000:01:00.0", "fixture-A", a_mb, None, None),
            GPUDevice(1, U_B, "0000:02:00.0", "fixture-B", b_mb, None, None),
        ),
        live_by_uuid_mib=live or None,
    )


def _triple_gpu_topology(*, a_mb=16000, b_mb=16000, c_mb=16000):
    return topology_from_inventory(
        (
            GPUDevice(0, U_A, "0000:01:00.0", "fixture-A", a_mb, None, None),
            GPUDevice(1, U_B, "0000:02:00.0", "fixture-B", b_mb, None, None),
            GPUDevice(2, U_C, "0000:03:00.0", "fixture-C", c_mb, None, None),
        ),
    )


def _meminfo(*, avail_mb=64000, swap_free_mb=0, swap_used_mb=0):
    return {"ram_total_mb": 128000, "ram_available_mb": avail_mb,
            "swap_free_mb": swap_free_mb, "swap_used_mb": swap_used_mb}


def _resolve(**overrides):
    kwargs = dict(
        selected_backend="ollama",
        discovered_candidates=[_candidate()],
        topology=_single_gpu_topology(),
        host_meminfo=_meminfo(),
        weight_bytes=8 * GB,
        kv_cache_bytes=1 * GB,
        requested_context=8192,
        allow_ram_spill=False,
    )
    kwargs.update(overrides)
    return resolve_runtime(**kwargs)


# --------------------------------------------------------------------------
# backend authority (§2)
# --------------------------------------------------------------------------
def test_missing_backend_authority_does_not_guess():
    res = _resolve(selected_backend=None)
    assert res.status is RuntimeResolutionStatus.NO_BACKEND_SELECTED
    assert res.resolved is None


def test_unknown_backend_is_distinct_from_no_backend_selected():
    """Stage 3B.3A carryover 3: an explicit unsupported backend is a
    materially different state from 'no backend selected' -- callers must
    distinguish them without reading `detail` prose."""
    none_selected = _resolve(selected_backend=None)
    unsupported = _resolve(selected_backend="vllm")
    assert none_selected.status is RuntimeResolutionStatus.NO_BACKEND_SELECTED
    assert unsupported.status is RuntimeResolutionStatus.UNSUPPORTED_BACKEND_SELECTED
    assert none_selected.status is not unsupported.status
    assert unsupported.resolved is None


def test_unknown_backend_string_variants_all_unsupported():
    for name in ("vllm", "tensorrt_llm", "unknown_backend", "OLLAMA"):
        res = _resolve(selected_backend=name)
        assert res.status is RuntimeResolutionStatus.UNSUPPORTED_BACKEND_SELECTED, name


def test_recommended_flag_has_no_authority_over_resolution():
    """Two healthy candidates on the SAME endpoint (provably identity-
    equivalent) -- the stable tie-break is profile name. Flipping which one
    carries recommended=True must not change the winner."""
    a = _candidate(name="aaa", recommended=False)
    b = _candidate(name="bbb", recommended=True)
    first = _resolve(discovered_candidates=[a, b])
    assert first.is_resolved
    assert first.resolved.runtime_profile_name == "aaa"  # tie-break, not 'recommended'

    a2 = _candidate(name="aaa", recommended=True)
    b2 = _candidate(name="bbb", recommended=False)
    second = _resolve(discovered_candidates=[a2, b2])
    assert second.resolved.runtime_profile_name == "aaa"  # unchanged


def test_recommended_function_mutation_has_no_authority(monkeypatch):
    """Directly monkeypatch runtime_profiles._recommended to return the
    inverse backend. The resolution must be byte-identical -- the resolver
    never imports or calls it. Reads vacuous today; it is the regression
    guard if someone later wires _recommended into the authority path."""
    import llm_modelbench.runtime_profiles as rp

    baseline = _resolve().to_dict()

    def _inverted(profile, health, gpu_count):  # pragma: no cover - guard
        return True  # would mark every candidate "recommended"

    monkeypatch.setattr(rp, "_recommended", _inverted)
    after = _resolve().to_dict()
    assert after == baseline


def test_recommended_gpu_count_heuristic_is_never_consulted():
    """A dual-GPU host would make runtime_profiles._recommended prefer
    llama_cpp -- but with selected_backend='ollama' and a healthy ollama
    candidate, resolution stays ollama."""
    res = _resolve(
        selected_backend="ollama",
        discovered_candidates=[_candidate(backend="ollama", recommended=False)],
        topology=_dual_gpu_topology(),
        weight_bytes=4 * GB,
        kv_cache_bytes=512 * 1024 * 1024,
    )
    assert res.is_resolved
    assert res.resolved.backend == "ollama"


# --------------------------------------------------------------------------
# determinism (§3, §10)
# --------------------------------------------------------------------------
def test_repeated_identical_resolution_is_identical():
    a = _resolve()
    b = _resolve()
    assert a.to_dict() == b.to_dict()


def test_shuffled_candidate_order_gives_identical_result():
    cands = [
        _candidate(name="ollama-local", endpoint="http://127.0.0.1:11434"),
        _candidate(name="ollama-alt-unhealthy", endpoint="http://127.0.0.1:11500", health="unreachable"),
    ]
    forward = _resolve(discovered_candidates=list(cands))
    backward = _resolve(discovered_candidates=list(reversed(cands)))
    assert forward.to_dict() == backward.to_dict()
    assert forward.is_resolved


def test_multiple_healthy_distinct_endpoints_is_ambiguous_not_guessed():
    cands = [
        _candidate(name="p1", endpoint="http://127.0.0.1:11434"),
        _candidate(name="p2", endpoint="http://127.0.0.1:11500"),
    ]
    res = _resolve(discovered_candidates=cands)
    assert res.status is RuntimeResolutionStatus.RUNTIME_AMBIGUOUS
    assert res.resolved is None
    # order-independent
    res_rev = _resolve(discovered_candidates=list(reversed(cands)))
    assert res_rev.status is RuntimeResolutionStatus.RUNTIME_AMBIGUOUS


# --------------------------------------------------------------------------
# same-endpoint physical-GPU divergence (DEFECT-3B.2-AUDIT-02)
# --------------------------------------------------------------------------
def test_same_endpoint_equivalent_gpu_identity_resolves_deterministically():
    """Two healthy candidates, same endpoint, same physical GPU set (even
    with the UUID tuple written in a different order -- RuntimeProfile
    dedups but does not sort). Provably equivalent -> stable name tie-break,
    deterministic RESOLVED."""
    a = _candidate(name="bbb", physical_gpu_uuids=(U_A, U_B))
    b = _candidate(name="aaa", physical_gpu_uuids=(U_B, U_A))  # same set, reordered
    res = _resolve(discovered_candidates=[a, b], topology=_dual_gpu_topology(),
                   weight_bytes=4 * GB, kv_cache_bytes=512 * 1024 * 1024)
    assert res.is_resolved
    assert res.resolved.runtime_profile_name == "aaa"  # tie-break winner
    # candidate order independent
    res_rev = _resolve(discovered_candidates=[b, a], topology=_dual_gpu_topology(),
                       weight_bytes=4 * GB, kv_cache_bytes=512 * 1024 * 1024)
    assert res_rev.to_dict() == res.to_dict()


def test_same_endpoint_divergent_gpu_identity_is_ambiguous():
    """Same endpoint, materially different physical GPU placement -> the
    (backend, endpoint) pair is NOT proof of equivalence. Must fail closed
    as RUNTIME_AMBIGUOUS, never resolved by lexical profile.name."""
    a = _candidate(name="aaa", physical_gpu_uuids=(U_A,))
    b = _candidate(name="bbb", physical_gpu_uuids=(U_B,))
    res = _resolve(discovered_candidates=[a, b], topology=_dual_gpu_topology(),
                   weight_bytes=4 * GB, kv_cache_bytes=512 * 1024 * 1024)
    assert res.status is RuntimeResolutionStatus.RUNTIME_AMBIGUOUS
    assert res.resolved is None
    # candidate order independent
    res_rev = _resolve(discovered_candidates=[b, a], topology=_dual_gpu_topology(),
                       weight_bytes=4 * GB, kv_cache_bytes=512 * 1024 * 1024)
    assert res_rev.status is RuntimeResolutionStatus.RUNTIME_AMBIGUOUS


def test_same_endpoint_gpu_divergence_not_hidden_by_uuid_ordering():
    """The existing 'candidate order' tests vary candidate order, not the
    within-profile UUID tuple order. This pins that ("A","B") vs ("B","A")
    is treated as the SAME placement (frozenset compare), so a genuinely
    divergent case ((A,) vs (B,)) is what trips ambiguity -- not tuple
    ordering noise."""
    same = _resolve(
        discovered_candidates=[
            _candidate(name="p1", physical_gpu_uuids=(U_A, U_B)),
            _candidate(name="p2", physical_gpu_uuids=(U_B, U_A)),
        ],
        topology=_dual_gpu_topology(), weight_bytes=4 * GB,
        kv_cache_bytes=512 * 1024 * 1024,
    )
    assert same.is_resolved  # reordered UUIDs are NOT divergence

    divergent = _resolve(
        discovered_candidates=[
            _candidate(name="p1", physical_gpu_uuids=(U_A,)),
            _candidate(name="p2", physical_gpu_uuids=(U_A, U_B)),
        ],
        topology=_dual_gpu_topology(), weight_bytes=4 * GB,
        kv_cache_bytes=512 * 1024 * 1024,
    )
    assert divergent.status is RuntimeResolutionStatus.RUNTIME_AMBIGUOUS


def test_explicit_profile_disambiguates_multiple_endpoints():
    cands = [
        _candidate(name="p1", endpoint="http://127.0.0.1:11434"),
        _candidate(name="p2", endpoint="http://127.0.0.1:11500"),
    ]
    res = _resolve(discovered_candidates=cands, explicit_profile_name="p2")
    assert res.is_resolved
    assert res.resolved.endpoint == "http://127.0.0.1:11500"


def test_equivalent_candidate_tie_break_is_deterministic_and_documented():
    """Same backend + same endpoint, differ only in name/provenance/source
    -- resolve to the lexicographically-first name, every time, regardless
    of input order."""
    x = _candidate(name="zzz", source=("process",))
    y = _candidate(name="aaa", source=("saved_profile",))
    assert _resolve(discovered_candidates=[x, y]).resolved.runtime_profile_name == "aaa"
    assert _resolve(discovered_candidates=[y, x]).resolved.runtime_profile_name == "aaa"


# --------------------------------------------------------------------------
# discovery / endpoint states (§6)
# --------------------------------------------------------------------------
def test_runtime_unreachable_is_structured_unavailable():
    res = _resolve(discovered_candidates=[_candidate(health="unreachable")])
    assert res.status is RuntimeResolutionStatus.RUNTIME_UNAVAILABLE
    assert res.resolved is None


def test_runtime_unhealthy_is_structured_incompatible():
    res = _resolve(discovered_candidates=[_candidate(health="unhealthy")])
    assert res.status is RuntimeResolutionStatus.RUNTIME_INCOMPATIBLE


def test_no_candidates_for_backend_is_no_usable_endpoint():
    res = _resolve(discovered_candidates=[_candidate(backend="llama_cpp")])
    assert res.status is RuntimeResolutionStatus.NO_USABLE_ENDPOINT


def test_backend_executable_not_installed_and_no_endpoint_is_backend_unavailable():
    class _Exe:
        def __init__(self, backend, state):
            self.backend, self.state = backend, state

    res = _resolve(
        selected_backend="llama_cpp",
        discovered_candidates=[_candidate(backend="ollama")],
        backend_executables=[_Exe("llama_cpp", "not_installed"), _Exe("ollama", "installed")],
    )
    assert res.status is RuntimeResolutionStatus.BACKEND_UNAVAILABLE


def test_explicit_profile_absent_is_no_usable_endpoint():
    res = _resolve(explicit_profile_name="does-not-exist")
    assert res.status is RuntimeResolutionStatus.NO_USABLE_ENDPOINT


# --------------------------------------------------------------------------
# identity / provenance (§6)
# --------------------------------------------------------------------------
def test_required_content_addressed_identity_missing_fails_closed():
    res = _resolve(require_content_addressed_model_identity=True, model_primary_sha256=None)
    assert res.status is RuntimeResolutionStatus.IDENTITY_INSUFFICIENT


def test_required_content_addressed_identity_present_passes():
    res = _resolve(require_content_addressed_model_identity=True, model_primary_sha256="sha256:abc")
    assert res.is_resolved


def test_resolved_recipe_retains_model_primary_sha256_for_managed_materialisation():
    # Stage 3B.3C's managed llama-server spawn needs the resolved recipe to
    # carry the artifact identity it was resolved against, so it can prove it
    # is launching exactly that artifact and not an arbitrary path.
    res = _resolve(model_primary_sha256="sha256:deadbeef")
    assert res.is_resolved
    assert res.resolved.model_primary_sha256 == "sha256:deadbeef"
    assert res.resolved.to_dict()["model_primary_sha256"] == "sha256:deadbeef"


def test_resolved_recipe_model_primary_sha256_is_none_when_not_supplied():
    res = _resolve()  # _resolve() supplies no model_primary_sha256
    assert res.is_resolved
    assert res.resolved.model_primary_sha256 is None
    assert res.resolved.to_dict()["model_primary_sha256"] is None


def test_resolved_recipe_model_primary_sha256_is_trimmed():
    res = _resolve(model_primary_sha256="  sha256:abc  ")
    assert res.resolved.model_primary_sha256 == "sha256:abc"


# --------------------------------------------------------------------------
# capability gate (§6) -- evidence consumed, never computed
# --------------------------------------------------------------------------
def test_required_capability_without_evidence_fails_closed():
    res = _resolve(required_capabilities=[RequiredCapability("tools", None)])
    assert res.status is RuntimeResolutionStatus.CAPABILITY_EVIDENCE_INSUFFICIENT


def test_required_capability_measured_unsupported_is_incompatible():
    res = _resolve(required_capabilities=[
        RequiredCapability("tools", MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    ])
    assert res.status is RuntimeResolutionStatus.CAPABILITY_INCOMPATIBLE


def test_required_capability_inconclusive_is_evidence_insufficient():
    res = _resolve(required_capabilities=[
        RequiredCapability("tools", MeasuredCapabilityState.PROBE_INCONCLUSIVE)
    ])
    assert res.status is RuntimeResolutionStatus.CAPABILITY_EVIDENCE_INSUFFICIENT


def test_required_capability_supported_does_not_block():
    res = _resolve(required_capabilities=[
        RequiredCapability("text", MeasuredCapabilityState.MEASURED_SUPPORTED)
    ])
    assert res.is_resolved


# --------------------------------------------------------------------------
# fit / placement (§5) -- reuse the frozen machinery
# --------------------------------------------------------------------------
def test_compatible_runtime_and_fit_is_approved_full_gpu():
    res = _resolve(weight_bytes=8 * GB, kv_cache_bytes=1 * GB, topology=_single_gpu_topology(installed_mb=24000))
    assert res.is_resolved
    assert isinstance(res.resolved, ResolvedRuntime)
    assert res.resolved.placement_class == "full_gpu"
    assert res.resolved.selected_physical_gpu_uuids == (U_A,)


def test_one_gpu_fit_chooses_one_gpu_even_when_two_available():
    res = _resolve(
        topology=_dual_gpu_topology(a_mb=24000, b_mb=24000),
        weight_bytes=8 * GB, kv_cache_bytes=1 * GB,
    )
    assert res.is_resolved
    assert res.resolved.selected_physical_gpu_uuids == (U_A,)
    assert res.resolved.placement_class == "full_gpu"


def test_minimum_multi_gpu_only_when_one_gpu_cannot_fit():
    # 20 GiB workload, two 16000 MB (~15.6 GiB, *0.88 safe ~13.7 GiB) cards:
    # no single card fits, the pair does.
    res = _resolve(
        topology=_dual_gpu_topology(a_mb=16000, b_mb=16000),
        weight_bytes=20 * GB, kv_cache_bytes=1 * GB,
    )
    assert res.is_resolved
    assert res.resolved.placement_class == "multi_gpu"
    assert set(res.resolved.selected_physical_gpu_uuids) == {U_A, U_B}
    assert res.resolved.execution_settings.strategy == "layer_split"


def test_environment_infeasible_when_workload_exceeds_all_gpus_and_no_spill():
    res = _resolve(
        topology=_dual_gpu_topology(a_mb=16000, b_mb=16000),
        weight_bytes=60 * GB, kv_cache_bytes=4 * GB,
        allow_ram_spill=False,
    )
    assert res.status is RuntimeResolutionStatus.ENVIRONMENT_INFEASIBLE
    assert res.resolved is None


def test_fit_unknown_when_weight_missing():
    res = _resolve(weight_bytes=None)
    assert res.status is RuntimeResolutionStatus.FIT_UNKNOWN


def test_fit_unknown_when_no_gpu_inventory():
    empty = topology_from_inventory(())
    res = _resolve(topology=empty)
    assert res.status is RuntimeResolutionStatus.FIT_UNKNOWN


# --------------------------------------------------------------------------
# RAM spill (§6, §7, §8)
# --------------------------------------------------------------------------
def test_ram_spill_never_activates_without_explicit_permission():
    res = _resolve(
        topology=_dual_gpu_topology(a_mb=16000, b_mb=16000),
        weight_bytes=40 * GB, kv_cache_bytes=2 * GB,
        allow_ram_spill=False,
        host_meminfo=_meminfo(avail_mb=200000),  # plenty of RAM -- must not matter
    )
    assert res.status is RuntimeResolutionStatus.ENVIRONMENT_INFEASIBLE
    assert res.resolved is None


def test_explicit_ram_spill_resolves_when_safe_host_ram_is_sufficient():
    res = _resolve(
        topology=_dual_gpu_topology(a_mb=16000, b_mb=16000),
        weight_bytes=36 * GB, kv_cache_bytes=1 * GB,
        allow_ram_spill=True,
        host_meminfo=_meminfo(avail_mb=200000),
    )
    assert res.is_resolved
    assert res.resolved.placement_class == "ram_spill"
    assert res.resolved.allow_ram_spill is True
    assert res.resolved.execution_settings.allow_cpu_spill is True
    assert res.resolved.estimated_ram_spill_bytes and res.resolved.estimated_ram_spill_bytes > 0


def test_allow_cpu_spill_tracks_permission_not_placement_outcome():
    """allow_cpu_spill is identity-bearing as the resolved RAM-spill
    *permission*, not the resulting placement (matching
    runtime_identity.collect_runtime_identity, Stage 3.2C-2b). A run that
    was granted --allow-ram-spill but ends up fitting fully on GPU still
    records allow_cpu_spill=True; the actual placement is carried by
    placement_class / estimated_ram_spill_bytes."""
    res = _resolve(
        topology=_single_gpu_topology(installed_mb=48000),
        weight_bytes=8 * GB, kv_cache_bytes=1 * GB,
        allow_ram_spill=True,
    )
    assert res.is_resolved
    assert res.resolved.placement_class in ("full_gpu", "multi_gpu")
    assert res.resolved.estimated_ram_spill_bytes in (None, 0)
    assert res.resolved.allow_ram_spill is True
    assert res.resolved.execution_settings.allow_cpu_spill is True


def test_allow_cpu_spill_stays_none_without_permission_even_near_capacity():
    """The ordinary run (no permission) keeps allow_cpu_spill=None -- the
    historical default -- so its identity hash is unchanged."""
    res = _resolve(allow_ram_spill=False)
    assert res.is_resolved
    assert res.resolved.execution_settings.allow_cpu_spill is None


def test_explicit_ram_spill_still_fails_when_safe_physical_ram_insufficient():
    res = _resolve(
        topology=_dual_gpu_topology(a_mb=16000, b_mb=16000),
        weight_bytes=36 * GB, kv_cache_bytes=1 * GB,
        allow_ram_spill=True,
        host_meminfo=_meminfo(avail_mb=4000),  # tiny physical RAM
    )
    assert res.status is RuntimeResolutionStatus.ENVIRONMENT_INFEASIBLE


def test_swap_is_never_counted_as_capacity():
    # Tiny physical RAM but a huge SwapFree -- must still be infeasible.
    res = _resolve(
        topology=_dual_gpu_topology(a_mb=16000, b_mb=16000),
        weight_bytes=36 * GB, kv_cache_bytes=1 * GB,
        allow_ram_spill=True,
        host_meminfo=_meminfo(avail_mb=2000, swap_free_mb=500000, swap_used_mb=100),
    )
    assert res.status is RuntimeResolutionStatus.ENVIRONMENT_INFEASIBLE


def test_ram_spill_inconclusive_host_evidence_is_fit_unknown():
    res = _resolve(
        topology=_dual_gpu_topology(a_mb=16000, b_mb=16000),
        weight_bytes=36 * GB, kv_cache_bytes=1 * GB,
        allow_ram_spill=True,
        host_meminfo={},  # no MemAvailable
    )
    assert res.status is RuntimeResolutionStatus.FIT_UNKNOWN


# --------------------------------------------------------------------------
# no-spawn boundary (§9)
# --------------------------------------------------------------------------
def test_resolver_never_launches_a_runtime(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("runtime_resolution attempted to spawn a process")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(subprocess, "call", _boom)
    monkeypatch.setattr(subprocess, "check_output", _boom)
    monkeypatch.setattr(subprocess, "check_call", _boom)
    res = _resolve()
    assert res.is_resolved  # and no AssertionError was raised


def test_resolver_does_no_network_io(monkeypatch):
    import socket
    import urllib.request

    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("runtime_resolution opened a socket")))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("runtime_resolution made an HTTP request")))
    res = _resolve()
    assert res.is_resolved


def test_resolved_recipe_serializes_with_stable_physical_gpu_identity():
    res = _resolve()
    d = res.resolved.to_dict()
    assert d["selected_physical_gpu_uuids"] == [U_A]
    assert d["placement_class"] in ("full_gpu", "multi_gpu", "ram_spill")
    assert d["runtime_profile_identity_stable_key"]  # non-empty stable key


# --------------------------------------------------------------------------
# OWNER DECISION 3B.3C-OD1 -- deterministic multi-GPU tensor split
# --------------------------------------------------------------------------
from math import gcd  # noqa: E402

from llm_modelbench.topology_budget import TopologyBudget  # noqa: E402


def _expected_split(topology: TopologyBudget, uuids):
    caps = []
    by_uuid = {d.uuid: d for d in topology.devices}
    for u in uuids:
        caps.append(int(by_uuid[u].safe_capacity_bytes))
    g = 0
    for c in caps:
        g = gcd(g, c)
    return tuple(c // g for c in caps)


def _resolve_multi(**overrides):
    kwargs = dict(
        selected_backend="ollama",
        discovered_candidates=[_candidate()],
        topology=_dual_gpu_topology(a_mb=16000, b_mb=16000),
        host_meminfo=_meminfo(),
        weight_bytes=20 * GB,
        kv_cache_bytes=1 * GB,
        requested_context=8192,
        allow_ram_spill=False,
    )
    kwargs.update(overrides)
    return resolve_runtime(**kwargs)


def test_single_gpu_recipe_carries_no_tensor_split():
    res = _resolve()
    assert res.resolved.placement_class == "full_gpu"
    assert res.resolved.tensor_split_weights is None
    assert res.resolved.to_dict()["tensor_split_weights"] is None


def test_two_required_gpus_get_an_exact_deterministic_split():
    top = _dual_gpu_topology(a_mb=16000, b_mb=16000)
    res = _resolve_multi(topology=top)
    assert res.resolved.placement_class == "multi_gpu"
    uuids = res.resolved.selected_physical_gpu_uuids
    assert res.resolved.tensor_split_weights == _expected_split(top, uuids)


def test_three_required_gpus_get_an_exact_deterministic_split():
    top = _triple_gpu_topology(a_mb=12000, b_mb=12000, c_mb=12000)
    res = _resolve_multi(
        topology=top, weight_bytes=26 * GB, kv_cache_bytes=1 * GB,
    )
    assert res.resolved.placement_class == "multi_gpu"
    uuids = res.resolved.selected_physical_gpu_uuids
    assert len(uuids) == 3
    assert res.resolved.tensor_split_weights == _expected_split(top, uuids)


def test_minimum_number_of_gpus_is_preserved_in_the_split():
    # three cards available, workload fits on two -> split has exactly 2 weights
    top = _triple_gpu_topology(a_mb=16000, b_mb=16000, c_mb=16000)
    res = _resolve_multi(topology=top, weight_bytes=20 * GB, kv_cache_bytes=1 * GB)
    assert res.resolved.placement_class == "multi_gpu"
    assert len(res.resolved.selected_physical_gpu_uuids) == 2
    assert len(res.resolved.tensor_split_weights) == 2


def test_heterogeneous_capacities_produce_proportional_deterministic_weights():
    top = _dual_gpu_topology(a_mb=20000, b_mb=12000)
    res = _resolve_multi(topology=top, weight_bytes=24 * GB, kv_cache_bytes=1 * GB)
    assert res.resolved.placement_class == "multi_gpu"
    w = res.resolved.tensor_split_weights
    uuids = res.resolved.selected_physical_gpu_uuids
    assert w == _expected_split(top, uuids)
    # proportional: weight ratio tracks safe-capacity ratio
    by_uuid = {d.uuid: d for d in top.devices}
    cap0 = by_uuid[uuids[0]].safe_capacity_bytes
    cap1 = by_uuid[uuids[1]].safe_capacity_bytes
    assert w[0] * cap1 == w[1] * cap0


def test_equal_capacities_produce_equivalent_weights():
    top = _dual_gpu_topology(a_mb=16000, b_mb=16000)
    res = _resolve_multi(topology=top)
    assert res.resolved.tensor_split_weights == (1, 1)


def test_shuffled_hardware_inventory_produces_identical_resolved_recipe():
    inv_fwd = (
        GPUDevice(0, U_A, "0000:01:00.0", "A", 20000, None, None),
        GPUDevice(1, U_B, "0000:02:00.0", "B", 12000, None, None),
    )
    inv_rev = tuple(reversed(inv_fwd))
    t1 = topology_from_inventory(inv_fwd)
    t2 = topology_from_inventory(inv_rev)
    r1 = _resolve_multi(topology=t1, weight_bytes=24 * GB, kv_cache_bytes=1 * GB)
    r2 = _resolve_multi(topology=t2, weight_bytes=24 * GB, kv_cache_bytes=1 * GB)
    assert r1.resolved.selected_physical_gpu_uuids == r2.resolved.selected_physical_gpu_uuids
    assert r1.resolved.tensor_split_weights == r2.resolved.tensor_split_weights
    assert r1.resolved.to_dict() == r2.resolved.to_dict()


def test_shuffled_inventory_resolving_to_same_order_gives_same_request_identity_key():
    # 3B.3C FINAL IDENTITY CORRECTION, owner test #2: a shuffled *hardware
    # inventory* that resolves to the same primary-first GPU order must yield
    # the same MaterialisationRequest.identity_key -- inventory enumeration
    # order is not identity, the resolved primary-first order is.
    from llm_modelbench.runtime_lifecycle import MaterialisationRequest

    inv_fwd = (
        GPUDevice(0, U_A, "0000:01:00.0", "A", 20000, None, None),
        GPUDevice(1, U_B, "0000:02:00.0", "B", 12000, None, None),
    )
    inv_rev = tuple(reversed(inv_fwd))
    r1 = _resolve_multi(topology=topology_from_inventory(inv_fwd),
                        weight_bytes=24 * GB, kv_cache_bytes=1 * GB)
    r2 = _resolve_multi(topology=topology_from_inventory(inv_rev),
                        weight_bytes=24 * GB, kv_cache_bytes=1 * GB)
    k1 = MaterialisationRequest.from_resolution(r1).identity_key()
    k2 = MaterialisationRequest.from_resolution(r2).identity_key()
    assert k1 == k2


def test_same_gpu_set_different_resolved_primary_order_gives_different_identity_key():
    # 3B.3C FINAL IDENTITY CORRECTION, owner test #3: same physical cards,
    # host ordinals swapped so the resolver's primary-first order genuinely
    # flips (U_A first vs U_B first). The generated launch command differs
    # (CUDA_VISIBLE_DEVICES order, --tensor-split alignment), so the identity
    # key must differ.
    from llm_modelbench.runtime_lifecycle import MaterialisationRequest

    inv_a_primary = (
        GPUDevice(0, U_A, "0000:01:00.0", "A", 20000, None, None),
        GPUDevice(1, U_B, "0000:02:00.0", "B", 12000, None, None),
    )
    inv_b_primary = (
        GPUDevice(1, U_A, "0000:01:00.0", "A", 20000, None, None),
        GPUDevice(0, U_B, "0000:02:00.0", "B", 12000, None, None),
    )
    r_a = _resolve_multi(topology=topology_from_inventory(inv_a_primary),
                         weight_bytes=24 * GB, kv_cache_bytes=1 * GB)
    r_b = _resolve_multi(topology=topology_from_inventory(inv_b_primary),
                         weight_bytes=24 * GB, kv_cache_bytes=1 * GB)
    # sanity: the resolved primary-first order really did flip
    assert r_a.resolved.selected_physical_gpu_uuids[0] == U_A
    assert r_b.resolved.selected_physical_gpu_uuids[0] == U_B
    k_a = MaterialisationRequest.from_resolution(r_a).identity_key()
    k_b = MaterialisationRequest.from_resolution(r_b).identity_key()
    assert k_a != k_b


def test_transient_gpu_ordinal_changes_do_not_alter_the_per_uuid_split():
    # Same physical cards + capacities, host ordinals swapped. The pool is
    # ordered "primary (GPU0) first", so a genuine primary change legitimately
    # reorders selected_physical_gpu_uuids -- and the split tuple moves in
    # lockstep. The invariant that must hold is the physical-UUID -> weight
    # binding, not the tuple index.
    inv1 = (
        GPUDevice(0, U_A, "0000:01:00.0", "A", 20000, None, None),
        GPUDevice(1, U_B, "0000:02:00.0", "B", 12000, None, None),
    )
    inv2 = (
        GPUDevice(1, U_A, "0000:01:00.0", "A", 20000, None, None),
        GPUDevice(0, U_B, "0000:02:00.0", "B", 12000, None, None),
    )
    r1 = _resolve_multi(topology=topology_from_inventory(inv1),
                        weight_bytes=24 * GB, kv_cache_bytes=1 * GB)
    r2 = _resolve_multi(topology=topology_from_inventory(inv2),
                        weight_bytes=24 * GB, kv_cache_bytes=1 * GB)
    assert dict(zip(r1.resolved.selected_physical_gpu_uuids,
                    r1.resolved.tensor_split_weights)) == \
           dict(zip(r2.resolved.selected_physical_gpu_uuids,
                    r2.resolved.tensor_split_weights))


def test_split_does_not_use_live_free_vram():
    # identical physical capacity; one run has a device reporting low live-free
    # VRAM below its safe capacity. The split must be identical either way
    # (safe_capacity_bytes is static; effective_now_bytes would fold live_free in).
    base = _dual_gpu_topology(a_mb=20000, b_mb=12000)
    starved = _dual_gpu_topology(a_mb=20000, b_mb=12000, a_free_mb=1000, b_free_mb=1000)
    r_base = _resolve_multi(topology=base, weight_bytes=24 * GB, kv_cache_bytes=1 * GB)
    # starved live-free would make the pool not fit at all via effective_now;
    # force multi by allowing spill so ram_spill path still selects the pool.
    r_starved = _resolve_multi(
        topology=starved, weight_bytes=24 * GB, kv_cache_bytes=1 * GB,
        allow_ram_spill=True, host_meminfo=_meminfo(avail_mb=200000),
    )
    assert r_base.resolved.tensor_split_weights == r_starved.resolved.tensor_split_weights


def test_selected_uuid_order_is_primary_first():
    top = _triple_gpu_topology(a_mb=12000, b_mb=12000, c_mb=12000)
    res = _resolve_multi(topology=top, weight_bytes=26 * GB, kv_cache_bytes=1 * GB)
    # placement priority is host-ordinal ascending; U_A is ordinal 0
    assert res.resolved.selected_physical_gpu_uuids[0] == U_A


def test_unselected_gpu_never_appears_in_split_or_pool():
    top = _triple_gpu_topology(a_mb=16000, b_mb=16000, c_mb=16000)
    res = _resolve_multi(topology=top, weight_bytes=20 * GB, kv_cache_bytes=1 * GB)
    assert U_C not in res.resolved.selected_physical_gpu_uuids
    assert len(res.resolved.tensor_split_weights) == len(res.resolved.selected_physical_gpu_uuids)


def test_unknown_selected_capacity_fails_closed_no_recipe():
    # a selected card with no installed capacity -> safe_capacity_bytes is None
    inv = (
        GPUDevice(0, U_A, "0000:01:00.0", "A", 16000, None, None),
        GPUDevice(1, U_B, "0000:02:00.0", "B", None, None, None),
    )
    res = _resolve_multi(topology=topology_from_inventory(inv),
                         weight_bytes=20 * GB, kv_cache_bytes=1 * GB)
    assert res.resolved is None
    assert res.status in (
        RuntimeResolutionStatus.FIT_UNKNOWN,
        RuntimeResolutionStatus.ENVIRONMENT_INFEASIBLE,
    )


def test_candidate_run_repeated_twice_is_structurally_identical():
    top = _dual_gpu_topology(a_mb=20000, b_mb=12000)
    a = _resolve_multi(topology=top, weight_bytes=24 * GB, kv_cache_bytes=1 * GB)
    b = _resolve_multi(topology=top, weight_bytes=24 * GB, kv_cache_bytes=1 * GB)
    assert a.resolved.to_dict() == b.resolved.to_dict()


def test_multi_gpu_ram_spill_recipe_also_carries_the_split():
    top = _dual_gpu_topology(a_mb=16000, b_mb=12000)
    res = _resolve_multi(
        topology=top, weight_bytes=30 * GB, kv_cache_bytes=1 * GB,
        allow_ram_spill=True, host_meminfo=_meminfo(avail_mb=200000),
    )
    assert res.resolved.placement_class == "ram_spill"
    assert len(res.resolved.selected_physical_gpu_uuids) == 2
    assert res.resolved.tensor_split_weights == _expected_split(
        top, res.resolved.selected_physical_gpu_uuids
    )
