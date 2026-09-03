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


def _dual_gpu_topology(*, a_mb=16000, b_mb=16000):
    return topology_from_inventory(
        (
            GPUDevice(0, U_A, "0000:01:00.0", "fixture-A", a_mb, None, None),
            GPUDevice(1, U_B, "0000:02:00.0", "fixture-B", b_mb, None, None),
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
