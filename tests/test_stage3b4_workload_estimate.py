"""Anvil Stage 3B.4 -- production workload-estimate unblock.

Resolves DEFECT-3B.3E-01: `cli._resolve_and_materialise_for_run` supplied no
`weight_bytes`, so `resolve_runtime()` returned `FIT_UNKNOWN` for every
non-mock backend and no run reached `materialise()`.

Stage 3B.4:
* managed llama_cpp (single explicit --models + resolved local GGUF) supplies
  `weight_bytes = verified GGUF file size` and asserts
  `owned_placement_required=True` -> the resolver runs §5 and reaches a real
  fit / preflight / materialisation outcome, never a bare FIT_UNKNOWN for a
  missing estimate;
* Ollama (amendment §23) and llama_cpp with no resolved GGUF are reuse-only:
  the cli seam asserts `owned_placement_required=False`, the resolver skips
  §5/§6 (no managed placement decision), and the run reaches
  `REUSED_EXTERNAL` -- no local GGUF, no spawn.

No real GPU / Ollama / llama.cpp / inference. Fake seams + fake preflight.
"""
from __future__ import annotations

import hashlib

import pytest

from llm_modelbench import cli
from llm_modelbench import runtime_materialisation as rm
from llm_modelbench.config import Config
from llm_modelbench.identity import resolve_runtime_profile_identity
from llm_modelbench.llama_server_materialisation import (
    MaterialisationStatus,
    materialise as real_materialise,
)
from llm_modelbench.runtime_identity import RuntimeExecutionSettings
from llm_modelbench.runtime_lifecycle import MaterialisationRequest
from llm_modelbench.runtime_profiles import RuntimeCandidate, RuntimeProfile
from llm_modelbench.runtime_resolution import (
    ResolvedRuntime,
    RuntimeResolution,
    RuntimeResolutionStatus,
    resolve_runtime,
)
from llm_modelbench.topology_budget import topology_from_inventory
from llm_modelbench.hardware import GPUDevice

U_A = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
GB = 1024 * 1024 * 1024
_BYTES = b"GGUF\x00 fake weights for the 3B.4 workload-estimate test"
_SHA = hashlib.sha256(_BYTES).hexdigest()


@pytest.fixture()
def gguf(tmp_path):
    p = tmp_path / "bench-model.gguf"
    p.write_bytes(_BYTES)
    return p


# ---------------------------------------------------------------------------
# resolve_runtime -- the owned_placement_required contract
# ---------------------------------------------------------------------------
def _candidate(*, backend="ollama", endpoint="http://127.0.0.1:11434", health="healthy"):
    return RuntimeCandidate(
        profile=RuntimeProfile(name="p", backend=backend, endpoint=endpoint,
                               provenance="configured"),
        health=health, source=("saved_profile",), detail=f"{health} fx",
    )


def _single_gpu_topology(installed_mb=24000):
    return topology_from_inventory(
        (GPUDevice(0, U_A, "0000:01:00.0", "fx", installed_mb, None, None),),
    )


def _resolve(**over):
    kw = dict(
        selected_backend="ollama",
        discovered_candidates=[_candidate()],
        topology=_single_gpu_topology(),
        host_meminfo={"ram_total_mb": 128000, "ram_available_mb": 64000,
                      "swap_free_mb": 0, "swap_used_mb": 0},
        weight_bytes=None,
        kv_cache_bytes=None,
        requested_context=8192,
        allow_ram_spill=False,
    )
    kw.update(over)
    return resolve_runtime(**kw)


def test_default_owned_placement_required_missing_weight_is_fit_unknown():
    """Unchanged historical behaviour: the default (owned_placement_required
    absent -> True) still fails closed at FIT_UNKNOWN when no weight is given.
    Reuse-only is never a default."""
    res = _resolve()  # no owned_placement_required kwarg
    assert res.status is RuntimeResolutionStatus.FIT_UNKNOWN
    assert res.resolved is None


def test_reuse_only_skips_fit_gate_and_resolves():
    res = _resolve(owned_placement_required=False)
    assert res.status is RuntimeResolutionStatus.RESOLVED
    assert res.resolved is not None
    assert res.resolved.owned_placement is False
    assert res.resolved.placement_class is None
    assert res.resolved.selected_physical_gpu_uuids == ()
    assert res.resolved.tensor_split_weights is None
    assert "no managed placement decision" in res.detail
    assert res.resolved.requested_context == 8192


def test_reuse_only_recipe_does_not_bake_operator_context_into_identity():
    """The reuse-only recipe retains the operator's requested context as
    evidence (`requested_context`) but must NOT assert it in the identity-
    bearing execution settings: ModelBench launches nothing on this path, an
    external llama-server fixes --ctx-size at its own launch, and a context
    ModelBench never set must not trip `context_changed` campaign-compat."""
    res = _resolve(owned_placement_required=False, requested_context=8192)
    assert res.resolved.requested_context == 8192
    assert res.resolved.execution_settings.context_size is None
    # identity is independent of the operator's requested context on this path
    other = _resolve(owned_placement_required=False, requested_context=4096)
    assert (res.resolved.runtime_profile_identity.stable_key()
            == other.resolved.runtime_profile_identity.stable_key())


def test_reuse_only_assertion_wins_over_a_supplied_estimate():
    """A stray weight_bytes cannot silently re-enable a placement decision the
    caller says ModelBench is not making."""
    res = _resolve(owned_placement_required=False, weight_bytes=8 * GB,
                   kv_cache_bytes=1 * GB)
    assert res.status is RuntimeResolutionStatus.RESOLVED
    assert res.resolved.owned_placement is False
    assert res.resolved.placement_class is None
    assert res.workload_fit is None  # §5 never ran


def test_reuse_only_still_runs_backend_authority():
    res = _resolve(selected_backend=None, owned_placement_required=False)
    assert res.status is RuntimeResolutionStatus.NO_BACKEND_SELECTED


def test_reuse_only_still_runs_candidate_health_gate():
    res = _resolve(discovered_candidates=[_candidate(health="unreachable")],
                   owned_placement_required=False)
    assert res.status is RuntimeResolutionStatus.RUNTIME_UNAVAILABLE


def test_reuse_only_still_runs_identity_sufficiency_gate():
    res = _resolve(owned_placement_required=False,
                   require_content_addressed_model_identity=True,
                   model_primary_sha256=None)
    assert res.status is RuntimeResolutionStatus.IDENTITY_INSUFFICIENT


def test_reuse_only_still_runs_required_capability_gate():
    from llm_modelbench.runtime_resolution import RequiredCapability
    res = _resolve(owned_placement_required=False,
                   required_capabilities=[RequiredCapability("tools", None)])
    assert res.status is RuntimeResolutionStatus.CAPABILITY_EVIDENCE_INSUFFICIENT


def test_managed_capable_with_weight_reaches_a_real_fit_outcome():
    """weight_bytes alone (kv_cache_bytes None) -> candidate_single_gpu_fit ->
    full_gpu -> RESOLVED. Not FIT_UNKNOWN."""
    res = _resolve(selected_backend="llama_cpp",
                   discovered_candidates=[_candidate(backend="llama_cpp",
                                                     endpoint="http://127.0.0.1:8081")],
                   weight_bytes=4 * GB, owned_placement_required=True)
    assert res.status is RuntimeResolutionStatus.RESOLVED
    assert res.resolved.owned_placement is True
    assert res.resolved.placement_class == "full_gpu"


def test_managed_capable_without_weight_is_still_fit_unknown():
    res = _resolve(selected_backend="llama_cpp",
                   discovered_candidates=[_candidate(backend="llama_cpp",
                                                     endpoint="http://127.0.0.1:8081")],
                   weight_bytes=None, owned_placement_required=True)
    assert res.status is RuntimeResolutionStatus.FIT_UNKNOWN


def test_managed_capable_weight_exceeds_pool_is_environment_infeasible_not_fit_unknown():
    res = _resolve(selected_backend="llama_cpp",
                   discovered_candidates=[_candidate(backend="llama_cpp",
                                                     endpoint="http://127.0.0.1:8081")],
                   topology=_single_gpu_topology(installed_mb=8000),
                   weight_bytes=40 * GB, owned_placement_required=True)
    assert res.status is RuntimeResolutionStatus.ENVIRONMENT_INFEASIBLE


# ---------------------------------------------------------------------------
# MANAGED-SPAWN REACHABILITY CORRECTION -- managed llama-server must be
# reachable with NO already-running external llama-server, given the complete
# managed-input set. The operator declares llama_cpp intent via a configured
# llama_cpp runtime profile; when its endpoint is not serving, discovery
# yields an UNHEALTHY llama_cpp candidate (not a missing one).
# ---------------------------------------------------------------------------
class _Exe:
    def __init__(self, backend, state):
        self.backend = backend
        self.state = state
        self.executable_available = state == "installed"


_LLAMA_INSTALLED = [_Exe("llama_cpp", "installed"), _Exe("ollama", "installed")]


@pytest.mark.parametrize("health", ["unreachable", "unhealthy", "unknown"])
def test_managed_llama_cpp_no_healthy_external_reaches_resolved_owned(health):
    """No compatible external runtime to reuse (the configured llama_cpp
    endpoint is down) + complete managed inputs + llama-server executable
    installed -> RESOLVED owned-placement recipe that materialise() will
    spawn. This must NOT depend on a healthy external endpoint or on the
    external_still_healthy demotion path."""
    res = _resolve(
        selected_backend="llama_cpp",
        discovered_candidates=[_candidate(backend="llama_cpp",
                                          endpoint="http://127.0.0.1:8081",
                                          health=health)],
        topology=_single_gpu_topology(installed_mb=24000),
        weight_bytes=4 * GB,
        owned_placement_required=True,
        backend_executables=_LLAMA_INSTALLED,
    )
    assert res.status is RuntimeResolutionStatus.RESOLVED
    assert res.resolved.owned_placement is True
    assert res.resolved.placement_class == "full_gpu"
    # the recipe endpoint is the operator's configured (non-serving) endpoint
    assert res.resolved.endpoint == "http://127.0.0.1:8081"
    assert res.resolved.runtime_profile_name == "p"
    # HONEST: no fake healthy candidate is introduced
    assert res.selected_candidate is not None
    assert res.selected_candidate.health != "healthy"
    # evidence names the managed-from-cold decision
    assert "no reusable external runtime" in res.detail
    assert "managed llama-server launch" in res.detail


def test_managed_llama_cpp_no_healthy_external_then_materialise_spawns():
    """End-to-end structural: the RESOLVED owned recipe from a no-healthy
    resolution reaches spawn_managed via materialise(), and
    external_still_healthy is NEVER consulted (MUT-2 guard)."""
    res = _resolve(
        selected_backend="llama_cpp",
        discovered_candidates=[_candidate(backend="llama_cpp",
                                          endpoint="http://127.0.0.1:8081",
                                          health="unreachable")],
        topology=_single_gpu_topology(installed_mb=24000),
        weight_bytes=4 * GB,
        owned_placement_required=True,
        backend_executables=_LLAMA_INSTALLED,
    )
    req = MaterialisationRequest.from_resolution(res)
    spawned = {"n": 0}

    def _probe_must_not_be_called(r):
        pytest.fail("external_still_healthy consulted on a no-external managed spawn")

    out = real_materialise(
        req,
        spawn_managed=lambda r: spawned.__setitem__("n", spawned["n"] + 1)
        or __import__("llm_modelbench.llama_server_materialisation", fromlist=["x"]).ManagedMaterialisationOutcome(
            status=MaterialisationStatus.SPAWNED_READY, detail="ok"),
        external_still_healthy=_probe_must_not_be_called,
    )
    assert out.status is MaterialisationStatus.SPAWNED_READY
    assert spawned["n"] == 1


def test_managed_llama_cpp_remote_endpoint_is_never_a_local_managed_launch():
    """A remote-host llama_cpp profile probes as health 'unsupported'
    ('non-local endpoint discovery is disabled'). A managed llama-server is
    always spawned on loopback -- a remote profile must NOT silently become a
    local spawn whose evidence then records the remote address."""
    res = _resolve(
        selected_backend="llama_cpp",
        discovered_candidates=[_candidate(backend="llama_cpp",
                                          endpoint="http://192.168.1.50:8080",
                                          health="unsupported")],
        topology=_single_gpu_topology(installed_mb=24000),
        weight_bytes=4 * GB,
        owned_placement_required=True,
        backend_executables=_LLAMA_INSTALLED,
    )
    assert res.status is not RuntimeResolutionStatus.RESOLVED
    assert res.resolved is None


def test_managed_llama_cpp_no_external_missing_executable_fails_structurally():
    """No reusable external + missing llama-server executable -> structural
    refusal, NOT a managed launch."""
    res = _resolve(
        selected_backend="llama_cpp",
        discovered_candidates=[_candidate(backend="llama_cpp",
                                          endpoint="http://127.0.0.1:8081",
                                          health="unreachable")],
        topology=_single_gpu_topology(installed_mb=24000),
        weight_bytes=4 * GB,
        owned_placement_required=True,
        backend_executables=[_Exe("llama_cpp", "not_installed")],
    )
    assert res.status is not RuntimeResolutionStatus.RESOLVED
    assert res.resolved is None


def test_managed_llama_cpp_no_external_missing_weight_fails_structurally():
    """No reusable external + no weight estimate -> FIT_UNKNOWN, never a
    managed launch on an un-estimated workload."""
    res = _resolve(
        selected_backend="llama_cpp",
        discovered_candidates=[_candidate(backend="llama_cpp",
                                          endpoint="http://127.0.0.1:8081",
                                          health="unreachable")],
        topology=_single_gpu_topology(installed_mb=24000),
        weight_bytes=None,
        owned_placement_required=True,
        backend_executables=_LLAMA_INSTALLED,
    )
    assert res.status is RuntimeResolutionStatus.FIT_UNKNOWN
    assert res.resolved is None


def test_managed_llama_cpp_no_external_weight_exceeds_pool_is_infeasible():
    res = _resolve(
        selected_backend="llama_cpp",
        discovered_candidates=[_candidate(backend="llama_cpp",
                                          endpoint="http://127.0.0.1:8081",
                                          health="unreachable")],
        topology=_single_gpu_topology(installed_mb=8000),
        weight_bytes=40 * GB,
        owned_placement_required=True,
        backend_executables=_LLAMA_INSTALLED,
    )
    assert res.status is RuntimeResolutionStatus.ENVIRONMENT_INFEASIBLE


def test_healthy_external_llama_cpp_still_reuses_never_spawns():
    """REUSE PRESERVED: a healthy external llama_cpp candidate resolves the
    old way (reuse-eligible), and materialise() reuses it -- no spawn."""
    res = _resolve(
        selected_backend="llama_cpp",
        discovered_candidates=[_candidate(backend="llama_cpp",
                                          endpoint="http://127.0.0.1:8081",
                                          health="healthy")],
        topology=_single_gpu_topology(installed_mb=24000),
        weight_bytes=4 * GB,
        owned_placement_required=True,
        backend_executables=_LLAMA_INSTALLED,
    )
    assert res.status is RuntimeResolutionStatus.RESOLVED
    assert res.selected_candidate.health == "healthy"
    req = MaterialisationRequest.from_resolution(res)
    out = real_materialise(
        req,
        spawn_managed=lambda r: pytest.fail("healthy external must be reused, not spawned"),
        external_still_healthy=None,
    )
    assert out.status is MaterialisationStatus.REUSED_EXTERNAL


def test_unhealthy_external_plus_complete_inputs_can_spawn_without_demotion():
    """The required contract, explicitly: unhealthy external candidate +
    complete managed inputs -> managed launch eligible, and the managed
    spawn does NOT depend on external_still_healthy demotion (the resolution
    itself carries an owned recipe)."""
    res = _resolve(
        selected_backend="llama_cpp",
        discovered_candidates=[_candidate(backend="llama_cpp",
                                          endpoint="http://127.0.0.1:8081",
                                          health="unhealthy")],
        topology=_single_gpu_topology(installed_mb=24000),
        weight_bytes=4 * GB,
        owned_placement_required=True,
        backend_executables=_LLAMA_INSTALLED,
    )
    assert res.status is RuntimeResolutionStatus.RESOLVED
    assert res.resolved.owned_placement is True


def test_ollama_no_healthy_external_never_becomes_a_managed_launch():
    """OLLAMA PRESERVED: no spawn, no service mutation, no fallback -- an
    unhealthy Ollama candidate stays unresolved even with a llama-server
    executable installed and a weight estimate present."""
    res = _resolve(
        selected_backend="ollama",
        discovered_candidates=[_candidate(backend="ollama",
                                          endpoint="http://127.0.0.1:11434",
                                          health="unreachable")],
        topology=_single_gpu_topology(installed_mb=24000),
        weight_bytes=4 * GB,
        owned_placement_required=True,
        backend_executables=_LLAMA_INSTALLED,
    )
    assert res.status is not RuntimeResolutionStatus.RESOLVED
    assert res.resolved is None


def test_reuse_only_assertion_still_wins_for_llama_cpp_no_external():
    """A caller that asserts owned_placement_required=False for llama_cpp
    with no healthy external gets a reuse-only recipe (never a managed
    launch) even with the executable installed and a weight estimate -- the
    caller's assertion is authoritative (MUT-3 at the resolver boundary)."""
    res = _resolve(
        selected_backend="llama_cpp",
        discovered_candidates=[_candidate(backend="llama_cpp",
                                          endpoint="http://127.0.0.1:8081",
                                          health="healthy")],
        topology=_single_gpu_topology(installed_mb=24000),
        weight_bytes=4 * GB,
        owned_placement_required=False,
        backend_executables=_LLAMA_INSTALLED,
    )
    assert res.status is RuntimeResolutionStatus.RESOLVED
    assert res.resolved.owned_placement is False


def test_runtime_ambiguous_is_not_swallowed_by_managed_launch():
    """Two healthy llama_cpp candidates on distinct endpoints stay
    RUNTIME_AMBIGUOUS -- the managed-launch branch is only for the
    no-healthy case."""
    res = _resolve(
        selected_backend="llama_cpp",
        discovered_candidates=[
            _candidate(backend="llama_cpp", endpoint="http://127.0.0.1:8081",
                       health="healthy"),
            _candidate(backend="llama_cpp", endpoint="http://127.0.0.1:8082",
                       health="healthy"),
        ],
        topology=_single_gpu_topology(installed_mb=24000),
        weight_bytes=4 * GB,
        owned_placement_required=True,
        backend_executables=_LLAMA_INSTALLED,
    )
    assert res.status is RuntimeResolutionStatus.RUNTIME_AMBIGUOUS


# ---------------------------------------------------------------------------
# ResolvedRuntime marker validation
# ---------------------------------------------------------------------------
def _reuse_recipe(**over):
    s = RuntimeExecutionSettings(strategy=None, context_size=8192)
    kw = dict(
        backend="ollama", endpoint="http://127.0.0.1:11434",
        runtime_profile_name="p", execution_settings=s,
        runtime_profile_identity=resolve_runtime_profile_identity(
            backend="ollama", execution_settings=s),
        selected_physical_gpu_uuids=(), placement_class=None,
        requested_context=8192, allow_ram_spill=False,
        estimated_ram_spill_bytes=None, owned_placement=False,
    )
    kw.update(over)
    return ResolvedRuntime(**kw)


def test_reuse_only_recipe_rejects_a_fabricated_placement():
    with pytest.raises(ValueError, match="placement_class must be None"):
        _reuse_recipe(placement_class="full_gpu")


def test_reuse_only_recipe_rejects_selected_gpus():
    with pytest.raises(ValueError, match="selects no physical GPU"):
        _reuse_recipe(selected_physical_gpu_uuids=(U_A,))


def test_reuse_only_recipe_rejects_a_tensor_split():
    with pytest.raises(ValueError, match="no tensor split"):
        _reuse_recipe(tensor_split_weights=(1, 1))


def test_owned_recipe_still_requires_a_real_placement_label():
    with pytest.raises(ValueError, match="placement_class must be one of"):
        _reuse_recipe(owned_placement=True, placement_class=None,
                      selected_physical_gpu_uuids=())


def test_reuse_only_identity_key_is_stable_and_non_colliding():
    a = MaterialisationRequest.from_resolution(
        RuntimeResolution(status=RuntimeResolutionStatus.RESOLVED, reason="r",
                          detail="d", resolved=_reuse_recipe(),
                          selected_candidate=_candidate()))
    b = MaterialisationRequest.from_resolution(
        RuntimeResolution(status=RuntimeResolutionStatus.RESOLVED, reason="r",
                          detail="d", resolved=_reuse_recipe(),
                          selected_candidate=_candidate()))
    assert a.identity_key() == b.identity_key()
    assert "reuse_only" in a.identity_key()
    # cannot collide with an owned full_gpu recipe on the same endpoint
    owned = _reuse_recipe(owned_placement=True, placement_class="full_gpu",
                          selected_physical_gpu_uuids=(U_A,))
    owned_req = MaterialisationRequest.from_resolution(
        RuntimeResolution(status=RuntimeResolutionStatus.RESOLVED, reason="r",
                          detail="d", resolved=owned, selected_candidate=_candidate()))
    assert owned_req.identity_key() != a.identity_key()


# ---------------------------------------------------------------------------
# materialise() -- reuse-only never spawns
# ---------------------------------------------------------------------------
def _reuse_request(backend="llama_cpp"):
    return MaterialisationRequest.from_resolution(
        RuntimeResolution(
            status=RuntimeResolutionStatus.RESOLVED, reason="r", detail="d",
            resolved=_reuse_recipe(backend=backend,
                                   endpoint="http://127.0.0.1:8081"
                                   if backend == "llama_cpp"
                                   else "http://127.0.0.1:11434"),
            selected_candidate=_candidate(
                backend=backend,
                endpoint="http://127.0.0.1:8081" if backend == "llama_cpp"
                else "http://127.0.0.1:11434"),
        )
    )


def test_reuse_only_reuses_healthy_external_endpoint():
    out = real_materialise(_reuse_request("llama_cpp"),
                           spawn_managed=lambda r: pytest.fail("no spawn"),
                           external_still_healthy=None)
    assert out.status is MaterialisationStatus.REUSED_EXTERNAL


@pytest.mark.parametrize("backend", ["llama_cpp", "ollama"])
def test_reuse_only_demotion_is_a_structured_refusal_never_a_spawn(backend):
    out = real_materialise(
        _reuse_request(backend),
        spawn_managed=lambda r: pytest.fail("reuse-only must never spawn"),
        external_still_healthy=lambda r: False,  # endpoint died
    )
    assert out.status is MaterialisationStatus.EXTERNAL_RUNTIME_REQUIRED
    assert "reuse-only" in out.detail
    assert "never spawns" in out.detail


# ---------------------------------------------------------------------------
# cli seam -- the owned_placement / weight_bytes decision
# ---------------------------------------------------------------------------
class _FakePreflight:
    def __init__(self, backend, endpoint):
        self.blocked = False
        self.blocker = None
        self.selected_candidate = RuntimeCandidate(
            profile=RuntimeProfile(name="p", backend=backend, endpoint=endpoint,
                                   provenance="configured"),
            health="healthy", source=("saved_profile",), detail="fx",
        )
        self.candidates = [self.selected_candidate]
        self.gpu_inventory = []
        self.topology = object()


def _wire(monkeypatch, *, backend, models, cfg, cap):
    import llm_modelbench.runtime_profiles as rp
    import llm_modelbench.hardware as hw
    monkeypatch.setattr(cli, "load_profiles", lambda store: ([], None))
    monkeypatch.setattr(cli, "resolve_operational_preflight",
                        lambda *a, **k: _FakePreflight(backend, "http://127.0.0.1:8081"))
    monkeypatch.setattr(rp, "discover_backend_executables", lambda **k: None)
    monkeypatch.setattr(cli, "_llama_server_executable_path", lambda be: "/opt/llama-server")
    monkeypatch.setattr(hw, "host_memory_snapshot", lambda: {})
    monkeypatch.setattr(rm, "production_seams", lambda **k: object())

    def _capture(**kwargs):
        cap.update(kwargs)
        return rm.RuntimeMaterialisationOutcome(
            ok=False, backend=kwargs.get("selected_backend") or "",
            resolution_status="fx", refusal_reason="stub",
            artifact_resolution=kwargs.get("artifact_resolution"),
        )

    monkeypatch.setattr(rm, "resolve_and_materialise_runtime", _capture)

    class _Args:
        pass
    a = _Args()
    a.models = models
    a.unattended = False
    a.runtime_profile = None
    a.runtime_profiles_file = None
    return a


def test_cli_managed_llama_cpp_supplies_gguf_file_size_as_weight_bytes(monkeypatch, gguf):
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": str(gguf)}
    cap = {}
    args = _wire(monkeypatch, backend="llama_cpp", models="bench-model", cfg=cfg, cap=cap)
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert cap["owned_placement_required"] is True
    assert cap["weight_bytes"] == len(_BYTES)
    ar = cap["artifact_resolution"]
    assert ar["weight_bytes"] == len(_BYTES)
    assert ar["weight_bytes_source"] == "verified_local_gguf_file_size"
    assert ar["kv_cache_bytes"] is None
    assert ar["workload_estimate_status"] == "supplied"
    assert ar["owned_placement"] is True


def test_cli_managed_weight_bytes_is_file_size_not_caller_claim(monkeypatch, gguf):
    """The estimate comes from the resolved+hashed artifact path, never a
    config- or caller-claimed number."""
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": str(gguf)}
    cap = {}
    args = _wire(monkeypatch, backend="llama_cpp", models="bench-model", cfg=cfg, cap=cap)
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert cap["weight_bytes"] == gguf.stat().st_size == len(_BYTES)


def test_cli_rewriting_the_gguf_changes_the_estimate_and_hash(monkeypatch, gguf):
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": str(gguf)}
    cap1 = {}
    args = _wire(monkeypatch, backend="llama_cpp", models="bench-model", cfg=cfg, cap=cap1)
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    gguf.write_bytes(_BYTES + b" MUCH more weight now")
    cap2 = {}
    args = _wire(monkeypatch, backend="llama_cpp", models="bench-model", cfg=cfg, cap=cap2)
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert cap2["weight_bytes"] != cap1["weight_bytes"]
    assert cap2["model_primary_sha256"] != cap1["model_primary_sha256"]


def test_cli_ollama_is_reuse_only_even_with_a_configured_gguf(monkeypatch, gguf):
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": str(gguf)}
    cap = {}
    args = _wire(monkeypatch, backend="ollama", models="bench-model", cfg=cfg, cap=cap)
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert cap["owned_placement_required"] is False
    assert cap["weight_bytes"] is None
    assert cap["model_primary_sha256"] is None
    assert cap["artifact_resolution"]["reuse_only"] is True
    assert cap["artifact_resolution"]["workload_estimate_status"] == "not_required_reuse_only"


def test_cli_multi_model_llama_cpp_is_reuse_only(monkeypatch, gguf):
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": str(gguf), "other": str(gguf)}
    cap = {}
    args = _wire(monkeypatch, backend="llama_cpp", models="bench-model;other", cfg=cfg, cap=cap)
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert cap["owned_placement_required"] is False
    assert cap["weight_bytes"] is None


def test_cli_select_llama_cpp_is_reuse_only(monkeypatch, gguf):
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": str(gguf)}
    cap = {}
    args = _wire(monkeypatch, backend="llama_cpp", models=None, cfg=cfg, cap=cap)
    args.select = True
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert cap["owned_placement_required"] is False
    assert cap["weight_bytes"] is None


def test_cli_all_installed_default_llama_cpp_is_reuse_only(monkeypatch, gguf):
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": str(gguf)}
    cap = {}
    args = _wire(monkeypatch, backend="llama_cpp", models=None, cfg=cfg, cap=cap)
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert cap["owned_placement_required"] is False
    assert cap["weight_bytes"] is None


def test_cli_missing_configured_gguf_is_reuse_only_not_a_blocked_spawn(monkeypatch):
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": "/nonexistent/bench-model.gguf"}
    cap = {}
    args = _wire(monkeypatch, backend="llama_cpp", models="bench-model", cfg=cfg, cap=cap)
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert cap["owned_placement_required"] is False
    assert cap["weight_bytes"] is None
    assert cap["artifact_resolution"]["status"] == "missing"
    assert cap["artifact_resolution"]["blocked_managed_spawn"] is False


# ---------------------------------------------------------------------------
# end-to-end composition: managed-capable + reuse-only both reach materialise
# ---------------------------------------------------------------------------
def _resolution_for(backend, endpoint, health="healthy"):
    s = RuntimeExecutionSettings(strategy="single_device", context_size=8192)
    recipe = ResolvedRuntime(
        backend=backend, endpoint=endpoint, runtime_profile_name="p",
        execution_settings=s,
        runtime_profile_identity=resolve_runtime_profile_identity(
            backend=backend, execution_settings=s),
        selected_physical_gpu_uuids=(U_A,), placement_class="full_gpu",
        requested_context=8192, allow_ram_spill=False, estimated_ram_spill_bytes=None,
        model_primary_sha256=_SHA,
    )
    return RuntimeResolution(
        status=RuntimeResolutionStatus.RESOLVED, reason="resolved", detail="fx",
        resolved=recipe,
        selected_candidate=RuntimeCandidate(
            profile=RuntimeProfile(name="p", backend=backend, endpoint=endpoint,
                                   provenance="configured"),
            health=health, source=("saved_profile",), detail="fx"),
    )


def _reuse_seams():
    return rm.MaterialisationSeams(
        spawn_managed=lambda r: pytest.fail("no spawn on the reuse path"),
        external_still_healthy=None,
    )


def test_composition_managed_capable_no_longer_bare_fit_unknown():
    """The managed-capable composition with weight_bytes reaches materialise
    (REUSED_EXTERNAL against the healthy endpoint), not a FIT_UNKNOWN refusal
    for a missing estimate."""
    out = rm.resolve_and_materialise_runtime(
        selected_backend="llama_cpp",
        discovered_candidates=[],
        topology=object(),
        host_meminfo={},
        seams=_reuse_seams(),
        weight_bytes=len(_BYTES),
        owned_placement_required=True,
        model_primary_sha256=_SHA,
        resolve_fn=lambda **k: _resolution_for("llama_cpp", "http://127.0.0.1:8081"),
        materialise_fn=real_materialise,
    )
    assert out.ok
    assert out.materialisation_status == "reused_external"


# ---------------------------------------------------------------------------
# cli seam -- managed-spawn reachability: a configured llama_cpp profile whose
# endpoint is DOWN is managed-launch eligible, not a dead-end refusal.
# ---------------------------------------------------------------------------
class _BlockedPreflight:
    """resolve_operational_preflight result when select_runtime raised: the
    explicit/default llama_cpp profile's endpoint is not serving, so its
    candidate is present but unhealthy."""

    def __init__(self, *, profile_backend="llama_cpp",
                 endpoint="http://127.0.0.1:8081", reason="unhealthy_profile",
                 candidate_health="unreachable", topology=None):
        from llm_modelbench.preflight import PreflightBlocker

        self.blocked = True
        self.blocker = PreflightBlocker(reason=reason, detail="endpoint down")
        self.selected_candidate = None
        self.candidates = [
            RuntimeCandidate(
                profile=RuntimeProfile(name="llama-local", backend=profile_backend,
                                       endpoint=endpoint, provenance="configured"),
                health=candidate_health, source=("saved_profile",),
                detail=f"{candidate_health} fx",
            )
        ]
        self.gpu_inventory = []
        self.topology = topology if topology is not None else _single_gpu_topology(24000)


class _Exe2:
    def __init__(self, backend, state):
        self.backend, self.state = backend, state
        self.executable_available = state == "installed"


def _wire_blocked(monkeypatch, *, cfg, cap, models="bench-model",
                  profiles=None, default="llama-local", reason="unhealthy_profile",
                  preflight=None, exes=(("llama_cpp", "installed"),)):
    import llm_modelbench.runtime_profiles as rp
    import llm_modelbench.hardware as hw

    profs = profiles if profiles is not None else [
        RuntimeProfile(name="llama-local", backend="llama_cpp",
                       endpoint="http://127.0.0.1:8081", provenance="configured"),
    ]
    monkeypatch.setattr(cli, "load_profiles", lambda store: (profs, default))
    monkeypatch.setattr(
        cli, "resolve_operational_preflight",
        lambda *a, **k: preflight or _BlockedPreflight(reason=reason),
    )
    monkeypatch.setattr(rp, "discover_backend_executables",
                        lambda **k: [_Exe2(b, s) for b, s in exes])
    monkeypatch.setattr(cli, "_llama_server_executable_path", lambda be: "/opt/llama-server")
    monkeypatch.setattr(hw, "host_memory_snapshot", lambda: {})
    monkeypatch.setattr(rm, "production_seams", lambda **k: object())

    def _capture(**kwargs):
        cap.update(kwargs)
        return rm.RuntimeMaterialisationOutcome(
            ok=False, backend=kwargs.get("selected_backend") or "",
            resolution_status="fx", refusal_reason="stub",
            artifact_resolution=kwargs.get("artifact_resolution"),
        )

    monkeypatch.setattr(rm, "resolve_and_materialise_runtime", _capture)

    class _Args:
        pass
    a = _Args()
    a.models = models
    a.unattended = False
    a.runtime_profile = None
    a.runtime_profiles_file = None
    return a


def test_cli_blocked_llama_cpp_profile_with_gguf_is_managed_launch_not_refusal(monkeypatch, gguf):
    """The MANAGED-SPAWN REACHABILITY contract at the cli seam: a configured
    llama_cpp profile whose endpoint is down + a resolved GGUF + installed
    executable reaches resolve_and_materialise_runtime with
    selected_backend='llama_cpp' and owned_placement_required=True -- not a
    NO_USABLE_ENDPOINT dead end."""
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": str(gguf)}
    cap = {}
    args = _wire_blocked(monkeypatch, cfg=cfg, cap=cap)
    out = cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert cap["selected_backend"] == "llama_cpp"
    assert cap["owned_placement_required"] is True
    assert cap["weight_bytes"] == len(_BYTES)
    assert cap["explicit_profile_name"] == "llama-local"
    # the unhealthy candidate is passed through for the resolver to place on
    assert any(c.health != "healthy" for c in cap["discovered_candidates"])
    assert out.ok is False  # stubbed capture


def test_cli_blocked_llama_cpp_no_gguf_stays_a_refusal(monkeypatch):
    """No resolved GGUF -> not managed-launch eligible -> the existing
    structured refusal stands (no silent ollama fallback, no spawn)."""
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": "/nonexistent/x.gguf"}
    cap = {}
    args = _wire_blocked(monkeypatch, cfg=cfg, cap=cap)
    out = cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert "resolve_and_materialise_runtime" not in cap  # never called
    assert out.ok is False
    assert "no_usable_endpoint" in out.refusal_reason


def test_cli_blocked_llama_cpp_missing_executable_stays_a_refusal(monkeypatch, gguf):
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": str(gguf)}
    cap = {}
    args = _wire_blocked(monkeypatch, cfg=cfg, cap=cap,
                         exes=(("llama_cpp", "not_installed"),))
    out = cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert not cap
    assert out.ok is False
    assert "no_usable_endpoint" in out.refusal_reason


def test_cli_blocked_ollama_profile_never_becomes_a_managed_launch(monkeypatch, gguf):
    """An unhealthy Ollama profile stays reuse-only refusal -- no llama_cpp
    fallback, even with a configured GGUF and llama-server installed."""
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": str(gguf)}
    cap = {}
    profs = [RuntimeProfile(name=" oll ".strip(), backend="ollama",
                            endpoint="http://127.0.0.1:11434", provenance="configured")]
    args = _wire_blocked(
        monkeypatch, cfg=cfg, cap=cap, profiles=profs, default="oll",
        preflight=_BlockedPreflight(profile_backend="ollama",
                                    endpoint="http://127.0.0.1:11434",
                                    candidate_health="unreachable"),
    )
    out = cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert "selected_backend" not in cap
    assert out.ok is False


def test_cli_blocked_no_topology_is_a_refusal_not_a_crash(monkeypatch, gguf):
    """MUT / advisor guard: a blocked preflight with no topology must be a
    structured refusal, never an AttributeError into the resolver."""
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": str(gguf)}
    cap = {}
    args = _wire_blocked(
        monkeypatch, cfg=cfg, cap=cap,
        preflight=_BlockedPreflight(topology=False),
    )
    out = cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert not cap
    assert out.ok is False
    assert "no_usable_endpoint" in out.refusal_reason


def test_cli_blocked_managed_launch_sets_owned_placement_true_mut3(monkeypatch, gguf):
    """MUT-3 owner: routing the no-external llama_cpp blocked path to
    reuse-only (owned_placement_required=False) instead of managed must be
    caught HERE -- a composition-only test would not."""
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": str(gguf)}
    cap = {}
    args = _wire_blocked(monkeypatch, cfg=cfg, cap=cap)
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert cap["owned_placement_required"] is True
