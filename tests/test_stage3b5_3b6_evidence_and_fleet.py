"""Anvil Stage 3B.5 + 3B.6 -- inline runtime evidence and sequential fleet
validation.

Per the source map (local_only/anvil/stage-3b.5-3b.6-source-map.md), most of
the evidence schema already existed; this stage's real net-new scope is:

* CUDA_VISIBLE_DEVICES read from the real env overlay (never re-derived);
* an ``actual_placement`` block present on every evidence record (ok and
  refused) with explicit not_observed/not_applicable markers, never a
  silent omission;
* a ``ram_spill`` placement-qualified warning;
* confirmation tests proving the already-safe sequential-fleet architecture
  (single materialisation call site, single-model artifact gate) rather than
  new safety machinery.

Pure composition: no CUDA, no llama.cpp binary, no Ollama, no real process,
no real inference.
"""
from __future__ import annotations

import json
import os

from llm_modelbench.identity import resolve_runtime_profile_identity
from llm_modelbench.runtime_identity import RuntimeExecutionSettings
from llm_modelbench.runtime_profiles import RuntimeCandidate, RuntimeProfile
from llm_modelbench.runtime_resolution import (
    ResolvedRuntime,
    RuntimeResolution,
    RuntimeResolutionStatus,
)
from llm_modelbench.runtime_lifecycle import (
    LaunchProcessProof,
    RuntimeLifecycleController,
    RuntimeOwnership,
    materialise_owned_runtime,
)
from llm_modelbench.llama_server_materialisation import (
    ManagedMaterialisationOutcome,
    MaterialisationStatus,
)
from llm_modelbench.runtime_materialisation import (
    MaterialisationSeams,
    materialisation_evidence,
    resolve_and_materialise_runtime,
)
from llm_modelbench.runtime_process_linux import read_process_rss_bytes

U_A = "GPU-00000000-1111-2222-3333-444444444444"
U_B = "GPU-99999999-8888-7777-6666-555555555555"
SHA = "sha256:" + "d" * 64


# ---------------------------------------------------------------------------
# fixtures (mirrors test_stage3b3d_runtime_materialisation.py's vocabulary)
# ---------------------------------------------------------------------------
def _recipe(
    *,
    backend="llama_cpp",
    endpoint="http://127.0.0.1:8081",
    gpu_uuids=(U_A,),
    placement_class="full_gpu",
    requested_context=8192,
    allow_ram_spill=False,
    model_primary_sha256=SHA,
    tensor_split_weights=None,
):
    strategy = "single_device" if len(gpu_uuids) <= 1 else "layer_split"
    settings = RuntimeExecutionSettings(strategy=strategy, context_size=requested_context)
    return ResolvedRuntime(
        backend=backend,
        endpoint=endpoint,
        runtime_profile_name="llama-local",
        execution_settings=settings,
        runtime_profile_identity=resolve_runtime_profile_identity(
            backend=backend, execution_settings=settings
        ),
        selected_physical_gpu_uuids=tuple(gpu_uuids),
        placement_class=placement_class,
        requested_context=requested_context,
        allow_ram_spill=allow_ram_spill,
        estimated_ram_spill_bytes=None,
        model_primary_sha256=model_primary_sha256,
        tensor_split_weights=tensor_split_weights,
    )


def _resolution(status=RuntimeResolutionStatus.RESOLVED, *, recipe=None, health="healthy",
                backend="llama_cpp", endpoint="http://127.0.0.1:8081"):
    recipe = recipe if recipe is not None else _recipe(backend=backend, endpoint=endpoint)
    cand = RuntimeCandidate(
        profile=RuntimeProfile(name="llama-local", backend=backend, endpoint=endpoint, provenance="configured"),
        health=health, source=("saved_profile",), detail=f"{health} fixture",
    )
    return RuntimeResolution(
        status=status, reason=status.value, detail="fixture",
        resolved=recipe if status is RuntimeResolutionStatus.RESOLVED else None,
        selected_candidate=cand,
    )


def _proof(pid=4321):
    return LaunchProcessProof(
        pid=pid, process_start_time_ticks=99, executable_path="/opt/llama-server",
        command_argv=("/opt/llama-server", "--model", "/m.gguf", "--port", "8090"),
        parent_pid=1,
    )


class _FakeController(RuntimeLifecycleController):
    def __init__(self, result, *, cleanup_exc=None):
        self.cleanup_calls = 0

        def _cleanup(owned):
            self.cleanup_calls += 1
            if cleanup_exc is not None:
                raise cleanup_exc

        if result.ownership is RuntimeOwnership.MODELBENCH_OWNED:
            super().__init__(result, cleanup_fn=_cleanup, revalidate_fn=lambda owned: owned.launch_proof)
        else:
            super().__init__(result)


def _spawn_ready(request, *, pid=4321, env_overlay=None):
    result = materialise_owned_runtime(
        request, launch_proof=_proof(pid), launched_at="2026-09-04T00:00:00Z"
    )
    return ManagedMaterialisationOutcome(
        status=MaterialisationStatus.SPAWNED_READY,
        detail="fake spawned ready",
        result=result,
        process=object(),
        endpoint=request.recipe.endpoint.replace("8081", "9099"),
        diagnostic_tail="loaded model; server listening",
        launched_argv=("/opt/llama-server", "--model", "/m.gguf", "--port", "9099", "--ctx-size", "8192"),
        attribution="ours",
        env_overlay=env_overlay,
    )


def _seams(*, spawn=None, external_still_healthy=None, controller_factory=None, cleanup_exc=None):
    def _default_cf(outcome):
        return _FakeController(outcome.result, cleanup_exc=cleanup_exc)

    return MaterialisationSeams(
        spawn_managed=spawn or (lambda r: _spawn_ready(r)),
        external_still_healthy=external_still_healthy,
        controller_factory=controller_factory or _default_cf,
    )


def _ok_managed_outcome(*, gpu_uuids=(U_A,), placement_class="full_gpu",
                         tensor_split_weights=None, env_overlay=None, allow_ram_spill=False):
    recipe = _recipe(gpu_uuids=gpu_uuids, placement_class=placement_class,
                     tensor_split_weights=tensor_split_weights, allow_ram_spill=allow_ram_spill)
    return resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(external_still_healthy=lambda r: False,
                                      spawn=lambda r: _spawn_ready(r, env_overlay=env_overlay)),
        resolve_fn=lambda **kw: _resolution(recipe=recipe, health="healthy"),
    )


def _reused_external_outcome():
    return resolve_and_materialise_runtime(
        selected_backend="ollama", discovered_candidates=[], topology=object(),
        host_meminfo={},
        seams=_seams(external_still_healthy=lambda r: True,
                     spawn=lambda r: (_ for _ in ()).throw(AssertionError("must not spawn"))),
        resolve_fn=lambda **kw: _resolution(backend="ollama", health="healthy"),
    )


# ===========================================================================
# CUDA_VISIBLE_DEVICES -- real overlay, never re-derived, always a reason
# ===========================================================================
def test_cuda_visible_devices_reads_the_real_env_overlay_not_a_guess():
    out = _ok_managed_outcome(gpu_uuids=(U_A, U_B), placement_class="multi_gpu",
                              tensor_split_weights=(1, 1),
                              env_overlay={"CUDA_VISIBLE_DEVICES": f"{U_A},{U_B}"})
    ev = materialisation_evidence(out)
    cvd = ev["materialisation"]["cuda_visible_devices"]
    assert cvd == {"value": f"{U_A},{U_B}", "reason": "observed_at_launch"}


def test_cuda_visible_devices_no_gpu_pinning_has_explicit_reason_not_bare_none():
    out = _ok_managed_outcome(env_overlay={})
    ev = materialisation_evidence(out)
    cvd = ev["materialisation"]["cuda_visible_devices"]
    assert cvd["value"] is None
    assert cvd["reason"] == "managed_launch_set_no_gpu_pinning"


def test_cuda_visible_devices_no_overlay_recorded_has_explicit_reason():
    out = _ok_managed_outcome(env_overlay=None)
    ev = materialisation_evidence(out)
    cvd = ev["materialisation"]["cuda_visible_devices"]
    assert cvd["value"] is None
    assert cvd["reason"] == "no_env_overlay_recorded"


def test_cuda_visible_devices_external_reuse_has_reuse_only_reason():
    out = _reused_external_outcome()
    ev = materialisation_evidence(out)
    cvd = ev["materialisation"]["cuda_visible_devices"]
    assert cvd == {"value": None, "reason": "reuse_only_no_managed_launch"}


# ===========================================================================
# actual_placement -- present on every record, explicit not_observed
# ===========================================================================
def test_actual_placement_present_on_ok_managed_record():
    out = _ok_managed_outcome()
    ev = materialisation_evidence(out, rss_bytes_launch_ready=123456, rss_bytes_post_execution=234567)
    ap = ev["actual_placement"]
    assert ap["process_rss_bytes"] == {
        "launch_ready": 123456, "post_execution": 234567, "reason": "observed",
    }
    assert ap["gpu_memory_observed"] is False
    assert ap["device_placement_observed"] is False


def test_actual_placement_rss_not_observed_when_no_samples_given():
    out = _ok_managed_outcome()
    ev = materialisation_evidence(out)
    ap = ev["actual_placement"]
    assert ap["process_rss_bytes"] == {
        "launch_ready": None, "post_execution": None, "reason": "not_observed",
    }


def test_actual_placement_present_on_external_reuse_record_not_applicable():
    out = _reused_external_outcome()
    ev = materialisation_evidence(out)
    ap = ev["actual_placement"]
    assert ap["process_rss_bytes"]["reason"] == "no_owned_process"
    assert ap["process_rss_bytes"]["launch_ready"] is None


def test_actual_placement_present_on_refused_outcome_mutation_13_guard():
    # MUT-13: dropping this block on the refusal path must be caught.
    out = resolve_and_materialise_runtime(
        selected_backend="llama_cpp", discovered_candidates=[], topology=object(),
        host_meminfo={}, seams=_seams(),
        resolve_fn=lambda **kw: _resolution(RuntimeResolutionStatus.RUNTIME_AMBIGUOUS),
    )
    assert out.ok is False
    ev = materialisation_evidence(out)
    assert "actual_placement" in ev
    assert ev["actual_placement"]["process_rss_bytes"]["reason"] == "no_owned_process"


def test_actual_placement_cross_references_runtime_telemetry_without_recollecting():
    out = _ok_managed_outcome()
    ev = materialisation_evidence(
        out, runtime_telemetry_ref={"artifact": "runtime_telemetry.json", "status": "collected"},
    )
    tel = ev["actual_placement"]["runtime_telemetry"]
    assert tel == {
        "present": True, "artifact": "runtime_telemetry.json",
        "status": "collected", "captured": "pre_benchmark_only",
    }


def test_actual_placement_runtime_telemetry_absent_is_explicit():
    out = _ok_managed_outcome()
    ev = materialisation_evidence(out, runtime_telemetry_ref=None)
    assert ev["actual_placement"]["runtime_telemetry"] == {
        "present": False, "artifact": None, "captured": None,
    }


# ===========================================================================
# ram_spill placement-qualified warning
# ===========================================================================
def test_ram_spill_placement_gets_explicit_not_full_gpu_warning():
    out = _ok_managed_outcome(placement_class="ram_spill", allow_ram_spill=True)
    ev = materialisation_evidence(out)
    assert "placement_ram_spill_not_full_gpu_equivalent" in ev.get("warnings", [])


def test_full_gpu_placement_has_no_ram_spill_warning():
    out = _ok_managed_outcome(placement_class="full_gpu")
    ev = materialisation_evidence(out)
    assert "placement_ram_spill_not_full_gpu_equivalent" not in ev.get("warnings", [])


# ===========================================================================
# persisted-file mutation guard (MUT-1 sibling): read the JSON file, not the
# in-memory dict -- sort_keys=True sorts object keys, not array elements.
# ===========================================================================
def test_persisted_evidence_file_preserves_gpu_uuid_order(tmp_path):
    out = _ok_managed_outcome(gpu_uuids=(U_B, U_A), placement_class="multi_gpu",
                              tensor_split_weights=(3, 5))
    ev = materialisation_evidence(out)
    path = tmp_path / "materialisation_evidence.json"
    path.write_text(json.dumps(ev, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["resolution"]["selected_physical_gpu_uuids"] == [U_B, U_A]
    assert reloaded["resolution"]["tensor_split_weights"] == [3, 5]


# ===========================================================================
# 3B.6 -- managed multi-model policy already enforced at the artifact gate
# (Checkpoint 4 finding: mutations #8/#9 have no code path in
# runtime_materialisation.py itself -- the gate lives in
# local_artifact_resolver.py / cli._resolve_managed_llama_artifact).
# ===========================================================================
def test_managed_artifact_resolution_refuses_none_selection_default_all_installed():
    # cli._resolve_managed_llama_artifact passes None for anything that is
    # not exactly one explicit --models entry (multi-model, default
    # all-installed, --select) -- the single call site that ever reaches
    # resolve_local_gguf_artifact for a managed run.
    from llm_modelbench.local_artifact_resolver import (
        LocalArtifactStatus,
        resolve_local_gguf_artifact,
    )

    result = resolve_local_gguf_artifact(None)
    assert result.status is LocalArtifactStatus.NO_MODEL_REF
    assert result.ok is False


def test_managed_artifact_resolution_allows_exactly_one_explicit_model(tmp_path):
    from llm_modelbench.local_artifact_resolver import resolve_local_gguf_artifact

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\x00" * 64)
    result = resolve_local_gguf_artifact("solo-model", artifacts_map={"solo-model": str(gguf)})
    assert result.ok is True


def test_cli_resolve_managed_llama_artifact_gates_on_exactly_one_explicit_models_entry():
    """The actual multi-model/default/--select -> NO_MODEL_REF gate lives in
    cli._resolve_managed_llama_artifact (it unwraps requested[0] only when
    len(requested) == 1), not in resolve_local_gguf_artifact's own
    signature (which only ever sees a single str or None). This is the real
    mutation #8/#9 target: removing the len(requested) == 1 restriction
    there is what would let multi-model reach managed-spawn eligibility."""
    import inspect

    from llm_modelbench import cli

    src = inspect.getsource(cli._resolve_managed_llama_artifact)
    assert "len(requested) == 1" in src, (
        "the single-explicit-model gate moved or was removed -- "
        "multi-model/default/--select could now reach managed spawn "
        "eligibility; this must route to OWNER DECISION REQUIRED, not pass "
        "silently"
    )


# ===========================================================================
# MUTATION #8/#9 re-targeted onto the real gate (per advisor guidance): a
# test proving the multi-model refusal is REAL is only meaningful if
# removing the len(requested) == 1 restriction in
# cli._resolve_managed_llama_artifact makes a multi-model ref reach
# eligibility. resolve_local_gguf_artifact's own signature only ever accepts
# a single str or None -- it is not itself handed a list -- so the
# structural gate to defend is the caller's len(...) == 1 check, exercised
# above (test_cli_resolve_managed_llama_artifact_gates_on_exactly_one...).
# A non-string / non-None-shaped ref (e.g. a stray list) is defended
# separately here to show resolve_local_gguf_artifact fails closed rather
# than silently coercing an unexpected shape.
# ===========================================================================
def test_artifact_resolver_fails_closed_on_a_non_string_ref_rather_than_coercing():
    from llm_modelbench.local_artifact_resolver import (
        LocalArtifactStatus,
        resolve_local_gguf_artifact,
    )

    for bad in [["a", "b"], ("a", "b"), 123]:
        result = resolve_local_gguf_artifact(bad)  # type: ignore[arg-type]
        assert result.status is LocalArtifactStatus.NO_MODEL_REF, bad


# ===========================================================================
# 3B.6 -- sequential fleet: single materialisation call site, one client for
# every selected model (per-run granularity, honestly proven, not per-model).
# ===========================================================================
def test_cmd_run_materialises_runtime_exactly_once_before_model_selection():
    """cli.cmd_run calls _resolve_and_materialise_for_run exactly once, and
    that call happens before selected_models is resolved -- confirmed by
    reading cli.py's control flow (Checkpoint 4). This test pins the
    source-level invariant: the function exists, is called from cmd_run
    exactly once in source, and _resolve_model_selection is a distinct,
    later call."""
    import inspect

    from llm_modelbench import cli

    src = inspect.getsource(cli.cmd_run)
    assert src.count("_resolve_and_materialise_for_run(") == 1
    mat_pos = src.index("_resolve_and_materialise_for_run(")
    sel_pos = src.index("_resolve_model_selection(")
    assert mat_pos < sel_pos, (
        "materialisation must happen before model selection -- if this ever "
        "reorders, the per-run granularity claim in the evidence design "
        "(source-map Checkpoint 4) becomes false"
    )


def test_raw_result_rows_carry_no_per_row_materialisation_identity():
    """Confirms the structural reason 'identity leak between rows' cannot
    happen today: runner.py's per-row dict construction carries no
    endpoint/pid/backend field -- only model/task/scores/timings. If a
    future change ever stamps materialisation identity onto rows, this test
    must be revisited alongside a real per-row leakage test."""
    import inspect

    from llm_modelbench import runner

    src = inspect.getsource(runner.run)
    row_start = src.index('row = {\n')
    row_block = src[row_start:row_start + 2000]
    for forbidden in ("endpoint", "materialisation", "pid\"", "\"backend\""):
        assert forbidden not in row_block, (
            f"{forbidden!r} found in the raw row construction -- if a row "
            "now carries per-row runtime identity, the 'no leak between "
            "rows because there is nothing per-row to leak' proof is stale "
            "and needs a real cross-model leakage test, not this guard."
        )


def test_managed_spawn_env_overlay_field_exists_on_outcome_type():
    """Structural guard: ManagedMaterialisationOutcome must carry the real
    env_overlay (Anvil Stage 3B.5) so cuda_visible_devices evidence can read
    it instead of re-deriving from the GPU UUID order."""
    from dataclasses import fields

    names = {f.name for f in fields(ManagedMaterialisationOutcome)}
    assert "env_overlay" in names


def test_read_process_rss_bytes_parses_a_real_live_process():
    """The only real (non-fake-pid) exercise of the /proc/<pid>/status VmRSS
    parser: the test process itself is always running and always has a
    positive resident set, so this is the sole guard that actually drives
    the ``VmRSS:`` prefix match, the ``kB`` unit check and the *1024
    conversion through a genuine kernel-produced line -- every other test in
    this module passes a synthetic pid (4321) that resolves to None before
    any of that parsing logic runs."""
    rss = read_process_rss_bytes(os.getpid())
    assert isinstance(rss, int)
    assert rss > 0


def test_read_process_rss_bytes_returns_none_for_a_pid_that_does_not_exist():
    # PID 1 is reserved/init and unreadable to us in virtually every
    # sandbox; a made-up very large pid is reliably absent from /proc.
    assert read_process_rss_bytes(2**30) is None
