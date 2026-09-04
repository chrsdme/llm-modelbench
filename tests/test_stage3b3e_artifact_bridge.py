"""Anvil Stage 3B.3E -- artifact resolver bridge into the managed
llama-server path + the generalised owned-cleanup-failure warning.

No real GPU / Ollama / llama.cpp / inference. The composition
(`resolve_and_materialise_runtime`) is driven with fake seams; the cli
wiring (`_resolve_and_materialise_for_run`) is driven with a fake preflight.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from llm_modelbench import cli
from llm_modelbench import runtime_materialisation as rm
from llm_modelbench.config import Config
from llm_modelbench.llama_server_materialisation import (
    ManagedMaterialisationOutcome,
    MaterialisationStatus,
    materialise as real_materialise,
    spawn_managed_llama_server,
)
from llm_modelbench.runtime_lifecycle import (
    LaunchProcessProof,
    MaterialisationRequest,
    RuntimeLifecycleController,
    RuntimeOwnership,
    materialise_owned_runtime,
)
from llm_modelbench.runtime_identity import RuntimeExecutionSettings
from llm_modelbench.identity import resolve_runtime_profile_identity
from llm_modelbench.runtime_profiles import RuntimeCandidate, RuntimeProfile
from llm_modelbench.runtime_resolution import (
    ResolvedRuntime,
    RuntimeResolution,
    RuntimeResolutionStatus,
)

U_A = "GPU-00000000-1111-2222-3333-444444444444"
_BYTES = b"GGUF\x00 fake weights for the 3B.3E bridge test"
_SHA = hashlib.sha256(_BYTES).hexdigest()


@pytest.fixture()
def gguf(tmp_path):
    p = tmp_path / "bench-model.gguf"
    p.write_bytes(_BYTES)
    return p


# ---------------------------------------------------------------------------
# recipe / resolution fixtures
# ---------------------------------------------------------------------------
def _recipe(*, backend="llama_cpp", endpoint="http://127.0.0.1:8081", sha=_SHA):
    settings = RuntimeExecutionSettings(strategy="single_device", context_size=8192)
    return ResolvedRuntime(
        backend=backend, endpoint=endpoint, runtime_profile_name="llama-local",
        execution_settings=settings,
        runtime_profile_identity=resolve_runtime_profile_identity(
            backend=backend, execution_settings=settings),
        selected_physical_gpu_uuids=(U_A,), placement_class="full_gpu",
        requested_context=8192, allow_ram_spill=False, estimated_ram_spill_bytes=None,
        model_primary_sha256=sha,
    )


def _resolution(*, backend="llama_cpp", endpoint="http://127.0.0.1:8081", health="healthy", sha=_SHA):
    cand = RuntimeCandidate(
        profile=RuntimeProfile(name="llama-local", backend=backend, endpoint=endpoint, provenance="configured"),
        health=health, source=("saved_profile",), detail="fx",
    )
    return RuntimeResolution(
        status=RuntimeResolutionStatus.RESOLVED, reason="resolved", detail="fx",
        resolved=_recipe(backend=backend, endpoint=endpoint, sha=sha), selected_candidate=cand,
    )


def _proof(pid=7777):
    return LaunchProcessProof(
        pid=pid, process_start_time_ticks=5, executable_path="/opt/llama-server",
        command_argv=("/opt/llama-server", "--model", "/m.gguf"), parent_pid=1,
    )


class _CountingController(RuntimeLifecycleController):
    def __init__(self, result, *, cleanup_exc=None):
        self.cleanup_calls = 0
        self._exc = cleanup_exc

        def _rev(owned):
            return owned.launch_proof

        def _cln(owned):
            self.cleanup_calls += 1
            if self._exc:
                raise self._exc

        if result.ownership is RuntimeOwnership.MODELBENCH_OWNED:
            super().__init__(result, cleanup_fn=_cln, revalidate_fn=_rev)
        else:
            super().__init__(result)


# ===========================================================================
# 1. the managed path no longer fails with RESOLVED_RECIPE_INCOMPLETE when a
#    verified artifact is supplied -- and the spawn adapter receives it.
# ===========================================================================
def test_managed_spawn_receives_resolver_path_and_verified_sha(gguf):
    captured = {}

    def _fake_spawn(request):
        # this is the production `_spawn` closure's target -- assert it got the
        # resolver's path + verified sha, and that the recipe sha matches.
        captured["recipe_sha"] = request.recipe.model_primary_sha256
        result = materialise_owned_runtime(
            request, launch_proof=_proof(), launched_at="2026-09-04T00:00:00Z")
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.SPAWNED_READY, detail="ready", result=result,
            process=object(), endpoint="http://127.0.0.1:9099", diagnostic_tail="ok",
            launched_argv=("/opt/llama-server", "--model", str(gguf)), attribution="ours",
        )

    seams = rm.MaterialisationSeams(
        spawn_managed=_fake_spawn,
        external_still_healthy=lambda r: False,  # demote reuse -> managed spawn
        controller_factory=lambda o: _CountingController(o.result),
    )
    snap = {"status": "resolved", "resolved_path": str(gguf), "verified_sha256": _SHA,
            "blocked_managed_spawn": False}
    out = rm.resolve_and_materialise_runtime(
        selected_backend="llama_cpp",
        discovered_candidates=[],
        topology=object(),
        host_meminfo={},
        seams=seams,
        model_primary_sha256=_SHA,
        artifact_resolution=snap,
        resolve_fn=lambda **k: _resolution(),
    )
    assert out.ok
    assert out.materialisation_status == "spawned_ready"
    assert captured["recipe_sha"] == _SHA
    assert out.artifact_resolution["resolved_path"] == str(gguf)


def test_real_spawn_adapter_accepts_verified_artifact_no_recipe_incomplete(gguf):
    """End-to-end through the *real* spawn_managed_llama_server: a verified
    path + sha passes the model-authority gates (no RESOLVED_RECIPE_INCOMPLETE,
    no ARTIFACT_IDENTITY_MISMATCH, no MODEL_CONTENT_MISMATCH) and progresses
    to a later stage keyed off the injected process seam."""
    req = MaterialisationRequest.from_resolution(_resolution())

    def _probe(exe):
        from llm_modelbench.llama_server_materialisation import REQUIRED_LLAMA_SERVER_CLI_OPTIONS
        return frozenset(REQUIRED_LLAMA_SERVER_CLI_OPTIONS)

    out = spawn_managed_llama_server(
        req,
        executable_path="/opt/llama-server",
        model_path=str(gguf),
        model_primary_sha256=_SHA,
        hardware_inventory=[],  # empty -> fails LATER at GPU identity translation
        observe_identity=lambda pid: _proof(pid),
        readiness_probe=lambda *a, **k: True,
        port_attribution=lambda *a, **k: True,
        context_conformance=lambda *a, **k: True,
        now_iso=lambda: "2026-09-04T00:00:00Z",
        popen=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("popen")),
        port_bindable=lambda host, port: True,
        cli_contract_probe=_probe,
    )
    # a verified artifact clears every model/artifact-authority gate; the
    # failure is a *downstream* one (GPU identity), never one of these:
    _ARTIFACT_GATES = {
        MaterialisationStatus.RESOLVED_RECIPE_INCOMPLETE,
        MaterialisationStatus.ARTIFACT_IDENTITY_MISMATCH,
        MaterialisationStatus.MODEL_CONTENT_MISMATCH,
    }
    assert out.status not in _ARTIFACT_GATES
    assert out.status is MaterialisationStatus.GPU_IDENTITY_UNTRANSLATABLE


def test_real_spawn_still_recipe_incomplete_without_artifact():
    """The 3B.3D behaviour is preserved when NO artifact is supplied."""
    req = MaterialisationRequest.from_resolution(
        _resolution(sha=None)  # recipe carries no sha
    )
    out = spawn_managed_llama_server(
        req,
        executable_path="/opt/llama-server",
        model_path=None,
        model_primary_sha256=None,
        hardware_inventory=[],
        observe_identity=lambda pid: _proof(pid),
        readiness_probe=lambda *a, **k: True,
        port_attribution=lambda *a, **k: True,
        context_conformance=lambda *a, **k: True,
        now_iso=lambda: "2026-09-04T00:00:00Z",
    )
    assert out.status is MaterialisationStatus.RESOLVED_RECIPE_INCOMPLETE


def test_caller_claimed_hash_cannot_override_bytes_at_the_spawn_boundary(tmp_path):
    """path_B + claimed SHA_A is rejected -- bdb7850's content check, now
    reached because 3B.3E supplies a path. The resolver hashes bytes; if a
    wrong sha somehow reached the recipe, the content hasher catches it."""
    real = tmp_path / "real.gguf"
    real.write_bytes(b"the real bytes")
    wrong_sha = "f" * 64
    req = MaterialisationRequest.from_resolution(_resolution(sha=wrong_sha))
    out = spawn_managed_llama_server(
        req,
        executable_path="/opt/llama-server",
        model_path=str(real),
        model_primary_sha256=wrong_sha,   # matches the recipe claim...
        hardware_inventory=[],
        observe_identity=lambda pid: _proof(pid),
        readiness_probe=lambda *a, **k: True,
        port_attribution=lambda *a, **k: True,
        context_conformance=lambda *a, **k: True,
        now_iso=lambda: "2026-09-04T00:00:00Z",
    )
    # ...but the actual bytes hash to something else -> fail closed
    assert out.status is MaterialisationStatus.MODEL_CONTENT_MISMATCH


# ===========================================================================
# 2. external / Ollama reuse never requires a local artifact
# ===========================================================================
def _reuse_seams():
    def _cf(o):
        return _CountingController(o.result)
    return rm.MaterialisationSeams(
        spawn_managed=lambda r: (_ for _ in ()).throw(AssertionError("no spawn on reuse")),
        controller_factory=_cf,
    )


def test_external_llama_server_reuse_needs_no_artifact():
    out = rm.resolve_and_materialise_runtime(
        selected_backend="llama_cpp",
        discovered_candidates=[],
        topology=object(),
        host_meminfo={},
        seams=_reuse_seams(),
        model_primary_sha256=None,
        artifact_resolution={"status": "not_configured", "blocked_managed_spawn": False},
        resolve_fn=lambda **k: _resolution(health="healthy"),
        materialise_fn=real_materialise,
    )
    assert out.ok and out.materialisation_status == "reused_external"


def test_ollama_reuse_needs_no_artifact():
    out = rm.resolve_and_materialise_runtime(
        selected_backend="ollama",
        discovered_candidates=[],
        topology=object(),
        host_meminfo={},
        seams=_reuse_seams(),
        model_primary_sha256=None,
        artifact_resolution={"status": "no_model_ref", "blocked_managed_spawn": False},
        resolve_fn=lambda **k: _resolution(backend="ollama", endpoint="http://127.0.0.1:11434"),
        materialise_fn=real_materialise,
    )
    assert out.ok and out.materialisation_status == "reused_external"


# ===========================================================================
# 3. cli wiring: resolver only feeds the managed llama_cpp path
# ===========================================================================
class _FakePreflight:
    def __init__(self, backend, endpoint):
        self.blocked = False
        self.blocker = None
        self.selected_candidate = RuntimeCandidate(
            profile=RuntimeProfile(name="p", backend=backend, endpoint=endpoint, provenance="configured"),
            health="healthy", source=("saved_profile",), detail="fx",
        )
        self.candidates = [self.selected_candidate]
        self.gpu_inventory = []
        self.topology = object()


def _wire_cli(monkeypatch, *, backend, models, cfg, capture):
    import llm_modelbench.runtime_profiles as rp
    import llm_modelbench.hardware as hw
    monkeypatch.setattr(cli, "load_profiles", lambda store: ([], None))
    monkeypatch.setattr(
        cli, "resolve_operational_preflight",
        lambda *a, **k: _FakePreflight(backend, "http://127.0.0.1:8081"),
    )
    monkeypatch.setattr(rp, "discover_backend_executables", lambda **k: None)
    monkeypatch.setattr(cli, "_llama_server_executable_path", lambda be: "/opt/llama-server")
    monkeypatch.setattr(hw, "host_memory_snapshot", lambda: {})

    def _capture_ram(**kwargs):
        capture.update(kwargs)
        return rm.RuntimeMaterialisationOutcome(
            ok=False, backend=kwargs.get("selected_backend") or "", resolution_status="fx",
            refusal_reason="stub", artifact_resolution=kwargs.get("artifact_resolution"),
        )

    monkeypatch.setattr(rm, "production_seams", lambda **k: object())
    monkeypatch.setattr(rm, "resolve_and_materialise_runtime", _capture_ram)

    class _Args:
        pass
    args = _Args()
    args.models = models
    args.unattended = False
    args.runtime_profile = None
    args.runtime_profiles_file = None
    return args


def test_cli_resolves_artifact_for_single_model_llama_cpp(monkeypatch, gguf):
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": str(gguf)}
    cap = {}
    args = _wire_cli(monkeypatch, backend="llama_cpp", models="bench-model", cfg=cfg, capture=cap)
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert cap["model_primary_sha256"] == _SHA
    assert cap["artifact_resolution"]["resolved_path"] == str(gguf)
    assert cap["artifact_resolution"]["blocked_managed_spawn"] is False


def test_cli_does_not_resolve_artifact_for_multi_model(monkeypatch, gguf):
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": str(gguf)}
    cap = {}
    args = _wire_cli(monkeypatch, backend="llama_cpp", models="bench-model;other", cfg=cfg, capture=cap)
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert cap["model_primary_sha256"] is None
    assert cap["artifact_resolution"]["status"] == "no_model_ref"
    assert cap["artifact_resolution"]["blocked_managed_spawn"] is True


def test_cli_does_not_resolve_artifact_for_ollama(monkeypatch, gguf):
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": str(gguf)}
    cap = {}
    args = _wire_cli(monkeypatch, backend="ollama", models="bench-model", cfg=cfg, capture=cap)
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert cap["model_primary_sha256"] is None
    assert cap["artifact_resolution"]["status"] == "no_model_ref"
    assert cap["artifact_resolution"]["blocked_managed_spawn"] is False


def test_cli_never_passes_a_path_or_sha_when_artifact_unresolved(monkeypatch):
    """MUT-4 guard: an unresolved artifact must reach production_seams as
    model_path=None / model_primary_sha256=None so the spawn adapter fails
    closed at RESOLVED_RECIPE_INCOMPLETE. A mutation that drops the
    ``if artifact.ok`` guard would leak a stale/None-but-truthy path here."""
    # a MISSING artifact: resolved_path is set (the bad path) but ok is False,
    # so a dropped ``if artifact.ok`` guard WOULD leak that path onward.
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": "/nonexistent/bench-model.gguf"}
    seams_cap = {}
    args = _wire_cli(monkeypatch, backend="llama_cpp", models="bench-model", cfg=cfg, capture={})
    monkeypatch.setattr(rm, "production_seams",
                        lambda **k: seams_cap.update(k) or object())
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert seams_cap["model_path"] is None
    assert seams_cap["model_primary_sha256"] is None


def test_cli_refusal_names_the_artifact_cause_when_managed_spawn_blocked(monkeypatch):
    cfg = Config()  # nothing configured
    cap = {}
    args = _wire_cli(monkeypatch, backend="llama_cpp", models="bench-model", cfg=cfg, capture=cap)
    monkeypatch.setattr(
        rm, "resolve_and_materialise_runtime",
        lambda **k: rm.RuntimeMaterialisationOutcome(
            ok=False, backend="llama_cpp", resolution_status="resolved",
            materialisation_status="resolved_recipe_incomplete",
            refusal_reason="runtime_not_materialised: resolved_recipe_incomplete: no sha",
            artifact_resolution=k.get("artifact_resolution"),
        ),
    )
    out = cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert "local GGUF artifact unresolved: not_configured" in out.refusal_reason


def test_cli_artifact_failure_does_not_short_circuit_reuse(monkeypatch):
    """No early SystemExit: an unresolved artifact still lets the composition
    run (and reuse a healthy external server)."""
    cfg = Config()  # nothing configured
    cap = {}
    args = _wire_cli(monkeypatch, backend="llama_cpp", models="bench-model", cfg=cfg, capture=cap)
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    # the composition WAS called (no SystemExit before it)
    assert "selected_backend" in cap
    assert cap["model_primary_sha256"] is None
    assert cap["artifact_resolution"]["status"] == "not_configured"
    assert cap["artifact_resolution"]["blocked_managed_spawn"] is True


# ===========================================================================
# 4. evidence
# ===========================================================================
def _owned_outcome(*, cleanup_exc=None, artifact_resolution=None):
    req = MaterialisationRequest.from_resolution(_resolution())
    result = materialise_owned_runtime(req, launch_proof=_proof(), launched_at="2026-09-04T00:00:00Z")
    mat = ManagedMaterialisationOutcome(
        status=MaterialisationStatus.SPAWNED_READY, detail="ready", result=result,
        process=object(), endpoint="http://127.0.0.1:9099", diagnostic_tail="ok",
        launched_argv=("/opt/llama-server", "--model", "/m.gguf", "--port", "9099"),
        attribution="ours",
    )
    ctrl = _CountingController(result, cleanup_exc=cleanup_exc)
    return rm.RuntimeMaterialisationOutcome(
        ok=True, backend="llama_cpp", resolution_status="resolved",
        resolution=_resolution(), materialisation_status="spawned_ready",
        materialisation=mat, endpoint="http://127.0.0.1:9099", controller=ctrl,
        identity_key=req.identity_key(), artifact_resolution=artifact_resolution,
    )


def test_evidence_records_artifact_resolution_on_ok(gguf):
    snap = {"status": "resolved", "resolved_path": str(gguf), "verified_sha256": _SHA,
            "blocked_managed_spawn": False}
    rec = rm.materialisation_evidence(_owned_outcome(artifact_resolution=snap), benchmark_completed=True)
    assert rec["artifact_resolution"]["verified_sha256"] == _SHA
    json.dumps(rec)


def test_evidence_records_artifact_resolution_on_refusal():
    snap = {"status": "not_configured", "resolved_path": None, "verified_sha256": None,
            "blocked_managed_spawn": True}
    refused = rm.RuntimeMaterialisationOutcome(
        ok=False, backend="llama_cpp", resolution_status="resolved",
        resolution=_resolution(), materialisation_status="resolved_recipe_incomplete",
        refusal_reason="runtime_not_materialised: resolved_recipe_incomplete: no sha",
        artifact_resolution=snap,
    )
    rec = rm.materialisation_evidence(refused, benchmark_completed=False, failure_stage=None)
    assert rec["artifact_resolution"]["blocked_managed_spawn"] is True
    assert rec["ok"] is False


# ===========================================================================
# 5. generalised owned-cleanup-failure warning (item 7 / accepted debt)
# ===========================================================================
def _bad_cleanup(detail="SIGTERM ignored"):
    from llm_modelbench.runtime_lifecycle import CleanupOutcome, CleanupResult
    return CleanupResult(outcome=CleanupOutcome.GRACEFUL_FAILED, detail=detail,
                         destructive_action_performed=True)


def _ok_cleanup():
    from llm_modelbench.runtime_lifecycle import CleanupOutcome, CleanupResult
    return CleanupResult(outcome=CleanupOutcome.SUCCEEDED, detail="stopped",
                         destructive_action_performed=True)


def test_cleanup_warning_generalises_to_pre_run_gates():
    rec = rm.materialisation_evidence(
        _owned_outcome(), cleanup_result=_bad_cleanup(),
        benchmark_completed=False, failure_stage="pre_run_gates",
    )
    assert "cleanup_failed_before_benchmark" in rec["warnings"]
    assert "cleanup_failed_after_client_construction_failure" not in rec["warnings"]


def test_client_construction_key_unchanged():
    rec = rm.materialisation_evidence(
        _owned_outcome(), cleanup_result=_bad_cleanup(),
        benchmark_completed=False, failure_stage="client_construction",
    )
    assert rec["warnings"] == ["cleanup_failed_after_client_construction_failure"]


def test_no_cleanup_warning_when_cleanup_ok():
    rec = rm.materialisation_evidence(
        _owned_outcome(), cleanup_result=_ok_cleanup(),
        benchmark_completed=False, failure_stage="pre_run_gates",
    )
    assert "warnings" not in rec


def test_no_generalised_warning_when_benchmark_completed():
    rec = rm.materialisation_evidence(
        _owned_outcome(), cleanup_result=_bad_cleanup(),
        benchmark_completed=True, failure_stage="pre_run_gates",
    )
    # a completed benchmark uses the pre-existing key, not the new one
    assert rec["warnings"] == ["cleanup_failed_on_successful_benchmark"]


def test_persist_prints_one_message_for_pre_run_gate_cleanup_failure(capsys, tmp_path):
    out = _owned_outcome()
    object.__setattr__(out.controller, "_last_cleanup", _bad_cleanup("ignored SIGTERM"))
    cli._persist_materialisation_evidence(
        tmp_path, out, benchmark_completed=False, failure_stage="pre_run_gates",
    )
    printed = capsys.readouterr().out
    assert "refused before the benchmark started" in printed
    assert printed.count("check and stop it") == 1


# ===========================================================================
# 6. config validation
# ===========================================================================
def test_config_rejects_relative_gguf_artifact_path(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"gguf_artifacts": {"m": "relative/path.gguf"}}))
    with pytest.raises(SystemExit, match="absolute path"):
        Config.load(str(p))


def test_config_rejects_relative_gguf_root(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"gguf_root": "models"}))
    with pytest.raises(SystemExit, match="absolute path"):
        Config.load(str(p))


def test_config_accepts_valid_gguf_config(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"gguf_artifacts": {"m": "/opt/models/m.gguf"}, "gguf_root": "/opt/models"}))
    cfg = Config.load(str(p))
    assert cfg.gguf_artifacts == {"m": "/opt/models/m.gguf"}
    assert cfg.gguf_root == "/opt/models"
    assert "gguf_artifacts" in cfg.to_dict()
