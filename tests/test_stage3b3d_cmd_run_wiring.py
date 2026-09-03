"""Anvil Stage 3B.3D -- cmd_run wiring of the resolve->materialise->lifecycle
composition into the real benchmark execution path.

No real GPU / no real Ollama / no real llama.cpp / no real inference. Each
test drives ``cmd_run`` only as far as the specific integration point it
checks, then stops it (a stubbed next step raises a sentinel). The full
composition itself is covered in ``test_stage3b3d_runtime_materialisation.py``.
"""
from __future__ import annotations

import json

import pytest

from llm_modelbench import cli
from llm_modelbench.config import Config
from llm_modelbench.runtime_lifecycle import (
    CleanupOutcome,
    LaunchProcessProof,
    MaterialisationRequest,
    RuntimeLifecycleController,
    RuntimeOwnership,
    materialise_owned_runtime,
    reuse_external_runtime,
)
from llm_modelbench.runtime_profiles import RuntimeCandidate, RuntimeProfile
from llm_modelbench.runtime_resolution import (
    ResolvedRuntime,
    RuntimeResolution,
    RuntimeResolutionStatus,
)
from llm_modelbench.runtime_identity import RuntimeExecutionSettings
from llm_modelbench.identity import resolve_runtime_profile_identity
from llm_modelbench.llama_server_materialisation import (
    ManagedMaterialisationOutcome,
    MaterialisationStatus,
)
from llm_modelbench import runtime_materialisation as rm

U_A = "GPU-00000000-1111-2222-3333-444444444444"
SHA = "sha256:" + "e" * 64
_STOP = RuntimeError("stop after wiring point")


def _recipe(endpoint="http://127.0.0.1:8081", backend="llama_cpp"):
    settings = RuntimeExecutionSettings(strategy="single_device", context_size=8192)
    return ResolvedRuntime(
        backend=backend, endpoint=endpoint, runtime_profile_name="llama-local",
        execution_settings=settings,
        runtime_profile_identity=resolve_runtime_profile_identity(
            backend=backend, execution_settings=settings),
        selected_physical_gpu_uuids=(U_A,), placement_class="full_gpu",
        requested_context=8192, allow_ram_spill=False, estimated_ram_spill_bytes=None,
        model_primary_sha256=SHA,
    )


def _resolution(endpoint="http://127.0.0.1:8081", backend="llama_cpp"):
    cand = RuntimeCandidate(
        profile=RuntimeProfile(name="llama-local", backend=backend, endpoint=endpoint, provenance="configured"),
        health="healthy", source=("saved_profile",), detail="fx",
    )
    return RuntimeResolution(
        status=RuntimeResolutionStatus.RESOLVED, reason="resolved", detail="fx",
        resolved=_recipe(endpoint, backend), selected_candidate=cand,
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


def _owned_outcome(endpoint="http://127.0.0.1:9099", *, cleanup_exc=None):
    req = MaterialisationRequest.from_resolution(_resolution())
    result = materialise_owned_runtime(req, launch_proof=_proof(), launched_at="2026-09-04T00:00:00Z")
    mat = ManagedMaterialisationOutcome(
        status=MaterialisationStatus.SPAWNED_READY, detail="ready", result=result,
        process=object(), endpoint=endpoint, diagnostic_tail="ok",
        launched_argv=("/opt/llama-server", "--model", "/m.gguf", "--port", "9099"),
        attribution="ours",
    )
    ctrl = _CountingController(result, cleanup_exc=cleanup_exc)
    return rm.RuntimeMaterialisationOutcome(
        ok=True, backend="llama_cpp", resolution_status="resolved",
        resolution=_resolution(), materialisation_status="spawned_ready",
        materialisation=mat, endpoint=endpoint, controller=ctrl,
        identity_key=req.identity_key(),
    )


def _reuse_outcome(endpoint="http://127.0.0.1:8081"):
    req = MaterialisationRequest.from_resolution(_resolution(endpoint=endpoint))
    mat = ManagedMaterialisationOutcome(
        status=MaterialisationStatus.REUSED_EXTERNAL, detail="reuse",
        result=reuse_external_runtime(req), endpoint=endpoint,
    )
    ctrl = _CountingController(mat.result)
    return rm.RuntimeMaterialisationOutcome(
        ok=True, backend="llama_cpp", resolution_status="resolved",
        resolution=_resolution(endpoint=endpoint), materialisation_status="reused_external",
        materialisation=mat, endpoint=endpoint, controller=ctrl,
        identity_key=req.identity_key(),
    )


def _refused_outcome(reason="runtime_not_resolved: runtime_ambiguous: two endpoints"):
    return rm.RuntimeMaterialisationOutcome(
        ok=False, backend="llama_cpp", resolution_status="runtime_ambiguous",
        resolution=None, refusal_reason=reason,
    )


def _run_args(tmp_path, *extra):
    return cli.build_parser().parse_args(
        ["run", "--tasks", "py_anagram", "--out", str(tmp_path),
         "--run-id", "r", "--yes", "--no-resume", *extra]
    )


class _StubClient:
    def __init__(self, endpoint):
        self.endpoint = endpoint


def _stop_after_client(monkeypatch, *, capture=None):
    """Make cmd_run raise _STOP right after the client is built (the next step
    is _run_dir / _resolve_model_selection). Records the client if asked."""
    orig = cli._run_dir

    def _boom(args):
        if capture is not None:
            capture["reached"] = True
        raise _STOP

    monkeypatch.setattr(cli, "_run_dir", _boom)
    return orig


# ===========================================================================
def test_mock_run_never_reaches_resolve_or_materialise(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli, "_resolve_and_materialise_for_run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no materialisation in mock mode")),
        raising=False,
    )
    monkeypatch.setattr(rm, "resolve_runtime",
                        lambda **k: (_ for _ in ()).throw(AssertionError("mock must not resolve")))
    monkeypatch.setattr(rm, "materialise",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("mock must not materialise")))
    _stop_after_client(monkeypatch)
    args = _run_args(tmp_path, "--mock")
    with pytest.raises(RuntimeError, match="stop after wiring point"):
        cli.cmd_run(args, Config())  # reached the stop -> mock client built, no resolve/materialise


def test_non_mock_run_calls_the_materialisation_path(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(cli, "_resolve_and_materialise_for_run",
                        lambda args, cfg, **k: seen.setdefault("called", True) and None or _reuse_outcome(),
                        raising=False)
    monkeypatch.setattr(cli, "_client_for_materialised_endpoint",
                        lambda endpoint, cfg, *, backend: _StubClient(endpoint), raising=False)
    _stop_after_client(monkeypatch)
    args = _run_args(tmp_path)
    with pytest.raises(RuntimeError, match="stop after wiring point"):
        cli.cmd_run(args, Config())
    assert seen.get("called") is True


def test_client_is_built_against_the_materialised_endpoint_not_a_stale_one(monkeypatch, tmp_path):
    built = {}
    monkeypatch.setattr(cli, "_resolve_and_materialise_for_run",
                        lambda args, cfg, **k: _owned_outcome(endpoint="http://127.0.0.1:9099"),
                        raising=False)

    def _adapter(endpoint, cfg, *, backend):
        built["endpoint"] = endpoint
        built["backend"] = backend
        return _StubClient(endpoint)

    monkeypatch.setattr(cli, "_client_for_materialised_endpoint", _adapter, raising=False)
    _stop_after_client(monkeypatch)
    args = _run_args(tmp_path)
    cfg = Config()
    cfg.ollama_url = "http://127.0.0.1:8081"  # the resolved (pre-managed) endpoint -- must NOT win
    with pytest.raises(RuntimeError, match="stop after wiring point"):
        cli.cmd_run(args, cfg)
    assert built["endpoint"] == "http://127.0.0.1:9099"
    assert built["backend"] == "llama_cpp"


def test_client_for_materialised_endpoint_honours_the_resolver_backend():
    cfg = Config()
    llama = cli._client_for_materialised_endpoint("http://127.0.0.1:9099", cfg, backend="llama_cpp")
    assert type(llama).__name__ == "LlamaCppBackendAdapter"
    ollama = cli._client_for_materialised_endpoint("http://127.0.0.1:11434", cfg, backend="ollama")
    assert type(ollama).__name__ == "OllamaBackendAdapter"
    assert cfg.ollama_url == "http://127.0.0.1:11434"  # ollama endpoint threaded onto cfg


@pytest.mark.parametrize("reason,token", [
    ("runtime_not_resolved: runtime_ambiguous: two endpoints", "runtime_not_resolved"),
    ("runtime_not_materialised: resolved_recipe_incomplete_for_materialisation: no artifact", "runtime_not_materialised"),
    ("runtime_not_resolved: environment_infeasible: no vram", "environment_infeasible"),
])
def test_non_ok_outcome_is_a_structured_systemexit_with_no_rows(monkeypatch, tmp_path, reason, token):
    monkeypatch.setattr(cli, "_resolve_and_materialise_for_run",
                        lambda args, cfg, **k: _refused_outcome(reason), raising=False)
    args = _run_args(tmp_path)
    with pytest.raises(SystemExit) as ei:
        cli.cmd_run(args, Config())
    assert "run refused" in str(ei.value) and token in str(ei.value)
    assert not (tmp_path / "r" / "raw_results.jsonl").exists()
    assert not (tmp_path / "r" / "materialisation_evidence.json").exists()  # nothing materialised


# --- lifecycle scope + evidence: drive cmd_run through a fake runner.run ----
def _wire_full_run(monkeypatch, outcome, *, run_impl):
    monkeypatch.setattr(cli, "_resolve_and_materialise_for_run",
                        lambda args, cfg, **k: outcome, raising=False)
    monkeypatch.setattr(cli, "_client_for_materialised_endpoint",
                        lambda endpoint, cfg, *, backend: _StubClient(endpoint), raising=False)
    monkeypatch.setattr(cli, "_resolve_model_selection", lambda args, client: ["m:latest"])
    monkeypatch.setattr(cli, "collect_runtime_identity",
                        lambda **k: object(), raising=False)
    monkeypatch.setattr(cli, "_plan_for_args", lambda *a, **k: {"capability_profiles": None})
    monkeypatch.setattr(cli, "_confirm_plan", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_require_host_code_opt_in", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_write_run_ranking_scope", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_ranking_dir_for", lambda *a, **k: None)

    class _C:
        endpoint = "x"

        def tags(self):
            return [{"name": "m:latest"}]

    monkeypatch.setattr(cli, "_client_for_materialised_endpoint",
                        lambda endpoint, cfg, *, backend: _C(), raising=False)

    import llm_modelbench.runner as runner_mod
    monkeypatch.setattr(runner_mod, "run", run_impl)
    monkeypatch.setattr(runner_mod, "assess_run_validity",
                        lambda d: {"status": "valid", "harness_error_rows": 0})
    monkeypatch.setattr(cli, "report", __import__("types").SimpleNamespace(build=lambda *a, **k: None), raising=False)
    import llm_modelbench.report as report_mod
    monkeypatch.setattr(report_mod, "build", lambda *a, **k: None)


def test_owned_runtime_cleaned_on_success_and_evidence_persisted(monkeypatch, tmp_path):
    outcome = _owned_outcome()

    def _run(client, cfg, **kw):
        kw["out_dir"].mkdir(parents=True, exist_ok=True)
        (kw["out_dir"] / "raw_results.jsonl").write_text("", encoding="utf-8")
        return kw["out_dir"]

    _wire_full_run(monkeypatch, outcome, run_impl=_run)
    cli.cmd_run(_run_args(tmp_path), Config())
    assert outcome.controller.cleanup_calls == 1
    ev = json.loads((tmp_path / "r" / "materialisation_evidence.json").read_text())
    assert ev["ok"] is True
    assert ev["materialisation"]["ownership"] == "modelbench_owned"
    assert ev["cleanup"]["outcome"] == CleanupOutcome.SUCCEEDED.value
    assert ev["identity_key"] == outcome.identity_key


def test_owned_runtime_cleaned_on_benchmark_exception(monkeypatch, tmp_path):
    outcome = _owned_outcome()

    def _run(client, cfg, **kw):
        raise RuntimeError("benchmark blew up")

    _wire_full_run(monkeypatch, outcome, run_impl=_run)
    with pytest.raises(RuntimeError, match="benchmark blew up"):
        cli.cmd_run(_run_args(tmp_path), Config())
    assert outcome.controller.cleanup_calls == 1
    ev = json.loads((tmp_path / "r" / "materialisation_evidence.json").read_text())
    assert ev["cleanup"]["observed"] is True


def test_cleanup_failure_on_successful_benchmark_is_not_lost(monkeypatch, tmp_path, capsys):
    outcome = _owned_outcome(cleanup_exc=RuntimeError("teardown failed"))

    def _run(client, cfg, **kw):
        kw["out_dir"].mkdir(parents=True, exist_ok=True)
        (kw["out_dir"] / "raw_results.jsonl").write_text("", encoding="utf-8")
        return kw["out_dir"]

    _wire_full_run(monkeypatch, outcome, run_impl=_run)
    cli.cmd_run(_run_args(tmp_path), Config())
    ev = json.loads((tmp_path / "r" / "materialisation_evidence.json").read_text())
    assert ev["cleanup"]["outcome"] == CleanupOutcome.GRACEFUL_FAILED.value
    assert "cleanup_failed_on_successful_benchmark" in ev.get("warnings", [])
    assert "cleanup" in capsys.readouterr().out.lower()


def test_external_reuse_runtime_is_never_cleaned(monkeypatch, tmp_path):
    outcome = _reuse_outcome()

    def _run(client, cfg, **kw):
        kw["out_dir"].mkdir(parents=True, exist_ok=True)
        (kw["out_dir"] / "raw_results.jsonl").write_text("", encoding="utf-8")
        return kw["out_dir"]

    _wire_full_run(monkeypatch, outcome, run_impl=_run)
    cli.cmd_run(_run_args(tmp_path), Config())
    assert outcome.controller.cleanup_calls == 0
    ev = json.loads((tmp_path / "r" / "materialisation_evidence.json").read_text())
    assert ev["materialisation"]["ownership"] == "external_reused"


def test_no_direct_subprocess_or_popen_in_cli_run_path():
    import inspect
    src = inspect.getsource(cli)
    for banned in ("subprocess.Popen", "os.system(", "Popen(", "shell=True"):
        assert banned not in src
