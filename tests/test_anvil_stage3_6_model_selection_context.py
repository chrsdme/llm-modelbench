"""Anvil Stage 3.6 -- model re-selection surfacing acceptance tests.

Covers the mandatory Stage 3.6 tests: known-model re-selection surfacing
(section 18), new model (section 19), canonical-vs-fastest (section 20),
lowest-VRAM honest unavailable (section 21), largest verified context
(section 22), protocol separation (section 23), read-only (section 24),
and the unattended plan path (section 25).
"""
from __future__ import annotations

import json
from pathlib import Path

from llm_modelbench.model_selection_context import (
    build_model_selection_context,
    render_model_selection_context,
    _prior_canonical_runtime,
    _fastest_observed,
    _largest_verified_context,
)


# ---------------------------------------------------------------------------
# fixture helpers -- write real prior-run evidence to disk
# ---------------------------------------------------------------------------


def _write_run(
    runs_dir: Path,
    run_id: str,
    *,
    bindings: dict | None = None,
    resume_divergent: list | None = None,
    rows: list | None = None,
) -> Path:
    run = runs_dir / run_id
    run.mkdir(parents=True)
    (run / "raw_results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in (rows or []))
    )
    if bindings is not None or resume_divergent is not None:
        payload = {"schema_version": 1, "bindings": bindings or {}}
        if resume_divergent is not None:
            payload["resume_divergent_bindings"] = resume_divergent
        (run / "benchmark_bindings.json").write_text(json.dumps(payload, indent=2))
    return run


def _binding_entry(*, binding_key: str, proto_key: str, profile_key: str,
                   protocol_id: str = "llm-modelbench-core", version: str = "1") -> dict:
    return {
        "binding": {
            "binding_key": binding_key,
            "benchmark_protocol_identity_key": proto_key,
            "runtime_profile_identity_key": profile_key,
            "allowed_adaptations_used": ["context_size"],
            "provenance": "anvil_stage3.2d_deterministic_resolution",
        },
        "protocol": {"protocol_id": protocol_id, "version": version},
    }


def _bench_row(model: str, *, task: str, tps: float, binding_key: str,
               proto_key: str = "proto-v1-key",
               protocol_id: str = "llm-modelbench-core", version: str = "1") -> dict:
    return {
        "model": model, "task": task, "tps": tps, "error_kind": None,
        "benchmark_binding_key": binding_key,
        "benchmark_protocol_identity_key": proto_key,
        "benchmark_protocol_id": protocol_id, "benchmark_protocol_version": version,
    }


def _needle_row(model: str, *, max_verified_ctx, attempted: bool = True) -> dict:
    row = {"model": model, "task": "needle", "needle_attempted": [] if attempted else None}
    if max_verified_ctx is not None:
        row["max_verified_ctx"] = max_verified_ctx
    return row


def _active_entry(model: str, *, families=None, evidence_hash="cap-hash-1",
                  disagreements=None, warnings=None) -> dict:
    return {
        "model": model,
        "families": list(["text"] if families is None else families),
        "capability_warnings": list(warnings or []),
        "capability_evidence_hash": evidence_hash,
        "native_legacy_capability_evidence_disagreements": list(disagreements or []),
    }


# ---------------------------------------------------------------------------
# section 18 -- known model re-selection surfaces the required information
# ---------------------------------------------------------------------------


class _FakeRuntimeIdentity:
    """Minimal stand-in for runtime_identity.RuntimeIdentity -- only the two
    attributes build_model_binding reads."""

    backend = "ollama"
    server_version = "0.1.0"


def test_known_model_reselection_surfaces_required_fields(tmp_path, monkeypatch):
    runs = tmp_path / "runs"

    # Pin the active protocol resolution so this test does not depend on the
    # real benchmark_binding machinery -- it exercises the surfacing, not
    # protocol construction.
    active_proto = {
        "protocol_id": "llm-modelbench-core", "version": "1",
        "identity_key": "pk", "benchmark_protocol_identity_key": "proto-v1-key",
        "active_binding_key": "bk-canonical",
    }
    monkeypatch.setattr(
        "llm_modelbench.model_selection_context._active_protocol_identity",
        lambda *a, **k: dict(active_proto),
    )

    _write_run(
        runs, "run_a",
        bindings={"llama3": _binding_entry(
            binding_key="bk-canonical", proto_key="proto-v1-key", profile_key="rp-gpu0")},
        rows=[
            _bench_row("llama3", task="rag", tps=42.0, binding_key="bk-canonical"),
            _bench_row("llama3", task="summ", tps=55.5, binding_key="bk-canonical"),
            _needle_row("llama3", max_verified_ctx=32768),
        ],
    )

    ctx = build_model_selection_context(
        [_active_entry("llama3", families=["text", "tools"])],
        runs_dir=runs,
        runtime_identities={"llama3": _FakeRuntimeIdentity()},
        models_rows={"llama3": {"name": "llama3", "digest": "sha256:abc"}},
    )
    obs = ctx.by_model["llama3"]

    assert obs.model == "llama3"
    assert obs.known is True
    assert obs.measured_capability_families == ["text", "tools"]
    assert obs.capability_evidence_hash == "cap-hash-1"
    assert obs.active_protocol_identity["benchmark_protocol_identity_key"] == "proto-v1-key"
    # canonical benchmark runtime -- from the binding artifact
    assert obs.canonical_benchmark_runtime["status"] == "resolved"
    assert obs.canonical_benchmark_runtime["binding_key"] == "bk-canonical"
    assert obs.canonical_benchmark_runtime["benchmark_protocol_id"] == "llm-modelbench-core"
    assert obs.canonical_benchmark_runtime["run_id"] == "run_a"
    # largest verified context -- the runner-persisted field
    assert obs.largest_verified_context == {
        "status": "resolved", "max_verified_ctx": 32768, "run_id": "run_a"}
    # fastest observed -- scoped to the canonical binding
    assert obs.fastest_observed["status"] == "resolved"
    assert obs.fastest_observed["tps"] == 55.5
    assert obs.fastest_observed["benchmark_binding_key"] == "bk-canonical"
    # lowest VRAM -- honest unavailable
    assert obs.lowest_vram_observed["status"] == "unavailable"

    # superseded four-role fields must not appear anywhere
    payload = json.dumps(ctx.to_dict())
    for forbidden in (
        "validated_runtime_profiles", "recommended_production_profile",
        "best_observed_profile", "human_validation_status", "human_correlation",
    ):
        assert forbidden not in payload

    rendered = render_model_selection_context(ctx)
    assert "llama3" in rendered
    assert "known" in rendered
    assert "canonical benchmark runtime:" in rendered
    assert "largest verified context: 32768" in rendered


# ---------------------------------------------------------------------------
# section 19 -- new / unknown model
# ---------------------------------------------------------------------------


def test_new_model_has_honest_unavailable_dispositions(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()

    ctx = build_model_selection_context(
        [_active_entry("brand-new", families=[], evidence_hash=None)],
        runs_dir=runs,
    )
    obs = ctx.by_model["brand-new"]

    assert obs.known is False
    assert obs.measured_capability_families == []
    assert obs.canonical_benchmark_runtime["status"] in {"unavailable", "deferred_pending_protocol_resolution"}
    assert obs.largest_verified_context["status"] == "unavailable"
    assert obs.fastest_observed["status"] == "unavailable"
    assert any("No prior authoritative evidence" in w for w in obs.warnings)
    # no invented capability support
    assert obs.measured_capability_families == []


def test_new_model_still_renders_and_does_not_raise(tmp_path):
    ctx = build_model_selection_context(
        [_active_entry("nope", families=[], evidence_hash=None)],
        runs_dir=tmp_path / "does-not-exist",
    )
    assert "new / unmeasured" in render_model_selection_context(ctx)


# ---------------------------------------------------------------------------
# section 20 -- canonical is NEVER replaced by fastest
# ---------------------------------------------------------------------------


def test_canonical_is_not_substituted_by_a_faster_other_profile(tmp_path):
    runs = tmp_path / "runs"
    proto_key = "proto-v1-key"
    # profile A is the canonical binding; profile B is a *different* binding
    # under the same protocol that happens to be faster.
    _write_run(
        runs, "run_canonical",
        bindings={"m": _binding_entry(
            binding_key="bk-A", proto_key=proto_key, profile_key="rp-A")},
        rows=[_bench_row("m", task="rag", tps=20.0, binding_key="bk-A")],
    )
    _write_run(
        runs, "run_faster",
        bindings={"m": _binding_entry(
            binding_key="bk-B", proto_key=proto_key, profile_key="rp-B")},
        rows=[_bench_row("m", task="rag", tps=99.0, binding_key="bk-B")],
    )

    # canonical resolves to the most recent binding artifact; force run_canonical newer
    import os
    import time
    newer = runs / "run_canonical" / "benchmark_bindings.json"
    os.utime(newer, (time.time() + 100, time.time() + 100))

    # canonical is resolved from binding artifacts only (never sees tps rows)
    canonical = _prior_canonical_runtime("m", runs, active_protocol_identity_key=proto_key)
    assert canonical["status"] == "resolved"
    assert canonical["binding_key"] == "bk-A"

    # fastest is a SEPARATE helper scoped to the *protocol*, not the
    # canonical binding. It surfaces the genuinely fastest observation
    # (99.0 under bk-B) AS HISTORY, tagged with B's own binding key -- so
    # the two facts are disjoint and visible: canonical is bk-A, and the
    # fastest run happened under bk-B. B is never promoted to canonical.
    fastest = _fastest_observed("m", runs, protocol_identity_key=proto_key)
    assert fastest["status"] == "resolved"
    assert fastest["tps"] == 99.0
    assert fastest["benchmark_binding_key"] == "bk-B"
    assert fastest["benchmark_binding_key"] != canonical["binding_key"]

    # and through the full builder with a pinned active protocol
    monkeypatch_proto = {
        "protocol_id": "llm-modelbench-core", "version": "1", "identity_key": "pk",
        "benchmark_protocol_identity_key": proto_key, "active_binding_key": "bk-A",
    }
    import llm_modelbench.model_selection_context as msc_mod
    orig = msc_mod._active_protocol_identity
    msc_mod._active_protocol_identity = lambda *a, **k: dict(monkeypatch_proto)
    try:
        ctx = build_model_selection_context(
            [_active_entry("m")], runs_dir=runs,
            runtime_identities={"m": _FakeRuntimeIdentity()},
        )
    finally:
        msc_mod._active_protocol_identity = orig
    obs = ctx.by_model["m"]
    # canonical stays bk-A; fastest is surfaced separately as the bk-B run
    assert obs.canonical_benchmark_runtime["binding_key"] == "bk-A"
    assert obs.fastest_observed["tps"] == 99.0
    assert obs.fastest_observed["benchmark_binding_key"] == "bk-B"


def test_fastest_helper_never_reads_binding_artifacts(tmp_path):
    """Structural: _fastest_observed only takes a protocol identity key."""
    runs = tmp_path / "runs"
    _write_run(runs, "r", rows=[_bench_row("m", task="t", tps=10.0, binding_key="bk")])
    # no benchmark_bindings.json written at all
    out = _fastest_observed("m", runs, protocol_identity_key="proto-v1-key")
    assert out["status"] == "resolved" and out["tps"] == 10.0
    # and with no scope -> unavailable, not a blind max
    assert _fastest_observed("m", runs, protocol_identity_key=None)["status"] == "unavailable"


# ---------------------------------------------------------------------------
# section 21 -- lowest-VRAM is honestly unavailable, not faked
# ---------------------------------------------------------------------------


def test_lowest_vram_is_always_unavailable_with_a_reason(tmp_path):
    runs = tmp_path / "runs"
    # even with needle rows that carry vram_peak_mb, we do not fake it
    _write_run(runs, "r", rows=[
        {"model": "m", "task": "needle", "vram_peak_mb": 8000,
         "needle_attempted": [{"num_ctx": 16000, "vram_peak_mb": 8000}]},
    ])
    ctx = build_model_selection_context([_active_entry("m")], runs_dir=runs)
    obs = ctx.by_model["m"]
    assert obs.lowest_vram_observed["status"] == "unavailable"
    assert "telemetry" in obs.lowest_vram_observed["reason"]


# ---------------------------------------------------------------------------
# section 22 -- largest verified context: 32K, never the failed 64K
# ---------------------------------------------------------------------------


def test_largest_verified_context_uses_persisted_field_not_attempted(tmp_path):
    runs = tmp_path / "runs"
    # The runner already collapses attempted probes to max_verified_ctx via
    # _max_verified_prefix (16K + 32K found, 64K failed -> 32768). We read
    # that field; a row that ATTEMPTED 64K but only VERIFIED 32K must yield 32768.
    _write_run(runs, "r1", rows=[_needle_row("m", max_verified_ctx=16384)])
    _write_run(runs, "r2", rows=[_needle_row("m", max_verified_ctx=32768)])
    out = _largest_verified_context("m", runs)
    assert out == {"status": "resolved", "max_verified_ctx": 32768, "run_id": "r2"}


def test_largest_verified_context_unavailable_when_no_depth_verified(tmp_path):
    runs = tmp_path / "runs"
    # needle was attempted but nothing verified
    _write_run(runs, "r", rows=[_needle_row("m", max_verified_ctx=None)])
    out = _largest_verified_context("m", runs)
    assert out["status"] == "unavailable"
    assert out["reason"] == "no_verified_needle_depth"


def test_largest_verified_context_unavailable_when_no_needle_evidence(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "r", rows=[_bench_row("m", task="rag", tps=1.0, binding_key="bk")])
    out = _largest_verified_context("m", runs)
    assert out["reason"] == "no_needle_evidence"


# ---------------------------------------------------------------------------
# section 23 -- protocol separation
# ---------------------------------------------------------------------------


def test_canonical_only_uses_protocol_compatible_history(tmp_path):
    runs = tmp_path / "runs"
    v1_key = "proto-v1-key"
    v2_key = "proto-v2-key"
    # v2 evidence exists and is faster; v1 evidence is what the run needs
    _write_run(
        runs, "run_v1",
        bindings={"m": _binding_entry(
            binding_key="bk-v1", proto_key=v1_key, profile_key="rp",
            version="1")},
        rows=[_bench_row("m", task="rag", tps=15.0, binding_key="bk-v1", version="1")],
    )
    _write_run(
        runs, "run_v2",
        bindings={"m": _binding_entry(
            binding_key="bk-v2", proto_key=v2_key, profile_key="rp",
            version="2")},
        rows=[_bench_row("m", task="rag", tps=88.0, binding_key="bk-v2", version="2")],
    )

    # active protocol identity key = v1
    out = _prior_canonical_runtime("m", runs, active_protocol_identity_key=v1_key)
    assert out["status"] == "resolved"
    assert out["binding_key"] == "bk-v1"
    assert out["benchmark_protocol_identity_key"] == v1_key
    # v2 (faster) is not selected


def test_canonical_deferred_when_active_protocol_unknown(tmp_path):
    runs = tmp_path / "runs"
    _write_run(
        runs, "run_v1",
        bindings={"m": _binding_entry(
            binding_key="bk-v1", proto_key="proto-v1-key", profile_key="rp")},
        rows=[],
    )
    # no active_protocol_identity_key -> deferred (section 10), not a wrong pick
    out = _prior_canonical_runtime("m", runs, active_protocol_identity_key=None)
    assert out["status"] == "deferred_pending_protocol_resolution"

    ctx = build_model_selection_context([_active_entry("m")], runs_dir=runs)
    obs = ctx.by_model["m"]
    assert obs.canonical_benchmark_runtime["status"] == "deferred_pending_protocol_resolution"
    assert any("deferred" in w.lower() for w in obs.warnings)


# ---------------------------------------------------------------------------
# section 24 -- read-only: no prior-run evidence is mutated
# ---------------------------------------------------------------------------


def test_building_context_does_not_mutate_prior_run_evidence(tmp_path):
    runs = tmp_path / "runs"
    proto_key = "pk"
    _write_run(
        runs, "run_a",
        bindings={"m": _binding_entry(binding_key="bk", proto_key=proto_key, profile_key="rp")},
        rows=[
            _bench_row("m", task="rag", tps=42.0, binding_key="bk"),
            _needle_row("m", max_verified_ctx=32768),
        ],
    )
    # Snapshot the WHOLE runs_dir tree (not just one run dir) so a new file
    # appearing anywhere under it -- including an EvidenceLedger -- is caught.
    def _tree():
        return {
            str(p.relative_to(runs)): p.read_bytes()
            for p in sorted(runs.rglob("*")) if p.is_file()
        }
    def _tree_mtimes():
        return {
            str(p.relative_to(runs)): p.stat().st_mtime_ns
            for p in sorted(runs.rglob("*")) if p.is_file()
        }

    before, mtimes = _tree(), _tree_mtimes()

    for _ in range(3):
        build_model_selection_context(
            [_active_entry("m")], runs_dir=runs,
            runtime_identities={"m": _FakeRuntimeIdentity()},
            models_rows={"m": {"name": "m"}},
        )

    assert _tree() == before
    assert _tree_mtimes() == mtimes
    # explicitly: no EvidenceLedger was created under runs_dir
    from llm_modelbench.capability_reprobe_execute import default_ledger_path
    assert not Path(default_ledger_path(runs)).exists()


def test_corrupt_prior_evidence_is_fail_soft(tmp_path):
    runs = tmp_path / "runs"
    run = runs / "bad_run"
    run.mkdir(parents=True)
    (run / "raw_results.jsonl").write_text('{"model": "m", "tps": 1.0\n{ truncated')
    (run / "benchmark_bindings.json").write_text("{ not json at all ")

    # must not raise -- every observation degrades to unavailable
    ctx = build_model_selection_context([_active_entry("m", families=["text"])], runs_dir=runs)
    obs = ctx.by_model["m"]
    assert obs.canonical_benchmark_runtime["status"] in {"unavailable", "deferred_pending_protocol_resolution"}
    assert obs.largest_verified_context["status"] == "unavailable"
    assert obs.fastest_observed["status"] == "unavailable"
    # capability families still surface (they come from the plan entry, not disk)
    assert obs.measured_capability_families == ["text"]


# ---------------------------------------------------------------------------
# section 25 -- unattended / plan path: context is in the plan dict
# ---------------------------------------------------------------------------


def test_context_rides_the_plan_dict_via_build_plan(tmp_path):
    from llm_modelbench.config import Config
    from llm_modelbench.ollama import MockClient
    from llm_modelbench.planner import build_plan, render_plan

    cfg = Config()
    cfg.vram_budget_gb = 12.0
    plan = build_plan(
        MockClient(), cfg, level="smoke", sample_mode="smart", auto_probe=True,
        runs_dir=tmp_path / "runs",
    )
    assert "model_selection_context" in plan
    msc = plan["model_selection_context"]
    assert msc["schema_version"] == 1
    assert {o["model"] for o in msc["observations"]} == {m["model"] for m in plan["active_models"]}
    # mock models carry interrogated capability evidence -> "known" in the
    # capability sense, but with NO prior benchmark history every historical
    # observation is honestly unavailable / deferred.
    for o in msc["observations"]:
        assert o["known"] is True
        assert o["canonical_benchmark_runtime"]["status"] in {
            "unavailable", "deferred_pending_protocol_resolution"}
        assert o["largest_verified_context"]["status"] == "unavailable"
        assert o["fastest_observed"]["status"] == "unavailable"
    # render_plan includes the block, still additive (loose substring pins hold)
    text = render_plan(plan)
    assert "unique /" in text
    assert "Prior model knowledge" in text


def test_confirm_plan_route_renders_context_for_a_pre_accepted_plan(tmp_path):
    """Section 14: the wizard / campaign-run route sets ``args._accepted_plan``
    and never rebuilds the plan; ``_confirm_plan`` -> ``render_plan`` must
    still surface the prior-knowledge block from the persisted dict."""
    from llm_modelbench.config import Config
    from llm_modelbench.ollama import MockClient
    from llm_modelbench.planner import build_plan
    from llm_modelbench import cli

    cfg = Config()
    cfg.vram_budget_gb = 12.0
    plan = build_plan(
        MockClient(), cfg, level="smoke", sample_mode="smart", auto_probe=True,
        runs_dir=tmp_path / "runs",
    )
    assert "model_selection_context" in plan

    import io
    import contextlib

    class _Args:
        yes = True
        plan_json = None

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli._confirm_plan(_Args(), plan)
    assert "Prior model knowledge" in buf.getvalue()


def test_render_drops_no_field_when_observation_schema_grows():
    """Section 14 regression guard: the renderer takes the plan dict
    directly, so adding an observation field cannot silently vanish before
    it reaches the operator."""
    from llm_modelbench.model_selection_context import render_model_selection_context

    payload = {
        "schema_version": 1,
        "observations": [{
            "model": "m", "known": True,
            "measured_capability_families": ["text"],
            "evidence_trust_class": "historical_valid",
            "canonical_benchmark_runtime": {"status": "deferred_pending_protocol_resolution"},
            "largest_verified_context": {"status": "resolved", "max_verified_ctx": 8192, "run_id": "r"},
            "fastest_observed": {"status": "unavailable", "reason": "x"},
            "lowest_vram_observed": {"status": "unavailable", "reason": "y"},
            "warnings": ["a warning line"],
            "a_future_field_the_dataclass_does_not_have": 123,
        }],
    }
    text = render_model_selection_context(payload)
    assert "m  [known]" in text
    assert "8192 tokens" in text
    assert "a warning line" in text
    assert "historical_valid" in text


def test_end_to_end_real_protocol_resolution_matches_real_binding(tmp_path):
    """No monkeypatch: build a real BenchmarkRuntimeBinding for the active
    run, write a prior run whose binding artifact carries the SAME real
    protocol identity key, and prove canonical resolves to it."""
    from llm_modelbench.config import Config
    from llm_modelbench.benchmark_binding import build_model_binding, binding_to_dict, protocol_to_dict
    from llm_modelbench.identity import ModelArtifactIdentity
    from llm_modelbench.tasks import TASKS

    cfg = Config()
    cfg.vram_budget_gb = 12.0
    tasks = list(TASKS)[:3]
    models_row = {"name": "m", "digest": "sha256:deadbeef"}

    protocol, binding = build_model_binding(
        model_artifact_identity=ModelArtifactIdentity.from_ollama_tag_row(models_row),
        selected_tasks=tasks, cfg=cfg, backend="ollama", backend_version="0.1.0",
        sample_mode="smart", judge_mode="off",
    )
    real_proto_key = binding.benchmark_protocol_identity_key
    real_binding_key = binding.binding_key()

    runs = tmp_path / "runs"
    run = runs / "prior"
    run.mkdir(parents=True)
    (run / "benchmark_bindings.json").write_text(json.dumps({
        "schema_version": 1,
        "bindings": {"m": {
            "binding": binding_to_dict(binding),
            "protocol": protocol_to_dict(protocol),
        }},
    }))
    (run / "raw_results.jsonl").write_text(
        json.dumps(_bench_row("m", task="rag", tps=33.0, binding_key=real_binding_key,
                              proto_key=real_proto_key)) + "\n"
        + json.dumps(_needle_row("m", max_verified_ctx=65536)) + "\n"
    )

    ctx = build_model_selection_context(
        [_active_entry("m", families=["text"])],
        runs_dir=runs,
        runtime_identities={"m": _FakeRuntimeIdentity()},
        models_rows={"m": models_row},
        cfg=cfg,
        tasks_by_model={"m": tasks},
        sample_mode="smart", judge_mode="off",
    )
    obs = ctx.by_model["m"]
    assert obs.active_protocol_identity["benchmark_protocol_identity_key"] == real_proto_key
    assert obs.canonical_benchmark_runtime["status"] == "resolved"
    assert obs.canonical_benchmark_runtime["binding_key"] == real_binding_key
    assert obs.canonical_benchmark_runtime["benchmark_protocol_identity_key"] == real_proto_key
    assert obs.fastest_observed["tps"] == 33.0
    assert obs.largest_verified_context["max_verified_ctx"] == 65536


def test_campaign_plan_payload_persists_context_and_equivalence_ignores_it():
    """Section 25: the non-interactive `campaign plan` path -- the context
    rides the plan dict into plan.json with no extra wiring, and it is
    excluded from plan-contract equivalence (it is display/provenance)."""
    from llm_modelbench import campaign

    plan = {
        "level": "smoke", "active_models": [{"model": "m", "tasks": []}],
        "model_selection_context": {"schema_version": 1, "observations": [
            {"model": "m", "known": True}]},
    }
    payload = campaign._campaign_plan_payload(
        campaign.resolve_paths("cid-xyz"), plan,
        configuration={"level": "smoke"}, created_at="2020-01-01T00:00:00+00:00",
    )
    assert payload["model_selection_context"] == plan["model_selection_context"]

    # a proposed re-plan whose only difference is the context is still equivalent
    other = dict(payload)
    other = json.loads(json.dumps(other))
    other["model_selection_context"] = {"schema_version": 1, "observations": [
        {"model": "m", "known": True, "fastest_observed": {"status": "resolved"}}]}
    assert campaign.campaign_plan_equivalent(payload, other) is True
    # but a real contract change is not equivalent
    other2 = json.loads(json.dumps(payload))
    other2["level"] = "full"
    assert campaign.campaign_plan_equivalent(payload, other2) is False


def test_disagreements_and_trust_surface_as_warnings(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    entry = _active_entry(
        "m", families=["text"],
        disagreements=[{"family": "vision", "native_applicable": False,
                        "legacy_applicable": True}],
    )
    ctx = build_model_selection_context(
        [entry], runs_dir=runs,
        capability_profiles={"m": {"evidence_trust_class": "historical_valid"}},
    )
    obs = ctx.by_model["m"]
    assert obs.evidence_trust_class == "historical_valid"
    assert any("disagreement" in w.lower() for w in obs.warnings)
    assert any("historical_valid" in w for w in obs.warnings)
