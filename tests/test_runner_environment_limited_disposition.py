"""Anvil Stage 2.6B: the named acceptance test for the harness_error ->
environment_limited correction. Owner-supplied advice: a needle task
where every context depth is excluded purely by the pre-flight VRAM/KV
budget check (`_needle_environment_skip`) must produce
`error_kind: "environment_limited"`, `score: None`, and preserve the
underlying low-level reason -- NOT `score: 0`, NOT `harness_error`, NOT
`capability_unsupported`. This is EXPECTED_CORRECTION, not a
legacy/new mismatch to eliminate: before this slice every such row
collapsed to `error_kind: "harness_error"` regardless of cause
(`runner.py`'s `if not hits:` branch hardcoded it).

`"environment_limited"` (not the advice's descriptive phrase
"environment_does_not_fit") is deliberately the string produced --
confirmed by reading `campaign.classify_recovery_row()`
(`error_kind == "environment_limited"` -> `disposition: "environment_limited"`,
`retry: False`), `campaign.TERMINAL_DISPOSITIONS`, and
`rankings_v3._STATUS_WEIGHT["environment_limited"]`: this is already a
real, wired, row-level vocabulary term recognized by downstream
consumers, just never previously produced by the harness. Using the
advice's literal phrase instead would have created a second, orphaned
vocabulary term alongside `evidence.py`'s already-unwired
`EvalStatus.ENVIRONMENT_SKIPPED` -- exactly the kind of gap this
correction exists to close, not repeat.
"""
import json

from llm_modelbench import outcome, repair
from llm_modelbench.capabilities import CAPABILITY_SCHEMA_VERSION, PROBE_PROTOCOL_VERSION, current_capability_identity
from llm_modelbench.config import Config
from llm_modelbench.ollama import MockClient

MODEL = "qwen2.5-coder:14b"


def _profile(client, model=MODEL):
    identity = current_capability_identity(client, model)
    return {model: {
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "capability_identity": identity,
        "declared_capabilities": ["completion"],
        "measured_capabilities": {
            "text": {"state": "measured_supported", "route_scored_tasks": True},
            "long_context": {"state": "measured_supported", "route_scored_tasks": True},
        },
    }}


def _run_needle_forcing_environment_skip(tmp_path, monkeypatch):
    from llm_modelbench import runner as runner_module

    def _always_over_budget(kv, wanted_ctx, safe_floor=32768):
        return {
            "reason": "kv_cache_exceeds_vram_budget",
            "estimated_total_gb": 999.0,
            "vram_budget_gb": 8.0,
            "kv_exceeds_budget": True,
        }

    monkeypatch.setattr(runner_module, "_needle_environment_skip", _always_over_budget)
    # The needle task's largest depth (65536) exceeds MockClient's default
    # context_length() (32768), which would otherwise exclude that depth
    # via the unrelated exceeds_context_length_max/model_capability path
    # -- a genuine model-capability limit, not an environment one. Raise
    # it here so every depth reaches the (patched) environment check,
    # keeping this test's "purely environment" precondition true for all
    # probed depths, not just most of them.
    monkeypatch.setattr(MockClient, "context_length", lambda self, model: 131072)
    client = MockClient()
    out_dir = runner_module.run(
        client, Config(fingerprint=False), level="full", out_dir=tmp_path,
        include=None, exclude=None, skip_offload=False, categories=None,
        task_ids=["needle"], resume=False, live_ui="off", fingerprint_enabled=False,
        selected_models=[MODEL], capability_profiles=_profile(client), auto_probe=False,
        gpu_inventory=(),
    )
    rows = [json.loads(line) for line in (out_dir / "raw_results.jsonl").read_text().splitlines() if line.strip()]
    needle_rows = [r for r in rows if r.get("task") == "needle"]
    assert len(needle_rows) == 1
    return needle_rows[0], out_dir


def test_pure_environment_skip_needle_row_is_environment_limited_not_harness_error(tmp_path, monkeypatch):
    row, out_dir = _run_needle_forcing_environment_skip(tmp_path, monkeypatch)

    # Capability-supported and environment-not-fit stay distinct all the
    # way through execution: the model reached and ran the needle task at
    # all only because its capability decision (long_context/text
    # measured_supported) passed the new_measured_supported_families()
    # gate -- it is not in skipped_models. The environment limit is a
    # per-task-execution disposition, not a capability re-decision.
    skipped_models = json.loads((out_dir / "skipped_models.json").read_text())
    assert not any(item.get("model") == MODEL for item in skipped_models)

    assert row["error_kind"] == "environment_limited"
    assert row["score"] is None
    assert row["environment_skip_reason"] == "kv_cache_exceeds_vram_budget"
    assert row["needle_coverage"] == 0.0
    # The specific low-level evidence survives in full, not just the
    # summary reason -- the layered model the advice asked for
    # (typed disposition: environment_limited; specific evidence:
    # kv_cache_exceeds_vram_budget, estimated_total_gb, budget_gb).
    assert row["needle_skipped"]
    assert all(item.get("skip_class") == "environment" for item in row["needle_skipped"])
    assert row["needle_skipped"][0]["estimated_total_gb"] == 999.0
    assert row["needle_skipped"][0]["vram_budget_gb"] == 8.0

    assert row["error_kind"] != "harness_error"
    assert row["error_kind"] != "capability_unsupported"
    assert row["score"] != 0


def test_environment_limited_row_maps_to_not_attempted_not_harness_error():
    row = {"error_kind": "environment_limited", "score": None, "reason": "no scored needle probes"}
    result = outcome.row_to_outcome(row)
    assert isinstance(result, outcome.NotAttempted)
    assert result.kind == "environment_limited"
    assert result.actor == "harness"


def test_environment_limited_row_does_not_force_category_score_to_harness_error():
    scored = outcome.Scored(80.0)
    env_limited = outcome.NotAttempted("environment_limited", "harness")
    score, coverage, blocker = outcome.category_score([scored, env_limited], [1.0, 1.0])
    # A real HarnessError in the mix forces the whole category to
    # "unknown" (blocker="harness_error"); an environment-limited row
    # must not trigger that same fail-closed-on-corruption behavior --
    # it's known-good evidence about a configuration limit, not
    # corrupted/unknown evidence. It should only lower coverage.
    assert blocker != "harness_error"
    assert coverage == 0.5


def test_environment_limited_needle_row_reaches_needle_aware_repair_triage(tmp_path, monkeypatch):
    # Before this slice, every such row hardcoded error_kind="harness_error"
    # and was swallowed by repair.build_plan()'s generic harness_error
    # branch (-> manual_harness_triage), never reaching
    # repair._needle_observation() -- the purpose-built VRAM/KV triage
    # logic already gated on needle_coverage/needle_skipped, not
    # error_kind. This proves that path is reachable again.
    row, out_dir = _run_needle_forcing_environment_skip(tmp_path, monkeypatch)
    run_dir = out_dir
    plan = repair.build_plan(run_dir.parent, run_id=run_dir.name, include_missing=False)
    observation_kinds = {obs.get("kind") for obs in plan.observations}
    action_kinds = {action.kind for action in plan.actions}
    assert "manual_harness_triage" not in observation_kinds
    assert observation_kinds & {"needle_not_automatically_repairable", "needle_incomplete_unclassified"} or "retry_needle_guarded" in action_kinds
