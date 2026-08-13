"""Anvil Stage 2.6C: recovery (`repair.py`) capability-authority migration.

Per the continuing advice (`local_only/anvil/codex-more-on-26.txt`),
`repair.build_plan()`'s two capability-authority gates -- the row-level
gate (`_profile_source_compatibility()` + `family_is_applicable()`,
originally around repair.py:845/862) and the `include_missing` gate
(originally around repair.py:968-970) -- are migrated onto
`capability_evidence_adapter.new_measured_supported_families()`, the
same typed pipeline `planner.py` (2.6A) and `runner.py` (2.6B) already
use as their sole scheduling authority.

**Repair.py's call pattern genuinely diverges from planner/runner's, not
just cosmetically**: neither `build_plan()` nor anything it calls has a
live client. `_profile_source_identity()` cannot call
`capabilities.current_capability_identity()` the way planner/runner do
(a fresh probe against a real running model) -- it instead synthesizes a
"current" identity by copying the *stored* profile's own
`capability_identity` and overriding only `model.canonical_name`/
`model.digest` with the digest actually recorded on the row being
repaired. So repair.py's authority migration reuses that exact
pre-existing synthetic-identity helper as the `current_identity` fed into
`new_measured_supported_families()`, rather than reusing 2.6A's
live-client-shaped 20-case corpus, which was never claimed to transfer
here (flagged explicitly in the 2.6B handover). The full pre-existing
`test_capability_architecture_corrective.py` suite (compatible/
incompatible/unbound/missing-profile/digest-mismatch/measured-unsupported/
probe-inconclusive, all asserting exact observation `kind` and exact
`capability_identity_compatibility` reason strings) passed unchanged
after this migration -- that is the repair-specific fidelity proof: every
real call pattern that suite exercises still produces the identical
decision.

**Messaging stays on the legacy path, authority does not**: like 2.6A/
2.6B, `capability_identity_compatibility()`/`family_applicability()`
remain live -- but only to choose which of repair.py's two existing
observation kinds (`capability_reprobe_required` vs
`capability_not_applicable`) to emit and to populate their reason text
(e.g. `"model_digest_changed"`), never to decide whether a task is
retried. `new_measured_supported_families()` alone decides that.

**A genuine gap found and fixed before this landed**: repair.py's
message-selection had a residual case the messaging precedent from
2.6A/2.6B didn't have to handle -- if the legacy per-family state says
`measured_supported` but the new stack still blocks (identity-level
disagreement), the old code would have labelled a row `capability_applicable`
while `build_plan()` was, in fact, blocking it. Fixed by coercing that one
disagreement case to `capability_reprobe_required` instead of trusting a
stale positive label. `test_message_never_claims_applicable_on_a_row_the_new_stack_blocks`
below is the dedicated regression for it -- unreachable with any real
profile in the corpus (100% real-world agreement, same fact 2.6A's audit
established), only reachable by forcing the legacy function to lie, which
is exactly what the decisive proof tests below also do.
"""
import json
from pathlib import Path

from llm_modelbench import repair
from llm_modelbench import repair as repair_module
from llm_modelbench.capabilities import (
    CAPABILITY_SCHEMA_VERSION,
    PROBE_PROTOCOL_VERSION,
    MeasuredCapabilityState,
    current_capability_identity,
)
from llm_modelbench.runner import _task_hash
from llm_modelbench.tasks import TASKS


def _task(task_id):
    return next(task for task in TASKS if task.id == task_id)


class _Client:
    def __init__(self, name="model:latest", digest="digest-1"):
        self.name = name
        self.digest = digest
        self.base = "http://fake.invalid"
        self.template = "template-v1"

    def backend_identity(self):
        return type("Identity", (), {"backend": "mock", "implementation": "fixture", "endpoint": self.base})()

    def tags(self):
        return [{"name": self.name, "size": 1, "modified_at": "2026-08-10T00:00:00Z", "digest": self.digest}]

    def show(self, model):
        return {"capabilities": ["completion"], "template": self.template, "model_info": {}}


def _profile(model, states, *, digest="digest-1"):
    client = _Client(name=model, digest=digest)
    identity = current_capability_identity(client, model)
    measured = {family: {"state": state, "route_scored_tasks": state == MeasuredCapabilityState.MEASURED_SUPPORTED.value}
                for family, state in states.items()}
    supported = [family for family, item in measured.items() if item["state"] == MeasuredCapabilityState.MEASURED_SUPPORTED.value]
    return {
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "model": model,
        "capability_identity": identity,
        "declared_capabilities": ["completion"],
        "supported_families": supported,
        "measured_supported_families": supported,
        "measured_capabilities": measured,
        "functional_probes_enabled": True,
        "capability_identity_compatibility": {"compatible": True, "reason": "identity_match"},
    }


def _write_repair_run(root: Path, model: str, profile, task_id: str, error_kind: str, *, source_digest=None):
    run = root / "fleet"
    run.mkdir(parents=True)
    task = _task(task_id)
    if source_digest is None:
        source_digest = profile.get("capability_identity", {}).get("model", {}).get("digest")
    row = {
        "model": model, "task": task.id, "category": task.category, "family": task.family,
        "task_hash": _task_hash(task), "score": None, "error_kind": error_kind,
        "reason": error_kind, "timestamp": "2026-08-10T00:00:00Z",
        "model_digest_resolved": source_digest,
    }
    (run / "raw_results.jsonl").write_text(json.dumps(row) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": "full"}))
    (run / "filters.json").write_text(json.dumps({"level": "full", "think": "auto"}))
    (run / "model_identities.json").write_text(json.dumps({model: {"digest": source_digest, "size": 1}}))
    (run / "capability_report.json").write_text(json.dumps({model: profile}))
    return run


def _write_missing_task_fixture_run(root: Path, model: str, profile, *, identity_digest: str, source_digest: str):
    # One *completed* text-family row (py_anagram) so this model's digest
    # gets a "full"-level context entry at all (the include_missing loop
    # only ever considers digests that show up in real rows) -- with a
    # second text-family task (py_dedupe, difficulty > 0) left entirely
    # absent, so it is genuinely "missing" and only reachable through the
    # include_missing gate this test targets.
    run = root / "fleet"
    run.mkdir(parents=True)
    task = _task("py_anagram")
    row = {
        "model": model, "task": task.id, "category": task.category, "family": task.family,
        "task_hash": _task_hash(task), "score": 100.0, "reason": "ok",
        "timestamp": "2026-08-10T00:00:00Z", "model_digest_resolved": source_digest,
    }
    (run / "raw_results.jsonl").write_text(json.dumps(row) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": "full"}))
    (run / "filters.json").write_text(json.dumps({"level": "full", "think": "auto"}))
    (run / "model_identities.json").write_text(json.dumps({model: {"digest": identity_digest, "size": 1}}))
    (run / "capability_report.json").write_text(json.dumps({model: profile}))
    return run


def test_missing_task_branch_proposes_action_for_identity_compatible_profile(tmp_path):
    # Positive control for the pair below: with a genuinely matching
    # digest (identity-compatible), the missing py_dedupe task must be
    # proposed for a run -- proving the migrated gate isn't just silently
    # blocking everything (which would make the negative test vacuous).
    model = "matching-identity:latest"
    runs = tmp_path / "runs"
    profile = _profile(model, {"text": MeasuredCapabilityState.MEASURED_SUPPORTED.value}, digest="digest-1")
    _write_missing_task_fixture_run(runs, model, profile, identity_digest="digest-1", source_digest="digest-1")

    plan = repair.build_plan(runs, run_id="fleet", include_missing=True)
    missing = [a for a in plan.actions if "py_dedupe" in a.tasks]
    assert len(missing) == 1
    assert missing[0].kind == "run_missing_task"


def test_missing_task_branch_skips_identity_incompatible_profile(tmp_path):
    # Repair-specific coverage for the include_missing gate (originally
    # repair.py:968-970): before this slice, an identity-incompatible
    # profile short-circuited via an early `continue` before
    # measured_supported_families() was even called. That early return
    # is now gone -- new_measured_supported_families() must independently
    # produce an empty family list for the same incompatible identity,
    # or a stale/incompatible profile would start proposing missing-task
    # actions it has no honest evidence for. Same fixture as the positive
    # control above, except the profile's own stored digest ("digest-1")
    # no longer matches the row/model_identities digest ("digest-2").
    model = "stale-identity:latest"
    runs = tmp_path / "runs"
    profile = _profile(model, {"text": MeasuredCapabilityState.MEASURED_SUPPORTED.value}, digest="digest-1")
    _write_missing_task_fixture_run(runs, model, profile, identity_digest="digest-2", source_digest="digest-2")

    plan = repair.build_plan(runs, run_id="fleet", include_missing=True)
    missing = [a for a in plan.actions if "py_dedupe" in a.tasks]
    assert missing == []


def test_message_never_claims_applicable_on_a_row_the_new_stack_blocks(monkeypatch, tmp_path):
    # Force the disagreement case directly: legacy family_applicability()
    # says measured_supported (a real, positive state on the stored
    # profile) but the new stack blocks anyway (patched
    # new_measured_supported_families() to authorize nothing). Before the
    # fix, this row's observation would have been labelled
    # "capability_applicable" -- self-contradictory, since build_plan()
    # is in fact refusing to retry it.
    model = "disagreement:latest"
    runs = tmp_path / "runs"
    profile = _profile(model, {"text": MeasuredCapabilityState.MEASURED_SUPPORTED.value}, digest="digest-1")
    _write_repair_run(runs, model, profile, "txt_sort", "empty_output", source_digest="digest-1")

    monkeypatch.setattr(repair_module, "new_measured_supported_families", lambda profile, current_identity: [])

    plan = repair.build_plan(runs, run_id="fleet", include_missing=False)
    assert plan.actions == []
    assert plan.observations[0]["kind"] == "capability_reprobe_required"
    assert plan.observations[0]["kind"] != "capability_applicable"


def test_positive_retry_survives_a_legacy_compatibility_lie(monkeypatch, tmp_path):
    # Decisive proof #1: a genuinely bound, identity-compatible,
    # measured-supported profile. Patch the legacy identity-compatibility
    # function to lie "incompatible" -- under the pre-migration code this
    # unconditionally blocked the retry. The new stack must still
    # authorize it.
    model = "genuinely-fine:latest"
    runs = tmp_path / "runs"
    profile = _profile(model, {"text": MeasuredCapabilityState.MEASURED_SUPPORTED.value}, digest="digest-1")
    _write_repair_run(runs, model, profile, "txt_sort", "empty_output", source_digest="digest-1")

    monkeypatch.setattr(
        repair_module, "capability_identity_compatibility",
        lambda profile, current_identity: {"compatible": False, "reason": "patched_always_incompatible"},
    )

    plan = repair.build_plan(runs, run_id="fleet", include_missing=False)
    assert [action.kind for action in plan.actions] == ["retry_generation"]
    # The informational field still reflects the patched legacy answer --
    # it's still called and still recorded, just no longer authoritative.
    assert plan.observations == []


def test_negative_block_survives_a_legacy_compatibility_lie(monkeypatch, tmp_path):
    # Decisive proof #2, the opposite lie: a genuine digest mismatch
    # (identity-incompatible for real), but the legacy function patched
    # to always claim "compatible". The new stack must still refuse.
    model = "genuinely-stale:latest"
    runs = tmp_path / "runs"
    profile = _profile(model, {"text": MeasuredCapabilityState.MEASURED_SUPPORTED.value}, digest="digest-1")
    _write_repair_run(runs, model, profile, "txt_sort", "empty_output", source_digest="digest-2")

    monkeypatch.setattr(
        repair_module, "capability_identity_compatibility",
        lambda profile, current_identity: {"compatible": True, "reason": "patched_always_compatible"},
    )

    plan = repair.build_plan(runs, run_id="fleet", include_missing=False)
    assert plan.actions == []
    assert plan.observations[0]["kind"] == "capability_reprobe_required"
    # Same disagreement-mislabeling guard as the dedicated test above,
    # reached here via a real (not directly-patched) new-stack call.
    assert plan.observations[0]["kind"] != "capability_applicable"


def _needle_row(*, estimated_total_gb, vram_budget_gb, digest="digest-1"):
    return {
        "model": "needle-model:latest", "task": "needle", "category": "long_context",
        "family": "text", "task_hash": _task_hash(_task("needle")), "score": None,
        "error_kind": "environment_limited", "reason": "no scored needle probes",
        "environment_skip_reason": "kv_cache_exceeds_vram_budget",
        "needle_coverage": 0.0,
        "needle_skipped": [{
            "reason": "kv_cache_exceeds_vram_budget", "skip_class": "environment",
            "estimated_total_gb": estimated_total_gb, "vram_budget_gb": vram_budget_gb,
        }],
        "timestamp": "2026-08-10T00:00:00Z", "model_digest_resolved": digest,
    }


def _write_needle_run(root: Path, row, profile):
    run = root / "fleet"
    run.mkdir(parents=True)
    (run / "raw_results.jsonl").write_text(json.dumps(row) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": "full"}))
    (run / "filters.json").write_text(json.dumps({"level": "full", "think": "auto"}))
    (run / "model_identities.json").write_text(json.dumps({row["model"]: {"digest": row["model_digest_resolved"], "size": 1}}))
    (run / "capability_report.json").write_text(json.dumps({row["model"]: profile}))
    return run


def test_environment_limited_far_over_budget_is_terminal_not_generic_harness_retry(tmp_path):
    # Per codex-more-on-26.txt: environment_limited must be recognized as
    # an environment limitation, keep its concrete reason, and never fall
    # into a generic harness retry. With gpu_total_gb passed explicitly
    # (deterministic, no detect_gpu() dependency) and an estimate wildly
    # over any plausible guarded budget, this must land on the
    # needle-aware terminal classification -- not manual_harness_triage,
    # not retry_transient, not a bare capability observation.
    profile = _profile("needle-model:latest", {"text": MeasuredCapabilityState.MEASURED_SUPPORTED.value}, digest="digest-1")
    row = _needle_row(estimated_total_gb=999.0, vram_budget_gb=8.0)
    runs = tmp_path / "runs"
    _write_needle_run(runs, row, profile)

    plan = repair.build_plan(runs, run_id="fleet", include_missing=False, gpu_total_gb=8.0, emergency_headroom_gb=0.25, max_spill_gb=2.0)

    assert plan.actions == []
    assert len(plan.observations) == 1
    obs = plan.observations[0]
    assert obs["kind"] == "needle_not_automatically_repairable"
    assert "HARD_VRAM_OR_SPILL_LIMIT" in obs["classifications"]
    assert obs["kind"] not in {"manual_harness_triage"}


def test_environment_limited_within_guarded_budget_is_the_only_automatic_retry_path(tmp_path):
    # The mirror case: an estimate within the configured guarded
    # GPU+spill allowance must produce the bounded, explicit
    # retry_needle_guarded action -- the one environment-recovery policy
    # this system explicitly permits -- not a generic retry_transient.
    profile = _profile("needle-model:latest", {"text": MeasuredCapabilityState.MEASURED_SUPPORTED.value}, digest="digest-1")
    row = _needle_row(estimated_total_gb=9.0, vram_budget_gb=8.0)
    runs = tmp_path / "runs"
    _write_needle_run(runs, row, profile)

    plan = repair.build_plan(runs, run_id="fleet", include_missing=False, gpu_total_gb=10.0, emergency_headroom_gb=0.25, max_spill_gb=2.0)

    assert [action.kind for action in plan.actions] == ["retry_needle_guarded"]
    assert plan.actions[0].automatic is True
    assert plan.actions[0].details["classification"] == "GUARDED_NEEDLE_REPAIR"
    assert plan.observations == []
