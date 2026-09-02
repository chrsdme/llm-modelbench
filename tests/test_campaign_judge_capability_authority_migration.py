"""Anvil Stage 2.6D, Task #9 -- the 15 required judge-eligibility test
scenarios from `codex-advise_pre2.6D.txt` part 7, exercised against the
real, current (post-migration) production functions: `campaign.build_judge_selection()`,
`campaign.qualify_judge()`/`select_qualified_campaign_judges()`,
`campaign.resolve_independent_judge_for_row()`, and `judge_dumps.judge_run()`.

Per the migration's central design decision (see the "## Stage 2.6D"
section of `local_only/anvil/ANVIL_PROGRESS.md`): positive judge capability
eligibility comes solely from `campaign._judge_capability_authorized_families()`
(the typed Stage 2 stack); `_judge_measured_text_state()`/
`_candidate_capability_identity_compatibility()` (both legacy) are consulted
only to pick a rejection message. Qualification (judge-qualification-v1) and
independence (no self-judging) remain entirely separate, untouched gates
downstream of capability eligibility -- this file's scenarios 10-13 are
deliberately about proving that separation held, not about re-testing
`judge_qualification.py`'s own protocol (already covered by
`tests/test_stage1b_judge_qualification.py`) or `judge_dumps.py`'s
structural-fallback/exhaustion state machine in full depth (already covered
by `tests/test_stage1d_judge_integration.py`) -- both of which passed
unchanged after this migration, which is itself part of the fidelity proof.

Scenario 7 (ambiguous projection) is marked N/A below, by construction, not
oversight: `campaign.py`'s judge-candidate adapter (`_judge_capability_authorized_families()`
-> `new_measured_supported_families()` -> `adapt_legacy_profile_family_to_observation()`)
produces at most one `CapabilityObservation` per family from a single stored
profile -- there is no way for a single candidate dict to produce the two
*disagreeing* observations `AMBIGUOUS_COMPATIBLE_OBSERVATIONS` requires. The
projection layer's own ambiguity handling is covered directly by
`tests/test_capability_projection.py::test_contradictory_compatible_observations_are_ambiguous`
and `::test_ambiguous_decision_is_not_applicable`.
"""
import json
from pathlib import Path

import pytest

from llm_modelbench import campaign
from llm_modelbench import judge_dumps
from llm_modelbench.capabilities import CAPABILITY_SCHEMA_VERSION, PROBE_PROTOCOL_VERSION
from llm_modelbench.runner import _task_hash
from llm_modelbench.tasks import TASKS


def _identity(name, digest, *, protocol_version=PROBE_PROTOCOL_VERSION, template_hash="template-v1", backend="mock"):
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "model": {"canonical_name": name, "backend_model_id": name, "digest": digest, "size": 1, "details": {}},
        "backend": {"backend": backend, "implementation": "fixture", "endpoint": "http://fake.invalid"},
        "runtime": {"endpoint": "http://fake.invalid", "implementation": "fixture"},
        "template_config": {"available": True, "hash": template_hash, "material": {"template": template_hash}},
        "probe_protocol_version": protocol_version,
    }


def _bind(item, **identity_kwargs):
    item.setdefault("probe_protocol_version", PROBE_PROTOCOL_VERSION)
    item["capability_identity"] = _identity(item["name"], item["digest"], **identity_kwargs)
    item["capability_identity_compatibility"] = {"compatible": True, "reason": "identity_match"}
    return item


def _measured_text(name, digest, **extra):
    item = {
        "name": name, "digest": digest,
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "measured_capabilities": {"text": {"state": "measured_supported"}},
    }
    item.update(extra)
    return _bind(item)


def _policy(**kwargs):
    defaults = {
        "requested_primary": None, "configured_fallbacks": (), "excluded_families": (),
        "allow_excluded_primary": False, "automatic_selection": True, "enabled": True,
    }
    defaults.update(kwargs)
    return campaign.JudgePolicy(**defaults)


def _subjective_task():
    return next(task for task in TASKS if task.scorer == "subjective")


def _write_subjective_run(root: Path, rows):
    task = _subjective_task()
    run = root / "run"
    raw_rows = []
    for row in rows:
        model = row["model"]
        digest = row["digest"]
        dump = run / "subjective" / task.id / f"{model}.md"
        dump.parent.mkdir(parents=True, exist_ok=True)
        dump.write_text(f"# TASK {task.id}\n\n## OUTPUT\nanswer from {model}\n")
        raw_rows.append({
            "model": model,
            "model_digest": digest,
            "model_digest_resolved": digest,
            "task": task.id,
            "category": task.category,
            "family": task.family,
            "task_hash": _task_hash(task),
            "score": None,
            "error_kind": None,
            "subjective_path": str(dump.relative_to(run)),
            "timestamp": "2026-08-10T00:00:00Z",
        })
    (run / "raw_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw_rows))
    return run


class _RecordingClient:
    def __init__(self, failures=None):
        self.failures = dict(failures or {})
        self.calls = []

    def chat(self, model, prompt, **kwargs):
        self.calls.append(model)
        failure = self.failures.get(model)
        if failure:
            return dict(failure)
        return {"ok": True, "text": '{"score": 88, "confidence": 1, "verdict": "synthetic"}'}


# 1. Compatible measured text support + qualification pass + independent
# model -> candidate can become usable.
def test_01_compatible_measured_qualified_independent_candidate_is_usable(monkeypatch):
    selection = campaign.build_judge_selection(
        [_measured_text("judge-a", "digest-judge-a")], [], _policy(requested_primary="judge-a"),
    )
    assert [item["name"] for item in selection.final_eligible_order] == ["judge-a"]

    monkeypatch.setattr(campaign, "qualify_judge", lambda client, candidate, *, ledger=None: {
        "model": candidate["name"], "digest": candidate["digest"], "qualified": True,
        "aggregate_disposition": "qualified", "protocol_version": "judge-qualification-v1",
    })
    qualified, qualifications = campaign.select_qualified_campaign_judges(object(), selection)
    assert [item["name"] for item in qualified] == ["judge-a"]

    row = {"model": "source-model", "digest": "digest-source"}
    resolution = campaign.resolve_independent_judge_for_row(row, qualified)
    assert resolution["status"] == "selected_independent_judge"
    assert resolution["judge_model"] == "judge-a"


# 2. Declared text capability only -> not positively capability-eligible.
def test_02_declared_only_is_not_capability_eligible():
    item = {
        "name": "declared-only", "digest": "digest-declared",
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "capabilities": ["completion"],
    }
    assert campaign._judge_capability_rejection(item) is not None
    selection = campaign.build_judge_selection([item], [], _policy(requested_primary="declared-only"))
    assert selection.final_eligible_order == []


# 3. supported_families=["text"] only -> not positively capability-eligible.
def test_03_supported_families_only_is_not_capability_eligible():
    item = {
        "name": "families-only", "digest": "digest-families",
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "supported_families": ["text"],
    }
    assert campaign._judge_capability_rejection(item) is not None
    selection = campaign.build_judge_selection([item], [], _policy(requested_primary="families-only"))
    assert selection.final_eligible_order == []


# 4. Missing measured evidence -> fail closed.
def test_04_missing_measured_evidence_fails_closed():
    item = {"name": "no-evidence", "digest": "digest-none", "capability_schema_version": CAPABILITY_SCHEMA_VERSION}
    assert campaign._judge_capability_authorized_families(item) == []
    assert campaign._judge_capability_rejection(item) is not None


# 5. Legacy/unbound capability evidence -> fail closed.
def test_05_legacy_unbound_profile_fails_closed():
    item = {"name": "legacy", "digest": "digest-legacy", "measured_capabilities": {"text": {"state": "measured_supported"}}}
    assert campaign._judge_capability_authorized_families(item) == []
    assert campaign._judge_capability_rejection(item) == "capability_reprobe_required"


# 6. Identity-incompatible measured evidence -> fail closed.
def test_06_identity_incompatible_measured_evidence_fails_closed():
    item = _measured_text("drifted", "digest-current")
    # A real precomputed field already caught a non-digest identity drift
    # (Task #8's own finding): the synthetic current-identity path only
    # independently re-observes digest, so this signal is the only way a
    # non-digest drift is visible to the typed authority at all.
    item["capability_identity_compatibility"] = {"compatible": False, "reason": "backend_changed"}
    assert campaign._judge_capability_authorized_families(item) == []
    assert campaign._judge_capability_rejection(item) == "capability_reprobe_required"


# 7. Ambiguous projection -> N/A by construction for the judge candidate
# path; see module docstring. Documented, not silently skipped.
def test_07_ambiguous_projection_is_out_of_scope_for_single_profile_judge_candidates():
    pytest.skip(
        "N/A by construction: a single judge candidate profile adapts to at "
        "most one CapabilityObservation per family, so AMBIGUOUS_COMPATIBLE_OBSERVATIONS "
        "(which requires >=2 disagreeing observations) cannot be reached "
        "through campaign.py's judge-candidate path. Covered directly at the "
        "projection layer by test_capability_projection.py::"
        "test_contradictory_compatible_observations_are_ambiguous and "
        "::test_ambiguous_decision_is_not_applicable."
    )


# 8. Measured unsupported -> capability ineligible.
def test_08_measured_unsupported_is_capability_ineligible():
    item = _bind({
        "name": "unsupported", "digest": "digest-unsupported",
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "measured_capabilities": {"text": {"state": "measured_unsupported"}},
    })
    assert campaign._judge_capability_authorized_families(item) == []
    assert campaign._judge_capability_rejection(item) is not None


# 9. Embedding-only candidate -> cannot become a text judge.
def test_09_embedding_only_candidate_cannot_become_text_judge():
    item = _bind({
        "name": "embedder", "digest": "digest-embed",
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "measured_capabilities": {"embedding": {"state": "measured_supported"}},
        "supported_families": ["embedding"],
    })
    assert "text" not in campaign._judge_capability_authorized_families(item)
    assert campaign._judge_capability_rejection(item) == "non_generative_embedding_only"
    selection = campaign.build_judge_selection([item], [], _policy(requested_primary="embedder"))
    assert selection.final_eligible_order == []
    assert selection.rejection_reasons[0]["reason"] == "non_generative_embedding_only"


# 10. Measured text supported + judge qualification failure -> qualification
# rejection, not capability rejection.
def test_10_qualification_failure_is_distinct_from_capability_rejection(monkeypatch):
    candidate = _measured_text("fails-qualification", "digest-q")
    assert campaign._judge_capability_rejection(candidate) is None  # capability itself is fine

    from llm_modelbench import judge_qualification

    monkeypatch.setattr(judge_qualification, "qualify_candidate", lambda client, cand, repeats=2: {
        "model": cand["name"], "digest": cand["digest"], "qualified": False,
        "aggregate_disposition": "malformed_structured_output", "protocol_version": "judge-qualification-v1",
        "checks": {},
    })
    result = campaign.qualify_judge(object(), candidate)
    assert result["qualified"] is False
    assert result["selection_rationale"] == "malformed_structured_output"
    assert result["selection_rationale"] not in {
        "capability_reprobe_required", "non_generative_embedding_only",
        "non_generative_vision_only", "unknown_or_non_generative_capability",
    }


# 11. Measured text supported + qualification pass + self/same-digest
# relation -> independence rejection, not capability rejection.
def test_11_same_digest_relation_is_independence_rejection_not_capability(monkeypatch):
    candidate = _measured_text("same-as-source", "digest-shared")
    assert campaign._judge_capability_rejection(candidate) is None

    monkeypatch.setattr(campaign, "qualify_judge", lambda client, cand, *, ledger=None: {
        "model": cand["name"], "digest": cand["digest"], "qualified": True,
        "aggregate_disposition": "qualified", "protocol_version": "judge-qualification-v1",
    })
    selection = campaign.build_judge_selection([candidate], [], _policy(requested_primary="same-as-source"))
    qualified, _ = campaign.select_qualified_campaign_judges(object(), selection)
    assert qualified != []

    row = {"model": "same-as-source", "digest": "digest-shared"}
    resolution = campaign.resolve_independent_judge_for_row(row, qualified)
    assert resolution["status"] == "awaiting_independent_judge"
    assert resolution["attempts"][0]["status"] == "rejected_self_identity"


def _fake_qualify_all(monkeypatch):
    # Real judge_qualification.qualify_candidate() runs the full
    # judge-qualification-v1 protocol (structured JSON scoring, rubric
    # consistency, repeat-stability) against a real client -- out of scope
    # to drive end-to-end here (already covered by
    # tests/test_stage1b_judge_qualification.py). Scenarios 12/13 are about
    # proving the *migrated capability gate* still feeds a working
    # fallback/exhaustion pipeline, so qualification itself is stubbed
    # "always qualified", matching test_stage1d_judge_integration.py's own
    # established pattern for this exact kind of test.
    monkeypatch.setattr(campaign, "qualify_judge", lambda client, cand, *, ledger=None: {
        "model": cand["name"], "digest": cand["digest"], "qualified": True,
        "aggregate_disposition": "qualified", "protocol_version": "judge-qualification-v1",
    })


# 12. First independent qualified judge structurally fails -> existing
# fallback remains intact, driven through the migrated capability gate.
def test_12_structural_failure_falls_back_to_next_independent_judge(monkeypatch, tmp_path):
    inventory = [
        _measured_text("preferred", "digest-preferred"),
        _measured_text("fallback", "digest-fallback"),
    ]
    selection = campaign.build_judge_selection(
        inventory, [{"name": "source", "digest": "digest-source"}],
        _policy(requested_primary="preferred", configured_fallbacks=("fallback",)),
    )
    assert [item["name"] for item in selection.final_eligible_order] == ["preferred", "fallback"]

    _fake_qualify_all(monkeypatch)
    qualified, _ = campaign.select_qualified_campaign_judges(object(), selection)
    assert [item["name"] for item in qualified] == ["preferred", "fallback"]

    run = _write_subjective_run(tmp_path, [{"model": "source", "digest": "digest-source"}])
    client = _RecordingClient(failures={"preferred": {"ok": False, "error": "unsupported request schema", "http_status": 415}})
    result = judge_dumps.judge_run(client, run, judge_model="preferred", qualified_judges=qualified)
    entry = result["entries"][0]

    assert client.calls == ["preferred", "fallback"]
    assert entry["status"] == "judged"
    assert entry["judge_model"] == "fallback"
    assert entry["judgement_attempts"][0]["status"] == "rejected_structural_incompatibility"


# 13. All usable independent judges exhausted -> existing terminal/waiting
# state preserved, no fake subjective score.
def test_13_exhausted_independent_judges_preserve_terminal_state_no_fake_score(monkeypatch, tmp_path):
    inventory = [_measured_text("only-judge", "digest-only")]
    selection = campaign.build_judge_selection(
        inventory, [{"name": "source", "digest": "digest-source"}], _policy(requested_primary="only-judge"),
    )
    _fake_qualify_all(monkeypatch)
    qualified, _ = campaign.select_qualified_campaign_judges(object(), selection)
    assert [item["name"] for item in qualified] == ["only-judge"]

    run = _write_subjective_run(tmp_path, [{"model": "source", "digest": "digest-source"}])
    client = _RecordingClient(failures={"only-judge": {"ok": False, "error": "unsupported request schema", "http_status": 415}})
    result = judge_dumps.judge_run(client, run, judge_model="only-judge", qualified_judges=qualified)
    entry = result["entries"][0]

    assert entry["status"] == "judge_exhausted_unavailable"
    assert entry.get("score") in (None, 0) or "score" not in entry
    assert entry["judgement_attempts"][0]["status"] == "rejected_structural_incompatibility"


# 14. Decisive proof #1: legacy eligibility function monkeypatched to lie
# YES, new typed authority genuinely says NO -> candidate remains
# capability-ineligible.
#
# `_judge_measured_text_state()` is patched directly (rather than the lower
# primitives `family_applicability()`/`capability_identity_compatibility()`,
# as repair.py's 2.6C decisive proofs do) because real judge candidates
# always take `_candidate_capability_identity_compatibility()`'s branch (2)
# -- trusting a precomputed field, never calling `capability_identity_compatibility()`
# directly (confirmed by the Stage 2.6D design note's traced cli.py call
# site) -- so patching those lower primitives would not actually make a
# realistic judge candidate's legacy decision lie. `_judge_measured_text_state()`
# is the actual, whole per-candidate legacy text-eligibility function this
# migration preserved specifically "as a regression oracle" (its own
# docstring); patching it directly is the faithful way to force *it* to lie
# for this component.
def test_14_positive_lie_from_legacy_cannot_admit_a_candidate_the_new_stack_blocks(monkeypatch):
    # Genuinely no measured text evidence at all -- the new stack must say NO.
    item = {
        "name": "no-real-evidence", "digest": "digest-lie-yes",
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
    }
    from llm_modelbench.capabilities import MeasuredCapabilityState

    monkeypatch.setattr(campaign, "_judge_measured_text_state", lambda candidate: MeasuredCapabilityState.MEASURED_SUPPORTED.value)

    # The legacy oracle now lies YES.
    assert campaign._judge_measured_text_state(item) == MeasuredCapabilityState.MEASURED_SUPPORTED.value

    # The real (unpatched) typed authority never calls
    # _judge_measured_text_state() at all -- it must still correctly refuse.
    assert campaign._judge_capability_authorized_families(item) == []
    assert campaign._judge_capability_rejection(item) is not None

    selection = campaign.build_judge_selection([item], [], _policy(requested_primary="no-real-evidence"))
    assert selection.final_eligible_order == []


# 15. Decisive proof #2: legacy eligibility function monkeypatched to lie
# NO, new typed authority genuinely says YES -> capability gate follows the
# new typed authority; qualification and independence still apply
# afterward.
def test_15_negative_lie_from_legacy_does_not_block_a_candidate_the_new_stack_admits(monkeypatch):
    # Genuinely compatible, measured-supported candidate -- the new stack
    # must say YES on its own evidence.
    item = _measured_text("genuinely-fine", "digest-lie-no")
    from llm_modelbench.capabilities import MeasuredCapabilityState

    monkeypatch.setattr(campaign, "_judge_measured_text_state", lambda candidate: MeasuredCapabilityState.MEASURED_UNSUPPORTED.value)

    # The legacy oracle now lies NO.
    assert campaign._judge_measured_text_state(item) != MeasuredCapabilityState.MEASURED_SUPPORTED.value

    # The real typed authority, unaffected by the patches, still admits it.
    assert "text" in campaign._judge_capability_authorized_families(item)
    assert campaign._judge_capability_rejection(item) is None

    selection = campaign.build_judge_selection([item], [], _policy(requested_primary="genuinely-fine"))
    assert [c["name"] for c in selection.final_eligible_order] == ["genuinely-fine"]

    # Qualification and independence still apply after the capability gate
    # admits the candidate -- the capability lie does not bypass them.
    monkeypatch.setattr(campaign, "qualify_judge", lambda client, cand, *, ledger=None: {
        "model": cand["name"], "digest": cand["digest"], "qualified": False,
        "aggregate_disposition": "rejected_structural_incompatibility", "protocol_version": "judge-qualification-v1",
    })
    qualified, qualifications = campaign.select_qualified_campaign_judges(object(), selection)
    assert qualified == []
    assert qualifications[0]["aggregate_disposition"] == "rejected_structural_incompatibility"
