"""Anvil Stage 3.5C: the judge capability-eligibility gate consumes native
authoritative ``EvidenceLedger`` evidence.

Stage 3.5B threaded an optional ``ledger`` through
``campaign._judge_capability_rejection`` /
``_judge_capability_authorized_families`` / ``build_judge_selection`` /
``qualify_judge`` / ``select_qualified_campaign_judges_for_rows`` and the
``judge_dumps`` manual ``--judge-model`` path, so the gate prefers a
current, identity-compatible native ``CapabilityObservation`` over the
legacy compatibility adapter (Stage 2.9 ``effective_measured_supported_families``
semantics).

Every assertion here goes through the *campaign / judge_dumps* functions
with a real ``CapabilityObservation`` written to a real ``EvidenceLedger``
-- never ``effective_measured_supported_families`` directly, which would
still pass with the Stage 3.5B wiring removed (the Stage 3.3C
vacuous-test lesson).

Fixtures mirror ``test_capability_evidence_adapter_effective_authority.py``
(the real ``interrogate_model`` profile shape); native observations are
appended straight from the same identity so the two cannot silently drift.
"""
from pathlib import Path

import pytest

from llm_modelbench import campaign, judge_dumps
from llm_modelbench.capabilities import (
    CAPABILITY_SCHEMA_VERSION,
    PROBE_PROTOCOL_VERSION,
    MeasuredCapabilityState,
)
from llm_modelbench.capabilities import _canonical_hash as legacy_canonical_hash
from llm_modelbench.capability_evidence_adapter import typed_identity_from_capability_identity
from llm_modelbench.capability_observation import (
    CapabilityObservation,
)
from llm_modelbench.capability_reprobe_execute import default_ledger_path
from llm_modelbench.evidence import EvidenceLedger


# --------------------------------------------------------------------------
# fixtures: a real interrogate_model-shaped capability identity + profile,
# and a native observation appended from the identical identity.
# --------------------------------------------------------------------------



# Anvil Stage 3B.2: append_capability_observation now requires an explicit
# EvidenceTrustClass (owner's frozen rule). These tests exercise ledger /
# projection behaviour, not trust classification, so they pass an explicit
# CANONICAL_COMPATIBLE via this thin shim rather than at every call site.
from llm_modelbench.capability_observation import append_capability_observation as _acobs_real
from llm_modelbench.evidence import EvidenceTrustClass as _ETC
# These files exercise ledger / projection / adapter behaviour, not the
# trust decision itself -- so they pin an explicit CANONICAL_COMPATIBLE
# rather than thread a trust class through every call site. The write-time
# trust decision is proven in tests/test_capability_trust.py and the three
# writer tests in tests/test_capability_reprobe_execute.py. The helper is
# deliberately NOT named like the real function so it cannot be mistaken
# for the production writer (which has no default and never assumes canonical).
def _append_with_explicit_canonical_trust(ledger, observation, *, trust_class=_ETC.CANONICAL_COMPATIBLE, provenance=()):
    return _acobs_real(ledger, observation, trust_class=trust_class, provenance=provenance)
append_capability_observation = _append_with_explicit_canonical_trust

def _template_config(*, num_ctx=8192):
    material = {
        "template": "{{ .System }}\n{{ .Prompt }}",
        "parameters": f"num_ctx {num_ctx}",
        "modelfile": None,
        "system": None,
        "model_info": {"llama.context_length": num_ctx},
    }
    return {"available": True, "hash": legacy_canonical_hash(material), "material": material}


def _capability_identity(*, digest="sha256:judge-a", canonical_name="judge-a", endpoint="http://127.0.0.1:11434"):
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "model": {
            "canonical_name": canonical_name,
            "backend_model_id": canonical_name,
            "digest": digest,
            "size": 9_000_000_000,
            "details": {"quantization_level": "Q4_K_M"},
        },
        "backend": {"backend": "ollama", "implementation": "OllamaClient", "endpoint": endpoint},
        "runtime": {"endpoint": endpoint, "implementation": "OllamaClient"},
        "template_config": _template_config(),
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "identity_hash": "irrelevant-composite-hash",
    }


def _judge_candidate(*, name="judge-a", digest="sha256:judge-a", legacy_text_state=MeasuredCapabilityState.MEASURED_SUPPORTED):
    """A judge candidate in the shape ``cli.py`` builds for the automatic
    campaign judging path: a stored capability profile spread with a
    freshly-observed digest and role fields. ``_candidate_current_capability_identity``
    resolves it via branch 2 (stored identity + fresh digest)."""
    identity = _capability_identity(digest=digest, canonical_name=name)
    return {
        "name": name,
        "digest": digest,
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "capability_identity": identity,
        "capability_identity_compatibility": {"compatible": True, "reason": "identity_match"},
        "measured_capabilities": {
            "text": {"state": legacy_text_state.value, "legacy_probe_state": "responded_ok", "route_scored_tasks": True},
        },
        "measured_supported_families": (
            ["text"] if legacy_text_state == MeasuredCapabilityState.MEASURED_SUPPORTED else []
        ),
        "declared_capabilities": ["completion", "tools"],
        "capabilities": ["completion"],
        "context": 8192,
        "priority": 0,
        "calibrated": False,
    }


def _native_observation(*, name="judge-a", digest="sha256:judge-a", family="text", state=MeasuredCapabilityState.MEASURED_SUPPORTED):
    typed = typed_identity_from_capability_identity(
        _capability_identity(digest=digest, canonical_name=name),
        protocol_version=PROBE_PROTOCOL_VERSION,
    )
    return CapabilityObservation(
        model_identity=typed.model_identity,
        runtime_profile_identity=typed.runtime_profile_identity,
        capability=family,
        result=state,
        probe_protocol_version=PROBE_PROTOCOL_VERSION,
        capability_schema_version=CAPABILITY_SCHEMA_VERSION,
        template_config_hash=typed.template_hash,
        endpoint_identity=typed.endpoint_identity,
    )


def _policy(**kwargs):
    defaults = {
        "requested_primary": None, "configured_fallbacks": (), "excluded_families": (),
        "allow_excluded_primary": False, "automatic_selection": True, "enabled": True,
    }
    defaults.update(kwargs)
    return campaign.JudgePolicy(**defaults)


# --------------------------------------------------------------------------
# 1. ledger=None keeps the pre-3.5 legacy-adapter behaviour, byte for byte.
# --------------------------------------------------------------------------


def test_no_ledger_is_identical_to_the_pre_3_5_legacy_path():
    # A legacy profile with measured_supported text + a compatible identity
    # is admitted by the legacy adapter alone (Stage 2.6D behaviour).
    # ledger=None must reproduce that exactly.
    supported = _judge_candidate(legacy_text_state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    assert campaign._judge_capability_authorized_families(supported) == ["text"]
    assert campaign._judge_capability_rejection(supported) is None
    assert campaign._judge_capability_rejection(supported, ledger=None) is None

    # A legacy profile with measured_unsupported text is rejected -- same
    # with and without an explicit ledger=None.
    unsupported = _judge_candidate(legacy_text_state=MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    assert campaign._judge_capability_authorized_families(unsupported) == []
    assert campaign._judge_capability_rejection(unsupported) is not None
    assert campaign._judge_capability_rejection(unsupported, ledger=None) is not None


# --------------------------------------------------------------------------
# 2. native SELECTED evidence is what admits a judge -- and it flows through
#    the campaign functions, not effective_measured_supported_families.
# --------------------------------------------------------------------------


def test_native_supported_observation_admits_a_legacy_unsupported_judge(tmp_path: Path):
    # Legacy profile: measured_unsupported (legacy alone would reject).
    # Native ledger: measured_supported -> native wins, judge admitted.
    candidate = _judge_candidate(legacy_text_state=MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    append_capability_observation(
        ledger, _native_observation(state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    )

    assert "text" in campaign._judge_capability_authorized_families(candidate, ledger=ledger)
    assert campaign._judge_capability_rejection(candidate, ledger=ledger) is None

    selection = campaign.build_judge_selection(
        [_judge_candidate(legacy_text_state=MeasuredCapabilityState.MEASURED_UNSUPPORTED)],
        [], _policy(requested_primary="judge-a"), ledger=ledger,
    )
    assert [item["name"] for item in selection.final_eligible_order] == ["judge-a"]


def test_without_the_ledger_the_legacy_unsupported_candidate_stays_rejected(tmp_path: Path):
    # Same candidate, same native evidence written -- but no ledger passed:
    # the native SUPPORTED observation is invisible and the gate falls
    # closed, proving the ledger is load-bearing.
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    append_capability_observation(
        ledger, _native_observation(state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    )
    selection_no_ledger = campaign.build_judge_selection(
        [_judge_candidate(legacy_text_state=MeasuredCapabilityState.MEASURED_UNSUPPORTED)],
        [], _policy(requested_primary="judge-a"),
    )
    assert selection_no_ledger.final_eligible_order == []
    assert selection_no_ledger.rejection_reasons[0]["reason"] == "unknown_or_non_generative_capability"


# --------------------------------------------------------------------------
# 3. native fails closed where legacy would admit (the restrictive
#    direction -- the one an operator would be most surprised by).
# --------------------------------------------------------------------------


def test_ambiguous_native_evidence_fails_closed_even_though_legacy_says_supported(tmp_path: Path):
    # Legacy profile: measured_supported. Two contradictory compatible native
    # observations -> native is ambiguous -> fail closed.
    candidate = _judge_candidate(legacy_text_state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    append_capability_observation(
        ledger, _native_observation(state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    )
    append_capability_observation(
        ledger, _native_observation(state=MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    )

    # Without a ledger the legacy adapter alone still does not positively
    # authorize (no native evidence at all) -- so to show the *restrictive*
    # effect we compare a ledger with a single SELECTED observation (admits)
    # against the ambiguous ledger (fails closed).
    admitting = EvidenceLedger(tmp_path / "admitting.jsonl")
    append_capability_observation(
        admitting, _native_observation(state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    )
    assert campaign._judge_capability_rejection(candidate, ledger=admitting) is None
    assert campaign._judge_capability_rejection(candidate, ledger=ledger) == "capability_reprobe_required"


def test_native_negative_overrides_a_legacy_positive(tmp_path: Path):
    candidate = _judge_candidate(legacy_text_state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    append_capability_observation(
        ledger, _native_observation(state=MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    )
    assert "text" not in campaign._judge_capability_authorized_families(candidate, ledger=ledger)
    assert campaign._judge_capability_rejection(candidate, ledger=ledger) is not None


# --------------------------------------------------------------------------
# 4. native-vs-legacy disagreement is surfaced additively -- recorded,
#    never a silent resolution -- and reason strings are unchanged.
# --------------------------------------------------------------------------


def test_disagreement_surfaces_as_capability_projection_drift_on_the_rejection(tmp_path: Path):
    # Legacy says supported; native says unsupported -> native wins (reject),
    # and the disagreement rides along as an additive field.
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    append_capability_observation(
        ledger, _native_observation(state=MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    )
    selection = campaign.build_judge_selection(
        [_judge_candidate(legacy_text_state=MeasuredCapabilityState.MEASURED_SUPPORTED)],
        [], _policy(requested_primary="judge-a"), ledger=ledger,
    )
    assert selection.final_eligible_order == []
    entry = selection.rejection_reasons[0]
    # Reason vocabulary unchanged (asserted elsewhere too -- pin it here).
    assert entry["reason"] in {
        "capability_reprobe_required",
        "non_generative_embedding_only",
        "unknown_or_non_generative_capability",
    }
    drift = entry["capability_projection_drift"]
    assert len(drift) == 1
    assert drift[0]["family"] == "text"
    assert drift[0]["native_applicable"] is False
    assert drift[0]["legacy_applicable"] is True


def test_disagreement_surfaces_on_an_admitted_candidate_too(tmp_path: Path):
    # Native says supported, legacy says unsupported -> admitted, but the
    # drift is still recorded on the eligible entry, not hidden.
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    append_capability_observation(
        ledger, _native_observation(state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    )
    selection = campaign.build_judge_selection(
        [_judge_candidate(legacy_text_state=MeasuredCapabilityState.MEASURED_UNSUPPORTED)],
        [], _policy(requested_primary="judge-a"), ledger=ledger,
    )
    assert [item["name"] for item in selection.final_eligible_order] == ["judge-a"]
    drift = selection.final_eligible_order[0]["capability_projection_drift"]
    assert drift[0]["family"] == "text"
    assert drift[0]["native_applicable"] is True
    assert drift[0]["legacy_applicable"] is False


def test_no_drift_key_when_ledger_is_absent(tmp_path: Path):
    selection = campaign.build_judge_selection(
        [_judge_candidate()], [], _policy(requested_primary="judge-a")
    )
    for entry in selection.rejection_reasons:
        assert "capability_projection_drift" not in entry


# --------------------------------------------------------------------------
# 5. ledger passthrough: the ledger reaches qualify_judge via
#    select_qualified_campaign_judges_for_rows -- so ledger=ledger at the
#    internal call sites cannot be silently deleted.
# --------------------------------------------------------------------------


def test_ledger_is_threaded_into_qualify_judge_for_rows(tmp_path: Path, monkeypatch):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    append_capability_observation(
        ledger, _native_observation(state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    )
    selection = campaign.build_judge_selection(
        [_judge_candidate()], [], _policy(requested_primary="judge-a"), ledger=ledger
    )
    assert selection.final_eligible_order  # native evidence admitted it

    seen = {}

    def _spy_qualify(client, candidate, *, repeats=2, ledger=None):
        seen["ledger"] = ledger
        return {
            "model": candidate["name"], "digest": candidate["digest"], "qualified": True,
            "aggregate_disposition": "qualified", "protocol_version": "judge-qualification-v1",
        }

    monkeypatch.setattr(campaign, "qualify_judge", _spy_qualify)
    source_rows = [{"model": "source-model", "digest": "digest-source", "task": "t", "run_id": "r"}]
    campaign.select_qualified_campaign_judges_for_rows(object(), selection, source_rows, ledger=ledger)
    assert seen["ledger"] is ledger


def test_ledger_is_threaded_into_qualify_judge_single(tmp_path: Path, monkeypatch):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    append_capability_observation(
        ledger, _native_observation(state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    )
    selection = campaign.build_judge_selection(
        [_judge_candidate()], [], _policy(requested_primary="judge-a"), ledger=ledger
    )
    seen = {}

    def _spy_qualify(client, candidate, *, repeats=2, ledger=None):
        seen["ledger"] = ledger
        return {"model": candidate["name"], "digest": candidate["digest"], "qualified": True,
                "aggregate_disposition": "qualified"}

    monkeypatch.setattr(campaign, "qualify_judge", _spy_qualify)
    campaign.select_qualified_campaign_judge(object(), selection, ledger=ledger)
    assert seen["ledger"] is ledger


def test_ledger_is_threaded_into_the_structural_continuation_path(tmp_path: Path, monkeypatch):
    # continue_qualification_after_runtime_structural_failure re-qualifies
    # the Stage 1A tail after a runtime structural failure -- its qualify_judge
    # call must also carry the ledger derived from the run being judged.
    run_dir = tmp_path / "runs" / "primary"
    run_dir.mkdir(parents=True)
    ledger = EvidenceLedger(default_ledger_path(run_dir.parent))
    append_capability_observation(
        ledger, _native_observation(name="judge-b", digest="sha256:judge-b",
                                    state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    )
    selection = campaign.build_judge_selection(
        [_judge_candidate(name="judge-b", digest="sha256:judge-b")],
        [], _policy(requested_primary="judge-b"), ledger=ledger,
    )
    assert selection.final_eligible_order

    seen = {}

    def _spy_qualify(client, candidate, *, repeats=2, ledger=None):
        seen["ledger"] = ledger
        return {"model": candidate["name"], "digest": candidate["digest"], "qualified": True,
                "aggregate_disposition": "qualified"}

    monkeypatch.setattr(campaign, "qualify_judge", _spy_qualify)
    source_rows = [{"model": "source-model", "digest": "digest-source"}]
    structural_entries = [{
        "source_row_hash": judge_dumps.source_row_hash(source_rows[0]),
        "status": "judge_exhausted_unavailable",
        "failure_disposition": "structural_incompatibility",
        "judgement_attempts": [],
    }]
    campaign.continue_qualification_after_runtime_structural_failure(
        object(), selection, [], [], source_rows, structural_entries, ledger=ledger,
    )
    assert seen.get("ledger") is ledger


# --------------------------------------------------------------------------
# 6. the manual --judge-model path -- its ONLY capability gate -- consumes
#    the native ledger under the run being judged.
# --------------------------------------------------------------------------


class _FakeManualClient:
    """Enough of an InferenceClient for build_manual_judge_candidate."""

    def __init__(self, digest="sha256:judge-a"):
        self._digest = digest

    def tags(self):
        return [{"name": "judge-a", "digest": self._digest}]

    def capability_hints(self, name):
        return ["completion"]

    def version(self):
        return {"version": "test"}


def _patch_manual_candidate(monkeypatch, *, legacy_state):
    def _fake_build(client, judge_model):
        cand = _judge_candidate(name=judge_model, legacy_text_state=legacy_state)
        cand["current_capability_identity"] = _capability_identity(canonical_name=judge_model)
        cand["manual_designation"] = True
        return cand

    monkeypatch.setattr(campaign, "build_manual_judge_candidate", _fake_build)


def test_manual_judge_pool_consumes_the_run_ledger(tmp_path: Path, monkeypatch):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "primary"
    run_dir.mkdir(parents=True)
    ledger = EvidenceLedger(default_ledger_path(runs_dir))
    append_capability_observation(
        ledger, _native_observation(name="judge-a", state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    )
    _patch_manual_candidate(monkeypatch, legacy_state=MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    monkeypatch.setattr(
        campaign, "qualify_judge",
        lambda client, candidate, *, ledger=None: {"model": candidate["name"], "digest": candidate["digest"],
                                          "qualified": True, "aggregate_disposition": "qualified"},
    )

    pool = judge_dumps._resolve_manual_judge_pool(
        _FakeManualClient(), "judge-a", ledger=campaign._run_capability_ledger(run_dir)
    )
    assert [j["name"] for j in pool] == ["judge-a"]


def test_manual_judge_without_native_evidence_is_rejected(tmp_path: Path, monkeypatch):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "primary"
    run_dir.mkdir(parents=True)
    # Empty ledger under the run + a legacy-unsupported candidate: neither
    # the native ledger nor the legacy adapter positively authorizes -> the
    # manual candidate is capability-ineligible.
    _patch_manual_candidate(monkeypatch, legacy_state=MeasuredCapabilityState.MEASURED_UNSUPPORTED)

    with pytest.raises(judge_dumps.ManualJudgeIneligibleError, match="capability-eligible"):
        judge_dumps._resolve_manual_judge_pool(
            _FakeManualClient(), "judge-a", ledger=campaign._run_capability_ledger(run_dir)
        )


# --------------------------------------------------------------------------
# 7. master_summary.json is not a judge-authority input -- changing it does
#    not move a judge decision.
# --------------------------------------------------------------------------


def test_master_summary_change_does_not_move_a_judge_decision(tmp_path: Path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    append_capability_observation(
        ledger, _native_observation(state=MeasuredCapabilityState.MEASURED_SUPPORTED)
    )
    before = campaign.build_judge_selection(
        [_judge_candidate()], [], _policy(requested_primary="judge-a"), ledger=ledger
    )
    # A candidate dict carrying an arbitrary master_summary-derived blob must
    # not change the outcome -- the gate reads native evidence + identity only.
    poisoned = _judge_candidate()
    poisoned["master_summary"] = {"rows": [{"model": "judge-a", "verdict": "banned"}]}
    poisoned["ranking"] = {"score": -999}
    after = campaign.build_judge_selection(
        [poisoned], [], _policy(requested_primary="judge-a"), ledger=ledger
    )
    assert [i["name"] for i in before.final_eligible_order] == [i["name"] for i in after.final_eligible_order]
