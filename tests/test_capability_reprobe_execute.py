"""Anvil Stage 2.7C: reprobe execution + validation.

Design rationale (before/after metrics split, ambiguous-vs-conflict
handling) is recorded in ``local_only/anvil/stage-2.7C-execution.md``, not
repeated here.
"""
import inspect

from llm_modelbench.capabilities import CAPABILITY_SCHEMA_VERSION, PROBE_PROTOCOL_VERSION, MeasuredCapabilityState
from llm_modelbench.capability_evidence_classification import EvidenceCellStatus, classify_fleet
from llm_modelbench.capability_observation import CapabilityObservation
from llm_modelbench.capability_projection import CapabilityProjectionStatus, project_capability_from_ledger
from llm_modelbench import capability_reprobe_execute as execute_mod
from llm_modelbench.capability_reprobe_execute import (
    default_ledger_path,
    execute_reprobe_actions,
    run_reprobe_execution,
)
from llm_modelbench.capability_reprobe_plan import ReprobeAction, ReprobeActionKind, plan_fleet_reprobes
from llm_modelbench.cli import build_parser, cmd_reprobe_execute
from llm_modelbench.config import Config
from llm_modelbench.evidence import EvidenceLedger, ProvenanceLink, ProvenanceRelation
from llm_modelbench.identity import ModelArtifactIdentity, RuntimeProfileIdentity
from llm_modelbench.ollama import MockClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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

def _model_identity(*, digest="digest-1"):
    return ModelArtifactIdentity(
        artifact_set_id=digest, primary_sha256=digest, size_bytes=9_000_000_000,
        format="ollama-blob", quantization="Q4_K_M", source="probe-model",
    )


def _runtime_identity(*, backend="ollama", template_hash="template-hash-1"):
    return RuntimeProfileIdentity(
        backend=backend, backend_version="0.5.1", protocol_version="capability-smoke-v2",
        template_hash=template_hash, runtime_configuration_hash="cfg-1",
    )


def _observation(**overrides):
    kwargs = dict(
        model_identity=_model_identity(), runtime_profile_identity=_runtime_identity(),
        capability="text", result=MeasuredCapabilityState.MEASURED_SUPPORTED,
        probe_protocol_version="capability-smoke-v2", capability_schema_version=2,
    )
    kwargs.update(overrides)
    return CapabilityObservation(**kwargs)


def _action(model="probe-model", capability="text", action=ReprobeActionKind.REPROBE, **overrides):
    kwargs = dict(
        model=model, capability=capability, classification="missing",
        classification_reason="fixture", action=action, action_reason="fixture",
        current_model_artifact_set_id=None, current_model_primary_sha256=None,
        current_runtime_profile_stable_key=None, current_backend=None,
        previous_evidence_hash=None, considered_evidence_hashes=(),
        typed_decision_reason=None, legacy_compatibility_reason=None, source_paths=(),
    )
    kwargs.update(overrides)
    return ReprobeAction(**kwargs)


class _FakeClient:
    """Minimal duck-typed client, in the style of
    tests/test_capability_workflow.py's inline fake classes -- only
    implements what interrogate_model(functional=True) actually calls for
    the families exercised here (text, tools)."""

    def __init__(self, *, digest="digest-1", chat_response=None, tools_response=None):
        self._digest = digest
        self._chat_response = chat_response or {"ok": True, "text": "AIW_TEXT_OK"}
        self._tools_response = tools_response or {"ok": False, "error": "not supported", "tool_calls": []}

    def tags(self):
        return [{"name": "probe-model", "digest": self._digest}]

    def capability_hints(self, model):
        return []

    def chat(self, model, prompt, **kwargs):
        return dict(self._chat_response)

    def chat_tools(self, *args, **kwargs):
        return dict(self._tools_response)


def _current_identity_kwargs(observation):
    return dict(
        current_model_identity=observation.model_identity,
        current_runtime_profile_identity=observation.runtime_profile_identity,
        current_probe_protocol_version=PROBE_PROTOCOL_VERSION,
        current_capability_schema_version=CAPABILITY_SCHEMA_VERSION,
        current_template_config_hash=observation.template_config_hash,
        current_endpoint_identity=observation.endpoint_identity,
    )


def _patch_fixed_observation(monkeypatch, observation):
    """Most of this file tests _execute_one's post-probe logic (prior-state
    lookup, supersession, after-state recomputation) against a hand-built
    ledger, not the real probe/adapter pipeline -- that end-to-end path is
    covered separately by the MockClient-based tests below. Patching
    interrogate_model/adapt_legacy_profile_family_to_observation to a fixed
    observation keeps identity fields deterministic and independent of
    whatever a fake client's client.show()/tags() shape happens to hash to."""
    monkeypatch.setattr(execute_mod, "interrogate_model", lambda *a, **k: {"stub": "profile"})
    monkeypatch.setattr(execute_mod, "adapt_legacy_profile_family_to_observation", lambda profile, family: observation)


# ---------------------------------------------------------------------------
# NO_ACTION cells are never touched
# ---------------------------------------------------------------------------

def test_no_action_cells_never_probed_or_appended(tmp_path):
    class _RaisingClient(_FakeClient):
        def chat(self, model, prompt, **kwargs):
            raise AssertionError("must not be probed: this cell was NO_ACTION")

    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    actions = (_action(action=ReprobeActionKind.NO_ACTION),)
    outcomes = execute_reprobe_actions(actions, _RaisingClient(), ledger)
    assert outcomes == ()
    assert list(ledger.all()) == []


def test_mixed_plan_only_touches_reprobe_actions(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    actions = (
        _action(model="probe-model", capability="text", action=ReprobeActionKind.REPROBE),
        _action(model="other-model", capability="tools", action=ReprobeActionKind.NO_ACTION),
    )
    client = _FakeClient()
    outcomes = execute_reprobe_actions(actions, client, ledger)
    assert len(outcomes) == 1
    assert outcomes[0].model == "probe-model"
    assert outcomes[0].appended is True


# ---------------------------------------------------------------------------
# First reprobe: nothing to supersede. Second: supersedes the first.
# ---------------------------------------------------------------------------

def test_first_reprobe_of_a_cell_has_nothing_to_supersede(tmp_path, monkeypatch):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    _patch_fixed_observation(monkeypatch, _observation())
    outcome = execute_reprobe_actions((_action(),), _FakeClient(), ledger)[0]
    assert outcome.appended is True
    assert outcome.prior_status == EvidenceCellStatus.MISSING.value
    assert outcome.superseded_record_ids == ()
    assert outcome.after_status == EvidenceCellStatus.CURRENT_VALID.value


def test_second_reprobe_of_the_same_cell_supersedes_the_first(tmp_path, monkeypatch):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    _patch_fixed_observation(monkeypatch, _observation(timestamp="2026-08-10T00:00:00Z"))
    first = execute_reprobe_actions((_action(),), _FakeClient(), ledger)[0]
    assert first.appended is True

    _patch_fixed_observation(monkeypatch, _observation(timestamp="2026-08-11T00:00:00Z"))
    second = execute_reprobe_actions((_action(),), _FakeClient(), ledger)[0]
    assert second.appended is True
    assert second.superseded_record_ids == (first.observation_id,)
    assert second.prior_status == EvidenceCellStatus.CURRENT_VALID.value
    assert second.after_status == EvidenceCellStatus.CURRENT_VALID.value

    # The terminal record for this cell is now the second observation, not
    # the first -- the projection resolves to exactly one selected id.
    projection = project_capability_from_ledger(
        ledger, capability="text", **_current_identity_kwargs(_observation()),
    )
    assert projection.status == CapabilityProjectionStatus.SELECTED
    assert projection.selected_record_id == second.observation_id


def test_no_ledger_mutation_when_probe_result_cannot_be_adapted(tmp_path, monkeypatch):
    monkeypatch.setattr(execute_mod, "interrogate_model", lambda *a, **k: {"stub": "profile"})
    monkeypatch.setattr(execute_mod, "adapt_legacy_profile_family_to_observation", lambda profile, family: None)
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    outcome = execute_reprobe_actions((_action(),), _FakeClient(), ledger)[0]
    assert outcome.appended is False
    assert outcome.error is None
    assert outcome.skip_reason
    assert list(ledger.all()) == []


def test_probe_exception_is_recorded_not_raised(tmp_path, monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("backend unreachable")

    monkeypatch.setattr(execute_mod, "interrogate_model", _raise)
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    outcome = execute_reprobe_actions((_action(),), _FakeClient(), ledger)[0]
    assert outcome.appended is False
    assert "backend unreachable" in outcome.error
    assert list(ledger.all()) == []


# ---------------------------------------------------------------------------
# Ambiguous prior -> supersede every predecessor. Structural conflict -> skip.
# ---------------------------------------------------------------------------

def test_ambiguous_prior_is_resolved_by_superseding_every_predecessor(tmp_path, monkeypatch):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    # Two disagreeing, mutually-compatible observations with no supersession
    # link between them -- a standing AMBIGUOUS_COMPATIBLE_OBSERVATIONS prior.
    a = append_capability_observation(ledger, _observation(timestamp="2026-08-10T00:00:00Z", result=MeasuredCapabilityState.MEASURED_SUPPORTED))
    b = append_capability_observation(ledger, _observation(timestamp="2026-08-11T00:00:00Z", result=MeasuredCapabilityState.MEASURED_UNSUPPORTED))
    projection = project_capability_from_ledger(ledger, capability="text", **_current_identity_kwargs(_observation()))
    assert projection.status == CapabilityProjectionStatus.AMBIGUOUS_COMPATIBLE_OBSERVATIONS
    assert {a.record_id, b.record_id} == set(projection.considered_observation_ids)

    _patch_fixed_observation(monkeypatch, _observation(timestamp="2026-08-12T00:00:00Z"))
    outcome = execute_reprobe_actions((_action(),), _FakeClient(), ledger)[0]
    assert outcome.appended is True
    assert outcome.prior_status == EvidenceCellStatus.AMBIGUOUS_COMPATIBLE_OBSERVATIONS.value
    assert set(outcome.superseded_record_ids) == {a.record_id, b.record_id}
    assert outcome.after_status == EvidenceCellStatus.CURRENT_VALID.value

    after = project_capability_from_ledger(ledger, capability="text", **_current_identity_kwargs(_observation()))
    assert after.status == CapabilityProjectionStatus.SELECTED
    assert after.selected_record_id == outcome.observation_id


def test_preexisting_supersession_conflict_is_skipped_not_repaired(tmp_path, monkeypatch):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    # A genuine fork: two records both claim to supersede the same predecessor.
    root = append_capability_observation(ledger, _observation(timestamp="2026-08-10T00:00:00Z"))
    append_capability_observation(
        ledger, _observation(timestamp="2026-08-11T00:00:00Z"),
        provenance=(ProvenanceLink(ProvenanceRelation.SUPERSEDES, root.record_id),),
    )
    append_capability_observation(
        ledger, _observation(timestamp="2026-08-12T00:00:00Z"),
        provenance=(ProvenanceLink(ProvenanceRelation.SUPERSEDES, root.record_id),),
    )
    before_ids = {record.record_id for record in ledger.all()}

    _patch_fixed_observation(monkeypatch, _observation(timestamp="2026-08-13T00:00:00Z"))
    outcome = execute_reprobe_actions((_action(),), _FakeClient(), ledger)[0]
    assert outcome.appended is False
    assert outcome.error is None
    assert "SUPERSESSION_CONFLICT" in outcome.skip_reason
    assert outcome.prior_status == EvidenceCellStatus.SUPERSESSION_CONFLICT.value
    # Nothing was appended over the conflict.
    assert {record.record_id for record in ledger.all()} == before_ids


# ---------------------------------------------------------------------------
# Anvil Stage 2.9: full lifecycle closure. MISSING -> REPROBE planned -> real
# probe executed -> native CapabilityObservation appended -> next
# classification reports CURRENT_VALID (or a valid terminal negative), not
# MISSING again -> next reprobe-plan reports NO_ACTION. Exercises the real
# end-to-end pipeline (classify_fleet -> plan_fleet_reprobes ->
# run_reprobe_execution -> classify_fleet -> plan_fleet_reprobes again), not
# a patched fixed observation -- this is the exact property advice item 4
# requires proven, not merely asserted.
# ---------------------------------------------------------------------------

def test_full_lifecycle_positive_missing_to_current_valid_to_no_action(tmp_path):
    runs_dir = tmp_path / "runs"
    campaigns_dir = tmp_path / "campaigns"
    runs_dir.mkdir()
    campaigns_dir.mkdir()
    client = _FakeClient()  # chat() -> {"ok": True, ...}: a genuine measured positive for "text"

    before = classify_fleet(client, runs_dir=runs_dir, campaigns_root=campaigns_dir)
    before_cell = next(c for c in before.cells if c.model == "probe-model" and c.capability == "text")
    assert before_cell.status == EvidenceCellStatus.MISSING

    plan = plan_fleet_reprobes(client, runs_dir=runs_dir, campaigns_root=campaigns_dir)
    actions = plan.filtered(model="probe-model", capability="text", only_required=True)
    assert len(actions) == 1
    assert actions[0].action == ReprobeActionKind.REPROBE

    ledger = EvidenceLedger(default_ledger_path(runs_dir))
    report = run_reprobe_execution(
        plan, client, ledger, runs_dir=runs_dir, campaigns_root=campaigns_dir, actions=actions,
    )
    assert report.native_evidence_after["appended"] == 1

    after = classify_fleet(client, runs_dir=runs_dir, campaigns_root=campaigns_dir)
    after_cell = next(c for c in after.cells if c.model == "probe-model" and c.capability == "text")
    assert after_cell.status == EvidenceCellStatus.CURRENT_VALID
    assert after_cell.typed_decision_reason == "measured_supported"
    assert after_cell.reason.startswith("selected native EvidenceLedger observation")

    plan2 = plan_fleet_reprobes(client, runs_dir=runs_dir, campaigns_root=campaigns_dir)
    actions2 = plan2.filtered(model="probe-model", capability="text", only_required=True)
    assert actions2 == ()


def test_full_lifecycle_negative_missing_to_terminal_negative_to_no_action(tmp_path):
    runs_dir = tmp_path / "runs"
    campaigns_dir = tmp_path / "campaigns"
    runs_dir.mkdir()
    campaigns_dir.mkdir()
    client = _FakeClient()  # chat_tools() -> {"ok": False, ...} by default: a genuine measured negative for "tools"

    before = classify_fleet(client, runs_dir=runs_dir, campaigns_root=campaigns_dir)
    before_cell = next(c for c in before.cells if c.model == "probe-model" and c.capability == "tools")
    assert before_cell.status == EvidenceCellStatus.MISSING

    plan = plan_fleet_reprobes(client, runs_dir=runs_dir, campaigns_root=campaigns_dir)
    actions = plan.filtered(model="probe-model", capability="tools", only_required=True)
    assert len(actions) == 1

    ledger = EvidenceLedger(default_ledger_path(runs_dir))
    report = run_reprobe_execution(
        plan, client, ledger, runs_dir=runs_dir, campaigns_root=campaigns_dir, actions=actions,
    )
    assert report.native_evidence_after["appended"] == 1

    after = classify_fleet(client, runs_dir=runs_dir, campaigns_root=campaigns_dir)
    after_cell = next(c for c in after.cells if c.model == "probe-model" and c.capability == "tools")
    # A valid terminal negative -- not MISSING, not PROBE_INCONCLUSIVE (which
    # would still be reprobe-worthy, per advice item 5).
    assert after_cell.status in (EvidenceCellStatus.MEASURED_UNSUPPORTED, EvidenceCellStatus.BACKEND_UNSUPPORTED)

    plan2 = plan_fleet_reprobes(client, runs_dir=runs_dir, campaigns_root=campaigns_dir)
    actions2 = plan2.filtered(model="probe-model", capability="tools", only_required=True)
    assert actions2 == ()


def test_native_probe_inconclusive_remains_reprobe_worthy_not_a_terminal_status(tmp_path):
    # A native PROBE_INCONCLUSIVE observation (metadata-only/non-functional)
    # must classify as PROBE_INCONCLUSIVE, not a fabricated CURRENT_VALID or
    # terminal negative -- and PROBE_INCONCLUSIVE must stay reprobe-worthy
    # (per advice item 5), unlike a genuine measured terminal state.
    from llm_modelbench.capability_evidence_classification import REPROBE_NOT_REQUIRED, classify_model_capability

    assert EvidenceCellStatus.PROBE_INCONCLUSIVE not in REPROBE_NOT_REQUIRED

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    ledger = EvidenceLedger(default_ledger_path(runs_dir))
    inconclusive = _observation(result=MeasuredCapabilityState.PROBE_INCONCLUSIVE)
    append_capability_observation(ledger, inconclusive)

    cell = classify_model_capability(
        "probe-model", "text", [], current_identity=None, ledger=ledger,
    )
    # No current_identity means no native lookup happens at all (see
    # classify_model_capability's own guard) -- this asserts that guard
    # rather than a native match, so cover the identity-present path too via
    # a real classify_fleet() call against a client whose identity matches.
    assert cell.status == EvidenceCellStatus.MISSING

    class _InconclusiveClient(_FakeClient):
        pass

    client = _InconclusiveClient()
    # Reuse the exact identity classify_fleet() itself would compute for
    # "probe-model" against this client, then append a matching native
    # PROBE_INCONCLUSIVE observation under that identity.
    from llm_modelbench.capabilities import current_capability_identity
    from llm_modelbench.capability_evidence_adapter import typed_identity_from_capability_identity

    live_identity = current_capability_identity(client, "probe-model")
    typed = typed_identity_from_capability_identity(live_identity, protocol_version=PROBE_PROTOCOL_VERSION)
    matching_observation = CapabilityObservation(
        model_identity=typed.model_identity,
        runtime_profile_identity=typed.runtime_profile_identity,
        capability="text",
        result=MeasuredCapabilityState.PROBE_INCONCLUSIVE,
        probe_protocol_version=PROBE_PROTOCOL_VERSION,
        capability_schema_version=CAPABILITY_SCHEMA_VERSION,
        template_config_hash=typed.template_hash,
        endpoint_identity=typed.endpoint_identity,
    )
    ledger2 = EvidenceLedger(default_ledger_path(runs_dir).with_name("ledger2.jsonl"))
    append_capability_observation(ledger2, matching_observation)

    report = classify_fleet(client, runs_dir=runs_dir, campaigns_root=tmp_path / "campaigns", ledger=ledger2)
    cell2 = next(c for c in report.cells if c.model == "probe-model" and c.capability == "text")
    assert cell2.status == EvidenceCellStatus.PROBE_INCONCLUSIVE
    assert cell2.status not in REPROBE_NOT_REQUIRED


# ---------------------------------------------------------------------------
# Before/after fleet metrics: two separate axes, per stage-2.7C-execution.md
# decision 1 -- the legacy axis is provably unchanged by construction.
# ---------------------------------------------------------------------------

def test_legacy_fleet_axis_is_unchanged_by_execution(tmp_path):
    runs_dir = tmp_path / "runs"
    campaigns_dir = tmp_path / "campaigns"
    runs_dir.mkdir()
    campaigns_dir.mkdir()
    client = MockClient()
    plan = plan_fleet_reprobes(client, runs_dir=runs_dir, campaigns_root=campaigns_dir)
    before = classify_fleet(client, runs_dir=runs_dir, campaigns_root=campaigns_dir)

    actions = plan.filtered(model="llama3.1:8b", capability="text", only_required=True)
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    report = run_reprobe_execution(
        plan, client, ledger, runs_dir=runs_dir, campaigns_root=campaigns_dir, actions=actions,
    )
    assert report.native_evidence_after["appended"] == 1

    after = classify_fleet(client, runs_dir=runs_dir, campaigns_root=campaigns_dir)
    # capability_report.json was never touched, so the legacy-evidence-axis
    # classification is byte-for-byte identical before and after, even
    # though native ledger evidence now exists for the reprobed cell.
    assert after.to_dict() == before.to_dict()
    assert report.fleet_before == before.to_dict()


def test_native_evidence_summary_breaks_down_by_capability(tmp_path):
    runs_dir = tmp_path / "runs"
    campaigns_dir = tmp_path / "campaigns"
    runs_dir.mkdir()
    campaigns_dir.mkdir()
    client = MockClient()
    plan = plan_fleet_reprobes(client, runs_dir=runs_dir, campaigns_root=campaigns_dir)
    actions = plan.filtered(model="llama3.1:8b", capability="text", only_required=True)
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    report = run_reprobe_execution(
        plan, client, ledger, runs_dir=runs_dir, campaigns_root=campaigns_dir, actions=actions,
    )
    assert report.native_evidence_after["by_capability"]["text"]["appended"] == 1
    assert report.native_evidence_after["by_capability"]["vision"]["appended"] == 0


# ---------------------------------------------------------------------------
# No model deletion path exists anywhere in this module.
# ---------------------------------------------------------------------------

def test_module_contains_no_deletion_or_pruning_logic():
    """No call-shaped deletion/pruning pattern anywhere in the module.
    (The prose docstring legitimately uses the word "deletes" to state
    this as a non-goal -- that is not itself a deletion call, so this
    checks for call syntax, not the bare word.)"""
    source = inspect.getsource(execute_mod)
    for token in ("rmtree(", ".unlink(", "os.remove(", "client.delete", ".prune("):
        assert token not in source.lower()


def test_default_ledger_path_has_no_prior_convention_to_collide_with(tmp_path):
    assert default_ledger_path(tmp_path / "runs") == tmp_path / "runs" / "capability_evidence_ledger.jsonl"


# ---------------------------------------------------------------------------
# CLI --apply gate: the dry-run guarantee lives entirely in cmd_reprobe_execute
# (execute_reprobe_actions itself always executes for real), so it needs its
# own direct test, not just module-level coverage.
# ---------------------------------------------------------------------------

def _cli_args(runs_dir, campaigns_dir, *, apply, model="llama3.1:8b", capability="text"):
    argv = [
        "reprobe-execute", "--mock",
        "--runs-dir", str(runs_dir), "--campaigns-dir", str(campaigns_dir),
        "--model", model, "--capability", capability,
    ]
    if apply:
        argv.append("--apply")
    return build_parser().parse_args(argv)


def test_dry_run_makes_no_probe_calls_and_writes_no_ledger(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    campaigns_dir = tmp_path / "campaigns"
    runs_dir.mkdir()
    campaigns_dir.mkdir()

    def _must_not_probe(*a, **k):
        raise AssertionError("dry-run must never call interrogate_model")

    monkeypatch.setattr(execute_mod, "interrogate_model", _must_not_probe)
    args = _cli_args(runs_dir, campaigns_dir, apply=False)
    cmd_reprobe_execute(args, Config())
    out = capsys.readouterr().out
    assert "dry-run only" in out
    assert not (runs_dir / "capability_evidence_ledger.jsonl").exists()


def test_apply_flag_actually_probes_and_writes_the_ledger(tmp_path):
    runs_dir = tmp_path / "runs"
    campaigns_dir = tmp_path / "campaigns"
    runs_dir.mkdir()
    campaigns_dir.mkdir()
    args = _cli_args(runs_dir, campaigns_dir, apply=True)
    cmd_reprobe_execute(args, Config())
    ledger_path = runs_dir / "capability_evidence_ledger.jsonl"
    assert ledger_path.exists()
    ledger = EvidenceLedger(ledger_path)
    assert len(list(ledger.all())) == 1


# ---------------------------------------------------------------------------
# Anvil Stage 3B.2 slice B: the reprobe-execute writer assigns an explicit
# EvidenceTrustClass computed from the current probe contract -- never a
# blanket canonical, never inferred from the fact that a probe just ran.
# ---------------------------------------------------------------------------

def test_writer_assigns_canonical_trust_for_a_complete_contract_probe(tmp_path, monkeypatch):
    from llm_modelbench.evidence import EvidenceTrustClass

    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    # _observation() has a content-addressed model identity, real runtime
    # identity, current protocol/schema, MEASURED_SUPPORTED -> complete contract.
    _patch_fixed_observation(monkeypatch, _observation())
    outcome = execute_reprobe_actions((_action(),), _FakeClient(), ledger)[0]
    assert outcome.appended is True
    record = ledger.get(outcome.observation_id)
    assert record.trust_class is EvidenceTrustClass.CANONICAL_COMPATIBLE


def test_writer_fails_closed_to_unknown_legacy_when_provenance_is_incomplete(tmp_path, monkeypatch):
    from llm_modelbench.evidence import EvidenceTrustClass

    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    # A fresh, current-schema observation whose model identity has NO
    # content-addressed provenance (name-hash artifact_set_id, primary_sha256
    # None). The probe still ran; trust must NOT follow from that.
    nameonly = _observation(
        model_identity=ModelArtifactIdentity.from_ollama_tag_row({"name": "mystery:latest"})
    )
    assert nameonly.capability_schema_version == CAPABILITY_SCHEMA_VERSION
    _patch_fixed_observation(monkeypatch, nameonly)
    outcome = execute_reprobe_actions((_action(),), _FakeClient(), ledger)[0]
    assert outcome.appended is True
    record = ledger.get(outcome.observation_id)
    assert record.trust_class is EvidenceTrustClass.UNKNOWN_LEGACY


def test_writer_does_not_default_to_canonical_by_omission(tmp_path):
    """append_capability_observation has no trust_class default: omitting it
    is a TypeError, so the EvidenceLedger CANONICAL_COMPATIBLE fallback can
    never be reached for a native observation by omission."""
    import pytest as _pytest
    from llm_modelbench.capability_observation import append_capability_observation as real_writer

    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    with _pytest.raises(TypeError):
        real_writer(ledger, _observation())
