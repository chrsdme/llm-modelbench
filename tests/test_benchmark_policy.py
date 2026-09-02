"""Anvil Stage 3.2D-0 -- deterministic BenchmarkProtocol construction."""
import dataclasses

import pytest

from llm_modelbench.benchmark_policy import (
    AGGREGATION_CONTRACT_VERSION, ALLOWED_ADAPTATIONS, PROTOCOL_ID, PROTOCOL_VERSION,
    SCORER_CONTRACT_VERSIONS, BenchmarkPolicyError, build_aggregation_policy_manifest,
    build_benchmark_protocol, build_output_budget_manifest,
    build_prompt_semantics_manifest, build_sampling_policy_manifest,
    resolve_scorer_versions,
)
from llm_modelbench.benchmark_protocol import BenchmarkProtocol
from llm_modelbench.config import Config
from llm_modelbench.tasks import TASKS, Task


def _cfg(**kw) -> Config:
    return dataclasses.replace(Config(), **kw)


def _task(id="t_demo", scorer="exact", **kw) -> Task:
    base = dict(id=id, category="demo", family="text", scorer=scorer,
                prompt="say hi", meta={"expected": "hi"})
    base.update(kw)
    return Task(**base)


# --- protocol identity: stability -------------------------------------------

def test_same_semantic_inputs_produce_the_same_protocol_identity():
    tasks = [_task("a"), _task("b", scorer="contains", meta={"needles": ["x"]})]
    a = build_benchmark_protocol(tasks, _cfg())
    b = build_benchmark_protocol([_task("a"), _task("b", scorer="contains", meta={"needles": ["x"]})], _cfg())
    assert a.identity_key() == b.identity_key()
    assert a.protocol_id == PROTOCOL_ID == "llm-modelbench-core"
    assert a.version == PROTOCOL_VERSION == "1"


def test_ordered_task_selection_is_identity_bearing():
    t1, t2 = _task("a"), _task("b")
    forward = build_benchmark_protocol([t1, t2], _cfg())
    reverse = build_benchmark_protocol([t2, t1], _cfg())
    assert forward.identity_key() != reverse.identity_key()


def test_different_valid_task_subset_differs_without_a_version_bump():
    full = build_benchmark_protocol([_task("a"), _task("b")], _cfg())
    subset = build_benchmark_protocol([_task("a")], _cfg())
    assert full.identity_key() != subset.identity_key()
    assert full.version == subset.version == "1"  # subset is not a methodology change


# --- protocol identity: sensitivity ----------------------------------------

def test_prompt_semantic_change_changes_identity():
    base = build_benchmark_protocol([_task("a", prompt="say hi")], _cfg())
    changed = build_benchmark_protocol([_task("a", prompt="say goodbye")], _cfg())
    assert base.prompt_semantics_hash != changed.prompt_semantics_hash
    assert base.identity_key() != changed.identity_key()


def test_prompt_bearing_meta_change_changes_identity():
    base = build_benchmark_protocol([_task("f", scorer="fim", meta={"suffix": "\nassert x == 1"})], _cfg())
    changed = build_benchmark_protocol([_task("f", scorer="fim", meta={"suffix": "\nassert x == 2"})], _cfg())
    assert base.prompt_semantics_hash != changed.prompt_semantics_hash


def test_presentation_only_meta_change_leaves_prompt_hash_unchanged():
    # `description` is the one real presentation-only meta key (fim_suffix_assertion).
    base = build_benchmark_protocol(
        [_task("f", scorer="fim", meta={"suffix": "\nassert x == 1", "description": "old text"})], _cfg())
    reworded = build_benchmark_protocol(
        [_task("f", scorer="fim", meta={"suffix": "\nassert x == 1", "description": "new wording"})], _cfg())
    assert base.prompt_semantics_hash == reworded.prompt_semantics_hash


def test_scoring_only_meta_change_leaves_prompt_hash_unchanged():
    base = build_benchmark_protocol([_task("a", scorer="exact", meta={"expected": "hi", "checks": ["a"]})], _cfg())
    changed = build_benchmark_protocol([_task("a", scorer="exact", meta={"expected": "hi", "checks": ["a", "b"]})], _cfg())
    assert base.prompt_semantics_hash == changed.prompt_semantics_hash


def test_sampling_change_changes_identity():
    base = build_benchmark_protocol([_task("a")], _cfg(temperature=0.0, seed=42))
    hot = build_benchmark_protocol([_task("a")], _cfg(temperature=0.7, seed=42))
    reseeded = build_benchmark_protocol([_task("a")], _cfg(temperature=0.0, seed=7))
    assert base.sampling_policy_hash != hot.sampling_policy_hash
    assert base.sampling_policy_hash != reseeded.sampling_policy_hash


def test_output_budget_change_changes_identity():
    base = build_benchmark_protocol([_task("a", num_predict=1024)], _cfg())
    wider = build_benchmark_protocol([_task("a", num_predict=4096)], _cfg())
    overridden = build_benchmark_protocol([_task("a", num_predict=1024)], _cfg(num_predict_override=512))
    assert base.output_budget_policy_hash != wider.output_budget_policy_hash
    assert base.output_budget_policy_hash != overridden.output_budget_policy_hash


def test_recovery_budget_policy_is_comparability_material():
    budget = build_output_budget_manifest([_task("a")], _cfg())
    assert budget["bounded_recovery"]["think_retry_num_predict"] == 4096
    assert "bounded_recovery" in budget and budget["bounded_recovery"]["policy"]


# --- scorer contract versions --------------------------------------------------

def test_allowed_adaptations_is_exactly_context_size_for_v1():
    proto = build_benchmark_protocol([_task("a")], _cfg())
    assert proto.allowed_adaptations == ("context_size",)
    assert ALLOWED_ADAPTATIONS == ("context_size",)


def test_unknown_scorer_fails_closed_never_defaults_to_one():
    with pytest.raises(BenchmarkPolicyError, match="no explicit contract version"):
        resolve_scorer_versions([_task("a", scorer="totally_made_up")])
    with pytest.raises(BenchmarkPolicyError):
        build_benchmark_protocol([_task("a", scorer="totally_made_up")], _cfg())


def test_regex_scorer_in_deterministic_but_unused_still_fails_closed():
    # `regex` is a real scoring.DETERMINISTIC key but no current task uses it,
    # so it deliberately has no contract version -- a future task using it
    # must add `regex: 1` explicitly (owner freeze section 4, rule 3).
    assert "regex" not in SCORER_CONTRACT_VERSIONS
    with pytest.raises(BenchmarkPolicyError):
        resolve_scorer_versions([_task("a", scorer="regex")])


def test_scorer_versions_shape_is_one_entry_per_task():
    versions = resolve_scorer_versions([_task("a", scorer="python"), _task("b", scorer="exact")])
    assert versions == (("a", "python@1"), ("b", "exact@1"))


def test_every_scorer_used_by_the_real_task_suite_has_a_contract_version():
    # The real motivating case: the frozen protocol must be constructible over
    # the entire shipped task suite without a fail-closed.
    used = {t.scorer for t in TASKS}
    missing = used - set(SCORER_CONTRACT_VERSIONS)
    assert not missing, f"shipped tasks use unversioned scorers: {sorted(missing)}"
    proto = build_benchmark_protocol(list(TASKS), _cfg())
    assert isinstance(proto, BenchmarkProtocol)
    assert len(proto.scorer_versions) == len(TASKS)


# --- manifest hygiene --------------------------------------------------------

def test_sampling_manifest_names_delegated_knobs_explicitly():
    manifest = build_sampling_policy_manifest(_cfg())
    for key in ("top_p", "top_k", "min_p", "repeat_penalty", "mirostat"):
        assert manifest[key] == "backend_default"
    assert "num_ctx" not in manifest  # context is an adaptation, not sampling


def test_prompt_manifest_excludes_scoring_and_presentation_meta():
    task = _task("a", scorer="exact",
                 meta={"expected": "hi", "checks": ["x"], "description": "blah", "suffix": "S"})
    entry = build_prompt_semantics_manifest([task])[0]
    assert entry["meta"] == {"suffix": "S"}


# --- Anvil Stage 3.2E: aggregation-policy identity --------------------------

def test_task_difficulty_change_changes_protocol_identity():
    # Same raw outputs, changed difficulty -> different aggregation_policy_hash
    # -> different BenchmarkProtocol identity. difficulty is the intra-category
    # weight in outcome.category_score; it must be identity-bearing.
    base = build_benchmark_protocol([_task("a", difficulty=1.0)], _cfg())
    harder = build_benchmark_protocol([_task("a", difficulty=2.0)], _cfg())
    assert base.aggregation_policy_hash != harder.aggregation_policy_hash
    assert base.identity_key() != harder.identity_key()
    # and it is NOT smuggled into prompt semantics
    assert base.prompt_semantics_hash == harder.prompt_semantics_hash


def test_difficulty_gate_boundary_is_recorded_and_identity_bearing():
    # difficulty > 0 (scored contributor) vs difficulty <= 0 (pass/fail gate)
    # is a semantic boundary the manifest must distinguish, not just a number.
    scored = build_aggregation_policy_manifest(
        [_task("a", difficulty=1.0)], _cfg(), sample_mode="smart", judge_mode="single")
    gate = build_aggregation_policy_manifest(
        [_task("a", difficulty=0.0)], _cfg(), sample_mode="smart", judge_mode="single")
    assert scored["gate_task_ids"] == []
    assert gate["gate_task_ids"] == ["a"]
    assert scored["task_difficulty"]["a"] == 1.0
    assert gate["task_difficulty"]["a"] == 0.0
    p_scored = build_benchmark_protocol([_task("a", difficulty=1.0)], _cfg())
    p_gate = build_benchmark_protocol([_task("a", difficulty=0.0)], _cfg())
    assert p_scored.identity_key() != p_gate.identity_key()


def test_category_weight_change_changes_protocol_identity():
    # A weight change for a category that materially participates moves the
    # identity. Injected via the manifest builder's DEFAULT_WEIGHTS read --
    # patch the module constant rather than mutating the shared global in place.
    import llm_modelbench.benchmark_policy as bp

    task = _task("a", category="coding_python")
    base = build_benchmark_protocol([task], _cfg())
    original = dict(bp.DEFAULT_WEIGHTS)
    try:
        bp.DEFAULT_WEIGHTS = dict(original, coding_python=original.get("coding_python", 0.0) + 0.05)
        bumped = build_benchmark_protocol([task], _cfg())
    finally:
        bp.DEFAULT_WEIGHTS = original
    assert base.aggregation_policy_hash != bumped.aggregation_policy_hash
    assert base.identity_key() != bumped.identity_key()


def test_category_weight_subset_only_participating_categories():
    # A weight change for a category NOT in this benchmark cannot change its
    # canonical result, so it must NOT change this protocol's identity --
    # while a change to a PARTICIPATING category's weight must (proven in the
    # test above and re-checked here so this test cannot pass vacuously if
    # the manifest ignored DEFAULT_WEIGHTS entirely).
    import llm_modelbench.benchmark_policy as bp

    task = _task("a", category="coding_python")  # 'ocr' does not participate
    base = build_benchmark_protocol([task], _cfg())
    original = dict(bp.DEFAULT_WEIGHTS)
    try:
        bp.DEFAULT_WEIGHTS = dict(original, ocr=original.get("ocr", 0.0) + 0.05)
        unrelated = build_benchmark_protocol([task], _cfg())
        bp.DEFAULT_WEIGHTS = dict(original, coding_python=original["coding_python"] + 0.05)
        participating = build_benchmark_protocol([task], _cfg())
        manifest = build_aggregation_policy_manifest(
            [task], _cfg(), sample_mode="smart", judge_mode="single")
    finally:
        bp.DEFAULT_WEIGHTS = original
    assert base.aggregation_policy_hash == unrelated.aggregation_policy_hash  # 'ocr' irrelevant
    assert base.aggregation_policy_hash != participating.aggregation_policy_hash  # not vacuous
    assert set(manifest["category_weights"]) == {"coding_python"}
    assert manifest["category_weights"]["coding_python"] == original["coding_python"] + 0.05


def test_requested_samples_change_changes_identity_where_sampling_applies():
    # A judge/subjective task under judging on: sample count depends on cfg.samples.
    subj = _task("s", scorer="subjective", judge=True)
    base = build_benchmark_protocol([subj], _cfg(samples=1), judge_mode="single")
    more = build_benchmark_protocol([subj], _cfg(samples=5), judge_mode="single")
    assert base.aggregation_policy_hash != more.aggregation_policy_hash
    assert base.identity_key() != more.identity_key()


def test_sample_mode_change_changes_identity():
    t = _task("a")
    smart = build_benchmark_protocol([t], _cfg(samples=3), sample_mode="smart")
    every = build_benchmark_protocol([t], _cfg(samples=3), sample_mode="all")
    assert smart.aggregation_policy_hash != every.aggregation_policy_hash
    # per_task_samples reflects the actual policy, not just cfg.samples
    m_smart = build_aggregation_policy_manifest([t], _cfg(samples=3), sample_mode="smart", judge_mode="single")
    m_every = build_aggregation_policy_manifest([t], _cfg(samples=3), sample_mode="all", judge_mode="single")
    assert m_smart["sample_policy"]["per_task_samples"]["a"] == 1
    assert m_every["sample_policy"]["per_task_samples"]["a"] == 3


def test_judge_mode_change_changes_sample_policy_identity():
    subj = _task("s", scorer="subjective", judge=True)
    on = build_benchmark_protocol([subj], _cfg(samples=4), judge_mode="single")
    off = build_benchmark_protocol([subj], _cfg(samples=4), judge_mode="off")
    assert on.aggregation_policy_hash != off.aggregation_policy_hash
    # the draw count actually moves (4 with judging on, 1 with it off) -- the
    # hash change is not merely from hashing the judge_mode string
    m_on = build_aggregation_policy_manifest([subj], _cfg(samples=4), sample_mode="smart", judge_mode="single")
    m_off = build_aggregation_policy_manifest([subj], _cfg(samples=4), sample_mode="smart", judge_mode="off")
    assert m_on["sample_policy"]["per_task_samples"]["s"] == 4
    assert m_off["sample_policy"]["per_task_samples"]["s"] == 1


def test_aggregation_contract_version_is_in_the_manifest():
    m = build_aggregation_policy_manifest(
        [_task("a")], _cfg(), sample_mode="smart", judge_mode="single")
    assert m["aggregation_contract_version"] == AGGREGATION_CONTRACT_VERSION == 1
    assert m["sample_policy"]["combination"] == "arithmetic_mean_numeric_v1"


def test_canonical_builder_always_populates_aggregation_policy_hash():
    proto = build_benchmark_protocol([_task("a")], _cfg())
    assert proto.aggregation_policy_hash
    assert len(proto.aggregation_policy_hash) == 16  # _stable_hash width


def test_legacy_protocol_without_aggregation_hash_stays_constructible():
    # A BenchmarkProtocol deserialized from before Stage 3.2E has no
    # aggregation_policy_hash -- the dataclass must still accept it (default "").
    from llm_modelbench.benchmark_protocol import BenchmarkProtocol

    legacy = BenchmarkProtocol(
        protocol_id="llm-modelbench-core", version="1", task_ids=("a",),
        prompt_semantics_hash="p", sampling_policy_hash="s",
        output_budget_policy_hash="o", scorer_versions=(("a", "exact@1"),),
    )
    assert legacy.aggregation_policy_hash == ""
    # and it has a different identity from a current one (intended -- 3.2E moved it)
    current = build_benchmark_protocol([_task("a", scorer="exact")], _cfg())
    assert legacy.identity_key() != current.identity_key()


# --- Anvil Stage 3.3A: canonical ranking aggregation-policy verifier --------

from llm_modelbench.benchmark_policy import (  # noqa: E402
    AGG_VERDICT_POLICY_DRIFT, AGG_VERDICT_UNVERIFIED_INCOMPLETE,
    AGG_VERDICT_UNVERIFIED_LEGACY, AGG_VERDICT_VERIFIED,
    recompute_aggregation_policy_hash, verify_recorded_aggregation_policy,
)


def _recorded_hash(tasks, *, samples=1, sample_mode="smart", judge_mode="single"):
    """The aggregation_policy_hash build_benchmark_protocol would record for
    this exact task set / sample policy under the current constants."""
    return build_benchmark_protocol(
        tasks, _cfg(samples=samples), sample_mode=sample_mode, judge_mode=judge_mode
    ).aggregation_policy_hash


def test_verifier_matching_policy_is_verified():
    tasks = [_task("a", category="coding_python", difficulty=1.0),
             _task("b", category="coding_python", difficulty=0.0)]
    by_id = {t.id: t for t in tasks}
    recorded = _recorded_hash(tasks)
    v = verify_recorded_aggregation_policy(
        recorded_hash=recorded, task_ids=["a", "b"],
        requested_samples=1, sample_mode="smart", judge_mode="single",
        tasks_by_id=by_id,
    )
    assert v.verdict == AGG_VERDICT_VERIFIED
    assert v.compatible is True
    assert v.recomputed_hash == recorded


def test_verifier_difficulty_drift_is_detected():
    # Run bound when 'a' had difficulty 1.0; today the suite says 2.0.
    bound_tasks = [_task("a", category="coding_python", difficulty=1.0)]
    recorded = _recorded_hash(bound_tasks)
    current = {"a": _task("a", category="coding_python", difficulty=2.0)}
    v = verify_recorded_aggregation_policy(
        recorded_hash=recorded, task_ids=["a"],
        requested_samples=1, sample_mode="smart", judge_mode="single",
        tasks_by_id=current,
    )
    assert v.verdict == AGG_VERDICT_POLICY_DRIFT
    assert v.compatible is False
    assert v.recorded_hash == recorded and v.recomputed_hash != recorded
    assert "difficulty" in v.reason or "current canonical" in v.reason


def test_verifier_participating_category_weight_drift_is_detected():
    import llm_modelbench.benchmark_policy as bp
    task = _task("a", category="coding_python", difficulty=1.0)
    recorded = _recorded_hash([task])
    original = dict(bp.DEFAULT_WEIGHTS)
    try:
        bp.DEFAULT_WEIGHTS = dict(original, coding_python=original["coding_python"] + 0.05)
        v = verify_recorded_aggregation_policy(
            recorded_hash=recorded, task_ids=["a"],
            requested_samples=1, sample_mode="smart", judge_mode="single",
            tasks_by_id={"a": task},
        )
    finally:
        bp.DEFAULT_WEIGHTS = original
    assert v.verdict == AGG_VERDICT_POLICY_DRIFT


def test_verifier_non_participating_category_weight_change_is_not_a_mismatch():
    import llm_modelbench.benchmark_policy as bp
    task = _task("a", category="coding_python", difficulty=1.0)  # 'ocr' absent
    recorded = _recorded_hash([task])
    original = dict(bp.DEFAULT_WEIGHTS)
    try:
        bp.DEFAULT_WEIGHTS = dict(original, ocr=original.get("ocr", 0.0) + 0.05)
        v = verify_recorded_aggregation_policy(
            recorded_hash=recorded, task_ids=["a"],
            requested_samples=1, sample_mode="smart", judge_mode="single",
            tasks_by_id={"a": task},
        )
        # and a participating change on the SAME setup IS caught (non-vacuous)
        bp.DEFAULT_WEIGHTS = dict(original, coding_python=original["coding_python"] + 0.05)
        v_part = verify_recorded_aggregation_policy(
            recorded_hash=recorded, task_ids=["a"],
            requested_samples=1, sample_mode="smart", judge_mode="single",
            tasks_by_id={"a": task},
        )
    finally:
        bp.DEFAULT_WEIGHTS = original
    assert v.verdict == AGG_VERDICT_VERIFIED
    assert v_part.verdict == AGG_VERDICT_POLICY_DRIFT


def test_verifier_sample_policy_drift_is_detected():
    # subjective task, judging on: draw count follows requested_samples.
    subj = _task("s", category="coding_python", scorer="subjective", difficulty=1.0)
    recorded = _recorded_hash([subj], samples=2, sample_mode="all", judge_mode="single")
    v = verify_recorded_aggregation_policy(
        recorded_hash=recorded, task_ids=["s"],
        requested_samples=5, sample_mode="all", judge_mode="single",
        tasks_by_id={"s": subj},
    )
    assert v.verdict == AGG_VERDICT_POLICY_DRIFT
    # sanity: same as recorded reproduces verified
    v_ok = verify_recorded_aggregation_policy(
        recorded_hash=recorded, task_ids=["s"],
        requested_samples=2, sample_mode="all", judge_mode="single",
        tasks_by_id={"s": subj},
    )
    assert v_ok.verdict == AGG_VERDICT_VERIFIED


def test_verifier_legacy_row_without_recorded_hash():
    v = verify_recorded_aggregation_policy(
        recorded_hash="", task_ids=[],
        requested_samples=1, sample_mode="smart", judge_mode="single",
        tasks_by_id={},
    )
    assert v.verdict == AGG_VERDICT_UNVERIFIED_LEGACY
    assert v.compatible is False
    assert v.recorded_hash == ""


def test_verifier_incomplete_run_config_is_not_a_drift_claim():
    task = _task("a", category="coding_python", difficulty=1.0)
    recorded = _recorded_hash([task])
    v = verify_recorded_aggregation_policy(
        recorded_hash=recorded, task_ids=["a"],
        requested_samples=None, sample_mode=None, judge_mode=None,
        tasks_by_id={"a": task},
    )
    assert v.verdict == AGG_VERDICT_UNVERIFIED_INCOMPLETE
    assert v.compatible is False


def test_verifier_unknown_task_is_incomplete_not_drift():
    task = _task("a", category="coding_python", difficulty=1.0)
    recorded = _recorded_hash([task])
    v = verify_recorded_aggregation_policy(
        recorded_hash=recorded, task_ids=["a", "gone_from_suite"],
        requested_samples=1, sample_mode="smart", judge_mode="single",
        tasks_by_id={"a": task},
    )
    assert v.verdict == AGG_VERDICT_UNVERIFIED_INCOMPLETE


def test_recompute_reuses_the_protocol_builder_hash_exactly():
    # The verifier's recompute must equal what build_benchmark_protocol records
    # for the same inputs -- i.e. it reuses build_aggregation_policy_manifest +
    # the "aggregation_policy_v1" _stable_hash idiom, not a parallel formula.
    tasks = [_task("a", category="coding_python", difficulty=1.5),
             _task("b", category="git", difficulty=0.0)]
    by_id = {t.id: t for t in tasks}
    via_protocol = build_benchmark_protocol(
        tasks, _cfg(samples=1), sample_mode="smart", judge_mode="single"
    ).aggregation_policy_hash
    via_verifier = recompute_aggregation_policy_hash(
        ["a", "b"], requested_samples=1, sample_mode="smart",
        judge_mode="single", tasks_by_id=by_id,
    )
    assert via_protocol == via_verifier
