"""Anvil Stage 3.2D-0 -- deterministic BenchmarkProtocol construction."""
import dataclasses

import pytest

from llm_modelbench.benchmark_policy import (
    ALLOWED_ADAPTATIONS, PROTOCOL_ID, PROTOCOL_VERSION, SCORER_CONTRACT_VERSIONS,
    BenchmarkPolicyError, build_benchmark_protocol, build_output_budget_manifest,
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
