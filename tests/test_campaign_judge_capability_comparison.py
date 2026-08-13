"""Anvil Stage 2.6D, Task #8 -- legacy-vs-new judge capability-eligibility
comparison gate, run across a real corpus of judge candidate dicts.

Unlike `tests/test_capability_migration_comparison_corpus.py`'s judge-path
cases (1-9), which that file's own docstring flags as *not* verified as the
literal function calls `campaign.py` makes internally (it models the
judge path using the generic `capability_identity_compatibility()` /
`measured_supported_families()` pair planner.py uses, because at the time
`campaign.py`'s own wrapper hadn't been migrated yet) -- this module calls
`campaign.py`'s actual pre- and post-2.6D functions directly via
`capability_migration_comparison.compare_judge_capability_eligibility()`,
so a discrepancy here reflects a real behavioral difference in the real
call path, not a harness approximation.

Per `codex-advise_pre2.6D.txt` part 5: classify every case MATCH /
EXPECTED_CORRECTION / UNEXPLAINED_DIFFERENCE and hard-gate on any
unexplained legacy-NO -> new-YES ("expansion") transition. Every case in
this corpus is asserted MATCH -- expected, since the 2.6D design note
already proved by direct experiment that the one behavioral difference the
migration could have introduced (the old function's redundant identity
recheck inside its own MEASURED_SUPPORTED branch) was already dead code:
`_judge_measured_text_state()` never reports MEASURED_SUPPORTED when that
recheck would fail. A 100%-MATCH corpus is therefore the expected,
positive result, not a weak test.
"""
import pytest

from llm_modelbench.capabilities import CAPABILITY_SCHEMA_VERSION, PROBE_PROTOCOL_VERSION
from llm_modelbench.capability_migration_comparison import (
    ComparisonDirection,
    ComparisonVerdict,
    compare_judge_capability_eligibility,
)

DIRECTION = ComparisonDirection
VERDICT = ComparisonVerdict


def _identity(name, digest, *, protocol_version=PROBE_PROTOCOL_VERSION, template_hash="template-v1", backend="mock"):
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "model": {"canonical_name": name, "backend_model_id": name, "digest": digest, "size": 1, "details": {}},
        "backend": {"backend": backend, "implementation": "fixture", "endpoint": "http://fake.invalid"},
        "runtime": {"endpoint": "http://fake.invalid", "implementation": "fixture"},
        "template_config": {"available": True, "hash": template_hash, "material": {"template": template_hash}},
        "probe_protocol_version": protocol_version,
    }


def _bind(
    item, *, identity_digest=None, protocol_version=PROBE_PROTOCOL_VERSION,
    template_hash="template-v1", backend="mock", compatibility=None,
):
    # Real interrogate_model() output carries probe_protocol_version both
    # top-level and nested inside capability_identity, always in sync
    # (capabilities.py:455/573) -- see the Stage 2.6D design note on the
    # dual-field finding.
    item.setdefault("probe_protocol_version", PROBE_PROTOCOL_VERSION)
    item["capability_identity"] = _identity(
        item["name"], identity_digest if identity_digest is not None else item["digest"],
        protocol_version=protocol_version, template_hash=template_hash, backend=backend,
    )
    # capability_identity_compatibility is the precomputed field real
    # judge candidates always carry (written earlier in the same campaign
    # run by planner.py's own live-client check -- confirmed by the
    # Stage 2.6D design note's traced call site). It defaults to a
    # genuine match here; callers simulating a real, already-detected
    # incompatibility (of any kind, not just digest) must pass
    # `compatibility=` explicitly so both the legacy oracle *and* the
    # post-fix typed authority (Task #8's own finding: non-digest drift
    # must be sourced from this field, since the synthetic "current"
    # identity below only ever independently re-observes digest) see the
    # same real signal, rather than the two sides silently disagreeing
    # because only one of them was told about the drift.
    item["capability_identity_compatibility"] = compatibility or {"compatible": True, "reason": "identity_match"}
    return item


def _measured(name, digest, states, **extra):
    item = {
        "name": name,
        "digest": digest,
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "measured_capabilities": {family: {"state": state} for family, state in states.items()},
    }
    item.update(extra)
    return item


CORPUS = []


def _case(case_id, item, *, expected_applicable):
    CORPUS.append((case_id, item, expected_applicable))
    return item


# 1. Compatible, measured-supported text, bound identity.
_case(
    "compatible_measured_text",
    _bind(_measured("judge-a", "digest-a", {"text": "measured_supported"})),
    expected_applicable=True,
)

# 2. Declared capability only, no measured evidence at all.
_case(
    "declared_only_no_measurement",
    {
        "name": "judge-b", "digest": "digest-b",
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "capabilities": ["completion"],
    },
    expected_applicable=False,
)

# 3. supported_families=["text"] only, no measured_capabilities.
_case(
    "supported_families_only",
    {
        "name": "judge-c", "digest": "digest-c",
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "supported_families": ["text"],
    },
    expected_applicable=False,
)

# 4. Legacy/unbound: no capability_schema_version at all.
_case(
    "legacy_unbound_profile",
    {"name": "judge-d", "digest": "digest-d", "measured_capabilities": {"text": {"state": "measured_supported"}}},
    expected_applicable=False,
)

# 5. Identity-incompatible: stored identity digest differs from the
# candidate's own freshly-observed digest. A real precomputed
# compatibility field would already flag this for the same reason (the
# typed authority's own fresh-digest override independently confirms it
# too, so this case would MATCH even without the explicit compatibility
# override -- it is set here for realism, not because it changes the
# outcome).
_case(
    "identity_incompatible_digest_mismatch",
    _bind(
        _measured("judge-e", "digest-e-current", {"text": "measured_supported"}),
        identity_digest="digest-e-stale",
        compatibility={"compatible": False, "reason": "model_digest_changed"},
    ),
    expected_applicable=False,
)

# 6. Measured unsupported.
_case(
    "measured_unsupported",
    _bind(_measured("judge-f", "digest-f", {"text": "measured_unsupported"})),
    expected_applicable=False,
)

# 7. Embedding-only candidate.
_case(
    "embedding_only",
    _bind(_measured("judge-g", "digest-g", {"embedding": "measured_supported"}, supported_families=["embedding"])),
    expected_applicable=False,
)

# 8. Vision-only candidate.
_case(
    "vision_only",
    _bind(_measured("judge-h", "digest-h", {"vision": "measured_supported"}, supported_families=["vision"])),
    expected_applicable=False,
)

# 9. Probe-inconclusive text state.
_case(
    "probe_inconclusive",
    _bind(_measured("judge-i", "digest-i", {"text": "probe_inconclusive"})),
    expected_applicable=False,
)

# 10. Stale probe_protocol_version (both copies, matching real capture
# behavior). The new side's own live-constant check (PROBE_PROTOCOL_VERSION,
# independent of the identity mapping) catches this on its own; the
# explicit compatibility override below is what makes the legacy oracle
# agree, matching a real precomputed field written by the same capture.
_stale = _bind(
    _measured("judge-j", "digest-j", {"text": "measured_supported"}), protocol_version="capability-smoke-v0",
    compatibility={"compatible": False, "reason": "probe_protocol_version_changed"},
)
_stale["probe_protocol_version"] = "capability-smoke-v0"
_case("stale_probe_protocol_version", _stale, expected_applicable=False)

# 11. Backend mismatch between stored identity and current runtime. This is
# the Task #8 finding case: the typed authority's synthetic "current"
# identity (campaign.py's `_candidate_current_capability_identity()`) only
# independently re-observes digest -- backend/template/endpoint are copied
# verbatim from the stored identity, so a backend-only drift is invisible
# to it unless it is told via this precomputed compatibility field (which
# `_candidate_current_capability_identity()` now consults and fails closed
# on, for any reason other than a stale digest). Without the compatibility
# override below, this case was a real, reproduced legacy-NO -> new-YES
# expansion before that fix landed in this same slice.
_case(
    "backend_mismatch",
    _bind(
        _measured("judge-k", "digest-k", {"text": "measured_supported"}), backend="other-backend",
        compatibility={"compatible": False, "reason": "backend_changed"},
    ),
    expected_applicable=False,
)

# 12. Template hash mismatch -- same structural gap and same fix as case 11,
# for the template_config.hash dimension instead of backend.
_case(
    "template_mismatch",
    _bind(
        _measured("judge-l", "digest-l", {"text": "measured_supported"}), template_hash="template-v2-current",
        compatibility={"compatible": False, "reason": "template_config_changed"},
    ),
    expected_applicable=False,
)

# 13. Missing digest entirely -- no independent freshness check possible.
_case(
    "missing_digest",
    _bind(_measured("judge-m", "", {"text": "measured_supported"})),
    expected_applicable=False,
)

# 14. A live current_capability_identity already attached (planner/runner-
# style), matching the stored identity -- compatible.
_live_match = _measured("judge-n", "digest-n", {"text": "measured_supported"})
_live_match = _bind(_live_match)
_live_match["current_capability_identity"] = _identity("judge-n", "digest-n")
_case("live_identity_present_and_matching", _live_match, expected_applicable=True)

# 15. A live current_capability_identity attached but mismatched.
_live_mismatch = _measured("judge-o", "digest-o", {"text": "measured_supported"})
_live_mismatch = _bind(_live_mismatch)
_live_mismatch["current_capability_identity"] = _identity("judge-o", "digest-o-drifted")
_case("live_identity_present_and_mismatched", _live_mismatch, expected_applicable=False)


@pytest.mark.parametrize("case_id,item,expected_applicable", CORPUS, ids=[c[0] for c in CORPUS])
def test_corpus_case_matches_and_agrees_with_expectation(case_id, item, expected_applicable):
    result = compare_judge_capability_eligibility(item)
    assert result.verdict == VERDICT.MATCH, (case_id, result)
    assert result.direction == DIRECTION.NONE, (case_id, result)
    assert result.new_applicable is expected_applicable, (case_id, result)
    assert result.legacy_applicable is expected_applicable, (case_id, result)


def test_hard_gate_no_unexplained_expansion_across_full_corpus():
    results = [compare_judge_capability_eligibility(item) for _, item, _ in CORPUS]
    expansions = [r for r in results if r.direction == DIRECTION.EXPANSION and r.verdict == VERDICT.UNEXPLAINED_DIFFERENCE]
    assert expansions == []


def test_hard_gate_no_unexplained_difference_of_any_direction_across_full_corpus():
    # Stronger than the required minimum: the design note's dead-code proof
    # predicts perfect agreement, not just "no dangerous expansions".
    results = [compare_judge_capability_eligibility(item) for _, item, _ in CORPUS]
    unexplained = [r for r in results if r.verdict == VERDICT.UNEXPLAINED_DIFFERENCE]
    assert unexplained == []


def test_known_correction_flag_classifies_a_forced_expansion_correctly(monkeypatch):
    # Confirms the harness's known_correction escape hatch (same convention
    # as compare_planner_capability_decision) actually changes the verdict
    # of a real difference, without any real corpus case needing it today.
    # Forces new-side YES on a candidate the frozen legacy reference says NO
    # for (declared-only, no measurement) -- an EXPANSION by construction.
    from llm_modelbench import campaign

    item = {
        "name": "judge-p", "digest": "digest-p",
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "capabilities": ["completion"],
    }
    monkeypatch.setattr(campaign, "_judge_capability_rejection", lambda candidate: None)

    unexplained = compare_judge_capability_eligibility(item, known_correction=False)
    assert unexplained.direction == DIRECTION.EXPANSION
    assert unexplained.verdict == VERDICT.UNEXPLAINED_DIFFERENCE

    explained = compare_judge_capability_eligibility(item, known_correction=True)
    assert explained.direction == DIRECTION.EXPANSION
    assert explained.verdict == VERDICT.EXPECTED_CORRECTION
