from llm_modelbench import campaign
from llm_modelbench.capabilities import CAPABILITY_SCHEMA_VERSION, PROBE_PROTOCOL_VERSION, MeasuredCapabilityState
from llm_modelbench.config import Config


def _identity(name, digest, *, backend="mock", endpoint="http://fake.invalid", template_hash="template-v1"):
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "model": {"canonical_name": name, "backend_model_id": name, "digest": digest, "size": 1, "details": {}},
        "backend": {"backend": backend, "implementation": "fixture", "endpoint": endpoint},
        "runtime": {"endpoint": endpoint, "implementation": "fixture"},
        "template_config": {"available": True, "hash": template_hash, "material": {"template": template_hash}},
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
    }


def _bind(item, *, compatible=True, reason="identity_match"):
    # Real interrogate_model() output always carries probe_protocol_version
    # both top-level and nested inside capability_identity (capabilities.py
    # sets both from the same PROBE_PROTOCOL_VERSION constant in one call,
    # see capabilities.py:455/573) -- the typed Stage 2.6D adapter reads the
    # top-level copy (capability_evidence_adapter.adapt_legacy_profile_family_to_observation),
    # so a fixture representing a real bound profile must set it too.
    item.setdefault("probe_protocol_version", PROBE_PROTOCOL_VERSION)
    item["capability_identity"] = _identity(item["name"], item["digest"])
    item["capability_identity_compatibility"] = {"compatible": compatible, "reason": reason}
    return item


def _measured_text(name, digest, **extra):
    item = {
        "name": name,
        "digest": digest,
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "measured_capabilities": {
            "text": {"state": MeasuredCapabilityState.MEASURED_SUPPORTED.value},
        },
    }
    item.update(extra)
    return _bind(item)


def _measured_embedding(name, digest, **extra):
    item = {
        "name": name,
        "digest": digest,
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "measured_capabilities": {
            "embedding": {"state": MeasuredCapabilityState.MEASURED_SUPPORTED.value},
        },
    }
    item.update(extra)
    return _bind(item)


def _policy(**kwargs):
    defaults = {
        "requested_primary": None,
        "configured_fallbacks": (),
        "excluded_families": ("qwen",),
        "allow_excluded_primary": False,
        "automatic_selection": True,
        "enabled": True,
    }
    defaults.update(kwargs)
    return campaign.JudgePolicy(**defaults)


def _select(inventory, policy=None, cohort=None):
    return campaign.build_judge_selection(inventory, cohort or [], policy or _policy())


def test_text_generative_candidate_is_eligible():
    result = _select([_measured_text("llama:8b", "l")])
    assert [item["name"] for item in result.final_eligible_order] == ["llama:8b"]
    assert result.selected["canonical_families"] == ["text"]


def test_embedding_only_candidate_is_rejected():
    result = _select([_measured_embedding("embedder", "e")])
    assert result.selected is None
    assert result.rejection_reasons[0]["reason"] == "non_generative_embedding_only"


def test_reranker_candidate_is_rejected():
    result = _select([{"name": "kalm-reranker:latest", "digest": "r", "capabilities": ["reranker"]}])
    assert result.selected is None
    assert result.rejection_reasons[0]["reason"] == "capability_reprobe_required"


def test_vision_text_candidate_is_eligible_but_vision_only_is_rejected():
    result = _select([
        {
            "name": "vision-only",
            "digest": "v",
            "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
            "measured_capabilities": {"vision": {"state": MeasuredCapabilityState.MEASURED_SUPPORTED.value}},
        },
        _measured_text("vision-text", "vt"),
    ])
    assert result.selected["name"] == "vision-text"
    assert result.rejection_reasons[0]["reason"] == "non_generative_vision_only"


def test_unknown_capability_fails_closed():
    result = _select([{"name": "mystery", "digest": "m", "capabilities": ["unknown_future_lane"]}])
    assert result.selected is None
    assert result.rejection_reasons[0]["reason"] == "capability_reprobe_required"


def test_configured_primary_precedence_is_exact_first_order():
    result = _select([
        _measured_text("auto-a", "a", calibrated=True),
        _measured_text("primary", "p"),
    ], _policy(requested_primary="primary"))
    assert [item["name"] for item in result.final_eligible_order] == ["primary", "auto-a"]
    assert result.selected["name"] == "primary"


def test_unavailable_and_ineligible_primary_rejection_is_preserved_before_fallback():
    result = _select([
        _measured_embedding("bad-primary", "b"),
        _measured_text("fallback", "f"),
    ], _policy(requested_primary="missing-primary", configured_fallbacks=("bad-primary", "fallback")))
    assert result.selected["name"] == "fallback"
    assert [(item["model"], item["reason"]) for item in result.rejection_reasons] == [
        ("missing-primary", "not_in_inventory"),
        ("bad-primary", "non_generative_embedding_only"),
    ]


def test_qwen_family_is_excluded_from_automatic_judging_by_default():
    result = _select([
        _measured_text("qwen9-future:72b", "q", calibrated=True),
        _measured_text("llama:8b", "l"),
    ])
    assert result.selected["name"] == "llama:8b"
    assert result.rejection_reasons[0]["reason"] == "excluded_family"


def test_excluded_primary_requires_explicit_override():
    inventory = [
        _measured_text("qwen9-future:72b", "q"),
        _measured_text("llama:8b", "l"),
    ]
    blocked = _select(inventory, _policy(requested_primary="qwen9-future:72b"))
    allowed = _select(inventory, _policy(requested_primary="qwen9-future:72b", allow_excluded_primary=True))
    assert blocked.selected["name"] == "llama:8b"
    assert blocked.rejection_reasons[0]["reason"] == "excluded_family"
    assert allowed.selected["name"] == "qwen9-future:72b"
    assert allowed.selected["excluded_family_override"] is True


def test_configured_fallback_order_duplicate_removal_and_automatic_exclusions():
    result = _select([
        _measured_text("auto", "a"),
        _measured_text("qwen-auto", "q", calibrated=True),
        _measured_text("fallback-b", "b"),
        _measured_text("fallback-a", "fa"),
    ], _policy(configured_fallbacks=("fallback-b", "fallback-a", "fallback-b")))
    assert result.configured_fallbacks == ["fallback-b", "fallback-a"]
    assert [item["name"] for item in result.final_eligible_order] == ["fallback-b", "fallback-a", "auto"]
    assert any(item["model"] == "qwen-auto" and item["reason"] == "excluded_family" for item in result.rejection_reasons)


def test_repeated_selection_is_deterministic_and_structured():
    inventory = [
        _measured_text("b", "2", priority=1),
        _measured_text("a", "1", priority=1),
    ]
    first = _select(inventory).to_dict()
    second = _select(list(reversed(inventory))).to_dict()
    assert first == second
    assert set(first) == {
        "requested_primary",
        "configured_fallbacks",
        "automatic_candidates",
        "exclusions",
        "rejection_reasons",
        "final_eligible_order",
        "selected",
        "policy",
    }


def test_tied_cohort_majority_family_has_deterministic_tie_breaking_with_reordered_inputs():
    inventory = [
        _measured_text("family-a", "a", architecture_family="a", calibrated=True),
        _measured_text("family-b", "b", architecture_family="b", calibrated=True),
        _measured_text("family-c", "c", architecture_family="c", calibrated=True),
    ]
    cohort_a = [{"name": "tested-a", "digest": "ta", "architecture_family": "a"},
                {"name": "tested-b", "digest": "tb", "architecture_family": "b"}]
    cohort_b = list(reversed(cohort_a))
    first = _select(inventory, cohort=cohort_a).to_dict()
    second = _select(list(reversed(inventory)), cohort=cohort_b).to_dict()
    assert first == second
    # Tied majority is broken by family name ("a"), then existing selection
    # policy avoids same-family judges before falling back to name order.
    assert [item["name"] for item in first["final_eligible_order"]] == ["family-b", "family-c", "family-a"]


def test_qualification_consumes_the_existing_selection_order(monkeypatch):
    selection = _select([
        _measured_text("first", "1"),
        _measured_text("second", "2"),
    ], _policy(configured_fallbacks=("second", "first")))
    consumed = []

    def fake_qualify(client, candidate):
        consumed.append(candidate["name"])
        return {"model": candidate["name"], "qualified": candidate["name"] == "first"}

    monkeypatch.setattr(campaign, "qualify_judge", fake_qualify)
    selected, chain = campaign.select_qualified_campaign_judge(object(), selection)
    assert [item["name"] for item in selection.final_eligible_order] == ["second", "first"]
    assert consumed == ["second", "first"]
    assert [item["model"] for item in chain] == ["second", "first"]
    assert selected["name"] == "first"


def test_empty_exclusion_policy_means_no_exclusions():
    result = _select([
        _measured_text("qwen2.5:14b", "q"),
    ], _policy(excluded_families=()))
    assert result.exclusions == []
    assert result.selected["name"] == "qwen2.5:14b"
    assert result.rejection_reasons == []


def test_legacy_wrapper_preserves_explicit_empty_exclusion_list():
    chosen = campaign.select_campaign_judge([
        _measured_text("qwen2.5:14b", "q"),
    ], [], excluded_families=[])
    assert chosen["name"] == "qwen2.5:14b"


def test_duplicate_same_name_same_digest_deduplicates_safely():
    result = _select([
        _measured_text("same", "d"),
        _measured_text("same", "d"),
    ])
    assert [item["name"] for item in result.final_eligible_order] == ["same"]
    assert result.rejection_reasons == []


def test_duplicate_same_name_different_digest_fails_closed_regardless_of_order():
    inventory = [
        _measured_text("same", "a"),
        _measured_text("same", "b"),
    ]
    first = _select(inventory).to_dict()
    second = _select(list(reversed(inventory))).to_dict()
    assert first == second
    assert first["selected"] is None
    assert first["final_eligible_order"] == []
    assert first["rejection_reasons"] == [{
        "model": "same",
        "source": "inventory",
        "reason": "conflicting_candidate_identity",
        "digests": ["a", "b"],
    }]


def test_contradictory_supported_families_cannot_override_embedding_capability():
    result = _select([_measured_embedding("bge-m3:latest", "bge", supported_families=["text"])])
    assert result.selected is None
    assert result.rejection_reasons[0]["reason"] == "non_generative_embedding_only"
    assert result.rejection_reasons[0]["canonical_families"] == ["embedding"]


def test_supported_families_text_alone_is_not_new_judge_eligibility_evidence():
    result = _select([{"name": "planned-text", "digest": "p", "supported_families": ["text"]}])
    assert result.selected is None
    assert result.rejection_reasons[0]["reason"] == "capability_reprobe_required"


def test_family_exclusion_uses_family_identity_or_controlled_name_fallback_not_substring():
    metadata_excluded = _select([
        _measured_text("neutral-name", "q", architecture_family="qwen"),
        _measured_text("neutral-qwen2-name", "qf2", architecture_family="qwen2"),
        _measured_text("notqwen-model", "n"),
        _measured_text("qwen2.5:14b", "q2"),
    ])
    rejected = {(item["model"], item["reason"]) for item in metadata_excluded.rejection_reasons}
    assert ("neutral-name", "excluded_family") in rejected
    assert ("neutral-qwen2-name", "excluded_family") in rejected
    assert ("qwen2.5:14b", "excluded_family") in rejected
    assert metadata_excluded.selected["name"] == "notqwen-model"


def test_synthetic_bge_m3_latest_is_never_eligible():
    result = _select([_measured_embedding("bge-m3:latest", "bge")])
    assert result.selected is None
    assert result.rejection_reasons[0]["reason"] == "non_generative_embedding_only"


def test_no_judge_policy_returns_empty_selection():
    result = _select([_measured_text("llama:8b", "l")], _policy(enabled=False))
    assert result.selected is None
    assert result.final_eligible_order == []
    assert result.policy["enabled"] is False


def test_policy_from_config_uses_judge_model_as_primary_and_candidates_as_fallbacks():
    cfg = Config()
    cfg.judge_model = "primary"
    cfg.judge_candidates = ["fallback"]
    cfg.judge_allow_excluded_primary = True
    policy = campaign.JudgePolicy.from_config(cfg)
    assert policy.requested_primary == "primary"
    assert policy.configured_fallbacks == ("fallback",)
    assert policy.allow_excluded_primary is True
