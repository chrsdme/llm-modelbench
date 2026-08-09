from llm_modelbench import campaign
from llm_modelbench.config import Config


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


def _select(inventory, policy=None):
    return campaign.build_judge_selection(inventory, [], policy or _policy())


def test_text_generative_candidate_is_eligible():
    result = _select([{"name": "llama:8b", "digest": "l", "capabilities": ["completion"]}])
    assert [item["name"] for item in result.final_eligible_order] == ["llama:8b"]
    assert result.selected["canonical_families"] == ["text"]


def test_embedding_only_candidate_is_rejected():
    result = _select([{"name": "embedder", "digest": "e", "capabilities": ["embedding"]}])
    assert result.selected is None
    assert result.rejection_reasons[0]["reason"] == "non_generative_embedding_only"


def test_reranker_candidate_is_rejected():
    result = _select([{"name": "kalm-reranker:latest", "digest": "r", "capabilities": ["reranker"]}])
    assert result.selected is None
    assert result.rejection_reasons[0]["reason"] == "non_generative_reranker"


def test_vision_text_candidate_is_eligible_but_vision_only_is_rejected():
    result = _select([
        {"name": "vision-only", "digest": "v", "supported_families": ["vision"]},
        {"name": "vision-text", "digest": "vt", "supported_families": ["vision", "text"]},
    ])
    assert result.selected["name"] == "vision-text"
    assert result.rejection_reasons[0]["reason"] == "non_generative_vision_only"


def test_unknown_capability_fails_closed():
    result = _select([{"name": "mystery", "digest": "m", "capabilities": ["unknown_future_lane"]}])
    assert result.selected is None
    assert result.rejection_reasons[0]["reason"] == "unknown_or_non_generative_capability"


def test_configured_primary_precedence_is_exact_first_order():
    result = _select([
        {"name": "auto-a", "digest": "a", "capabilities": ["completion"], "calibrated": True},
        {"name": "primary", "digest": "p", "capabilities": ["completion"]},
    ], _policy(requested_primary="primary"))
    assert [item["name"] for item in result.final_eligible_order] == ["primary", "auto-a"]
    assert result.selected["name"] == "primary"


def test_unavailable_and_ineligible_primary_rejection_is_preserved_before_fallback():
    result = _select([
        {"name": "bad-primary", "digest": "b", "capabilities": ["embedding"]},
        {"name": "fallback", "digest": "f", "capabilities": ["completion"]},
    ], _policy(requested_primary="missing-primary", configured_fallbacks=("bad-primary", "fallback")))
    assert result.selected["name"] == "fallback"
    assert [(item["model"], item["reason"]) for item in result.rejection_reasons] == [
        ("missing-primary", "not_in_inventory"),
        ("bad-primary", "non_generative_embedding_only"),
    ]


def test_qwen_family_is_excluded_from_automatic_judging_by_default():
    result = _select([
        {"name": "qwen9-future:72b", "digest": "q", "capabilities": ["completion"], "calibrated": True},
        {"name": "llama:8b", "digest": "l", "capabilities": ["completion"]},
    ])
    assert result.selected["name"] == "llama:8b"
    assert result.rejection_reasons[0]["reason"] == "excluded_family"


def test_excluded_primary_requires_explicit_override():
    inventory = [
        {"name": "qwen9-future:72b", "digest": "q", "capabilities": ["completion"]},
        {"name": "llama:8b", "digest": "l", "capabilities": ["completion"]},
    ]
    blocked = _select(inventory, _policy(requested_primary="qwen9-future:72b"))
    allowed = _select(inventory, _policy(requested_primary="qwen9-future:72b", allow_excluded_primary=True))
    assert blocked.selected["name"] == "llama:8b"
    assert blocked.rejection_reasons[0]["reason"] == "excluded_family"
    assert allowed.selected["name"] == "qwen9-future:72b"
    assert allowed.selected["excluded_family_override"] is True


def test_configured_fallback_order_duplicate_removal_and_automatic_exclusions():
    result = _select([
        {"name": "auto", "digest": "a", "capabilities": ["completion"]},
        {"name": "qwen-auto", "digest": "q", "capabilities": ["completion"], "calibrated": True},
        {"name": "fallback-b", "digest": "b", "capabilities": ["completion"]},
        {"name": "fallback-a", "digest": "fa", "capabilities": ["completion"]},
    ], _policy(configured_fallbacks=("fallback-b", "fallback-a", "fallback-b")))
    assert result.configured_fallbacks == ["fallback-b", "fallback-a"]
    assert [item["name"] for item in result.final_eligible_order] == ["fallback-b", "fallback-a", "auto"]
    assert any(item["model"] == "qwen-auto" and item["reason"] == "excluded_family" for item in result.rejection_reasons)


def test_repeated_selection_is_deterministic_and_structured():
    inventory = [
        {"name": "b", "digest": "2", "capabilities": ["completion"], "priority": 1},
        {"name": "a", "digest": "1", "capabilities": ["completion"], "priority": 1},
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


def test_synthetic_bge_m3_latest_is_never_eligible():
    result = _select([{"name": "bge-m3:latest", "digest": "bge", "capabilities": ["embedding"]}])
    assert result.selected is None
    assert result.rejection_reasons[0]["reason"] == "non_generative_embedding_only"


def test_no_judge_policy_returns_empty_selection():
    result = _select([{"name": "llama:8b", "digest": "l", "capabilities": ["completion"]}], _policy(enabled=False))
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
