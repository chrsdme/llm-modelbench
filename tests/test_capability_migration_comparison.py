"""Anvil Stage 2.6A (phase 1): legacy-vs-new capability decision
comparison harness. Fixtures mirror real `interrogate_model()`/
`current_capability_identity()` shapes (verified by reading those
functions directly), covering the representative scenarios named in
this session's Stage 2.6 continuing advice
(`local_only/anvil/codex-advice-stage2-continuing.txt`): matches,
negative-direction differences, and -- the dangerous direction --
positive-authority-expansion differences, which must never silently pass
as MATCH or auto-classify as EXPECTED_CORRECTION.
"""
from llm_modelbench.capabilities import CAPABILITY_SCHEMA_VERSION, PROBE_PROTOCOL_VERSION, MeasuredCapabilityState
from llm_modelbench.capabilities import _canonical_hash as legacy_canonical_hash
from llm_modelbench.capability_migration_comparison import (
    CapabilityDecisionComparison,
    ComparisonDirection,
    ComparisonVerdict,
    compare_planner_capability_decision,
)

DIRECTION = ComparisonDirection
VERDICT = ComparisonVerdict


def _template_config(*, num_ctx=8192):
    material = {
        "template": "{{ .System }}\n{{ .Prompt }}", "parameters": f"num_ctx {num_ctx}",
        "modelfile": None, "system": None, "model_info": {"llama.context_length": num_ctx},
    }
    return {"available": True, "hash": legacy_canonical_hash(material), "material": material}


def _capability_identity(*, digest="sha256:abc123", canonical_name="qwen2.5-coder:14b", endpoint="http://127.0.0.1:11434", protocol_version=PROBE_PROTOCOL_VERSION, num_ctx=8192, model_info_present=True):
    identity = {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "backend": {"backend": "ollama", "implementation": "OllamaClient", "endpoint": endpoint},
        "runtime": {"endpoint": endpoint, "implementation": "OllamaClient"},
        "template_config": _template_config(num_ctx=num_ctx),
        "probe_protocol_version": protocol_version,
        "identity_hash": "irrelevant-composite-hash",
    }
    if model_info_present:
        identity["model"] = {
            "canonical_name": canonical_name, "backend_model_id": canonical_name,
            "digest": digest, "size": 9_000_000_000, "details": {"quantization_level": "Q4_K_M"},
        }
    else:
        # A real-world malformed/short-circuited identity capture --
        # legacy's own field-by-field checks tolerate this (None == None
        # if `current_identity` is equally malformed); the new adapter
        # does not.
        identity["model"] = None
    return identity


def _profile(*, family="text", state=MeasuredCapabilityState.MEASURED_SUPPORTED, **identity_kwargs):
    return {
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "probe_protocol_version": identity_kwargs.get("protocol_version", PROBE_PROTOCOL_VERSION),
        "model": "qwen2.5-coder:14b",
        "capability_identity": _capability_identity(**identity_kwargs),
        "declared_capabilities": ["completion"],
        "measured_capabilities": {family: {"state": state.value, "legacy_probe_state": "responded_ok", "route_scored_tasks": state == MeasuredCapabilityState.MEASURED_SUPPORTED}},
        "measured_supported_families": [family] if state == MeasuredCapabilityState.MEASURED_SUPPORTED else [],
        "functional_probes_enabled": True,
        "routing_policy": "functional_probe_required",
        "warnings": [],
    }


def test_match_when_supported_and_identity_unchanged():
    profile = _profile()
    current_identity = _capability_identity()  # identical snapshot -- nothing changed
    result = compare_planner_capability_decision(profile, "text", legacy_current_identity=current_identity)
    assert result == CapabilityDecisionComparison(
        family="text", legacy_applicable=True, legacy_reason="measured_supported",
        new_applicable=True, new_reason="measured_supported",
        direction=DIRECTION.NONE, verdict=VERDICT.MATCH,
    )


def test_match_when_unsupported_both_sides():
    profile = _profile(state=MeasuredCapabilityState.MEASURED_UNSUPPORTED)
    current_identity = _capability_identity()
    result = compare_planner_capability_decision(profile, "text", legacy_current_identity=current_identity)
    assert result.legacy_applicable is False
    assert result.new_applicable is False
    assert result.verdict == VERDICT.MATCH
    assert result.direction == DIRECTION.NONE


def test_match_when_identity_changed_both_sides_fail_closed():
    profile = _profile(digest="sha256:old-digest")
    current_identity = _capability_identity(digest="sha256:new-digest")  # model was updated
    result = compare_planner_capability_decision(profile, "text", legacy_current_identity=current_identity)
    assert result.legacy_applicable is False
    assert result.new_applicable is False
    assert result.legacy_reason == "model_digest_changed"
    assert result.new_reason == "no_current_projection"
    assert result.verdict == VERDICT.MATCH
    assert result.direction == DIRECTION.NONE


def test_match_when_profile_is_legacy_unbound_shape():
    legacy_profile = {"model": "legacy:latest", "declared_capabilities": ["completion"], "supported_families": ["text"]}
    current_identity = _capability_identity()
    result = compare_planner_capability_decision(legacy_profile, "text", legacy_current_identity=current_identity)
    assert result.legacy_applicable is False
    assert result.legacy_reason == "legacy_or_unbound_capability_profile"
    assert result.new_applicable is False
    assert result.new_reason == "no_current_projection"
    assert result.verdict == VERDICT.MATCH


def test_contraction_when_adapter_strictness_catches_malformed_identity_legacy_missed():
    # A realistic case demonstrating why the migration matters: legacy's
    # field-by-field dict compatibility check tolerates a missing "model"
    # sub-dict on BOTH sides (None == None passes its equality check), so
    # if the family is otherwise measured-supported, legacy considers it
    # applicable. The new adapter requires real structured model identity
    # material and refuses -- correctly stricter, not a regression.
    profile = _profile(model_info_present=False)
    current_identity = _capability_identity(model_info_present=False)
    result = compare_planner_capability_decision(profile, "text", legacy_current_identity=current_identity)
    assert result.legacy_applicable is True
    assert result.new_applicable is False
    assert result.new_reason == "no_current_projection"
    assert result.direction == DIRECTION.CONTRACTION
    # Not automatically EXPECTED_CORRECTION -- a human must review and
    # explicitly mark it, per the harness's conservative default.
    assert result.verdict == VERDICT.UNEXPLAINED_DIFFERENCE


def test_contraction_can_be_marked_as_known_correction_explicitly():
    profile = _profile(model_info_present=False)
    current_identity = _capability_identity(model_info_present=False)
    result = compare_planner_capability_decision(
        profile, "text", legacy_current_identity=current_identity, known_correction=True,
    )
    assert result.direction == DIRECTION.CONTRACTION
    assert result.verdict == VERDICT.EXPECTED_CORRECTION


def test_expansion_when_new_side_accepts_a_protocol_version_legacy_hardcodes_against_constant():
    # A genuine, non-contrived semantic difference: legacy's
    # capability_identity_compatibility() compares the STORED protocol
    # version against the hardcoded PROBE_PROTOCOL_VERSION constant, not
    # against whatever "current" value is supplied -- so a profile stored
    # under a different (e.g. newer) protocol version is unconditionally
    # incompatible under legacy, regardless of what's actually running
    # now. The new stack takes current_probe_protocol_version as an
    # explicit parameter, so if a caller passes the SAME non-standard
    # value the profile was stored under, the new side finds it
    # compatible where legacy never could. This is exactly the dangerous
    # "non-applicable -> applicable" direction the harness must flag by
    # default, not wave through.
    custom_protocol = "capability-smoke-v3-preview"
    profile = _profile(protocol_version=custom_protocol)
    current_identity = _capability_identity(protocol_version=custom_protocol)
    result = compare_planner_capability_decision(
        profile, "text", legacy_current_identity=current_identity,
        current_probe_protocol_version=custom_protocol,
    )
    assert result.legacy_applicable is False
    assert result.legacy_reason == "probe_protocol_version_changed"
    assert result.new_applicable is True
    assert result.new_reason == "measured_supported"
    assert result.direction == DIRECTION.EXPANSION
    assert result.verdict == VERDICT.UNEXPLAINED_DIFFERENCE


def test_expansion_is_not_silently_waved_through_even_with_known_correction_available():
    # known_correction=True still requires an explicit, deliberate call
    # -- verifies the escape hatch exists for expansions too (a caller
    # CAN mark one reviewed-and-accepted), but the harness itself never
    # infers it.
    custom_protocol = "capability-smoke-v3-preview"
    profile = _profile(protocol_version=custom_protocol)
    current_identity = _capability_identity(protocol_version=custom_protocol)
    default_result = compare_planner_capability_decision(
        profile, "text", legacy_current_identity=current_identity, current_probe_protocol_version=custom_protocol,
    )
    reviewed_result = compare_planner_capability_decision(
        profile, "text", legacy_current_identity=current_identity, current_probe_protocol_version=custom_protocol,
        known_correction=True,
    )
    assert default_result.verdict == VERDICT.UNEXPLAINED_DIFFERENCE
    assert reviewed_result.verdict == VERDICT.EXPECTED_CORRECTION


def test_match_when_metadata_only_profile_end_to_end():
    # The single most common real profile shape: every model is
    # metadata-only interrogated by default (auto_probe=False), which
    # legacy labels every family PROBE_INCONCLUSIVE and excludes from
    # measured_supported_families(). This is the integration case --
    # exercised end-to-end through compare_planner_capability_decision()
    # itself, not just the adapter and the classification logic
    # separately -- proving both sides degrade to "not applicable" for
    # their own honest reasons (legacy: never in the measured-supported
    # list; new: PROBE_INCONCLUSIVE), not coincidentally.
    profile = _profile(state=MeasuredCapabilityState.PROBE_INCONCLUSIVE)
    profile["functional_probes_enabled"] = False
    profile["routing_policy"] = "metadata_only"
    current_identity = _capability_identity()
    result = compare_planner_capability_decision(profile, "text", legacy_current_identity=current_identity)
    assert result == CapabilityDecisionComparison(
        family="text", legacy_applicable=False, legacy_reason="not_measured_supported",
        new_applicable=False, new_reason="probe_inconclusive",
        direction=DIRECTION.NONE, verdict=VERDICT.MATCH,
    )


def test_match_when_current_identity_is_none():
    profile = _profile()
    result = compare_planner_capability_decision(profile, "text", legacy_current_identity=None)
    assert result.legacy_applicable is False
    assert result.legacy_reason == "current_capability_identity_missing"
    assert result.new_applicable is False
    assert result.new_reason == "no_current_projection"
    assert result.verdict == VERDICT.MATCH
