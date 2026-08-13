"""Anvil Stage 2.6A (phase 1): the legacy-profile -> CapabilityObservation
adapter. Profile fixtures below mirror the real shape
`capabilities.interrogate_model()` actually produces (verified by reading
that function directly, not guessed), so these tests exercise the adapter
against realistic legacy data, not an idealized shape it will never see in
production.
"""
import pytest

from llm_modelbench.capabilities import CAPABILITY_SCHEMA_VERSION, PROBE_PROTOCOL_VERSION, MeasuredCapabilityState
from llm_modelbench.capabilities import _canonical_hash as legacy_canonical_hash
from llm_modelbench.capability_evidence_adapter import adapt_legacy_profile_family_to_observation
from llm_modelbench.identity import ModelArtifactIdentity


def _template_config(*, num_ctx=8192):
    # Mirrors capabilities._template_config_identity()'s real return
    # shape exactly: {"available", "hash", "material"} -- the "hash" is
    # what capability_identity_compatibility() actually compares, so
    # fixtures must carry a real hash of the material, not an arbitrary
    # string, for the adapter's hash-reuse behavior to be meaningfully
    # tested.
    material = {
        "template": "{{ .System }}\n{{ .Prompt }}",
        "parameters": f"num_ctx {num_ctx}",
        "modelfile": None,
        "system": None,
        "model_info": {"llama.context_length": num_ctx},
    }
    return {"available": True, "hash": legacy_canonical_hash(material), "material": material}


def _capability_identity(*, digest="sha256:abc123", canonical_name="qwen2.5-coder:14b", backend="ollama", endpoint="http://127.0.0.1:11434", num_ctx=8192):
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "model": {
            "canonical_name": canonical_name,
            "backend_model_id": canonical_name,
            "digest": digest,
            "size": 9_000_000_000,
            "details": {"quantization_level": "Q4_K_M"},
        },
        "backend": {"backend": backend, "implementation": "OllamaClient", "endpoint": endpoint},
        "runtime": {"endpoint": endpoint, "implementation": "OllamaClient"},
        "template_config": _template_config(num_ctx=num_ctx),
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "identity_hash": "irrelevant-composite-hash",
    }


def _measured_supported_profile(*, family="text", digest="sha256:abc123", num_ctx=8192):
    return {
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "model": "qwen2.5-coder:14b",
        "capability_identity": _capability_identity(digest=digest, num_ctx=num_ctx),
        "declared_capabilities": ["completion", "tools"],
        "measured_capabilities": {
            family: {
                "state": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
                "legacy_probe_state": "responded_ok",
                "route_scored_tasks": True,
            },
        },
        "measured_supported_families": [family],
        "functional_probes_enabled": True,
        "routing_policy": "functional_probe_required",
        "warnings": [],
    }


def _metadata_only_profile(*, family="text"):
    profile = _measured_supported_profile(family=family)
    profile["measured_capabilities"] = {
        family: {
            "state": MeasuredCapabilityState.PROBE_INCONCLUSIVE.value,
            "legacy_probe_state": "not_probed_metadata_hint",
            "route_scored_tasks": False,
        },
    }
    profile["functional_probes_enabled"] = False
    profile["routing_policy"] = "metadata_only"
    profile["measured_supported_families"] = []
    return profile


def test_measured_supported_family_adapts_faithfully():
    profile = _measured_supported_profile()
    observation = adapt_legacy_profile_family_to_observation(profile, "text")
    assert observation is not None
    assert observation.capability == "text"
    assert observation.result == MeasuredCapabilityState.MEASURED_SUPPORTED
    assert observation.probe_protocol_version == PROBE_PROTOCOL_VERSION
    assert observation.capability_schema_version == CAPABILITY_SCHEMA_VERSION
    assert observation.endpoint_identity == "http://127.0.0.1:11434"
    assert observation.declared_hint == ("completion", "tools")


def test_metadata_only_hint_becomes_inconclusive_never_supported():
    # The single most important honesty property of this adapter: a
    # metadata-only/hint-only profile must never be upgraded into a
    # positive claim.
    profile = _metadata_only_profile()
    observation = adapt_legacy_profile_family_to_observation(profile, "text")
    assert observation is not None
    assert observation.result == MeasuredCapabilityState.PROBE_INCONCLUSIVE
    assert observation.result != MeasuredCapabilityState.MEASURED_SUPPORTED


@pytest.mark.parametrize(
    "state",
    [
        MeasuredCapabilityState.MEASURED_UNSUPPORTED,
        MeasuredCapabilityState.BACKEND_UNSUPPORTED,
        MeasuredCapabilityState.NOT_APPLICABLE,
    ],
)
def test_negative_states_adapt_faithfully(state):
    profile = _measured_supported_profile()
    profile["measured_capabilities"]["text"]["state"] = state.value
    observation = adapt_legacy_profile_family_to_observation(profile, "text")
    assert observation is not None
    assert observation.result == state


def test_model_identity_derived_from_real_digest_not_fabricated():
    profile = _measured_supported_profile(digest="sha256:real-digest-value")
    observation = adapt_legacy_profile_family_to_observation(profile, "text")
    assert observation.model_identity == ModelArtifactIdentity.from_ollama_tag_row(
        {"name": "qwen2.5-coder:14b", "digest": "sha256:real-digest-value", "size": 9_000_000_000,
         "details": {"quantization_level": "Q4_K_M"}}
    )


def test_two_profiles_with_same_underlying_identity_produce_equal_identity():
    # Determinism: adapting the same underlying model/backend/template
    # twice must produce the same identity, not a fresh synthetic one
    # each time.
    profile_a = _measured_supported_profile()
    profile_b = _measured_supported_profile()
    obs_a = adapt_legacy_profile_family_to_observation(profile_a, "text")
    obs_b = adapt_legacy_profile_family_to_observation(profile_b, "text")
    assert obs_a.model_identity == obs_b.model_identity
    assert obs_a.runtime_profile_identity == obs_b.runtime_profile_identity


def test_differing_template_config_produces_differing_template_hash():
    profile_a = _measured_supported_profile(num_ctx=8192)
    profile_b = _measured_supported_profile(num_ctx=32768)
    obs_a = adapt_legacy_profile_family_to_observation(profile_a, "text")
    obs_b = adapt_legacy_profile_family_to_observation(profile_b, "text")
    assert obs_a.runtime_profile_identity.template_hash != obs_b.runtime_profile_identity.template_hash
    assert obs_a.template_config_hash != obs_b.template_config_hash


def test_template_hash_reuses_legacy_precomputed_hash_verbatim():
    # The adapter must reuse capability_identity["template_config"]["hash"]
    # directly (the exact value capability_identity_compatibility() already
    # compares), not compute a second, uncorrelated hash of its own.
    profile = _measured_supported_profile()
    observation = adapt_legacy_profile_family_to_observation(profile, "text")
    legacy_hash = profile["capability_identity"]["template_config"]["hash"]
    assert observation.runtime_profile_identity.template_hash == legacy_hash
    assert observation.template_config_hash == legacy_hash


def test_missing_digest_falls_back_like_identity_py_itself_does():
    # ModelArtifactIdentity.from_ollama_tag_row() has its own documented
    # digest-absent fallback (hash of backend+name) -- the adapter must
    # not invent a stricter or looser rule than that factory already has.
    profile = _measured_supported_profile(digest=None)
    profile["capability_identity"]["model"]["digest"] = None
    observation = adapt_legacy_profile_family_to_observation(profile, "text")
    assert observation is not None
    assert observation.model_identity == ModelArtifactIdentity.from_ollama_tag_row(
        {"name": "qwen2.5-coder:14b", "digest": None, "size": 9_000_000_000,
         "details": {"quantization_level": "Q4_K_M"}}
    )


# --- refusal cases: None, never a best-effort guess ---


def test_refuses_schema_version_mismatch():
    profile = _measured_supported_profile()
    profile["capability_schema_version"] = 1
    assert adapt_legacy_profile_family_to_observation(profile, "text") is None


def test_refuses_missing_probe_protocol_version():
    profile = _measured_supported_profile()
    profile["probe_protocol_version"] = None
    assert adapt_legacy_profile_family_to_observation(profile, "text") is None


def test_refuses_missing_capability_identity():
    profile = _measured_supported_profile()
    del profile["capability_identity"]
    assert adapt_legacy_profile_family_to_observation(profile, "text") is None


def test_refuses_malformed_capability_identity():
    profile = _measured_supported_profile()
    profile["capability_identity"] = "not_a_dict"
    assert adapt_legacy_profile_family_to_observation(profile, "text") is None


def test_refuses_family_never_assessed():
    profile = _measured_supported_profile(family="text")
    assert adapt_legacy_profile_family_to_observation(profile, "vision") is None


def test_refuses_unrecognized_state_string():
    profile = _measured_supported_profile()
    profile["measured_capabilities"]["text"]["state"] = "not_a_real_state"
    assert adapt_legacy_profile_family_to_observation(profile, "text") is None


def test_refuses_legacy_or_unbound_profile_shape():
    # The exact real-world case this adapter must never silently convert:
    # today's "legacy_or_unbound_capability_profile" shape (no bound
    # identity object -- capabilities.capability_identity_compatibility()
    # itself already refuses these).
    legacy_profile = {
        "model": "legacy:latest",
        "declared_capabilities": ["completion"],
        "supported_families": ["text"],
    }
    assert adapt_legacy_profile_family_to_observation(legacy_profile, "text") is None


def test_refuses_non_mapping_profile():
    assert adapt_legacy_profile_family_to_observation("not_a_profile", "text") is None
    assert adapt_legacy_profile_family_to_observation(None, "text") is None
