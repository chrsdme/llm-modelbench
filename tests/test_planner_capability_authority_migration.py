"""Anvil Stage 2.6A phase 2: proves `planner.build_plan()`'s active/
skipped routing decision is actually driven by the new typed capability
stack (`planner._new_measured_supported_families()`), not merely
producing outputs that happen to coincide with the old
`capabilities.measured_supported_families()` answer.

Both tests here monkeypatch the OLD, still-imported
`capability_identity_compatibility()` to lie in the direction that would
flip the old code's routing decision, and confirm the real outcome does
NOT flip -- proving the new stack, not the patched function, is what
actually decides `active`/`skipped` post-migration. This is the live,
`interrogate_model(functional=True)`-produced, real-client-driven
end-to-end case (not an offline fixture-level equivalent) the migration
advice specifically called out as the one scenario old and new could
newly diverge on.
"""
from llm_modelbench import planner as planner_module
from llm_modelbench.capabilities import MeasuredCapabilityState, interrogate_model
from llm_modelbench.config import Config
from llm_modelbench.planner import build_plan


class _Client:
    def __init__(self, *, name="authority-migration:latest", digest="digest-1", text_ok=True):
        self.name = name
        self.digest = digest
        self.text_ok = text_ok
        self.base = "http://fake.invalid"

    def backend_identity(self):
        return type("Identity", (), {"backend": "mock", "implementation": "fixture", "endpoint": self.base})()

    def tags(self):
        return [{"name": self.name, "digest": self.digest, "size": 1}]

    def show(self, model):
        return {"capabilities": ["completion"], "template": "template-v1", "model_info": {"general.architecture": "fixture", "general.context_length": 4096}}

    def capability_hints(self, model):
        return ["completion"]

    def chat(self, model, prompt, **kwargs):
        return {"ok": self.text_ok, "text": "AIW_TEXT_OK" if self.text_ok else "", "error": None if self.text_ok else "not supported"}


def test_positive_scheduling_survives_a_legacy_compatibility_lie(monkeypatch):
    # A genuinely bound, compatible, measured-supported profile (real
    # interrogate_model() output, real client) -- the old code path would
    # unconditionally set fams=[] here since it gates entirely on
    # compatibility.get("compatible"). If the model is still scheduled
    # with the patched function lying "incompatible", the new stack (not
    # this patched function) is what's actually deciding.
    #
    # auto_probe=False is deliberate, not incidental: with auto_probe=True
    # the patched "incompatible" answer would land this model in
    # incompatible_profiles and trigger a REAL reprobe
    # (interrogate_models(...)), which replaces profiles[model] with a
    # freshly-built dict before the authority gate ever runs on it -- a
    # confound that would prove something else (a fresh reprobe passes on
    # its own merits) rather than what this test claims (the SAME
    # already-built profile, mislabeled "incompatible" by the patched old
    # function, is still scheduled by the new stack). With auto_probe=False
    # and the model already present in capability_profiles (not missing),
    # neither reprobe path fires, so profiles[model] stays exactly the
    # object built below, unmodified, all the way through.
    monkeypatch.setattr(
        planner_module, "capability_identity_compatibility",
        lambda profile, current_identity: {"compatible": False, "reason": "patched_always_incompatible"},
    )
    client = _Client()
    profile = interrogate_model(client, client.name, functional=True)
    assert profile["measured_capabilities"]["text"]["state"] == MeasuredCapabilityState.MEASURED_SUPPORTED.value

    plan = build_plan(client, Config(), level="short", selected_models=[client.name], auto_probe=False, capability_profiles={client.name: profile})

    assert plan["skipped_models"] == []
    assert plan["active_models"][0]["model"] == client.name
    assert "text" in plan["active_models"][0]["families"]
    # The informational field still reflects the (patched) legacy
    # function -- it's still called and still recorded, just no longer
    # authoritative.
    assert profile["capability_identity_compatibility"] == {"compatible": False, "reason": "patched_always_incompatible"}


def test_negative_skip_survives_a_legacy_compatibility_lie(monkeypatch):
    # The opposite lie: legacy always claims compatible. A profile the
    # new stack must still refuse (schema-v2 but no bound
    # capability_identity at all) must still be skipped -- proving the
    # patched "compatible" answer doesn't fool the new authority either.
    monkeypatch.setattr(
        planner_module, "capability_identity_compatibility",
        lambda profile, current_identity: {"compatible": True, "reason": "patched_always_compatible"},
    )
    client = _Client()
    unbound_profile = {
        "capability_schema_version": 2,
        "probe_protocol_version": "capability-smoke-v2",
        "measured_capabilities": {"text": {"state": MeasuredCapabilityState.MEASURED_SUPPORTED.value}},
        "declared_capabilities": ["completion"],
    }

    plan = build_plan(client, Config(), level="short", selected_models=[client.name], auto_probe=False, capability_profiles={client.name: unbound_profile})

    assert plan["active_models"] == []
    assert plan["skipped_models"][0] == {"model": client.name, "reason": "no_measured_supported_capabilities"}
    assert unbound_profile["capability_identity_compatibility"] == {"compatible": True, "reason": "patched_always_compatible"}
