"""Anvil Stage 2.6B: proves `runner.run()`'s active/skipped scheduling
decision -- and the family list actually used to select tasks -- is
driven by the new typed capability stack
(`capability_evidence_adapter.new_measured_supported_families()`), not
the legacy `capabilities.capability_identity_compatibility()` this
module still imports for the informational field and the reprobe-trigger
heuristic. Mirrors `tests/test_planner_capability_authority_migration.py`
one level up the stack (runner, not planner), reusing the same
monkeypatch-the-old-function-to-lie technique the migration advice
explicitly asked for.
"""
import json

from llm_modelbench import capabilities as capabilities_module
from llm_modelbench import runner as runner_module
from llm_modelbench.capabilities import CAPABILITY_SCHEMA_VERSION, PROBE_PROTOCOL_VERSION, MeasuredCapabilityState, current_capability_identity
from llm_modelbench.config import Config
from llm_modelbench.ollama import MockClient

MODEL = "qwen2.5-coder:14b"


def _bound_profile(client):
    identity = current_capability_identity(client, MODEL)
    return {MODEL: {
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "capability_identity": identity,
        "declared_capabilities": ["completion"],
        "measured_capabilities": {"text": {"state": MeasuredCapabilityState.MEASURED_SUPPORTED.value}},
    }}


def test_positive_scheduling_survives_a_legacy_compatibility_lie(monkeypatch, tmp_path):
    # A genuinely bound, identity-compatible, measured-supported profile.
    # The old code path gated fams entirely on
    # compatibility.get("compatible"), so patching that function to lie
    # "incompatible" would have unconditionally produced fams=[] (and a
    # skip) under the pre-migration code. auto_probe=False so the lie
    # can't trigger a real reprobe that would launder the result --
    # same deliberate confound-avoidance as the planner test.
    monkeypatch.setattr(
        capabilities_module, "capability_identity_compatibility",
        lambda profile, current_identity: {"compatible": False, "reason": "patched_always_incompatible"},
    )
    client = MockClient()
    profile = _bound_profile(client)

    out_dir = runner_module.run(
        client, Config(fingerprint=False), level="smoke", out_dir=tmp_path,
        include=None, exclude=None, skip_offload=False, categories=None,
        task_ids=["py_anagram"], resume=False, live_ui="off", fingerprint_enabled=False,
        selected_models=[MODEL], capability_profiles=profile, auto_probe=False,
    )

    skipped_models = json.loads((out_dir / "skipped_models.json").read_text())
    assert not any(item.get("model") == MODEL for item in skipped_models)
    raw_rows = [json.loads(line) for line in (out_dir / "raw_results.jsonl").read_text().splitlines() if line.strip()]
    assert any(row.get("model") == MODEL and row.get("task") == "py_anagram" for row in raw_rows)
    # The informational field still reflects the patched legacy answer --
    # it's still called and still recorded, just no longer authoritative.
    assert profile[MODEL]["capability_identity_compatibility"] == {"compatible": False, "reason": "patched_always_incompatible"}


def test_negative_skip_survives_a_legacy_compatibility_lie(monkeypatch, tmp_path):
    # The opposite lie: legacy always claims compatible. A profile the
    # new stack must still refuse (schema-v2 but no bound
    # capability_identity at all) must still be skipped.
    monkeypatch.setattr(
        capabilities_module, "capability_identity_compatibility",
        lambda profile, current_identity: {"compatible": True, "reason": "patched_always_compatible"},
    )
    client = MockClient()
    unbound_profile = {MODEL: {
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "measured_capabilities": {"text": {"state": MeasuredCapabilityState.MEASURED_SUPPORTED.value}},
        "declared_capabilities": ["completion"],
    }}

    out_dir = runner_module.run(
        client, Config(fingerprint=False), level="smoke", out_dir=tmp_path,
        include=None, exclude=None, skip_offload=False, categories=None,
        task_ids=["py_anagram"], resume=False, live_ui="off", fingerprint_enabled=False,
        selected_models=[MODEL], capability_profiles=unbound_profile, auto_probe=False,
    )

    skipped_models = json.loads((out_dir / "skipped_models.json").read_text())
    assert {"model": MODEL, "reason": "no_measured_supported_capabilities"} in skipped_models
    raw_path = out_dir / "raw_results.jsonl"
    raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()] if raw_path.exists() else []
    assert not any(row.get("model") == MODEL for row in raw_rows)
    assert unbound_profile[MODEL]["capability_identity_compatibility"] == {"compatible": True, "reason": "patched_always_compatible"}
