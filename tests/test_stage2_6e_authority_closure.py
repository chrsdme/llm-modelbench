"""Anvil Stage 2.6E -- consolidated authority-closure meta-regression.

Each of the four live capability-authority consumers already has its own
deep, dedicated decisive-lie proof pair (`test_planner_capability_authority_migration.py`,
`test_runner_capability_authority_migration.py`,
`test_repair_capability_authority_migration.py`,
`test_campaign_judge_capability_authority_migration.py`). This file does not
duplicate that depth -- per the 2.6E go-ahead advice ("this does not
necessarily require new duplicate tests. A consolidated test/report can
assert those existing guarantees remain present"), it re-derives one
compact positive-lie and one compact negative-lie proof per consumer,
purely so the closure claim "all four migrated consumers ignore legacy
authority if it lies, in both directions" is a single, durable, auditable,
machine-checked fact in one place -- not just a claim in prose scattered
across four other files.

If any one of these four fails, it means a legacy positive lie can once
again admit a candidate the typed Stage 2 stack would refuse (or a legacy
negative lie can once again block one the typed stack would admit) -- the
exact Stage 2.6E closure condition this file exists to guard.
"""
import json

from llm_modelbench import campaign, planner as planner_module, repair
from llm_modelbench import repair as repair_module
from llm_modelbench import runner as runner_module
from llm_modelbench.capabilities import (
    CAPABILITY_SCHEMA_VERSION,
    PROBE_PROTOCOL_VERSION,
    MeasuredCapabilityState,
    current_capability_identity,
    interrogate_model,
)
from llm_modelbench.config import Config
from llm_modelbench.ollama import MockClient
from llm_modelbench.planner import build_plan
from llm_modelbench.runner import _task_hash
from llm_modelbench.tasks import TASKS

MSC = MeasuredCapabilityState


class _LiveClient:
    def __init__(self, *, name="closure-audit:latest", digest="digest-1", text_ok=True):
        self.name, self.digest, self.text_ok, self.base = name, digest, text_ok, "http://fake.invalid"

    def backend_identity(self):
        return type("Identity", (), {"backend": "mock", "implementation": "fixture", "endpoint": self.base})()

    def tags(self):
        return [{"name": self.name, "digest": self.digest, "size": 1}]

    def show(self, model):
        return {"capabilities": ["completion"], "template": "template-v1", "model_info": {}}

    def capability_hints(self, model):
        return ["completion"]

    def chat(self, model, prompt, **kwargs):
        return {"ok": self.text_ok, "text": "AIW_TEXT_OK" if self.text_ok else "", "error": None if self.text_ok else "not supported"}


def test_planner_legacy_positive_lie_cannot_authorize(monkeypatch):
    monkeypatch.setattr(planner_module, "capability_identity_compatibility",
                         lambda profile, current_identity: {"compatible": False, "reason": "patched_always_incompatible"})
    client = _LiveClient(name="planner-closure:latest")
    profile = interrogate_model(client, client.name, functional=True)
    assert profile["measured_capabilities"]["text"]["state"] == MSC.MEASURED_SUPPORTED.value

    plan = build_plan(client, Config(), level="short", selected_models=[client.name], auto_probe=False, capability_profiles={client.name: profile})
    assert plan["skipped_models"] == []
    assert plan["active_models"][0]["model"] == client.name


def test_planner_legacy_negative_lie_cannot_suppress_valid_authority(monkeypatch):
    monkeypatch.setattr(planner_module, "capability_identity_compatibility",
                         lambda profile, current_identity: {"compatible": True, "reason": "patched_always_compatible"})
    client = _LiveClient(name="planner-closure-neg:latest")
    unbound_profile = {
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION, "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "measured_capabilities": {"text": {"state": MSC.MEASURED_SUPPORTED.value}}, "declared_capabilities": ["completion"],
    }
    plan = build_plan(client, Config(), level="short", selected_models=[client.name], auto_probe=False, capability_profiles={client.name: unbound_profile})
    assert plan["active_models"] == []


def test_runner_legacy_positive_lie_cannot_authorize(monkeypatch, tmp_path):
    from llm_modelbench import capabilities as capabilities_module
    monkeypatch.setattr(capabilities_module, "capability_identity_compatibility",
                         lambda profile, current_identity: {"compatible": False, "reason": "patched_always_incompatible"})
    model = "qwen2.5-coder:14b"
    client = MockClient()
    identity = current_capability_identity(client, model)
    profile = {model: {
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION, "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "capability_identity": identity, "declared_capabilities": ["completion"],
        "measured_capabilities": {"text": {"state": MSC.MEASURED_SUPPORTED.value}},
    }}
    out_dir = runner_module.run(
        client, Config(fingerprint=False), level="smoke", out_dir=tmp_path, include=None, exclude=None,
        skip_offload=False, categories=None, task_ids=["py_anagram"], resume=False, live_ui="off",
        fingerprint_enabled=False, selected_models=[model], capability_profiles=profile, auto_probe=False,
    )
    skipped = json.loads((out_dir / "skipped_models.json").read_text())
    assert not any(item.get("model") == model for item in skipped)


def test_runner_legacy_negative_lie_cannot_suppress_valid_authority(monkeypatch, tmp_path):
    from llm_modelbench import capabilities as capabilities_module
    monkeypatch.setattr(capabilities_module, "capability_identity_compatibility",
                         lambda profile, current_identity: {"compatible": True, "reason": "patched_always_compatible"})
    model = "qwen2.5-coder:14b"
    client = MockClient()
    unbound_profile = {model: {
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION, "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "measured_capabilities": {"text": {"state": MSC.MEASURED_SUPPORTED.value}}, "declared_capabilities": ["completion"],
    }}
    out_dir = runner_module.run(
        client, Config(fingerprint=False), level="smoke", out_dir=tmp_path, include=None, exclude=None,
        skip_offload=False, categories=None, task_ids=["py_anagram"], resume=False, live_ui="off",
        fingerprint_enabled=False, selected_models=[model], capability_profiles=unbound_profile, auto_probe=False,
    )
    skipped = json.loads((out_dir / "skipped_models.json").read_text())
    assert any(item.get("model") == model for item in skipped)


def _task(task_id):
    return next(t for t in TASKS if t.id == task_id)


def _repair_profile(model, digest):
    client = _LiveClient(name=model, digest=digest)
    identity = current_capability_identity(client, model)
    return {
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION, "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "model": model, "capability_identity": identity, "declared_capabilities": ["completion"],
        "measured_capabilities": {"text": {"state": MSC.MEASURED_SUPPORTED.value, "route_scored_tasks": True}},
    }


def _write_repair_run(root, model, profile, task_id, error_kind, *, source_digest):
    run = root / "fleet"
    run.mkdir(parents=True)
    task = _task(task_id)
    row = {
        "model": model, "task": task.id, "category": task.category, "family": task.family,
        "task_hash": _task_hash(task), "score": None, "error_kind": error_kind, "reason": error_kind,
        "timestamp": "2026-08-13T00:00:00Z", "model_digest_resolved": source_digest,
    }
    (run / "raw_results.jsonl").write_text(json.dumps(row) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": "full"}))
    (run / "filters.json").write_text(json.dumps({"level": "full", "think": "auto"}))
    (run / "model_identities.json").write_text(json.dumps({model: {"digest": source_digest, "size": 1}}))
    (run / "capability_report.json").write_text(json.dumps({model: profile}))
    return run


def test_repair_legacy_positive_lie_cannot_authorize(monkeypatch, tmp_path):
    model = "repair-closure:latest"
    profile = _repair_profile(model, "digest-1")
    _write_repair_run(tmp_path / "runs", model, profile, "txt_sort", "empty_output", source_digest="digest-1")

    monkeypatch.setattr(repair_module, "capability_identity_compatibility",
                         lambda profile, current_identity: {"compatible": False, "reason": "patched_always_incompatible"})
    plan = repair.build_plan(tmp_path / "runs", run_id="fleet", include_missing=False)
    assert [action.kind for action in plan.actions] == ["retry_generation"]


def test_repair_legacy_negative_lie_cannot_suppress_valid_authority(monkeypatch, tmp_path):
    model = "repair-closure-neg:latest"
    profile = _repair_profile(model, "digest-1")
    _write_repair_run(tmp_path / "runs", model, profile, "txt_sort", "empty_output", source_digest="digest-2")

    monkeypatch.setattr(repair_module, "capability_identity_compatibility",
                         lambda profile, current_identity: {"compatible": True, "reason": "patched_always_compatible"})
    plan = repair.build_plan(tmp_path / "runs", run_id="fleet", include_missing=False)
    assert plan.actions == []


def _judge_candidate(name, digest):
    client = _LiveClient(name=name, digest=digest)
    identity = current_capability_identity(client, name)
    return {
        "name": name, "digest": digest, "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "probe_protocol_version": PROBE_PROTOCOL_VERSION, "capability_identity": identity,
        "capability_identity_compatibility": {"compatible": True, "reason": "identity_match"},
        "measured_capabilities": {"text": {"state": MSC.MEASURED_SUPPORTED.value}},
    }


def test_judge_legacy_positive_lie_cannot_authorize(monkeypatch):
    monkeypatch.setattr(campaign, "_judge_measured_text_state", lambda candidate: MSC.MEASURED_SUPPORTED.value)
    item = {"name": "judge-closure:latest", "digest": "digest-x", "capability_schema_version": CAPABILITY_SCHEMA_VERSION}
    assert campaign._judge_capability_authorized_families(item) == []
    assert campaign._judge_capability_rejection(item) is not None


def test_judge_legacy_negative_lie_cannot_suppress_valid_authority(monkeypatch):
    monkeypatch.setattr(campaign, "_judge_measured_text_state", lambda candidate: MSC.MEASURED_UNSUPPORTED.value)
    item = _judge_candidate("judge-closure-neg:latest", "digest-y")
    assert "text" in campaign._judge_capability_authorized_families(item)
    assert campaign._judge_capability_rejection(item) is None
