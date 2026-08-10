import json
from pathlib import Path

from llm_modelbench import campaign, repair
from llm_modelbench.capabilities import (
    CAPABILITY_SCHEMA_VERSION,
    PROBE_PROTOCOL_VERSION,
    MeasuredCapabilityState,
    capability_identity_compatibility,
    current_capability_identity,
    interrogate_model,
)
from llm_modelbench.config import Config
from llm_modelbench.planner import build_plan
from llm_modelbench.runner import _task_hash
from llm_modelbench.tasks import TASKS


def _task(task_id):
    return next(task for task in TASKS if task.id == task_id)


class CapabilityClient:
    def __init__(
        self,
        *,
        name="model:latest",
        digest="digest-1",
        capabilities=None,
        text_ok=True,
        embedding_ok=False,
        tools_ok=False,
        tools_contract=True,
        fim_ok=False,
        fim_contract=True,
        vision_ok=False,
        transient_family=None,
        backend="mock",
        endpoint="http://fake.invalid",
        template="template-v1",
    ):
        self.name = name
        self.digest = digest
        self._capabilities = capabilities or ["completion"]
        self.text_ok = text_ok
        self.embedding_ok = embedding_ok
        self.tools_ok = tools_ok
        self.tools_contract = tools_contract
        self.fim_ok = fim_ok
        self.fim_contract = fim_contract
        self.vision_ok = vision_ok
        self.transient_family = transient_family
        self.backend = backend
        self.base = endpoint
        self.template = template

    def backend_identity(self):
        return type("Identity", (), {"backend": self.backend, "implementation": "fixture", "endpoint": self.base})()

    def tags(self):
        return [{"name": self.name, "size": 1, "digest": self.digest}]

    def show(self, model):
        return {
            "capabilities": list(self._capabilities),
            "template": self.template,
            "model_info": {"general.architecture": "fixture", "general.context_length": 4096},
        }

    def capabilities(self, model):
        return list(self._capabilities)

    def chat(self, model, prompt, **kwargs):
        if kwargs.get("images"):
            if self.transient_family == "vision":
                raise TimeoutError("temporary vision timeout")
            return {"ok": True, "text": "V7K9Q2" if self.vision_ok else "ordinary text"}
        if self.transient_family == "text":
            raise TimeoutError("temporary text timeout")
        return {"ok": self.text_ok, "text": "AIW_TEXT_OK" if self.text_ok else "", "error": None if self.text_ok else "not supported"}

    def embed(self, model, texts):
        if self.transient_family == "embedding":
            raise TimeoutError("temporary embedding timeout")
        if not self.embedding_ok:
            raise RuntimeError("embedding not supported")
        return [[1.0, 2.0], [3.0, 4.0]]

    def chat_tools(self, model, prompt, **kwargs):
        if self.transient_family == "tools":
            raise TimeoutError("temporary tools timeout")
        if not self.tools_ok:
            return {"ok": False, "error": "not supported", "tool_calls": []}
        if not self.tools_contract:
            return {"ok": True, "text": "{\"name\":\"lookup_weather\"}", "tool_calls": []}
        return {
            "ok": True,
            "tool_calls": [{"function": {"name": "lookup_weather", "arguments": {"city": "Paris", "units": "celsius"}}}],
        }

    def generate_suffix(self, model, prompt, *, suffix, **kwargs):
        if self.transient_family == "insert":
            raise TimeoutError("temporary insert timeout")
        if not self.fim_ok:
            return {"ok": False, "error": "not supported"}
        return {"ok": True, "text": "value.strip().lower()" if self.fim_contract else "'BLUE'"}


def _profile(model, states, *, digest="digest-1", backend="mock", endpoint="http://fake.invalid", template="template-v1"):
    client = CapabilityClient(name=model, digest=digest, capabilities=["completion"], backend=backend, endpoint=endpoint, template=template)
    identity = current_capability_identity(client, model)
    measured = {
        family: {"state": state, "route_scored_tasks": state == MeasuredCapabilityState.MEASURED_SUPPORTED.value}
        for family, state in states.items()
    }
    supported = [family for family, item in measured.items() if item["state"] == MeasuredCapabilityState.MEASURED_SUPPORTED.value]
    return {
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "model": model,
        "capability_identity": identity,
        "declared_capabilities": ["completion"],
        "supported_families": supported,
        "measured_supported_families": supported,
        "measured_capabilities": measured,
        "functional_probes_enabled": True,
    }


def _write_repair_run(root: Path, model: str, profile, task_id: str, error_kind: str):
    run = root / "fleet"
    run.mkdir(parents=True)
    task = _task(task_id)
    row = {
        "model": model,
        "model_digest_resolved": profile["capability_identity"]["model"]["digest"],
        "task": task.id,
        "category": task.category,
        "family": task.family,
        "task_hash": _task_hash(task),
        "score": None,
        "error_kind": error_kind,
        "reason": error_kind,
        "timestamp": "2026-08-10T00:00:00Z",
    }
    (run / "raw_results.jsonl").write_text(json.dumps(row) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": "full"}))
    (run / "filters.json").write_text(json.dumps({"level": "full", "think": "auto"}))
    (run / "model_identities.json").write_text(json.dumps({model: {"digest": row["model_digest_resolved"], "size": 1}}))
    (run / "capability_report.json").write_text(json.dumps({model: profile}))
    return run


def test_qwen3_embedding_metadata_tools_plans_embedding_only_and_not_judge_or_recovery(tmp_path):
    model = "qwen3-embedding:latest"
    client = CapabilityClient(name=model, capabilities=["embedding", "tools"], embedding_ok=True, text_ok=False)
    profile = interrogate_model(client, model, functional=True)
    plan = build_plan(client, Config(), level="short", selected_models=[model], auto_probe=True, capability_profiles={model: profile})

    assert plan["active_models"][0]["families"] == ["embedding"]
    assert all(_task(task_id).family == "embedding" for task_id in plan["active_models"][0]["tasks"])

    selection = campaign.build_judge_selection([{
        "name": model,
        "digest": "digest-1",
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "measured_capabilities": profile["measured_capabilities"],
    }], [], campaign.JudgePolicy(excluded_families=()))
    assert selection.selected is None

    runs = tmp_path / "runs"
    runs.mkdir()
    unsupported_text = _profile(model, {
        "embedding": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
        "text": MeasuredCapabilityState.MEASURED_UNSUPPORTED.value,
    })
    _write_repair_run(runs, model, unsupported_text, "txt_sort", "empty_output")
    recovery = repair.build_plan(runs, run_id="fleet", include_missing=False)
    assert not any(action.kind == "retry_generation" for action in recovery.actions)


def test_bge_m3_embedding_only_is_not_generation_judge():
    result = campaign.build_judge_selection([{
        "name": "bge-m3:latest",
        "digest": "bge",
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "measured_capabilities": {"embedding": {"state": MeasuredCapabilityState.MEASURED_SUPPORTED.value}},
    }], [], campaign.JudgePolicy(excluded_families=()))
    assert result.selected is None
    assert result.rejection_reasons[0]["reason"] == "non_generative_embedding_only"


def test_native_tool_contract_failure_is_measured_unsupported_and_not_planned():
    model = "toolish:latest"
    client = CapabilityClient(name=model, capabilities=["completion", "tools"], text_ok=True, tools_ok=True, tools_contract=False)
    profile = interrogate_model(client, model, functional=True)
    assert profile["measured_capabilities"]["tools"]["state"] == MeasuredCapabilityState.MEASURED_UNSUPPORTED.value
    assert "tools" not in profile["supported_families"]
    plan = build_plan(client, Config(), level="short", selected_models=[model], auto_probe=True, capability_profiles={model: profile})
    assert "agent_native_tool_call" not in plan["active_models"][0]["tasks"]


def test_fim_contract_failure_is_measured_unsupported_and_not_planned():
    model = "coder:latest"
    client = CapabilityClient(name=model, capabilities=["completion", "insert"], text_ok=True, fim_ok=True, fim_contract=False)
    profile = interrogate_model(client, model, functional=True)
    assert profile["measured_capabilities"]["insert"]["state"] == MeasuredCapabilityState.MEASURED_UNSUPPORTED.value
    plan = build_plan(client, Config(), level="short", selected_models=[model], auto_probe=True, capability_profiles={model: profile})
    assert "fim_suffix_assertion" not in plan["active_models"][0]["tasks"]


def test_inconclusive_probe_does_not_schedule_or_recover(tmp_path):
    model = "temporary-vlm:latest"
    client = CapabilityClient(name=model, capabilities=["completion", "vision"], text_ok=True, transient_family="vision")
    profile = interrogate_model(client, model, functional=True)
    assert profile["measured_capabilities"]["vision"]["state"] == MeasuredCapabilityState.PROBE_INCONCLUSIVE.value
    plan = build_plan(client, Config(), level="short", categories=["ocr"], selected_models=[model], auto_probe=True, capability_profiles={model: profile})
    assert plan["active_models"] == []

    runs = tmp_path / "runs"
    runs.mkdir()
    unresolved = _profile(model, {"text": MeasuredCapabilityState.PROBE_INCONCLUSIVE.value})
    _write_repair_run(runs, model, unresolved, "txt_sort", "empty_output")
    recovery = repair.build_plan(runs, run_id="fleet", include_missing=False)
    assert recovery.actions == []
    assert any(item["kind"] == "capability_reprobe_required" for item in recovery.observations)


def test_recovery_requires_positive_text_and_tool_applicability(tmp_path):
    model = "fixture:latest"

    unsupported_runs = tmp_path / "unsupported_text"
    unsupported_runs.mkdir()
    _write_repair_run(unsupported_runs, model, _profile(model, {
        "embedding": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
        "text": MeasuredCapabilityState.MEASURED_UNSUPPORTED.value,
    }), "txt_sort", "empty_output")
    unsupported = repair.build_plan(unsupported_runs, run_id="fleet", include_missing=False)
    assert not any(action.kind == "retry_generation" for action in unsupported.actions)

    tools_runs = tmp_path / "unsupported_tools"
    tools_runs.mkdir()
    _write_repair_run(tools_runs, model, _profile(model, {
        "text": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
        "tools": MeasuredCapabilityState.MEASURED_UNSUPPORTED.value,
    }), "agent_native_tool_call", "empty_output")
    tools = repair.build_plan(tools_runs, run_id="fleet", include_missing=False)
    assert tools.actions == []

    supported_runs = tmp_path / "supported_text"
    supported_runs.mkdir()
    _write_repair_run(supported_runs, model, _profile(model, {
        "text": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
    }), "txt_sort", "empty_output")
    supported = repair.build_plan(supported_runs, run_id="fleet", include_missing=False)
    assert [action.kind for action in supported.actions] == ["retry_generation"]


def test_capability_identity_rejects_digest_backend_template_and_protocol_changes():
    model = "fixture:latest"
    profile = _profile(model, {"text": MeasuredCapabilityState.MEASURED_SUPPORTED.value})
    assert capability_identity_compatibility(profile, current_capability_identity(CapabilityClient(name=model), model))["compatible"]
    assert capability_identity_compatibility(profile, current_capability_identity(CapabilityClient(name=model, digest="digest-2"), model))["reason"] == "model_digest_changed"
    assert capability_identity_compatibility(profile, current_capability_identity(CapabilityClient(name=model, backend="other"), model))["reason"] == "backend_changed"
    assert capability_identity_compatibility(profile, current_capability_identity(CapabilityClient(name=model, template="template-v2"), model))["reason"] == "template_config_changed"
    changed_protocol = dict(profile)
    changed_protocol["capability_identity"] = dict(profile["capability_identity"])
    changed_protocol["capability_identity"]["probe_protocol_version"] = "old"
    assert capability_identity_compatibility(changed_protocol, current_capability_identity(CapabilityClient(name=model), model))["reason"] == "probe_protocol_version_changed"


def test_legacy_profile_remains_readable_but_not_authoritative_for_new_plan():
    model = "legacy:latest"
    legacy = {"model": model, "declared_capabilities": ["completion"], "supported_families": ["text"]}
    client = CapabilityClient(name=model, capabilities=["completion"])
    plan = build_plan(client, Config(), level="short", selected_models=[model], auto_probe=False, capability_profiles={model: legacy})
    assert plan["active_models"] == []
    assert plan["skipped_models"][0]["reason"] == "no_measured_supported_capabilities"


def test_planner_recovery_readiness_consistency_for_unsupported_family(tmp_path):
    model = "unsupported:latest"
    profile = _profile(model, {
        "text": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
        "tools": MeasuredCapabilityState.MEASURED_UNSUPPORTED.value,
    })
    client = CapabilityClient(name=model, capabilities=["completion", "tools"])
    plan = build_plan(client, Config(), level="short", selected_models=[model], auto_probe=False, capability_profiles={model: profile})
    assert "agent_native_tool_call" not in plan["active_models"][0]["tasks"]

    runs = tmp_path / "runs"
    runs.mkdir()
    _write_repair_run(runs, model, profile, "agent_native_tool_call", "empty_output")
    recovery = repair.build_plan(runs, run_id="fleet", include_missing=False)
    assert recovery.actions == []

    paths = campaign.resolve_paths("cap_consistency", campaigns_root=tmp_path / "campaigns")
    campaign.create_campaign_dirs(paths)
    campaign.write_manifest(paths, campaign.CampaignManifest.new("cap_consistency", models=[model]))
    row = {
        "model": model,
        "model_digest_resolved": "digest-1",
        "task": "agent_native_tool_call",
        "task_hash": _task_hash(_task("agent_native_tool_call")),
        "score": None,
        "error_kind": "capability_unavailable",
        "reason": "capability unavailable",
    }
    paths.primary_raw_results.write_text(json.dumps(row) + "\n")
    summary = campaign.write_readiness(paths, [row])
    assert summary["readiness"] == "ready_for_adoption"
    assert summary["capability_unavailable"] == 1


def test_positive_multicapability_model_keeps_all_lanes():
    model = "future-coder-vlm:latest"
    client = CapabilityClient(
        name=model,
        capabilities=["completion", "tools", "vision", "insert"],
        text_ok=True,
        tools_ok=True,
        vision_ok=True,
        fim_ok=True,
    )
    profile = interrogate_model(client, model, functional=True)
    plan = build_plan(client, Config(), level="short", selected_models=[model], auto_probe=True, capability_profiles={model: profile})
    tasks = set(plan["active_models"][0]["tasks"])
    assert {"txt_sort", "agent_native_tool_call", "fim_suffix_assertion", "ocr_invoice"} <= tasks
