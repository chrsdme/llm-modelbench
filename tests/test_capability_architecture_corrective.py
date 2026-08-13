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
        modified_at="2026-08-10T00:00:00Z",
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
        self.modified_at = modified_at

    def backend_identity(self):
        return type("Identity", (), {"backend": self.backend, "implementation": "fixture", "endpoint": self.base})()

    def tags(self):
        row = {"name": self.name, "size": 1, "modified_at": self.modified_at}
        if self.digest is not None:
            row["digest"] = self.digest
        return [row]

    def show(self, model):
        return {
            "capabilities": list(self._capabilities),
            "template": self.template,
            "model_info": {"general.architecture": "fixture", "general.context_length": 4096},
        }

    def capability_hints(self, model):
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
        "capability_identity_compatibility": {"compatible": True, "reason": "identity_match"},
    }


def _write_repair_run(root: Path, model: str, profile, task_id: str, error_kind: str, *, source_digest=None, write_identity=True):
    run = root / "fleet"
    run.mkdir(parents=True)
    task = _task(task_id)
    if source_digest is None:
        source_digest = profile.get("capability_identity", {}).get("model", {}).get("digest")
    row = {
        "model": model,
        "task": task.id,
        "category": task.category,
        "family": task.family,
        "task_hash": _task_hash(task),
        "score": None,
        "error_kind": error_kind,
        "reason": error_kind,
        "timestamp": "2026-08-10T00:00:00Z",
    }
    if source_digest:
        row["model_digest_resolved"] = source_digest
    (run / "raw_results.jsonl").write_text(json.dumps(row) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": "full"}))
    (run / "filters.json").write_text(json.dumps({"level": "full", "think": "auto"}))
    identity = {"size": 1}
    if source_digest and write_identity:
        identity["digest"] = source_digest
    (run / "model_identities.json").write_text(json.dumps({model: identity}))
    (run / "capability_report.json").write_text(json.dumps({model: profile}))
    return run


def _write_unknown_repair_run(root: Path, model: str, profile, task_id: str, error_kind: str, *, source_digest="digest-1"):
    run = root / "fleet"
    run.mkdir(parents=True)
    row = {
        "model": model,
        "task": task_id,
        "task_hash": "unknown-task-hash",
        "score": None,
        "error_kind": error_kind,
        "reason": error_kind,
        "timestamp": "2026-08-10T00:00:00Z",
    }
    if source_digest:
        row["model_digest_resolved"] = source_digest
    (run / "raw_results.jsonl").write_text(json.dumps(row) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": "full"}))
    (run / "filters.json").write_text(json.dumps({"level": "full", "think": "auto"}))
    identity = {"size": 1}
    if source_digest:
        identity["digest"] = source_digest
    (run / "model_identities.json").write_text(json.dumps({model: identity}))
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
        "capability_identity": profile["capability_identity"],
        "capability_identity_compatibility": {"compatible": True, "reason": "identity_match"},
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
        "capability_identity": _profile("bge-m3:latest", {
            "embedding": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
        }, digest="bge")["capability_identity"],
        "capability_identity_compatibility": {"compatible": True, "reason": "identity_match"},
        "measured_capabilities": {"embedding": {"state": MeasuredCapabilityState.MEASURED_SUPPORTED.value}},
    }], [], campaign.JudgePolicy(excluded_families=()))
    assert result.selected is None
    assert result.rejection_reasons[0]["reason"] == "non_generative_embedding_only"


def test_judge_eligibility_requires_schema_v2_measured_text_support():
    policy = campaign.JudgePolicy(excluded_families=())

    legacy = campaign.build_judge_selection([{
        "name": "legacy-text",
        "digest": "legacy",
        "supported_families": ["text"],
    }], [], policy)
    assert legacy.selected is None
    assert legacy.rejection_reasons[0]["reason"] == "capability_reprobe_required"

    missing = campaign.build_judge_selection([{
        "name": "missing-profile",
        "digest": "missing",
    }], [], policy)
    assert missing.selected is None
    assert missing.rejection_reasons[0]["reason"] == "capability_reprobe_required"

    metadata = campaign.build_judge_selection([{
        "name": "metadata-only",
        "digest": "metadata",
        "capabilities": ["completion", "tools"],
    }], [], policy)
    assert metadata.selected is None
    assert metadata.rejection_reasons[0]["reason"] == "capability_reprobe_required"

    supported_profile = _profile("measured-text", {
        "text": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
    }, digest="text")
    supported = campaign.build_judge_selection([{**supported_profile, "name": "measured-text", "digest": "text"}], [], policy)
    assert supported.selected["name"] == "measured-text"

    unbound = campaign.build_judge_selection([{
        "name": "unbound-v2-text",
        "digest": "unbound",
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "measured_capabilities": {"text": {"state": MeasuredCapabilityState.MEASURED_SUPPORTED.value}},
    }], [], policy)
    assert unbound.selected is None
    assert unbound.rejection_reasons[0]["reason"] == "capability_reprobe_required"

    incomplete_profile = _profile("incomplete-text", {
        "text": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
    }, digest="incomplete")
    incomplete_profile["capability_identity"] = {"model": {"digest": "incomplete"}}
    incomplete = campaign.build_judge_selection([{**incomplete_profile, "name": "incomplete-text", "digest": "incomplete"}], [], policy)
    assert incomplete.selected is None
    assert incomplete.rejection_reasons[0]["reason"] == "capability_reprobe_required"

    mismatch_profile = _profile("mismatch-text", {
        "text": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
    }, digest="DIGEST_A")
    mismatch_profile["current_capability_identity"] = current_capability_identity(CapabilityClient(name="mismatch-text", digest="DIGEST_B"), "mismatch-text")
    mismatch = campaign.build_judge_selection([{**mismatch_profile, "name": "mismatch-text", "digest": "DIGEST_A"}], [], policy)
    assert mismatch.selected is None
    assert mismatch.rejection_reasons[0]["reason"] == "capability_reprobe_required"

    backend_mismatch = _profile("backend-mismatch", {
        "text": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
    }, digest="same")
    backend_mismatch["current_capability_identity"] = current_capability_identity(CapabilityClient(name="backend-mismatch", digest="same", backend="other"), "backend-mismatch")
    backend_result = campaign.build_judge_selection([{**backend_mismatch, "name": "backend-mismatch", "digest": "same"}], [], policy)
    assert backend_result.selected is None
    assert backend_result.rejection_reasons[0]["reason"] == "capability_reprobe_required"

    template_mismatch = _profile("template-mismatch", {
        "text": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
    }, digest="same")
    template_mismatch["current_capability_identity"] = current_capability_identity(CapabilityClient(name="template-mismatch", digest="same", template="template-v2"), "template-mismatch")
    template_result = campaign.build_judge_selection([{**template_mismatch, "name": "template-mismatch", "digest": "same"}], [], policy)
    assert template_result.selected is None
    assert template_result.rejection_reasons[0]["reason"] == "capability_reprobe_required"

    protocol_mismatch = _profile("protocol-mismatch", {
        "text": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
    }, digest="same")
    # A real protocol-version drift predates today's code, so both the
    # top-level probe_protocol_version and the copy nested inside
    # capability_identity carry the old value together -- interrogate_model()
    # sets both from the same constant in one call (capabilities.py:455/573).
    # Mutating only one would be an unrealistic shape neither the legacy nor
    # the typed Stage 2.6D adapter would ever see from real stored evidence.
    protocol_mismatch["capability_identity"] = dict(protocol_mismatch["capability_identity"])
    protocol_mismatch["capability_identity"]["probe_protocol_version"] = "old"
    protocol_mismatch["probe_protocol_version"] = "old"
    protocol_mismatch["current_capability_identity"] = current_capability_identity(CapabilityClient(name="protocol-mismatch", digest="same"), "protocol-mismatch")
    protocol_result = campaign.build_judge_selection([{**protocol_mismatch, "name": "protocol-mismatch", "digest": "same"}], [], policy)
    assert protocol_result.selected is None
    assert protocol_result.rejection_reasons[0]["reason"] == "capability_reprobe_required"

    unsupported = campaign.build_judge_selection([{
        "name": "measured-unsupported",
        "digest": "unsupported",
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "measured_capabilities": {"text": {"state": MeasuredCapabilityState.MEASURED_UNSUPPORTED.value}},
    }], [], policy)
    assert unsupported.selected is None
    assert unsupported.rejection_reasons[0]["reason"] == "unknown_or_non_generative_capability"

    inconclusive = campaign.build_judge_selection([{
        "name": "measured-inconclusive",
        "digest": "inconclusive",
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "measured_capabilities": {"text": {"state": MeasuredCapabilityState.PROBE_INCONCLUSIVE.value}},
    }], [], policy)
    assert inconclusive.selected is None
    assert inconclusive.rejection_reasons[0]["reason"] == "capability_reprobe_required"


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


def test_unknown_task_never_enters_generation_recovery(tmp_path):
    model = "fixture:latest"
    profile = _profile(model, {
        "text": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
        "tools": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
        "vision": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
    })

    empty_runs = tmp_path / "unknown_empty"
    empty_runs.mkdir()
    _write_unknown_repair_run(empty_runs, model, profile, "completely_unknown_task", "empty_output")
    empty = repair.build_plan(empty_runs, run_id="fleet", include_missing=False)
    assert empty.actions == []
    assert empty.observations[0]["kind"] == "unknown_task_not_repairable"

    thinking_runs = tmp_path / "unknown_thinking"
    thinking_runs.mkdir()
    _write_unknown_repair_run(thinking_runs, model, profile, "completely_unknown_task", "thinking_only")
    thinking = repair.build_plan(thinking_runs, run_id="fleet", include_missing=False)
    assert thinking.actions == []
    assert thinking.observations[0]["kind"] == "unknown_task_not_repairable"

    metadata_runs = tmp_path / "unknown_metadata"
    metadata_runs.mkdir()
    metadata_profile = {
        "model": model,
        "declared_capabilities": ["completion", "tools", "vision"],
        "supported_families": ["text", "tools", "vision"],
    }
    _write_unknown_repair_run(metadata_runs, model, metadata_profile, "completely_unknown_task", "empty_output")
    metadata = repair.build_plan(metadata_runs, run_id="fleet", include_missing=False)
    assert metadata.actions == []
    assert metadata.observations[0]["kind"] == "unknown_task_not_repairable"


def test_recovery_capability_profile_must_match_source_digest(tmp_path):
    model = "example:latest"
    same_runs = tmp_path / "same_digest"
    same_runs.mkdir()
    _write_repair_run(same_runs, model, _profile(model, {
        "text": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
    }, digest="DIGEST_A"), "txt_sort", "empty_output", source_digest="DIGEST_A")
    same = repair.build_plan(same_runs, run_id="fleet", include_missing=False)
    assert [action.kind for action in same.actions] == ["retry_generation"]

    mismatch_runs = tmp_path / "mismatch_digest"
    mismatch_runs.mkdir()
    _write_repair_run(mismatch_runs, model, _profile(model, {
        "text": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
    }, digest="DIGEST_B"), "txt_sort", "empty_output", source_digest="DIGEST_A")
    mismatch = repair.build_plan(mismatch_runs, run_id="fleet", include_missing=False)
    assert mismatch.actions == []
    assert mismatch.observations[0]["kind"] == "capability_reprobe_required"
    assert mismatch.observations[0]["capability_identity_compatibility"]["reason"] == "model_digest_changed"

    missing_profile_digest_runs = tmp_path / "missing_profile_digest"
    missing_profile_digest_runs.mkdir()
    _write_repair_run(missing_profile_digest_runs, model, _profile(model, {
        "text": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
    }, digest=None), "txt_sort", "empty_output", source_digest="DIGEST_A")
    missing_profile_digest = repair.build_plan(missing_profile_digest_runs, run_id="fleet", include_missing=False)
    assert missing_profile_digest.actions == []
    assert missing_profile_digest.observations[0]["capability_identity_compatibility"]["reason"] == "model_digest_changed"

    missing_source_digest_runs = tmp_path / "missing_source_digest"
    missing_source_digest_runs.mkdir()
    _write_repair_run(missing_source_digest_runs, model, _profile(model, {
        "text": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
    }, digest="DIGEST_A"), "txt_sort", "empty_output", source_digest="", write_identity=False)
    missing_source_digest = repair.build_plan(missing_source_digest_runs, run_id="fleet", include_missing=False)
    assert missing_source_digest.actions == []
    assert missing_source_digest.observations[0]["capability_identity_compatibility"]["reason"] == "source_digest_missing"


def test_specialized_recovery_lanes_cannot_cross_digest_boundaries(tmp_path):
    model = "native:latest"
    for task_id, family in [
        ("agent_native_tool_call", "tools"),
        ("fim_suffix_assertion", "insert"),
        ("ocr_invoice", "vision"),
    ]:
        runs = tmp_path / family
        runs.mkdir()
        _write_repair_run(runs, model, _profile(model, {
            family: MeasuredCapabilityState.MEASURED_SUPPORTED.value,
        }, digest="DIGEST_B"), task_id, "empty_output", source_digest="DIGEST_A")
        plan = repair.build_plan(runs, run_id="fleet", include_missing=False)
        assert plan.actions == []
        assert plan.observations[0]["kind"] == "capability_reprobe_required"
        assert plan.observations[0]["capability_identity_compatibility"]["reason"] == "model_digest_changed"


def test_known_specialized_recovery_lanes_stay_capability_gated(tmp_path):
    model = "native:latest"

    unsupported_tools = tmp_path / "tools_unsupported"
    unsupported_tools.mkdir()
    _write_repair_run(unsupported_tools, model, _profile(model, {
        "tools": MeasuredCapabilityState.MEASURED_UNSUPPORTED.value,
    }), "agent_native_tool_call", "empty_output")
    tools_plan = repair.build_plan(unsupported_tools, run_id="fleet", include_missing=False)
    assert tools_plan.actions == []
    assert tools_plan.observations[0]["kind"] == "capability_not_applicable"

    supported_fim = tmp_path / "fim_supported"
    supported_fim.mkdir()
    _write_repair_run(supported_fim, model, _profile(model, {
        "insert": MeasuredCapabilityState.MEASURED_SUPPORTED.value,
    }), "fim_suffix_assertion", "empty_output")
    fim_plan = repair.build_plan(supported_fim, run_id="fleet", include_missing=False)
    assert [action.kind for action in fim_plan.actions] == ["capability_gate"]
    assert fim_plan.actions[0].family == "insert"


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


def test_capability_identity_hash_ignores_modified_at_but_keeps_material_changes():
    model = "fixture:latest"
    first = current_capability_identity(CapabilityClient(name=model, modified_at="2026-08-10T00:00:00Z"), model)
    second = current_capability_identity(CapabilityClient(name=model, modified_at="2026-08-11T00:00:00Z"), model)
    assert first["identity_hash"] == second["identity_hash"]
    assert "modified_at" not in first["model"]

    digest_changed = current_capability_identity(CapabilityClient(name=model, digest="digest-2"), model)
    backend_changed = current_capability_identity(CapabilityClient(name=model, backend="other"), model)
    template_changed = current_capability_identity(CapabilityClient(name=model, template="template-v2"), model)
    assert first["identity_hash"] != digest_changed["identity_hash"]
    assert first["identity_hash"] != backend_changed["identity_hash"]
    assert first["identity_hash"] != template_changed["identity_hash"]


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


def test_recovery_blocks_known_task_on_legacy_profile_present_but_unbound(tmp_path):
    """Anvil Stage 2.0: closes a confirmed regression-matrix gap. The
    2026-08-10 closure audit's 20-case matrix (cases 13/15) claims a known
    task with a legacy or schema-v2-but-unbound capability profile (present,
    but no `capability_identity` key) fails closed -- but no test in this
    file or anywhere else in the suite exercised that exact code path
    (`repair._profile_source_compatibility`'s `legacy_or_unbound_capability_profile`
    branch) before this test. Verified empirically before writing this
    assertion, not just by reading the source.
    """
    model = "legacy-unbound:latest"
    runs = tmp_path / "runs"
    runs.mkdir()
    legacy_profile = {"model": model, "declared_capabilities": ["completion"], "supported_families": ["text"]}
    _write_repair_run(runs, model, legacy_profile, "txt_sort", "empty_output", source_digest="digest-1")
    plan = repair.build_plan(runs, run_id="fleet", include_missing=False)
    assert plan.actions == []
    assert plan.observations[0]["kind"] == "capability_reprobe_required"
    assert plan.observations[0]["capability_identity_compatibility"] == {
        "compatible": False,
        "reason": "legacy_or_unbound_capability_profile",
    }


def test_recovery_blocks_known_task_on_capability_profile_missing_entirely(tmp_path):
    """Anvil Stage 2.0: closes the second half of the same confirmed gap
    (case 14) -- a *missing* capability profile (no entry at all for the
    model in `capability_report.json`) takes a different code path than a
    present-but-unbound one: `repair._best_profile()`'s empty-profiles
    early return (repair.py's fallback dict) never reaches
    `_profile_source_compatibility`, so it carries no
    `capability_identity_compatibility` reason string at all. Still
    verified (empirically, not just by trace) to fail closed via the same
    `error_kind and not compatible` gate, just with an empty-dict shape
    instead of a labelled reason -- asserting that exact shape here so a
    future change that silently starts treating a missing profile as
    vacuously compatible cannot pass unnoticed.
    """
    model = "no-profile-at-all:latest"
    runs = tmp_path / "runs"
    run = runs / "fleet"
    run.mkdir(parents=True)
    task = _task("txt_sort")
    row = {
        "model": model, "task": task.id, "category": task.category, "family": task.family,
        "task_hash": _task_hash(task), "score": None, "error_kind": "empty_output",
        "reason": "empty_output", "timestamp": "2026-08-10T00:00:00Z",
        "model_digest_resolved": "digest-1",
    }
    (run / "raw_results.jsonl").write_text(json.dumps(row) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": "full"}))
    (run / "filters.json").write_text(json.dumps({"level": "full", "think": "auto"}))
    (run / "model_identities.json").write_text(json.dumps({model: {"digest": "digest-1", "size": 1}}))
    (run / "capability_report.json").write_text(json.dumps({}))  # no entry for `model` at all
    plan = repair.build_plan(runs, run_id="fleet", include_missing=False)
    assert plan.actions == []
    assert plan.observations[0]["kind"] == "capability_reprobe_required"
    assert plan.observations[0]["capability_identity_compatibility"] == {}


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
