import json
from pathlib import Path

from llm_modelbench import campaign, judge_dumps
from llm_modelbench.capabilities import PROBE_PROTOCOL_VERSION
from llm_modelbench.runner import _task_hash
from llm_modelbench.tasks import TASKS


def _identity(name, digest):
    return {
        "schema_version": campaign.CAPABILITY_SCHEMA_VERSION,
        "model": {"canonical_name": name, "backend_model_id": name, "digest": digest, "size": 1, "details": {}},
        "backend": {"backend": "mock", "implementation": "fixture", "endpoint": "http://fake.invalid"},
        "runtime": {"endpoint": "http://fake.invalid", "implementation": "fixture"},
        "template_config": {"available": True, "hash": "template-v1", "material": {"template": "template-v1"}},
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
    }


def _bind(item):
    item["capability_identity"] = _identity(item["name"], item["digest"])
    item["capability_identity_compatibility"] = {"compatible": True, "reason": "identity_match"}
    return item


def _measured_text(name, digest, **extra):
    item = {
        "name": name,
        "digest": digest,
        "capability_schema_version": campaign.CAPABILITY_SCHEMA_VERSION,
        "measured_capabilities": {"text": {"state": "measured_supported"}},
    }
    item.update(extra)
    return _bind(item)


def _measured_embedding(name, digest, **extra):
    item = {
        "name": name,
        "digest": digest,
        "capability_schema_version": campaign.CAPABILITY_SCHEMA_VERSION,
        "measured_capabilities": {"embedding": {"state": "measured_supported"}},
    }
    item.update(extra)
    return _bind(item)


def _subjective_task():
    return next(task for task in TASKS if task.scorer == "subjective")


def _write_subjective_run(root: Path, rows):
    task = _subjective_task()
    run = root / "run"
    raw_rows = []
    for row in rows:
        model = row["model"]
        digest = row["digest"]
        dump = run / "subjective" / task.id / f"{model}.md"
        dump.parent.mkdir(parents=True, exist_ok=True)
        dump.write_text(f"# TASK {task.id}\n\n## OUTPUT\nanswer from {model}\n")
        raw_rows.append({
            "model": model,
            "model_digest": digest,
            "model_digest_resolved": digest,
            "task": task.id,
            "category": task.category,
            "family": task.family,
            "task_hash": _task_hash(task),
            "score": None,
            "error_kind": None,
            "subjective_path": str(dump.relative_to(run)),
            "timestamp": "2026-01-01T00:00:00Z",
        })
    (run / "raw_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw_rows))
    return run


def _campaign_with_subjective_primary(tmp_path, campaign_id="judge_campaign"):
    paths, manifest = campaign.create_campaign(campaign_id, models=["source"], campaigns_root=tmp_path / "campaigns")
    run = _write_subjective_run(tmp_path / f"{campaign_id}_source", [{"model": "source", "digest": "digest-source"}])
    paths.primary_raw_results.write_text((run / "raw_results.jsonl").read_text())
    paths.primary_run_validity.write_text('{"status":"valid"}')
    for source in (run / "subjective").rglob("*"):
        if source.is_file():
            target = paths.primary_dir / source.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source.read_text())
    manifest = campaign.transition(paths, manifest, "planned")
    manifest = campaign.transition(paths, manifest, "generating")
    campaign.transition(paths, manifest, "judging")
    raw_row = json.loads(paths.primary_raw_results.read_text().splitlines()[0])
    return paths, raw_row


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


def _qualified(name: str, digest: str, *, runtime_backend: str = "mock-runtime"):
    return {
        "name": name,
        "digest": digest,
        "roles": ["judge"],
        "runtime_identity": {"backend": runtime_backend, "model": name},
        "qualified": True,
        "qualification": {
            "protocol_version": "judge-qualification-v1",
            "aggregate_disposition": "qualified",
            "runtime_identity": {"backend": runtime_backend, "model": name},
            "controls": [{"control_id": "synthetic", "passed": True}],
        },
    }


def _selection_evidence(selection_result, qualified, qualifications, coverage):
    return {
        "eligible": 1,
        "cohort": [{"name": "source", "digest": "digest-source"}],
        "judge": qualified[0] if qualified else None,
        "qualified_judges": qualified,
        "qualification": (qualified[0].get("qualification") if qualified else None),
        "qualification_chain": qualifications,
        "qualification_coverage": coverage,
        "posthoc_judge_model": (qualified[0] if qualified else {}).get("name"),
        "posthoc_judge_digest": (qualified[0] if qualified else {}).get("digest"),
        "judge_policy_selection": selection_result.to_dict(),
    }


def _selection(names):
    inventory = [
        _measured_text(name, f"digest-{name}", roles=["judge"])
        for name in names
    ]
    return campaign.build_judge_selection(
        inventory,
        [{"name": "source", "digest": "digest-source"}],
        _policy(requested_primary=names[0], configured_fallbacks=tuple(names[1:]), excluded_families=()),
    )


class RecordingJudgeClient:
    def __init__(self, failures=None, structural_models=()):
        self.failures = dict(failures or {})
        for model in structural_models:
            self.failures[model] = {"ok": False, "error": "unsupported request schema", "http_status": 415}
        self.calls = []

    def chat(self, model, prompt, **kwargs):
        self.calls.append(model)
        failure = self.failures.get(model)
        if failure == "timeout_exception":
            raise TimeoutError("deadline exceeded")
        if failure:
            return dict(failure)
        return {"ok": True, "text": '{"score": 88, "confidence": 1, "verdict": "synthetic"}'}


def test_stage1d_selection_qualification_pool_and_judged_sidecar_provenance(monkeypatch, tmp_path):
    inventory = [
        _measured_text("preferred", "digest-preferred", roles=["judge"]),
        _measured_text("fallback", "digest-fallback", roles=["judge"]),
        _measured_embedding("embedder", "digest-embed", roles=["judge"]),
        _measured_text("qwen2.5:7b", "digest-qwen", roles=["judge"]),
    ]
    selection = campaign.build_judge_selection(
        inventory,
        [{"name": "source", "digest": "digest-source"}],
        _policy(requested_primary="preferred", configured_fallbacks=("fallback", "embedder")),
    )
    assert [item["name"] for item in selection.final_eligible_order] == ["preferred", "fallback"]
    assert any(item["model"] == "embedder" and item["reason"] == "non_generative_embedding_only" for item in selection.rejection_reasons)
    assert any(item["model"] == "qwen2.5:7b" and item["reason"] == "excluded_family" for item in selection.rejection_reasons)

    def fake_qualify(client, candidate):
        return {
            "model": candidate["name"],
            "digest": candidate["digest"],
            "qualified": True,
            "aggregate_disposition": "qualified",
            "protocol_version": "judge-qualification-v1",
            "controls": [{"control_id": "synthetic", "passed": True}],
        }

    monkeypatch.setattr(campaign, "qualify_judge", fake_qualify)
    run = _write_subjective_run(tmp_path, [{"model": "source", "digest": "digest-source"}])
    qualified, qualifications, coverage = campaign.select_qualified_campaign_judges_for_rows(
        object(), selection, [json.loads((run / "raw_results.jsonl").read_text().splitlines()[0])]
    )
    result = judge_dumps.judge_run(RecordingJudgeClient(), run, judge_model="preferred", qualified_judges=qualified)
    entry = result["entries"][0]

    assert len(qualifications) == 1
    assert coverage["coverage_complete"] is True
    assert result["judged"] == 1
    assert entry["status"] == "judged"
    assert entry["judge_model"] == "preferred"
    assert entry["judge_model_digest"] == "digest-preferred"
    assert entry["source_model"] == "source"
    assert entry["source_model_digest"] == "digest-source"
    assert entry["identity_relation"] == "independent"
    assert entry["qualified_judge_pool"][0]["qualification_protocol_version"] == "judge-qualification-v1"
    assert entry["samples"][0]["output_sha256"]
    assert entry["judge_mode_configuration"] == {"judge_mode": "single", "num_ctx": None, "think": "auto"}


def test_stage1d_primary_self_falls_back_to_actual_independent_judge(tmp_path):
    run = _write_subjective_run(tmp_path, [{"model": "preferred", "digest": "digest-preferred"}])
    pool = [_qualified("preferred", "digest-preferred"), _qualified("fallback", "digest-fallback")]
    client = RecordingJudgeClient()

    result = judge_dumps.judge_run(client, run, judge_model="preferred", qualified_judges=pool)
    entry = result["entries"][0]

    assert client.calls == ["fallback"]
    assert entry["judge_model"] == "fallback"
    assert entry["judge_model_digest"] == "digest-fallback"
    assert entry["judge_resolution"]["attempts"][0]["status"] == "rejected_self_identity"
    assert entry["judgement_attempts"][-1]["identity_relation"] == "independent"


def test_stage1d_structural_qualification_failure_is_bounded_and_fallback_qualifies(monkeypatch):
    selection = campaign.build_judge_selection([
        _measured_text("preferred", "digest-preferred", roles=["judge"]),
        _measured_text("fallback", "digest-fallback", roles=["judge"]),
    ], [], _policy(requested_primary="preferred", configured_fallbacks=("fallback",)))
    calls = []

    def fake_qualify(client, candidate):
        calls.append(candidate["name"])
        if candidate["name"] == "preferred":
            return {
                "model": "preferred",
                "digest": "digest-preferred",
                "qualified": False,
                "aggregate_disposition": "rejected_structural_incompatibility",
                "protocol_version": "judge-qualification-v1",
                "checks": {"structural_incompatibility": True},
                "controls": [{"control_id": "obviously_correct", "failures": ["structural_incompatibility"]}],
            }
        return {
            "model": "fallback",
            "digest": "digest-fallback",
            "qualified": True,
            "aggregate_disposition": "qualified",
            "protocol_version": "judge-qualification-v1",
            "controls": [{"control_id": "synthetic", "passed": True}],
        }

    monkeypatch.setattr(campaign, "qualify_judge", fake_qualify)
    qualified, qualifications = campaign.select_qualified_campaign_judges(object(), selection)

    assert calls == ["preferred", "fallback"]
    assert [item["name"] for item in qualified] == ["fallback"]
    assert qualifications[0]["aggregate_disposition"] == "rejected_structural_incompatibility"


def test_stage1d_structural_judging_failure_records_once_and_uses_next_independent_judge(tmp_path):
    run = _write_subjective_run(tmp_path, [{"model": "source", "digest": "digest-source"}])
    pool = [_qualified("bad-judge", "digest-bad"), _qualified("good-judge", "digest-good")]
    client = RecordingJudgeClient(structural_models={"bad-judge"})

    result = judge_dumps.judge_run(client, run, judge_model="bad-judge", qualified_judges=pool)
    entry = result["entries"][0]

    assert client.calls == ["bad-judge", "good-judge"]
    assert result["judged"] == 1
    assert entry["judge_model"] == "good-judge"
    assert entry["judgement_attempts"][0]["status"] == "rejected_structural_incompatibility"
    assert entry["judgement_attempts"][0]["judge_model"] == "bad-judge"
    assert entry["judgement_attempts"][-1]["status"] == "judged"
    assert entry["qualified_judge_pool"][0]["name"] == "bad-judge"
    assert entry["judgement_attempts"][0]["sample"]["backend"]["http_status"] == 415
    assert entry["judgement_attempts"][0]["compatibility_fingerprint"]


def test_stage1d_exhausted_independent_judges_overlay_and_readiness_block(tmp_path):
    paths, manifest = campaign.create_campaign("judge_exhausted", models=["source"], campaigns_root=tmp_path / "campaigns")
    run = _write_subjective_run(tmp_path, [{"model": "source", "digest": "digest-source"}])
    paths.primary_raw_results.write_text((run / "raw_results.jsonl").read_text())
    paths.primary_run_validity.write_text('{"status":"valid"}')
    for source in (run / "subjective").rglob("*"):
        if source.is_file():
            target = paths.primary_dir / source.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source.read_text())
    manifest = campaign.transition(paths, manifest, "planned")
    manifest = campaign.transition(paths, manifest, "generating")
    campaign.transition(paths, manifest, "judging")

    client = RecordingJudgeClient(structural_models={"bad-judge"})
    result = judge_dumps.judge_run(
        client,
        paths.primary_dir,
        judge_model="bad-judge",
        qualified_judges=[_qualified("bad-judge", "digest-bad")],
    )
    paths.judge_results.write_text((paths.primary_dir / "judge_results.jsonl").read_text())
    raw_rows = [json.loads(paths.primary_raw_results.read_text().splitlines()[0])]
    rows = judge_dumps.apply_judgements(paths.primary_dir, raw_rows)
    summary = campaign.write_readiness(paths, rows, judge_available=True)
    effective = json.loads(paths.effective_rows.read_text().splitlines()[0])

    assert client.calls == ["bad-judge"]
    assert result["entries"][0]["status"] == "judge_exhausted_unavailable"
    assert rows[0]["disposition"] == "judge_exhausted_unavailable"
    assert effective["terminal_disposition"] == "judge_exhausted_unavailable"
    assert effective["provenance"]["judge_failure_disposition"] == "structural_incompatibility"
    assert summary["readiness"] == "not_ready_external_judge"
    assert "judge_exhausted_unavailable" in summary["blockers"]


def test_stage1d_unchanged_exhausted_judge_rerun_is_noop(tmp_path):
    run = _write_subjective_run(tmp_path, [{"model": "source", "digest": "digest-source"}])
    client = RecordingJudgeClient(structural_models={"bad-judge"})
    pool = [_qualified("bad-judge", "digest-bad")]

    first = judge_dumps.judge_run(client, run, judge_model="bad-judge", qualified_judges=pool)
    second = judge_dumps.judge_run(client, run, judge_model="bad-judge", qualified_judges=pool)

    assert first["entries"][0]["status"] == "judge_exhausted_unavailable"
    assert second["eligible"] == 0
    assert second["skipped"][0]["reason"] == "already_judge_exhausted_unavailable"
    assert client.calls == ["bad-judge"]
    assert len((run / "judge_results.jsonl").read_text().splitlines()) == 1


def test_stage1d_expanded_pool_reuses_known_bad_structural_fingerprint_without_calling_j1(tmp_path):
    run = _write_subjective_run(tmp_path, [{"model": "source", "digest": "digest-source"}])
    client = RecordingJudgeClient(structural_models={"bad-judge"})
    bad_only = [_qualified("bad-judge", "digest-bad")]
    expanded = [*bad_only, _qualified("good-judge", "digest-good")]

    first = judge_dumps.judge_run(client, run, judge_model="bad-judge", qualified_judges=bad_only)
    second = judge_dumps.judge_run(client, run, judge_model="bad-judge", qualified_judges=expanded)

    assert first["entries"][0]["status"] == "judge_exhausted_unavailable"
    assert client.calls == ["bad-judge", "good-judge"]
    assert second["judged"] == 1
    assert second["entries"][0]["judge_model"] == "good-judge"
    assert second["entries"][0]["judgement_attempts"][0]["status"] == "reused_structural_incompatibility"


def test_stage1d_changed_execution_fingerprint_reconsiders_prior_structural_judge(tmp_path):
    run = _write_subjective_run(tmp_path, [{"model": "source", "digest": "digest-source"}])
    client = RecordingJudgeClient(structural_models={"bad-judge"})
    bad_only = [_qualified("bad-judge", "digest-bad")]
    expanded = [*bad_only, _qualified("good-judge", "digest-good")]

    judge_dumps.judge_run(client, run, judge_model="bad-judge", qualified_judges=bad_only, think="auto")
    second = judge_dumps.judge_run(client, run, judge_model="bad-judge", qualified_judges=expanded, think="off")

    assert client.calls == ["bad-judge", "bad-judge", "good-judge"]
    assert second["judged"] == 1
    assert second["entries"][0]["judgement_attempts"][0]["status"] == "rejected_structural_incompatibility"


def test_stage1d_typed_backend_failures_preserve_dispositions_and_overlay_readiness(tmp_path):
    cases = [
        ("bad400", {"ok": False, "error": "bad request", "http_status": 400}, "structural_incompatibility", "judge_exhausted_unavailable"),
        ("bad404", {"ok": False, "error": "not found", "http_status": 404}, "structural_incompatibility", "judge_exhausted_unavailable"),
        ("bad405", {"ok": False, "error": "method", "http_status": 405}, "structural_incompatibility", "judge_exhausted_unavailable"),
        ("bad415", {"ok": False, "error": "media", "http_status": 415}, "structural_incompatibility", "judge_exhausted_unavailable"),
        ("bad422", {"ok": False, "error": "entity", "http_status": 422}, "structural_incompatibility", "judge_exhausted_unavailable"),
        ("timeout408", {"ok": False, "error": "deadline", "http_status": 408}, "timeout", "judge_error"),
        ("rate429", {"ok": False, "error": "rate", "http_status": 429}, "transient_backend_failure", "judge_error"),
        ("server500", {"ok": False, "error": "server", "http_status": 500}, "transient_backend_failure", "judge_error"),
        ("typed-timeout", "timeout_exception", "timeout", "judge_error"),
        ("malformed", {"ok": True, "text": "not json"}, "judge_output_failure", "judge_error"),
    ]
    for model, failure, disposition, status in cases:
        root = tmp_path / model
        run = _write_subjective_run(root, [{"model": "source", "digest": "digest-source"}])
        client = RecordingJudgeClient(failures={model: failure})
        result = judge_dumps.judge_run(client, run, judge_model=model, qualified_judges=[_qualified(model, f"digest-{model}")])
        entry = result["entries"][0]

        assert entry["status"] == status
        assert entry["failure_disposition"] == disposition
        assert entry["score"] is None
        if status == "judge_error":
            overlaid = judge_dumps.apply_judgements(run, [json.loads((run / "raw_results.jsonl").read_text().splitlines()[0])])
            assert overlaid[0]["posthoc_judged"] is False
            assert overlaid[0]["disposition"] == disposition
            assert overlaid[0]["score"] is None
            second = judge_dumps.judge_run(client, run, judge_model=model, qualified_judges=[_qualified(model, f"digest-{model}")])
            assert second["eligible"] == 0
            assert second["skipped"][0]["reason"] == "already_judge_error_for_resolved_judge"


def test_stage1d_non_structural_failure_effective_row_and_readiness_are_truthful(tmp_path):
    paths, manifest = campaign.create_campaign("judge_timeout", models=["source"], campaigns_root=tmp_path / "campaigns")
    run = _write_subjective_run(tmp_path, [{"model": "source", "digest": "digest-source"}])
    paths.primary_raw_results.write_text((run / "raw_results.jsonl").read_text())
    paths.primary_run_validity.write_text('{"status":"valid"}')
    for source in (run / "subjective").rglob("*"):
        if source.is_file():
            target = paths.primary_dir / source.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source.read_text())
    manifest = campaign.transition(paths, manifest, "planned")
    manifest = campaign.transition(paths, manifest, "generating")
    campaign.transition(paths, manifest, "judging")

    result = judge_dumps.judge_run(
        RecordingJudgeClient(failures={"timeout-judge": {"ok": False, "error": "deadline", "http_status": 408}}),
        paths.primary_dir,
        judge_model="timeout-judge",
        qualified_judges=[_qualified("timeout-judge", "digest-timeout")],
    )
    paths.judge_results.write_text((paths.primary_dir / "judge_results.jsonl").read_text())
    raw_rows = [json.loads(paths.primary_raw_results.read_text().splitlines()[0])]
    rows = judge_dumps.apply_judgements(paths.primary_dir, raw_rows)
    summary = campaign.write_readiness(paths, rows, judge_available=True)
    effective = json.loads(paths.effective_rows.read_text().splitlines()[0])

    assert result["entries"][0]["status"] == "judge_error"
    assert rows[0]["disposition"] == "timeout"
    assert effective["terminal_disposition"] == "timeout"
    assert effective["effective_score"] is None
    assert effective["provenance"]["judge_model"] == "timeout-judge"
    assert effective["provenance"]["judge_failure_disposition"] == "timeout"
    assert summary["readiness"] == "not_ready_external_judge"
    assert "judge_failure" in summary["blockers"]


def test_stage1d_mocked_campaign_artifacts_link_selection_fallback_judge_and_readiness(monkeypatch, tmp_path):
    paths, raw_row = _campaign_with_subjective_primary(tmp_path, "fallback_campaign")

    inventory = [
        _measured_text("j1", "digest-j1", roles=["judge"]),
        _measured_text("j2", "digest-j2", roles=["judge"]),
        _measured_embedding("embedder", "digest-embedder", roles=["judge"]),
    ]
    selection_result = campaign.build_judge_selection(
        inventory,
        [{"name": "source", "digest": "digest-source"}],
        _policy(requested_primary="j1", configured_fallbacks=("j2", "embedder"), excluded_families=()),
    )

    def fake_qualify(client, candidate):
        return {
            "model": candidate["name"],
            "digest": candidate["digest"],
            "qualified": True,
            "aggregate_disposition": "qualified",
            "protocol_version": "judge-qualification-v1",
            "runtime_identity": {"backend": "mock", "model": candidate["name"]},
            "controls": [{"control_id": "synthetic", "passed": True}],
        }

    monkeypatch.setattr(campaign, "qualify_judge", fake_qualify)
    qualified, qualifications, coverage = campaign.select_qualified_campaign_judges_for_rows(object(), selection_result, [raw_row])
    selection = _selection_evidence(selection_result, qualified, qualifications, coverage)
    campaign._atomic_write_text(paths.judge_dir / "judge_selection.json", json.dumps(selection, indent=2, sort_keys=True))

    judged, selection = campaign.judge_run_with_structural_continuation(
        RecordingJudgeClient(structural_models={"j1"}),
        paths.primary_dir,
        selection=selection_result,
        selection_evidence=selection,
        qualified_judges=qualified,
        qualifications=qualifications,
        source_rows=[raw_row],
        judge_model="j1",
        judge_mode="single",
        num_ctx=4096,
        think="off",
    )
    paths.judge_results.write_text((paths.primary_dir / "judge_results.jsonl").read_text())
    campaign._atomic_write_text(paths.judge_dir / "judge_selection.json", json.dumps(selection, indent=2, sort_keys=True))
    campaign._atomic_write_text(paths.judge_summary, json.dumps({**judged, "selection": selection}, indent=2, sort_keys=True))
    rows = judge_dumps.apply_judgements(paths.primary_dir, [raw_row])
    summary = campaign.write_readiness(paths, rows, judge_available=True)

    persisted_selection = json.loads((paths.judge_dir / "judge_selection.json").read_text())
    sidecars = [json.loads(line) for line in paths.judge_results.read_text().splitlines()]
    exhausted_sidecar = next(entry for entry in sidecars if entry["status"] == "judge_exhausted_unavailable")
    sidecar = next(entry for entry in sidecars if entry["status"] == "judged")
    effective = json.loads(paths.effective_rows.read_text().splitlines()[0])
    readiness = json.loads(paths.readiness_json.read_text())

    assert persisted_selection["judge_policy_selection"]["requested_primary"] == "j1"
    assert [item["name"] for item in persisted_selection["judge_policy_selection"]["final_eligible_order"]] == ["j1", "j2"]
    assert any(item["model"] == "embedder" and item["reason"] == "non_generative_embedding_only" for item in persisted_selection["judge_policy_selection"]["rejection_reasons"])
    assert persisted_selection["qualification_chain"][0]["protocol_version"] == "judge-qualification-v1"
    assert [item["name"] for item in persisted_selection["initial_qualified_judges"]] == ["j1"]
    assert persisted_selection["qualification_coverage"]["coverage_complete"] is True
    assert [item["name"] for item in persisted_selection["qualified_judges"]] == ["j1", "j2"]
    assert [item["name"] for item in persisted_selection["final_qualified_judges"]] == ["j1", "j2"]
    assert persisted_selection["qualification_continuations"][0]["reason"] == "qualification_continuation_after_structural_incompatibility"
    assert persisted_selection["qualification_continuations"][0]["added_qualified_judges"][0]["name"] == "j2"
    assert exhausted_sidecar["judge_model"] is None
    assert exhausted_sidecar["judgement_attempts"][0]["judge_model"] == "j1"
    assert exhausted_sidecar["judgement_attempts"][0]["status"] == "rejected_structural_incompatibility"
    assert sidecar["source_model"] == "source"
    assert sidecar["source_model_digest"] == "digest-source"
    assert sidecar["judge_model"] == "j2"
    assert sidecar["judge_model_digest"] == "digest-j2"
    assert sidecar["identity_relation"] == "independent"
    assert sidecar["judgement_attempts"][0]["judge_model"] == "j1"
    assert sidecar["judgement_attempts"][0]["status"] == "reused_structural_incompatibility"
    assert sidecar["task_hash"] == raw_row["task_hash"]
    assert sidecar["source_row_hash"]
    assert sidecar["samples"][0]["output_sha256"]
    assert sidecar["judge_mode_configuration"] == {"judge_mode": "single", "num_ctx": 4096, "think": "off"}
    assert effective["result_origin"] == "judged"
    assert effective["provenance"]["judge_model"] == "j2"
    assert effective["provenance"]["judge_model_digest"] == "digest-j2"
    assert summary["readiness"] == "ready_for_adoption"
    assert readiness["readiness"] == "ready_for_adoption"


def test_stage1d_continuation_skips_failed_tail_qualification_and_uses_j3(monkeypatch, tmp_path):
    paths, raw_row = _campaign_with_subjective_primary(tmp_path, "fallback_j3")
    selection_result = _selection(["j1", "j2", "j3"])
    calls = []

    def fake_qualify(client, candidate):
        calls.append(candidate["name"])
        if candidate["name"] == "j2":
            return {"model": "j2", "digest": "digest-j2", "qualified": False, "aggregate_disposition": "rejected_quality_controls", "protocol_version": "judge-qualification-v1"}
        return {"model": candidate["name"], "digest": candidate["digest"], "qualified": True, "aggregate_disposition": "qualified", "protocol_version": "judge-qualification-v1"}

    monkeypatch.setattr(campaign, "qualify_judge", fake_qualify)
    qualified, qualifications, coverage = campaign.select_qualified_campaign_judges_for_rows(object(), selection_result, [raw_row])
    judged, selection = campaign.judge_run_with_structural_continuation(
        RecordingJudgeClient(structural_models={"j1"}),
        paths.primary_dir,
        selection=selection_result,
        selection_evidence=_selection_evidence(selection_result, qualified, qualifications, coverage),
        qualified_judges=qualified,
        qualifications=qualifications,
        source_rows=[raw_row],
        judge_model="j1",
    )

    assert calls == ["j1", "j2", "j3"]
    assert judged["judged"] == 1
    assert judged["entries"][0]["judge_model"] == "j3"
    assert [item["model"] for item in selection["qualification_continuations"][0]["continued_qualifications"]] == ["j2", "j3"]
    assert selection["qualification_continuations"][0]["continued_qualifications"][0]["qualified"] is False


def test_stage1d_continuation_exhausts_when_tail_fails_qualification(monkeypatch, tmp_path):
    paths, raw_row = _campaign_with_subjective_primary(tmp_path, "fallback_exhausted")
    selection_result = _selection(["j1", "j2", "j3"])

    def fake_qualify(client, candidate):
        return {
            "model": candidate["name"],
            "digest": candidate["digest"],
            "qualified": candidate["name"] == "j1",
            "aggregate_disposition": "qualified" if candidate["name"] == "j1" else "rejected_quality_controls",
            "protocol_version": "judge-qualification-v1",
        }

    monkeypatch.setattr(campaign, "qualify_judge", fake_qualify)
    qualified, qualifications, coverage = campaign.select_qualified_campaign_judges_for_rows(object(), selection_result, [raw_row])
    judged, selection = campaign.judge_run_with_structural_continuation(
        RecordingJudgeClient(structural_models={"j1"}),
        paths.primary_dir,
        selection=selection_result,
        selection_evidence=_selection_evidence(selection_result, qualified, qualifications, coverage),
        qualified_judges=qualified,
        qualifications=qualifications,
        source_rows=[raw_row],
        judge_model="j1",
    )

    assert judged["entries"][0]["status"] == "judge_exhausted_unavailable"
    assert [item["name"] for item in selection["final_qualified_judges"]] == ["j1"]
    assert selection["qualification_continuations"][0]["coverage_complete"] is False
    assert [item["model"] for item in selection["qualification_continuations"][0]["continued_qualifications"]] == ["j2", "j3"]


def test_stage1d_continuation_reuses_known_bad_j1_after_pool_extension(monkeypatch, tmp_path):
    paths, raw_row = _campaign_with_subjective_primary(tmp_path, "fallback_reuse")
    selection_result = _selection(["j1", "j2"])

    def fake_qualify(client, candidate):
        return {"model": candidate["name"], "digest": candidate["digest"], "qualified": True, "aggregate_disposition": "qualified", "protocol_version": "judge-qualification-v1"}

    monkeypatch.setattr(campaign, "qualify_judge", fake_qualify)
    qualified, qualifications, coverage = campaign.select_qualified_campaign_judges_for_rows(object(), selection_result, [raw_row])
    client = RecordingJudgeClient(structural_models={"j1"})
    judged, updated_selection = campaign.judge_run_with_structural_continuation(
        client,
        paths.primary_dir,
        selection=selection_result,
        selection_evidence=_selection_evidence(selection_result, qualified, qualifications, coverage),
        qualified_judges=qualified,
        qualifications=qualifications,
        source_rows=[raw_row],
        judge_model="j1",
    )

    assert client.calls == ["j1", "j2"]
    assert judged["judged"] == 1
    assert judged["entries"][0]["judgement_attempts"][0]["status"] == "reused_structural_incompatibility"
    assert updated_selection["qualification_continuations"][0]["added_qualified_judges"][0]["name"] == "j2"


def test_stage1d_continuation_changed_fingerprint_reconsiders_j1(monkeypatch, tmp_path):
    paths, raw_row = _campaign_with_subjective_primary(tmp_path, "fallback_changed_fingerprint")
    selection_result = _selection(["j1", "j2"])

    def fake_qualify(client, candidate):
        return {"model": candidate["name"], "digest": candidate["digest"], "qualified": True, "aggregate_disposition": "qualified", "protocol_version": "judge-qualification-v1"}

    monkeypatch.setattr(campaign, "qualify_judge", fake_qualify)
    qualified, qualifications, coverage = campaign.select_qualified_campaign_judges_for_rows(object(), selection_result, [raw_row])
    client = RecordingJudgeClient(structural_models={"j1"})
    campaign.judge_run_with_structural_continuation(
        client,
        paths.primary_dir,
        selection=selection_result,
        selection_evidence=_selection_evidence(selection_result, qualified, qualifications, coverage),
        qualified_judges=qualified,
        qualifications=qualifications,
        source_rows=[raw_row],
        judge_model="j1",
        think="auto",
    )
    campaign.judge_run_with_structural_continuation(
        client,
        paths.primary_dir,
        selection=selection_result,
        selection_evidence=_selection_evidence(selection_result, qualified, qualifications, coverage),
        qualified_judges=qualified,
        qualifications=qualifications,
        source_rows=[raw_row],
        judge_model="j1",
        think="off",
    )

    assert client.calls == ["j1", "j2", "j1", "j2"]


def test_stage1d_non_structural_runtime_failure_does_not_continue_qualification(monkeypatch, tmp_path):
    paths, raw_row = _campaign_with_subjective_primary(tmp_path, "fallback_non_structural")
    selection_result = _selection(["j1", "j2"])
    calls = []

    def fake_qualify(client, candidate):
        calls.append(candidate["name"])
        return {"model": candidate["name"], "digest": candidate["digest"], "qualified": True, "aggregate_disposition": "qualified", "protocol_version": "judge-qualification-v1"}

    monkeypatch.setattr(campaign, "qualify_judge", fake_qualify)
    qualified, qualifications, coverage = campaign.select_qualified_campaign_judges_for_rows(object(), selection_result, [raw_row])
    judged, selection = campaign.judge_run_with_structural_continuation(
        RecordingJudgeClient(failures={"j1": {"ok": False, "error": "deadline", "http_status": 408}}),
        paths.primary_dir,
        selection=selection_result,
        selection_evidence=_selection_evidence(selection_result, qualified, qualifications, coverage),
        qualified_judges=qualified,
        qualifications=qualifications,
        source_rows=[raw_row],
        judge_model="j1",
    )

    assert calls == ["j1"]
    assert judged["entries"][0]["status"] == "judge_error"
    assert judged["entries"][0]["failure_disposition"] == "timeout"
    assert selection["qualification_continuations"] == []


def test_stage1d_incomplete_identity_and_manual_judge_dump_fail_closed(tmp_path):
    run = _write_subjective_run(tmp_path, [{"model": "source-a", "digest": "digest-source"}])
    client = RecordingJudgeClient()

    result = judge_dumps.judge_run(client, run, judge_model="manual-judge")
    entry = result["entries"][0]

    assert client.calls == []
    assert entry["status"] == "awaiting_independent_judge"
    assert entry["qualified_judge_pool"][0]["qualification_state"] == "manual_unqualified_designation"
    assert entry["qualified_judge_pool"][0]["qualification_protocol_version"] is None
    assert entry["judge_resolution"]["attempts"][0]["status"] == "skipped_indeterminate_identity"
