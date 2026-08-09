import json
from pathlib import Path

from llm_modelbench import campaign, judge_dumps
from llm_modelbench.runner import _task_hash
from llm_modelbench.tasks import TASKS


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


def _qualified(name: str, digest: str):
    return {
        "name": name,
        "digest": digest,
        "roles": ["judge"],
        "qualified": True,
        "qualification": {
            "protocol_version": "judge-qualification-v1",
            "aggregate_disposition": "qualified",
            "controls": [{"control_id": "synthetic", "passed": True}],
        },
    }


class RecordingJudgeClient:
    def __init__(self, structural_models=()):
        self.structural_models = set(structural_models)
        self.calls = []

    def chat(self, model, prompt, **kwargs):
        self.calls.append(model)
        if model in self.structural_models:
            return {"ok": False, "error": "HTTP 415 unsupported request schema"}
        return {"ok": True, "text": '{"score": 88, "confidence": 1, "verdict": "synthetic"}'}


def test_stage1d_selection_qualification_pool_and_judged_sidecar_provenance(monkeypatch, tmp_path):
    inventory = [
        {"name": "preferred", "digest": "digest-preferred", "roles": ["judge"], "capabilities": ["completion"]},
        {"name": "fallback", "digest": "digest-fallback", "roles": ["judge"], "capabilities": ["completion"]},
        {"name": "embedder", "digest": "digest-embed", "roles": ["judge"], "capabilities": ["embedding"]},
        {"name": "qwen2.5:7b", "digest": "digest-qwen", "roles": ["judge"], "capabilities": ["completion"]},
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
        {"name": "preferred", "digest": "digest-preferred", "roles": ["judge"], "capabilities": ["completion"]},
        {"name": "fallback", "digest": "digest-fallback", "roles": ["judge"], "capabilities": ["completion"]},
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
