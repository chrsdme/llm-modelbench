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


def _subjective_task():
    return next(task for task in TASKS if task.scorer == "subjective")


def _write_subjective_run(root: Path, rows):
    task = _subjective_task()
    run = root / "run"
    raw_rows = []
    identities = {}
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
        identities[model] = {"digest": digest}
    (run / "raw_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw_rows))
    (run / "model_identities.json").write_text(json.dumps(identities, sort_keys=True))
    return run


class RecordingJudgeClient:
    def __init__(self):
        self.models = []

    def chat(self, model, prompt, **kwargs):
        self.models.append(model)
        return {"ok": True, "text": '{"score": 88, "confidence": 1, "verdict": "synthetic"}'}


def _policy(**kwargs):
    defaults = {
        "requested_primary": None,
        "configured_fallbacks": (),
        "excluded_families": (),
        "allow_excluded_primary": False,
        "automatic_selection": True,
        "enabled": True,
    }
    defaults.update(kwargs)
    return campaign.JudgePolicy(**defaults)


def test_roles_are_separate_from_capabilities_and_multiple_roles_are_preserved():
    selection = campaign.build_judge_selection([
        _measured_text("judge-only", "j", roles=["judge"]),
        _measured_text("candidate-only", "c", roles=["benchmark_candidate"]),
        _measured_text("both", "b", roles=["judge", "benchmark_candidate"]),
    ], [], _policy(configured_fallbacks=("judge-only", "candidate-only", "both")))

    assert [item["name"] for item in selection.final_eligible_order] == ["judge-only", "both"]
    assert selection.final_eligible_order[1]["roles"] == ["benchmark_candidate", "judge"]
    assert any(item["model"] == "candidate-only" and item["reason"] == "missing_judge_role" for item in selection.rejection_reasons)


def test_benchmark_candidate_role_does_not_prevent_judge_eligibility():
    selection = campaign.build_judge_selection([
        _measured_text("both", "b", roles=["benchmark_candidate", "judge"]),
    ], [{"name": "both", "digest": "b"}], _policy())

    assert selection.selected["name"] == "both"
    assert selection.selected["roles"] == ["benchmark_candidate", "judge"]
    assert not any(item["reason"] == "in_tested_cohort" for item in selection.rejection_reasons)


def test_stable_identity_prefers_digest_for_alias_self_detection():
    row = {"model": "alias-under-test", "model_digest_resolved": "digest-same"}
    judge = {"name": "preferred-judge-alias", "digest": "digest-same", "roles": ["judge"]}
    other = {"name": "other-judge", "digest": "digest-other", "roles": ["judge"]}

    assert campaign.same_stable_model_identity(row, judge) is True
    assert campaign.same_stable_model_identity(row, other) is False


def test_independent_judge_resolution_skips_self_primary_and_uses_fallback():
    row = {"model": "source-alias", "model_digest_resolved": "digest-source"}
    qualified = [
        {"name": "primary-alias", "digest": "digest-source", "roles": ["judge"], "qualified": True},
        {"name": "fallback", "digest": "digest-fallback", "roles": ["judge"], "qualified": True},
    ]

    resolution = campaign.resolve_independent_judge_for_row(row, qualified)
    assert resolution["status"] == "selected_independent_judge"
    assert resolution["judge_model"] == "fallback"
    assert [attempt["status"] for attempt in resolution["attempts"]] == [
        "rejected_self_identity",
        "selected_independent_judge",
    ]


def test_no_independent_judge_leaves_row_pending_without_calling_judge(tmp_path):
    run = _write_subjective_run(tmp_path, [{"model": "selene-like", "digest": "digest-s"}])
    client = RecordingJudgeClient()

    result = judge_dumps.judge_run(
        client,
        run,
        judge_model="selene-like",
        qualified_judges=[{"name": "selene-like", "digest": "digest-s", "roles": ["judge", "benchmark_candidate"], "qualified": True}],
    )

    assert client.models == []
    assert result["attempted"] == 0
    assert result["judged"] == 0
    assert result["pending"] == 1
    entry = result["entries"][0]
    assert entry["status"] == "awaiting_independent_judge"
    assert entry["judge_resolution"]["status"] == "awaiting_independent_judge"


def test_judge_run_selects_independent_fallback_per_source_row_identity(tmp_path):
    run = _write_subjective_run(tmp_path, [{"model": "source-alias", "digest": "digest-source"}])
    client = RecordingJudgeClient()

    result = judge_dumps.judge_run(
        client,
        run,
        judge_model="primary-alias",
        qualified_judges=[
            {"name": "primary-alias", "digest": "digest-source", "roles": ["judge", "benchmark_candidate"], "qualified": True},
            {"name": "fallback", "digest": "digest-fallback", "roles": ["judge"], "qualified": True},
        ],
    )

    assert client.models == ["fallback"]
    assert result["judged"] == 1
    assert result["pending"] == 0
    entry = result["entries"][0]
    assert entry["judge_model"] == "fallback"
    assert entry["judge_model_digest"] == "digest-fallback"
    assert entry["judge_resolution"]["attempts"][0]["status"] == "rejected_self_identity"


def test_per_row_resolution_allows_models_with_both_roles_to_judge_other_rows(tmp_path):
    run = _write_subjective_run(tmp_path, [
        {"model": "judge-a", "digest": "digest-a"},
        {"model": "judge-b", "digest": "digest-b"},
    ])
    client = RecordingJudgeClient()

    result = judge_dumps.judge_run(
        client,
        run,
        judge_model="judge-a",
        qualified_judges=[
            {"name": "judge-a", "digest": "digest-a", "roles": ["judge", "benchmark_candidate"], "qualified": True},
            {"name": "judge-b", "digest": "digest-b", "roles": ["judge", "benchmark_candidate"], "qualified": True},
        ],
    )

    assert client.models == ["judge-b", "judge-a"]
    assert result["judged"] == 2
    assert [entry["source_model"] for entry in result["entries"]] == ["judge-a", "judge-b"]
    assert [entry["judge_model"] for entry in result["entries"]] == ["judge-b", "judge-a"]


def test_independent_resolution_is_deterministic_for_repeated_inputs():
    row = {"model": "m", "model_digest_resolved": "digest-m"}
    qualified = [
        {"name": "self", "digest": "digest-m", "roles": ["judge"], "qualified": True},
        {"name": "b", "digest": "digest-b", "roles": ["judge"], "qualified": True},
        {"name": "a", "digest": "digest-a", "roles": ["judge"], "qualified": True},
    ]

    first = campaign.resolve_independent_judge_for_row(row, qualified)
    second = campaign.resolve_independent_judge_for_row(dict(row), list(qualified))
    assert first == second
    assert first["judge_model"] == "b"


def test_fallback_judged_row_rerun_is_noop_without_duplicate_sidecar(tmp_path):
    run = _write_subjective_run(tmp_path, [{"model": "source", "digest": "digest-source"}])
    pool = [
        {"name": "preferred", "digest": "digest-source", "roles": ["judge"], "qualified": True},
        {"name": "fallback", "digest": "digest-fallback", "roles": ["judge"], "qualified": True},
    ]
    client = RecordingJudgeClient()

    first = judge_dumps.judge_run(client, run, judge_model="preferred", qualified_judges=pool)
    second = judge_dumps.judge_run(client, run, judge_model="preferred", qualified_judges=pool)
    entries = (run / "judge_results.jsonl").read_text().splitlines()

    assert first["judged"] == 1
    assert second["eligible"] == 0
    assert second["skipped"][0]["reason"] == "already_judged_by_resolved_independent_judge"
    assert client.models == ["fallback"]
    assert len(entries) == 1


def test_force_rerun_with_fallback_judge_is_explicit_and_appends(tmp_path):
    run = _write_subjective_run(tmp_path, [{"model": "source", "digest": "digest-source"}])
    pool = [{"name": "fallback", "digest": "digest-fallback", "roles": ["judge"], "qualified": True}]
    client = RecordingJudgeClient()

    judge_dumps.judge_run(client, run, judge_model="fallback", qualified_judges=pool)
    forced = judge_dumps.judge_run(client, run, judge_model="fallback", qualified_judges=pool, force=True)

    assert forced["judged"] == 1
    assert client.models == ["fallback", "fallback"]
    assert len((run / "judge_results.jsonl").read_text().splitlines()) == 2


def test_unchanged_pending_rerun_is_noop_but_new_independent_judge_can_judge(tmp_path):
    run = _write_subjective_run(tmp_path, [{"model": "self", "digest": "digest-self"}])
    self_only = [{"name": "self", "digest": "digest-self", "roles": ["judge", "benchmark_candidate"], "qualified": True}]
    with_fallback = [*self_only, {"name": "fallback", "digest": "digest-fallback", "roles": ["judge"], "qualified": True}]
    client = RecordingJudgeClient()

    first = judge_dumps.judge_run(client, run, judge_model="self", qualified_judges=self_only)
    second = judge_dumps.judge_run(client, run, judge_model="self", qualified_judges=self_only)
    third = judge_dumps.judge_run(client, run, judge_model="self", qualified_judges=with_fallback)

    assert first["pending"] == 1
    assert second["eligible"] == 0
    assert second["skipped"][0]["reason"] == "already_awaiting_independent_judge"
    assert third["judged"] == 1
    assert client.models == ["fallback"]
    assert [json.loads(line)["status"] for line in (run / "judge_results.jsonl").read_text().splitlines()] == [
        "awaiting_independent_judge",
        "judged",
    ]


def test_manual_judge_dumps_path_does_not_fabricate_qualification_and_fails_closed(tmp_path):
    run = _write_subjective_run(tmp_path, [{"model": "alias-a", "digest": "digest-a"}])
    client = RecordingJudgeClient()

    result = judge_dumps.judge_run(client, run, judge_model="alias-b")

    assert client.models == []
    assert result["pending"] == 1
    resolution = result["entries"][0]["judge_resolution"]
    assert resolution["attempts"][0]["status"] == "skipped_indeterminate_identity"
    assert resolution["attempts"][0]["judge_identity"]["roles"] == []
    assert resolution["attempts"][0]["judge_identity"]["digest"] is None


def test_pending_sidecar_propagates_to_effective_rows_and_blocks_readiness(tmp_path):
    paths, manifest = campaign.create_campaign("pending", models=["self"], campaigns_root=tmp_path / "campaigns")
    run = _write_subjective_run(tmp_path, [{"model": "self", "digest": "digest-self"}])
    paths.primary_raw_results.write_text((run / "raw_results.jsonl").read_text())
    paths.primary_run_validity.write_text('{"status":"valid"}')
    (paths.primary_dir / "model_identities.json").write_text('{"self":{"digest":"digest-self"}}')
    subjective_dir = paths.primary_dir / "subjective"
    subjective_dir.mkdir(parents=True, exist_ok=True)
    for source in (run / "subjective").rglob("*"):
        if source.is_file():
            target = paths.primary_dir / source.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source.read_text())
    manifest = campaign.transition(paths, manifest, "planned")
    manifest = campaign.transition(paths, manifest, "generating")
    campaign.transition(paths, manifest, "judging")

    judge_dumps.judge_run(
        RecordingJudgeClient(),
        paths.primary_dir,
        judge_model="self",
        qualified_judges=[{"name": "self", "digest": "digest-self", "roles": ["judge", "benchmark_candidate"], "qualified": True}],
    )
    rows = judge_dumps.apply_judgements(paths.primary_dir, [
        json.loads(paths.primary_raw_results.read_text().splitlines()[0])
    ])
    summary = campaign.write_readiness(paths, rows, judge_available=True)
    effective = json.loads(paths.effective_rows.read_text().splitlines()[0])

    assert rows[0]["posthoc_judged"] is False
    assert rows[0]["disposition"] == "awaiting_independent_judge"
    assert effective["terminal_disposition"] == "awaiting_independent_judge"
    assert effective["effective_score"] is None
    assert effective["provenance"]["judge_model"] is None
    assert summary["readiness"] == "not_ready_external_judge"
    assert "awaiting_independent_judge" in summary["blockers"]


def test_bounded_qualification_stops_when_independent_coverage_is_satisfied(monkeypatch):
    selection = campaign.build_judge_selection([
        _measured_text("self", "digest-self", roles=["judge", "benchmark_candidate"]),
        _measured_text("fallback", "digest-fallback", roles=["judge"]),
        _measured_text("extra", "digest-extra", roles=["judge"]),
    ], [], _policy(configured_fallbacks=("self", "fallback", "extra")))
    calls = []

    def fake_qualify(client, candidate):
        calls.append(candidate["name"])
        return {"model": candidate["name"], "digest": candidate["digest"], "qualified": True, "aggregate_disposition": "qualified"}

    monkeypatch.setattr(campaign, "qualify_judge", fake_qualify)
    qualified, qualifications, coverage = campaign.select_qualified_campaign_judges_for_rows(
        object(),
        selection,
        [{"model": "self-alias", "model_digest_resolved": "digest-self"}],
    )

    assert calls == ["self", "fallback"]
    assert [item["name"] for item in qualified] == ["self", "fallback"]
    assert [item["model"] for item in qualifications] == ["self", "fallback"]
    assert coverage["coverage_complete"] is True
    assert coverage["considered"][-1]["model"] == "extra"
    assert coverage["considered"][-1]["decision"] == "not_considered_coverage_satisfied"


def test_campaign_role_source_marks_same_model_as_judge_and_benchmark_candidate():
    policy = _policy(requested_primary="both")
    candidates = campaign.apply_campaign_roles_to_judge_candidates(
        [_measured_text("both", "digest-both")],
        [{"name": "both-alias", "digest": "digest-both"}],
        policy,
    )

    assert candidates[0]["roles"] == ["benchmark_candidate", "judge"]
    assert candidates[0]["role_sources"] == ["campaign_cohort", "judge_candidate_policy"]
