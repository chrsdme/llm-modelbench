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
        {"name": "judge-only", "digest": "j", "roles": ["judge"], "capabilities": ["completion"]},
        {"name": "candidate-only", "digest": "c", "roles": ["benchmark_candidate"], "capabilities": ["completion"]},
        {"name": "both", "digest": "b", "roles": ["judge", "benchmark_candidate"], "capabilities": ["completion"]},
    ], [], _policy(configured_fallbacks=("judge-only", "candidate-only", "both")))

    assert [item["name"] for item in selection.final_eligible_order] == ["judge-only", "both"]
    assert selection.final_eligible_order[1]["roles"] == ["benchmark_candidate", "judge"]
    assert any(item["model"] == "candidate-only" and item["reason"] == "missing_judge_role" for item in selection.rejection_reasons)


def test_benchmark_candidate_role_does_not_prevent_judge_eligibility():
    selection = campaign.build_judge_selection([
        {"name": "both", "digest": "b", "roles": ["benchmark_candidate", "judge"], "capabilities": ["completion"]},
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
