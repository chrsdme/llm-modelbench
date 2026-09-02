"""Anvil Stage 2.6E -- closure fix for judge_dumps.py's manual --judge-model
override path.

Before this fix, `_manual_judge_pool()` (used whenever `judge_run`/
`judge_everything` are invoked without an explicit `qualified_judges` list --
exactly what `cli.py`'s standalone `judge-dumps` command does) built a bare
one-entry pool tagged `qualification_state: "manual_unqualified_designation"`
and handed it straight to `campaign.resolve_independent_judge_for_row()`,
never calling `campaign.build_judge_selection()` or `campaign.qualify_judge()`
at all -- a real, live positive-judge-authority bypass (category C in the
2.6E audit), not merely a cosmetic gap. `tests/test_stage1c_model_roles.py`
and `tests/test_stage1d_judge_integration.py` each had one pre-existing test
characterizing that old bypass behavior (both updated in this same slice to
assert the new fail-closed `ManualJudgeIneligibleError` instead).

Per the go-ahead advice's preferred "Option A": named selection (`--judge-model
X`) still chooses *which* candidate to evaluate, but the candidate must
independently satisfy the exact same typed capability-authority and
qualification gates as an automatically-selected candidate --
`campaign.build_manual_judge_candidate()` runs a real functional
interrogation (never reuses/manufactures a profile), and
`judge_dumps._resolve_manual_judge_pool()` runs it through
`campaign._judge_capability_rejection()` and `campaign.qualify_judge()`
before ever admitting it, raising `ManualJudgeIneligibleError` with a clear
reason otherwise.
"""
import json
from pathlib import Path

import pytest

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
            "timestamp": "2026-08-13T00:00:00Z",
        })
    (run / "raw_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw_rows))
    return run


class _InterrogatableClient:
    """A client complete enough for a real capabilities.interrogate_model()
    functional text probe to succeed, plus real judging chat calls."""

    def __init__(self, name="manual-judge", digest="digest-manual"):
        self.name = name
        self.digest = digest
        self.base = "http://fake.invalid"
        self.calls = []

    def backend_identity(self):
        return type("Identity", (), {"backend": "mock", "implementation": "fixture", "endpoint": self.base})()

    def tags(self):
        return [{"name": self.name, "size": 1, "digest": self.digest}]

    def show(self, model):
        return {"capabilities": ["completion"], "template": "template-v1", "model_info": {}}

    def capability_hints(self, model):
        return ["completion"]

    def chat(self, model, prompt, **kwargs):
        self.calls.append(model)
        if "AIW_TEXT_OK" in prompt:
            return {"ok": True, "text": "AIW_TEXT_OK"}
        return {"ok": True, "text": '{"score": 88, "confidence": 1, "verdict": "synthetic"}'}


def test_genuinely_eligible_manual_judge_still_works(monkeypatch, tmp_path):
    run = _write_subjective_run(tmp_path, [{"model": "source-model", "digest": "digest-source"}])
    client = _InterrogatableClient()

    monkeypatch.setattr(campaign, "qualify_judge", lambda client, candidate, *, ledger=None: {
        "model": candidate["name"], "digest": candidate["digest"], "qualified": True,
        "aggregate_disposition": "qualified", "protocol_version": "judge-qualification-v1",
    })

    result = judge_dumps.judge_run(client, run, judge_model="manual-judge")
    entry = result["entries"][0]

    assert entry["status"] == "judged"
    assert entry["judge_model"] == "manual-judge"
    assert entry["judge_model_digest"] == "digest-manual"


def test_capability_ineligible_manual_judge_raises_not_silently_bypasses(tmp_path):
    run = _write_subjective_run(tmp_path, [{"model": "source-model", "digest": "digest-source"}])

    class _NeverAnswersClient(_InterrogatableClient):
        def chat(self, model, prompt, **kwargs):
            self.calls.append(model)
            return {"ok": False, "error": "connection refused"}

    client = _NeverAnswersClient()
    with pytest.raises(judge_dumps.ManualJudgeIneligibleError, match="not capability-eligible"):
        judge_dumps.judge_run(client, run, judge_model="manual-judge")


def test_qualification_failure_raises_distinctly_from_capability_failure(monkeypatch, tmp_path):
    run = _write_subjective_run(tmp_path, [{"model": "source-model", "digest": "digest-source"}])
    client = _InterrogatableClient()

    monkeypatch.setattr(campaign, "qualify_judge", lambda client, candidate, *, ledger=None: {
        "model": candidate["name"], "digest": candidate["digest"], "qualified": False,
        "aggregate_disposition": "rejected_unstable", "protocol_version": "judge-qualification-v1",
    })

    with pytest.raises(judge_dumps.ManualJudgeIneligibleError, match="failed judge qualification"):
        judge_dumps.judge_run(client, run, judge_model="manual-judge")


def test_missing_client_fails_closed_without_a_confusing_awaiting_state():
    with pytest.raises(judge_dumps.ManualJudgeIneligibleError, match="live client"):
        judge_dumps._resolve_manual_judge_pool(None, "manual-judge")


def test_judge_everything_resolves_manual_pool_once_not_per_run(monkeypatch, tmp_path):
    # Interrogation + qualification are real live-client calls; resolving
    # once and reusing across every run directory (Stage 1.3's "resolve
    # once, reuse" discipline) avoids repeating the same expensive live
    # work once per run for the same judge_model.
    # _write_subjective_run always writes under <root>/run -- build two
    # distinct run directories by renaming each into a shared runs_dir.
    runs_dir = tmp_path / "runs"
    run_a = _write_subjective_run(runs_dir / "a", [{"model": "source-model", "digest": "digest-source"}])
    run_b = _write_subjective_run(runs_dir / "b", [{"model": "source-model", "digest": "digest-source"}])
    run_a.rename(runs_dir / "run-a")
    run_b.rename(runs_dir / "run-b")

    client = _InterrogatableClient()
    resolve_calls = []
    real_resolve = campaign.build_manual_judge_candidate

    def counting_resolve(client, judge_model):
        resolve_calls.append(judge_model)
        return real_resolve(client, judge_model)

    monkeypatch.setattr(campaign, "build_manual_judge_candidate", counting_resolve)
    monkeypatch.setattr(campaign, "qualify_judge", lambda client, candidate, *, ledger=None: {
        "model": candidate["name"], "digest": candidate["digest"], "qualified": True,
        "aggregate_disposition": "qualified", "protocol_version": "judge-qualification-v1",
    })

    result = judge_dumps.judge_everything(client, runs_dir, judge_model="manual-judge")

    assert resolve_calls == ["manual-judge"]
    assert result["runs_scanned"] == 2
