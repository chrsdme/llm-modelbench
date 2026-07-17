import hashlib
import json
from pathlib import Path

from llm_modelbench import judge_dumps, rankings
from llm_modelbench.ollama import MockClient
from llm_modelbench.runner import _task_hash
from llm_modelbench.tasks import TASKS


def _subjective_task():
    return next(task for task in TASKS if task.scorer == "subjective")


def _write_run(root: Path, name: str, *, model="m", error=None):
    task = _subjective_task()
    run = root / name
    dump = run / "subjective" / task.id / f"{model}.md"
    dump.parent.mkdir(parents=True)
    dump.write_text(f"# TASK {task.id}\n\n## OUTPUT\nA useful answer from {model}.\n")
    row = {
        "model": model,
        "model_digest": f"digest-{model}",
        "task": task.id,
        "category": task.category,
        "family": task.family,
        "task_hash": _task_hash(task),
        "score": None,
        "error_kind": error,
        "subjective_path": str(dump.relative_to(run)),
        "timestamp": "2026-01-01T00:00:00Z",
    }
    raw = run / "raw_results.jsonl"
    raw.write_text(json.dumps(row) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": "short"}))
    (run / "model_identities.json").write_text(json.dumps({model: {"digest": f"digest-{model}"}}))
    return run, raw


def _client():
    return MockClient("http://127.0.0.1:11434", 42, 0.0, 10)


def test_posthoc_judge_preserves_raw_and_overlays_sidecar(tmp_path):
    run, raw = _write_run(tmp_path, "r1")
    before = hashlib.sha256(raw.read_bytes()).hexdigest()
    result = judge_dumps.judge_run(_client(), run, judge_model="mock-judge", judge_mode="single")
    after = hashlib.sha256(raw.read_bytes()).hexdigest()
    assert before == after
    assert result["judged"] == 1
    assert (run / "judge_results.jsonl").exists()
    original = [json.loads(raw.read_text())]
    overlaid = judge_dumps.apply_judgements(run, original)
    assert overlaid[0]["score"] == 88.0
    assert overlaid[0]["posthoc_judged"] is True


def test_everything_scans_runs_sequentially_and_skips_source_errors(tmp_path):
    _write_run(tmp_path, "r1", model="m1")
    _write_run(tmp_path, "r2", model="m2")
    _write_run(tmp_path, "r3", model="m3", error="thinking_only")
    result = judge_dumps.judge_everything(_client(), tmp_path, judge_model="mock-judge")
    assert result["runs_scanned"] == 3
    assert result["eligible"] == 2
    assert result["judged"] == 2
    assert result["skipped"] == 1
    assert not (tmp_path / "r3" / "judge_results.jsonl").exists()


def test_repeated_judge_is_resumable_unless_force(tmp_path):
    run, _ = _write_run(tmp_path, "r1")
    first = judge_dumps.judge_run(_client(), run, judge_model="mock-judge")
    second = judge_dumps.judge_run(_client(), run, judge_model="mock-judge")
    forced = judge_dumps.judge_run(_client(), run, judge_model="mock-judge", force=True)
    assert first["judged"] == 1
    assert second["eligible"] == 0
    assert forced["judged"] == 1


def test_judge_sidecar_change_reimports_rankings_and_history(tmp_path):
    runs = tmp_path / "runs"; runs.mkdir()
    run, _ = _write_run(runs, "r1")
    out = tmp_path / "rankings"
    rankings.write_rankings(runs, out)
    before = json.loads((out / "master_summary.json").read_text())[0]
    assert before["history"][0]["posthoc_judged"] is False
    judge_dumps.judge_run(_client(), run, judge_model="mock-judge")
    rankings.write_rankings(runs, out)
    after = json.loads((out / "master_summary.json").read_text())[0]
    assert after["history"][0]["posthoc_judged"] is True
    assert after["history"][0]["score"] == 88.0
