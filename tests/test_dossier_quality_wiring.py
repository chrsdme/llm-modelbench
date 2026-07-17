import json
from llm_modelbench import coverage
from llm_modelbench.cli import _quality_by_digest_from_ledger
from llm_modelbench.dossier import composite_score
from llm_modelbench.tasks import TASKS

def test_dossier_quality_is_loaded_from_ledger_run_artifacts(tmp_path):
    task_ids = [t.id for t in TASKS if t.category == "retrieval"]
    run = tmp_path / "run"; run.mkdir()
    rows = [{"model":"m", "task":tid, "category":"retrieval", "score":88.5, "reason":"ok"} for tid in task_ids]
    (run / "raw_results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (run / "model_identities.json").write_text(json.dumps({"m":{"digest":"digest"}}))
    ledger = coverage.update_ledger_from_run({}, raw_rows=rows, identities={"m":{"digest":"digest"}}, tasks=TASKS, benchmark_version="x", out_dir=str(run), timestamp="x")
    quality = _quality_by_digest_from_ledger(ledger)
    assert quality["digest"]["retrieval"] == 88.5
    assert composite_score("digest", ledger, quality["digest"], {"retrieval":1.0}, TASKS)["composite"] == 88.5
