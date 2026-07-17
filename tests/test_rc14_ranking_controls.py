import json
from pathlib import Path

from llm_modelbench import rankings
from llm_modelbench.cli import build_parser
from llm_modelbench.ranking_controls import set_model_excluded, write_run_scope, SCOPE_SEPARATE


def _write_run(runs: Path, run_id: str, model: str, score: float = 100.0):
    run = runs / run_id
    run.mkdir(parents=True)
    row = {
        "model": model,
        "task": "py_dedupe",
        "category": "coding_python",
        "family": "text",
        "score": score,
        "task_hash": rankings._CURRENT_HASHES["py_dedupe"],
        "level": "full",
        "timestamp": "2026-07-17T00:00:00Z",
    }
    (run / "raw_results.jsonl").write_text(json.dumps(row) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": "full"}))
    (run / "model_identities.json").write_text(json.dumps({model: {"digest": f"digest-{model}"}}))
    return run


def test_exclude_model_is_non_destructive_and_public_safe(tmp_path):
    runs = tmp_path / "runs"
    rankings_dir = tmp_path / "rankings"
    _write_run(runs, "full", "public-model:latest")

    first = rankings.write_rankings(runs, rankings_dir, force_rescan=True)
    assert first["models"] == 1
    set_model_excluded(rankings_dir, "public-model:latest", True, reason="playground diagnostic")
    second = rankings.write_rankings(runs, rankings_dir, force_rescan=True)
    assert second["models"] == 0

    raw = (rankings_dir / "master_raw.jsonl").read_text()
    assert "public-model:latest" in raw
    exclusions = json.loads((rankings_dir / "exclusions.json").read_text())
    entry = exclusions["excluded_models"]["public-model:latest"]
    assert entry["reason"] == "playground diagnostic"
    assert "operator" not in entry
    assert "operator" not in json.dumps(exclusions).lower()


def test_separate_run_is_not_imported_into_canonical_but_has_own_output(tmp_path):
    runs = tmp_path / "runs"
    canonical = tmp_path / "rankings"
    separate = tmp_path / "rankings-separate" / "diag"
    _write_run(runs, "canonical", "core-model:latest", 90.0)
    separate_run = _write_run(runs, "diag", "diag-model:latest", 100.0)
    write_run_scope(separate_run, scope=SCOPE_SEPARATE, rankings_dir=separate)

    main = rankings.write_rankings(runs, canonical, force_rescan=True)
    assert main["models"] == 1
    assert "diag-model:latest" not in (canonical / "master_summary.json").read_text()

    diag = rankings.write_rankings(runs, separate, force_rescan=True, include_separate=True, only_run_ids=["diag"])
    assert diag["models"] == 1
    assert "diag-model:latest" in (separate / "master_summary.json").read_text()


def test_ranking_flags_parse_on_evidence_commands():
    parser = build_parser()
    cases = [
        ["run", "--mock", "--yes", "--no-ranking-update"],
        ["run", "--mock", "--yes", "--separate-ranking"],
        ["repair", "--run-id", "r", "--apply", "--yes", "--no-ranking-update"],
        ["repair", "--run-id", "r", "--apply", "--yes", "--separate-ranking"],
        ["context-profile", "--model", "m", "--yes", "--no-ranking-update"],
        ["context-profile", "--model", "m", "--yes", "--separate-ranking"],
        ["judge-dumps", "--run-id", "r", "--yes", "--no-ranking-update"],
        ["judge-dumps", "--run-id", "r", "--yes", "--separate-ranking"],
        ["rankings", "--exclude-model", "m", "--reason", "diagnostic"],
        ["rankings", "--include-model", "m"],
        ["rankings", "--list-excluded"],
    ]
    for argv in cases:
        args = parser.parse_args(argv)
        assert args.cmd in {"run", "repair", "context-profile", "judge-dumps", "rankings"}
