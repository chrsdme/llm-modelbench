"""Anvil Stage 3.3A -- canonical ranking aggregation-policy verifier.

Exercises the real production flow: ``rankings.write_rankings`` ->
``import_new_runs`` (resolves each row's recorded aggregation policy via its
``benchmark_binding_key`` against ``benchmark_bindings.json``) ->
``build_summary`` (excludes provably-drifted rows from the canonical
composite, surfaces an operator-readable disposition).
"""
import dataclasses
import json

from llm_modelbench import rankings
from llm_modelbench.benchmark_binding import (
    binding_to_dict, build_model_binding, protocol_to_dict, row_binding_reference,
)
from llm_modelbench.config import Config
from llm_modelbench.identity import ModelArtifactIdentity
from llm_modelbench.tasks import TASKS

_TASKS_BY_ID = {t.id: t for t in TASKS}
# real deterministic-scored coding tasks (stable difficulties, no judge needed)
_SELECTED = ["py_anagram", "py_dedupe", "py_csv"]


def _cfg(**kw):
    return dataclasses.replace(Config(), **kw)


def _artifact_identity(model):
    return ModelArtifactIdentity.from_ollama_tag_row(
        {"model": model, "digest": f"sha256:{model}", "name": model}
    )


def _binding_entry(model, task_ids, *, cfg, sample_mode="smart", judge_mode="single"):
    tasks = [_TASKS_BY_ID[t] for t in task_ids]
    protocol, binding = build_model_binding(
        model_artifact_identity=_artifact_identity(model),
        selected_tasks=tasks,
        cfg=cfg,
        backend="mock",
        backend_version="mock",
        sample_mode=sample_mode,
        judge_mode=judge_mode,
    )
    return {
        "binding": binding_to_dict(binding),
        "protocol": protocol_to_dict(protocol),
        "row_reference": row_binding_reference(binding, protocol),
    }


def _write_run(
    runs_dir, run_id, *, model, digest, task_ids, bound_entry=None,
    sample_mode="smart", judge_mode="single", requested_samples=1, level="full",
):
    run = runs_dir / run_id
    run.mkdir(parents=True)
    ref = (bound_entry or {}).get("row_reference") or {}
    rows = []
    for tid in task_ids:
        row = {
            "model": model, "task": tid,
            "category": _TASKS_BY_ID[tid].category,
            "score": 90.0, "level": level,
            "timestamp": "2026-02-01T00:00:00Z",
            "task_hash": rankings._CURRENT_HASHES.get(tid),
        }
        row.update(ref)
        rows.append(row)
    (run / "raw_results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": level}))
    (run / "model_identities.json").write_text(json.dumps({model: {"digest": digest}}))
    (run / "filters.json").write_text(json.dumps({
        "level": level, "sample_mode": sample_mode, "judge_mode": judge_mode,
        "requested_samples": requested_samples,
    }))
    if bound_entry is not None:
        (run / "benchmark_bindings.json").write_text(json.dumps({
            "schema_version": 1, "bindings": {model: bound_entry},
        }, indent=2, sort_keys=True))
    return run


def _entry_for(summary, digest):
    return summary[digest]["aggregation_policy"]


def test_matching_policy_row_is_verified_and_counted(tmp_path):
    runs_dir = tmp_path / "runs"
    cfg = _cfg(samples=1)
    entry = _binding_entry("m", _SELECTED, cfg=cfg)
    _write_run(runs_dir, "r1", model="m", digest="d1", task_ids=_SELECTED,
               bound_entry=entry)
    out = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out)
    summary = {row["digest"]: row for row in json.loads((out / "master_summary.json").read_text())}
    disp = _entry_for(summary, "d1")
    assert disp["verdict_counts"] == {"verified": 3}
    assert disp["excluded_from_canonical"] == 0
    assert disp["heterogeneous"] is False
    # canonical score still computed
    assert summary["d1"]["overall_mean_score"] is not None


def test_difficulty_drift_excludes_rows_from_canonical_with_reason(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    cfg = _cfg(samples=1)
    entry = _binding_entry("m", _SELECTED, cfg=cfg)
    _write_run(runs_dir, "r1", model="m", digest="d1", task_ids=_SELECTED,
               bound_entry=entry)

    # Now the live suite disagrees: py_anagram difficulty moved.
    bumped = dataclasses.replace(_TASKS_BY_ID["py_anagram"], difficulty=9.9)
    patched = dict(_TASKS_BY_ID, py_anagram=bumped)
    monkeypatch.setattr(rankings, "_TASKS", patched)
    monkeypatch.setattr(
        rankings, "_TASK_DIFFICULTY",
        {k: v.difficulty for k, v in patched.items()},
    )

    out = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out)
    summary = {row["digest"]: row for row in json.loads((out / "master_summary.json").read_text())}
    disp = _entry_for(summary, "d1")
    assert disp["verdict_counts"].get("policy_drift") == 3
    assert disp["excluded_from_canonical"] == 3
    assert disp["drift_reasons"] and "current canonical" in disp["drift_reasons"][0]
    # every canonical row excluded -> no silent live-weights score
    assert summary["d1"]["overall_mean_score"] is None


def test_legacy_run_without_bindings_stays_unverified_and_still_ranks(tmp_path):
    runs_dir = tmp_path / "runs"
    _write_run(runs_dir, "legacy", model="m", digest="d1", task_ids=_SELECTED,
               bound_entry=None)
    out = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out)
    summary = {row["digest"]: row for row in json.loads((out / "master_summary.json").read_text())}
    disp = _entry_for(summary, "d1")
    assert disp["verdict_counts"] == {"unverified_legacy": 3}
    assert disp["excluded_from_canonical"] == 0
    assert summary["d1"]["overall_mean_score"] is not None


def test_resume_divergent_bindings_are_all_resolvable(tmp_path):
    # One run directory, rows referencing two different bindings: the model
    # binding plus one appended under resume_divergent_bindings. Both must
    # resolve -- never "bindings[model] applies to every row".
    runs_dir = tmp_path / "runs"
    cfg = _cfg(samples=1)
    entry_a = _binding_entry("m", _SELECTED, cfg=cfg, sample_mode="smart")
    entry_b = _binding_entry("m", _SELECTED, cfg=cfg, sample_mode="all")
    assert (entry_a["row_reference"]["benchmark_binding_key"]
            != entry_b["row_reference"]["benchmark_binding_key"])

    run = runs_dir / "r1"
    run.mkdir(parents=True)
    rows = []
    for i, tid in enumerate(_SELECTED):
        ref = (entry_a if i == 0 else entry_b)["row_reference"]
        row = {"model": "m", "task": tid, "category": _TASKS_BY_ID[tid].category,
               "score": 90.0, "level": "full", "timestamp": "2026-02-01T00:00:00Z",
               "task_hash": rankings._CURRENT_HASHES.get(tid), **ref}
        rows.append(row)
    (run / "raw_results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": "full"}))
    (run / "model_identities.json").write_text(json.dumps({"m": {"digest": "d1"}}))
    (run / "filters.json").write_text(json.dumps(
        {"level": "full", "sample_mode": "smart", "judge_mode": "single",
         "requested_samples": 1}))
    (run / "benchmark_bindings.json").write_text(json.dumps({
        "schema_version": 1,
        "bindings": {"m": entry_a},
        "resume_divergent_bindings": [{"model": "m", **entry_b}],
    }, indent=2, sort_keys=True))

    out = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out)
    raw = [json.loads(ln) for ln in (out / "master_raw.jsonl").read_text().splitlines() if ln.strip()]
    verdicts = {r["task"]: r["aggregation_policy_verdict"]["verdict"] for r in raw}
    # Both bindings resolved individually -- neither row fell through to
    # "unverified_legacy". row 0 references entry_a (sample_mode "smart", which
    # matches the run's filters.json) -> verified; rows 1-2 reference entry_b
    # (recorded under sample_mode "all") -> its recorded hash no longer matches
    # what the run's declared config produces -> policy_drift. The point: each
    # row's *own* benchmark_binding_key was resolved, incl. the one only present
    # under resume_divergent_bindings.
    assert "unverified_legacy" not in verdicts.values()
    assert verdicts[_SELECTED[0]] == "verified"
    assert verdicts[_SELECTED[1]] == "policy_drift"
    summary = {row["digest"]: row for row in json.loads((out / "master_summary.json").read_text())}
    disp = _entry_for(summary, "d1")
    assert disp["heterogeneous"] is True
    assert disp["excluded_from_canonical"] == 2


def test_partial_drift_within_one_run_is_heterogeneous(tmp_path, monkeypatch):
    # One run, one binding covering the whole selection, but the live suite
    # moved only py_anagram's difficulty. The recorded hash covers all three
    # tasks together, so the single recompute mismatches -> all three rows
    # drift. To get a genuine within-selection split, give two tasks their own
    # (still-current) binding and one task a binding whose recorded hash is
    # stale.
    runs_dir = tmp_path / "runs"
    cfg = _cfg(samples=1)
    ok_entry = _binding_entry("m", ["py_dedupe", "py_csv"], cfg=cfg)

    bumped = dataclasses.replace(_TASKS_BY_ID["py_anagram"], difficulty=9.9)
    stale_entry = _binding_entry_from_tasks("m", [bumped], cfg=cfg)

    run = runs_dir / "r1"
    run.mkdir(parents=True)
    rows = []
    for tid in ["py_dedupe", "py_csv"]:
        rows.append({"model": "m", "task": tid, "category": _TASKS_BY_ID[tid].category,
                     "score": 90.0, "level": "full", "timestamp": "2026-02-01T00:00:00Z",
                     "task_hash": rankings._CURRENT_HASHES.get(tid),
                     **ok_entry["row_reference"]})
    rows.append({"model": "m", "task": "py_anagram",
                 "category": _TASKS_BY_ID["py_anagram"].category, "score": 90.0,
                 "level": "full", "timestamp": "2026-02-01T00:00:00Z",
                 "task_hash": rankings._CURRENT_HASHES.get("py_anagram"),
                 **stale_entry["row_reference"]})
    (run / "raw_results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": "full"}))
    (run / "model_identities.json").write_text(json.dumps({"m": {"digest": "d1"}}))
    (run / "filters.json").write_text(json.dumps(
        {"level": "full", "sample_mode": "smart", "judge_mode": "single",
         "requested_samples": 1}))
    (run / "benchmark_bindings.json").write_text(json.dumps({
        "schema_version": 1, "bindings": {"m": ok_entry},
        "resume_divergent_bindings": [{"model": "m", **stale_entry}],
    }, indent=2, sort_keys=True))

    out = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out)
    summary = {row["digest"]: row for row in json.loads((out / "master_summary.json").read_text())}
    disp = _entry_for(summary, "d1")
    assert disp["verdict_counts"] == {"policy_drift": 1, "verified": 2}
    assert disp["heterogeneous"] is True
    assert disp["excluded_from_canonical"] == 1
    # the two verified rows still produce a canonical score
    assert summary["d1"]["overall_mean_score"] is not None


def _binding_entry_from_tasks(model, tasks, *, cfg, sample_mode="smart", judge_mode="single"):
    protocol, binding = build_model_binding(
        model_artifact_identity=_artifact_identity(model),
        selected_tasks=tasks,
        cfg=cfg,
        backend="mock",
        backend_version="mock",
        sample_mode=sample_mode,
        judge_mode=judge_mode,
    )
    return {
        "binding": binding_to_dict(binding),
        "protocol": protocol_to_dict(protocol),
        "row_reference": row_binding_reference(binding, protocol),
    }


def test_weight_override_report_path_is_untouched_by_the_verifier():
    # E7: the verifier is a canonical-ranking concern only. report.build's
    # aggregate(rows, cfg.weights, difficulty) override path (cmd_report
    # --weights) must remain a separate, unverified surface -- there is no
    # aggregation_policy_verdict wiring there.
    import inspect
    from llm_modelbench import report
    src = inspect.getsource(report)
    assert "aggregation_policy_verdict" not in src
    assert "verify_recorded_aggregation_policy" not in src
