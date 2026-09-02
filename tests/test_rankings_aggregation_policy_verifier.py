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

    # Now the live suite disagrees: py_anagram difficulty moved. Patch ONLY
    # rankings._TASKS -- the dict the verifier reads Task.difficulty from --
    # and deliberately leave rankings._TASK_DIFFICULTY (which only aggregate()
    # consumes) alone, to prove the verifier's drift source is the Task
    # objects, i.e. what a real tasks.py edit actually moves.
    bumped = dataclasses.replace(_TASKS_BY_ID["py_anagram"], difficulty=9.9)
    patched = dict(_TASKS_BY_ID, py_anagram=bumped)
    monkeypatch.setattr(rankings, "_TASKS", patched)

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


def test_verified_alongside_legacy_rows_is_heterogeneous(tmp_path):
    # A digest whose selected canonical rows come partly from a bound run
    # (verified) and partly from a legacy run (no recorded policy) is a
    # genuine policy mix -- an operator should see it. Neither is excluded
    # from the canonical composite (legacy rows still count).
    runs_dir = tmp_path / "runs"
    cfg = _cfg(samples=1)
    bound = _binding_entry("m", ["py_anagram", "py_dedupe"], cfg=cfg)
    # bound run: full level -> its rows win selection for those two tasks
    _write_run(runs_dir, "bound", model="m", digest="d1",
               task_ids=["py_anagram", "py_dedupe"], bound_entry=bound, level="full")
    # legacy run: adds py_csv only, no bindings
    _write_run(runs_dir, "legacy", model="m", digest="d1",
               task_ids=["py_csv"], bound_entry=None, level="full")
    out = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out)
    summary = {row["digest"]: row for row in json.loads((out / "master_summary.json").read_text())}
    disp = _entry_for(summary, "d1")
    assert disp["verdict_counts"] == {"verified": 2, "unverified_legacy": 1}
    assert disp["heterogeneous"] is True
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


def test_report_path_surfaces_the_verdict_without_excluding_rows():
    # 3.3B category B: report.build surfaces an aggregation-policy
    # disposition (same defect class as the canonical-ranking gap) but as
    # advisory metadata only -- it never drops rows from aggregate(), since
    # a per-run report's summary.json is not a cross-run comparability
    # input to rankings.
    import inspect
    from llm_modelbench import report
    src = inspect.getsource(report)
    assert "verify_recorded_aggregation_policy" in src
    assert "_aggregation_policy_provenance" in src
    # aggregate() is still called on the full row set -- no eligible-row
    # filtering was inserted into the report scoring path
    build_src = inspect.getsource(report.build)
    assert "aggregate(rows, cfg.weights, difficulty)" in build_src
    prov_src = inspect.getsource(report._aggregation_policy_provenance)
    # override must be distinguishable so a "verified" count is not read as
    # a canonical endorsement of an override-scored report
    assert "override_active" in prov_src and "weight_override_spec" in prov_src


def test_report_aggregation_policy_provenance_flags_override(tmp_path):
    from types import SimpleNamespace
    from llm_modelbench import report
    rows = [{"model": "m", "task": "py_anagram", "score": 90.0}]
    ctx = {"filters": {}, "sample_mode": "smart", "judge_mode": "single"}
    canonical = report._aggregation_policy_provenance(tmp_path, rows, SimpleNamespace(weight_override_spec=None), ctx)
    overridden = report._aggregation_policy_provenance(tmp_path, rows, SimpleNamespace(weight_override_spec="coding_python=2"), ctx)
    assert canonical["override_active"] is False and canonical["canonical_scorecard"] is True
    assert overridden["override_active"] is True and overridden["canonical_scorecard"] is False


def test_campaign_overlay_path_carries_the_verdict(tmp_path):
    # campaign.py:3944 canonical rankings go through rankings.write_rankings,
    # so the 3.3A verdict must appear with no campaign-specific wiring.
    runs_dir = tmp_path / "runs"
    cfg = _cfg(samples=1)
    entry = _binding_entry("m", _SELECTED, cfg=cfg)
    _write_run(runs_dir, "r1", model="m", digest="d1",
               task_ids=_SELECTED, bound_entry=entry)
    out = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out)
    summary = {row["digest"]: row for row in json.loads((out / "master_summary.json").read_text())}
    assert _entry_for(summary, "d1")["verdict_counts"] == {"verified": 3}


def test_dossier_advisory_surface_counts_verdicts_and_records_exclusion(tmp_path, monkeypatch):
    from llm_modelbench import cli
    runs_dir = tmp_path / "runs"
    cfg = _cfg(samples=1)
    entry = _binding_entry("m", _SELECTED, cfg=cfg)
    run = _write_run(runs_dir, "r1", model="m", digest="d1",
                     task_ids=_SELECTED, bound_entry=entry)
    ledger = {"d1": {"names_seen": ["m"], "categories": {
        "coding_python": {"out_dir": str(run)}}}}
    verdicts = cli._aggregation_policy_by_digest_from_ledger(ledger)
    assert verdicts["d1"]["verdict_counts"] == {"verified": 3}
    assert verdicts["d1"]["override_runs"] is False
    assert verdicts["d1"]["excluded_from_canonical_composite"] == 0

    # a per-run scoring override recorded in summary_meta.json is detected
    (run / "summary_meta.json").write_text(json.dumps({"level": "full", "weight_override": "coding_python=2"}))
    assert cli._aggregation_policy_by_digest_from_ledger(ledger)["d1"]["override_runs"] is True
    (run / "summary_meta.json").write_text(json.dumps({"level": "full"}))

    # difficulty drift on the live task table -> policy_drift: still counted +
    # reasoned in the advisory surface, and now marked excluded from canonical
    drifted = [dataclasses.replace(t, difficulty=t.difficulty + 5.0) if t.id in _SELECTED else t for t in TASKS]
    import llm_modelbench.tasks as tasks_mod
    monkeypatch.setattr(tasks_mod, "TASKS", drifted)
    verdicts2 = cli._aggregation_policy_by_digest_from_ledger(ledger)
    assert set(verdicts2["d1"]["verdict_counts"]) == {"policy_drift"}
    assert verdicts2["d1"]["verdict_counts"]["policy_drift"] == 3
    assert verdicts2["d1"]["drift_reasons"]
    assert verdicts2["d1"]["excluded_from_canonical_composite"] == 3


def _dossier_ledger(tmp_path, run, quality=90.0):
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps({"d1": {"names_seen": ["m"], "categories": {
        "coding_python": {"out_dir": str(run), "quality": quality}}}}))
    return ledger_path


def _run_dossier(cli, ledger_path, weights, capsys):
    from types import SimpleNamespace
    args = SimpleNamespace(ledger=str(ledger_path), weights=weights, out=None, json=True)
    cli.cmd_dossier(args, _cfg())
    return json.loads(capsys.readouterr().out)["d1"]


def test_canonical_dossier_composite_uses_verified_rows_only(tmp_path, monkeypatch, capsys):
    # A digest whose contributing rows come from two runs bound to tasks in
    # *different categories*: py_* (verified) and js_debounce (coding_js). Then
    # js_debounce's difficulty drifts, so only the coding_js run's recorded
    # aggregation-policy hash stops matching. The canonical quality composite
    # must be built from the verified (py_*) rows only -- the coding_js quality
    # must NOT appear in _quality_by_digest_from_ledger -- while the drifted
    # rows stay visible in the advisory surface (verdict_counts + drift_reasons).
    from llm_modelbench import cli
    runs_dir = tmp_path / "runs"
    cfg = _cfg(samples=1)
    verified_tasks = ["py_anagram", "py_dedupe"]
    drift_tasks = ["js_debounce"]
    entry_ok = _binding_entry("m", verified_tasks, cfg=cfg)
    entry_drift = _binding_entry("m", drift_tasks, cfg=cfg)
    run_ok = _write_run(runs_dir, "ok", model="m", digest="d1",
                        task_ids=verified_tasks, bound_entry=entry_ok)
    run_drift = _write_run(runs_dir, "drift", model="m", digest="d1",
                           task_ids=drift_tasks, bound_entry=entry_drift)
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps({"d1": {"names_seen": ["m"], "categories": {
        "coding_python": {"out_dir": str(run_ok), "quality": 90.0},
        "coding_js": {"out_dir": str(run_drift), "quality": 90.0}}}}))

    # drift js_debounce difficulty -> only run_drift's recorded hash stops matching
    drifted = [dataclasses.replace(t, difficulty=t.difficulty + 5.0) if t.id in drift_tasks else t for t in TASKS]
    import llm_modelbench.tasks as tasks_mod
    monkeypatch.setattr(tasks_mod, "TASKS", drifted)

    result = _run_dossier(cli, ledger_path, None, capsys)
    policy = result["aggregation_policy"]
    assert policy["verdict_counts"] == {"policy_drift": 1, "verified": 2}
    assert policy["excluded_from_canonical_composite"] == 1
    assert policy["drift_reasons"]  # drifted rows still visible in provenance
    assert policy["canonical_composite"] is True
    assert "canonical_composite_unavailable_reason" not in policy
    # discriminating assertion: the verified py_* run's quality is present, the
    # drifted coding_js run's quality is dropped from the canonical composite.
    # Fails immediately if the drift filter is removed from
    # _quality_by_digest_from_ledger.
    qbd = cli._quality_by_digest_from_ledger(json.loads(ledger_path.read_text()))
    assert "coding_python" in qbd["d1"]
    assert "coding_js" not in qbd["d1"]


def test_canonical_dossier_composite_unavailable_when_all_rows_drift(tmp_path, monkeypatch, capsys):
    from llm_modelbench import cli
    runs_dir = tmp_path / "runs"
    cfg = _cfg(samples=1)
    entry = _binding_entry("m", _SELECTED, cfg=cfg)
    run = _write_run(runs_dir, "r1", model="m", digest="d1",
                     task_ids=_SELECTED, bound_entry=entry)
    ledger_path = _dossier_ledger(tmp_path, run)

    drifted = [dataclasses.replace(t, difficulty=t.difficulty + 5.0) if t.id in _SELECTED else t for t in TASKS]
    import llm_modelbench.tasks as tasks_mod
    monkeypatch.setattr(tasks_mod, "TASKS", drifted)

    result = _run_dossier(cli, ledger_path, None, capsys)
    policy = result["aggregation_policy"]
    assert policy["excluded_from_canonical_composite"] == 3
    assert policy["all_rows_excluded_as_drift"] is True
    # not a live-policy number over incompatible rows -- honestly unavailable
    assert result["composite"] is None
    assert policy["canonical_composite_unavailable_reason"] == (
        "all comparable rows excluded as policy_drift"
    )
    # discriminating assertion: every row was drift-excluded, so the digest
    # produces no canonical quality at all. Fails if the drift filter is
    # removed from _quality_by_digest_from_ledger (the drifted rows would then
    # aggregate to a live-policy quality under today's constants).
    qbd = cli._quality_by_digest_from_ledger(json.loads(ledger_path.read_text()))
    assert qbd.get("d1", {}) == {}


def test_dossier_own_weights_override_marks_composite_noncanonical(tmp_path, monkeypatch, capsys):
    from llm_modelbench import cli
    runs_dir = tmp_path / "runs"
    cfg = _cfg(samples=1)
    entry = _binding_entry("m", _SELECTED, cfg=cfg)
    run = _write_run(runs_dir, "r1", model="m", digest="d1",
                     task_ids=_SELECTED, bound_entry=entry)
    ledger_path = _dossier_ledger(tmp_path, run)

    canonical = _run_dossier(cli, ledger_path, None, capsys)["aggregation_policy"]
    # keep the sum at 1.0 (validate_weights is strict): shift 0.05 py->js
    overridden = _run_dossier(cli, ledger_path, "coding_python=0.10,coding_js=0.15", capsys)["aggregation_policy"]
    assert canonical["dossier_weights_overridden"] is False and canonical["canonical_composite"] is True
    assert overridden["dossier_weights_overridden"] is True and overridden["canonical_composite"] is False
    # policy verification does not make the override canonical
    assert overridden["verdict_counts"] == {"verified": 3}


def test_dossier_legacy_rows_are_not_excluded(tmp_path, capsys):
    # A run with no bindings -> unverified_legacy rows. They keep contributing
    # to the dossier composite; no retroactive policy attribution.
    from llm_modelbench import cli
    runs_dir = tmp_path / "runs"
    run = _write_run(runs_dir, "legacy", model="m", digest="d1",
                     task_ids=_SELECTED, bound_entry=None)
    ledger_path = _dossier_ledger(tmp_path, run)
    result = _run_dossier(cli, ledger_path, None, capsys)
    policy = result["aggregation_policy"]
    assert policy["verdict_counts"] == {"unverified_legacy": 3}
    assert policy["excluded_from_canonical_composite"] == 0
    assert "canonical_composite_unavailable_reason" not in policy
    assert policy["canonical_composite"] is True
    # legacy rows still feed the quality composite -- no retroactive exclusion
    qbd = cli._quality_by_digest_from_ledger(json.loads(ledger_path.read_text()))
    assert qbd.get("d1", {}).get("coding_python") == 90.0
