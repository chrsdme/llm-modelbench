"""Bounded RC20 retention, migration, and root-hygiene matrix.

All fixtures are disposable and live below pytest's temporary directory.
"""
import hashlib
import json
import os

import pytest

from llm_modelbench import campaign


def _fixture(tmp_path, state="rejected", *, dumps=True, campaign_id="c", campaigns_root=None):
    paths, manifest = campaign.create_campaign(campaign_id, models=["m"], campaigns_root=campaigns_root or (tmp_path / "campaigns"))
    paths.primary_raw_results.write_text('{"model":"m","task":"exact","score":1}\n')
    paths.primary_run_validity.write_text('{"status":"valid"}\n')
    (paths.primary_dir / "model_identities.json").write_text('{"m":{"digest":"d"}}\n')
    paths.plan_json.write_text('{"task_hashes":{"exact":"h"}}\n')
    paths.inventory_json.write_text('[{"name":"m","digest":"d"}]\n')
    paths.capabilities_json.write_text('{"m":{"state":"confirmed_supported"}}\n')
    paths.effective_rows.write_text('{"model":"m","task":"exact","task_hash":"h","terminal_disposition":"scored"}\n')
    (paths.reports_dir / "readiness.json").write_text('{"readiness":"ready_for_adoption"}\n')
    (paths.reports_dir / "readiness.md").write_text("# ready\n")
    paths.readiness_json.write_text('{"readiness":"ready_for_adoption"}\n')
    (paths.candidate_rankings_dir / "master_raw.jsonl").write_text('{"model":"m","task":"exact","score":1}\n')
    (paths.candidate_rankings_dir / "master_summary.json").write_text('{}\n')
    if dumps:
        (paths.primary_dumps_dir / "disposable.txt").write_text("dump")
    for target in ("planned", "generating", "packaged", state):
        manifest = campaign.transition(paths, manifest, target)
    campaign.package_campaign(paths)
    return paths


@pytest.mark.parametrize("state", ["accepted", "rejected", "archived_diagnostic"])
def test_cleanup_terminal_states_are_eligible(tmp_path, state):
    paths = _fixture(tmp_path, state)
    result = campaign.cleanup_campaign(paths)
    assert result["eligible"] is True
    assert result["dry_run"] is True
    assert result["policy_version"] == "rc20-retention-1"


@pytest.mark.parametrize("state", ["created", "planned", "generating", "recovering", "judging", "packaged", "interrupted", "failed"])
def test_cleanup_nonterminal_states_are_refused(tmp_path, state):
    paths, manifest = campaign.create_campaign("c", models=["m"], campaigns_root=tmp_path / "campaigns")
    if state == "created":
        pass
    elif state == "planned":
        manifest = campaign.transition(paths, manifest, "planned")
    elif state in {"generating", "recovering", "judging", "interrupted"}:
        manifest = campaign.transition(paths, manifest, "planned")
        manifest = campaign.transition(paths, manifest, "generating")
        if state != "generating":
            manifest = campaign.transition(paths, manifest, state)
    elif state == "packaged":
        manifest = campaign.transition(paths, manifest, "planned")
        manifest = campaign.transition(paths, manifest, "generating")
        manifest = campaign.transition(paths, manifest, "packaged")
    else:
        manifest = campaign.transition(paths, manifest, "failed")
    result = campaign.cleanup_campaign(paths)
    assert result["eligible"] is False
    assert any("ineligible_state" in blocker for blocker in result["blockers"])
    with pytest.raises(campaign.CampaignError):
        campaign.cleanup_campaign(paths, apply=True)


def test_cleanup_dry_run_is_zero_mutation_and_apply_preserves_forensics(tmp_path):
    paths = _fixture(tmp_path)
    before = {p.relative_to(paths.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths.root.rglob("*") if p.is_file() and paths.primary_dumps_dir not in p.parents}
    preview = campaign.cleanup_campaign(paths)
    assert preview["files_removed"] == []
    assert paths.primary_dumps_dir.exists()
    after_preview = {p.relative_to(paths.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in paths.root.rglob("*") if p.is_file() and paths.primary_dumps_dir not in p.parents}
    assert before == after_preview
    applied = campaign.cleanup_campaign(paths, apply=True)
    assert applied["files_removed"] == ["evidence/primary/dumps/disposable.txt"]
    assert not paths.primary_dumps_dir.exists()
    for retained in (paths.primary_raw_results, paths.primary_run_validity, paths.effective_rows,
                     paths.reports_dir / "readiness.json", paths.reports_dir / "readiness.md",
                     paths.packages_dir / "c-review.zip", paths.checksums_json):
        assert retained.exists()
    assert (paths.root / "cleanup" / "cleanup_record.json").exists()
    second = campaign.cleanup_campaign(paths, apply=True)
    assert second["applied"] is True
    assert second["files_removed"] == []


def test_cleanup_lock_package_and_pending_matrix(tmp_path):
    paths = _fixture(tmp_path)
    paths.lock_file.write_text(json.dumps({"pid": os.getpid(), "hostname": campaign.socket.gethostname()}))
    assert "active_or_ambiguous_lock" in campaign.cleanup_campaign(paths)["blockers"]
    paths.lock_file.unlink()
    paths.recovery_result.write_text('{"status":"pending","actions":[{"status":"pending"}]}')
    result = campaign.cleanup_campaign(paths)
    assert "pending_recovery" in result["blockers"]
    paths.recovery_result.unlink()
    paths.judge_summary.write_text('{"pending":1}')
    assert "pending_judging" in campaign.cleanup_campaign(paths)["blockers"]
    paths.judge_summary.unlink()
    paths.effective_rows.write_text('{"terminal_disposition":"conflicting_evidence/manual_review"}\n')
    assert "unresolved_manual_item" in campaign.cleanup_campaign(paths)["blockers"]


def test_cleanup_stale_checksum_and_missing_evidence_refuse(tmp_path):
    paths = _fixture(tmp_path)
    paths.lock_file.write_text(json.dumps({"pid": 99999999, "hostname": campaign.socket.gethostname()}))
    preview = campaign.cleanup_campaign(paths)
    assert preview["eligible"] is True
    assert "proven_stale_same_host_lock" in preview["warnings"]
    paths.checksums_json.write_text('{"package":"bad","size":1}')
    assert "package_verification_failed" in campaign.cleanup_campaign(paths)["blockers"]
    with pytest.raises(campaign.CampaignError):
        campaign.cleanup_campaign(paths, apply=True)


@pytest.mark.parametrize("variant", ["recovery", "judge"])
def test_cleanup_missing_referenced_evidence_refuses(tmp_path, variant):
    paths = _fixture(tmp_path)
    if variant == "recovery":
        paths.effective_rows.write_text('{"result_origin":"recovered","recovery_child_id":"x"}\n')
    else:
        paths.effective_rows.write_text('{"result_origin":"judged","judge_row_hash":"x"}\n')
    assert "package_verification_failed" in campaign.cleanup_campaign(paths)["blockers"]


def test_cleanup_symlink_and_traversal_boundaries(tmp_path):
    paths = _fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (paths.primary_dumps_dir / "disposable.txt").unlink()
    paths.primary_dumps_dir.rmdir()
    paths.primary_dumps_dir.symlink_to(outside, target_is_directory=True)
    result = campaign.cleanup_campaign(paths)
    assert result["eligible"] is False
    assert any(item in result["blockers"] for item in ("unsafe_cleanup_target", "symlink_cleanup_target"))
    assert outside.exists()


def test_cleanup_failure_is_recoverable_and_never_touches_sibling(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    sibling = _fixture(tmp_path / "other")
    original = campaign.shutil.rmtree
    calls = {"n": 0}

    def fail_once(target, *args, **kwargs):
        if str(target).endswith("evidence/primary/dumps"):
            calls["n"] += 1
        if calls["n"] == 1 and str(target).endswith("evidence/primary/dumps"):
            raise OSError("injected cleanup failure")
        return original(target)

    monkeypatch.setattr(campaign.shutil, "rmtree", fail_once)
    with pytest.raises(campaign.CampaignError):
        campaign.cleanup_campaign(paths, apply=True)
    assert paths.primary_raw_results.exists()
    assert sibling.primary_raw_results.exists()
    monkeypatch.setattr(campaign.shutil, "rmtree", original)
    campaign.cleanup_campaign(paths, apply=True)
    assert sibling.primary_dumps_dir.exists()


def test_cleanup_all_reports_and_processes_only_eligible(tmp_path):
    shared = tmp_path / "shared"
    _fixture(shared, campaign_id="good", campaigns_root=shared / "campaigns")
    bad, manifest = campaign.create_campaign("bad", models=["m"], campaigns_root=shared / "campaigns")
    manifest = campaign.transition(bad, manifest, "planned")
    result = campaign.cleanup_all_campaigns(campaigns_root=shared / "campaigns", apply=True)
    by_id = {item["campaign_id"]: item for item in result["campaigns"]}
    assert by_id["good"]["applied"] is True
    assert by_id["bad"]["eligible"] is False
    assert bad.root.exists()


def test_migration_dry_run_and_apply_are_copy_only_and_provenanced(tmp_path):
    runs = tmp_path / "runs"
    source = runs / "old"
    source.mkdir(parents=True)
    (source / "raw_results.jsonl").write_text('{"model":"m","task":"exact","score":1}\n')
    (source / "run_validity.json").write_text('{"status":"valid"}\n')
    (source / "report.html").write_text("report")
    (source / "repair_results.jsonl").write_text('{"attempt":1}\n')
    (source / "judge_results.jsonl").write_text('{"score":1}\n')
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    preview = campaign.migrate_legacy_run("old", "new", runs_dir=runs, campaigns_root=tmp_path / "campaigns")
    assert preview["dry_run"] is True and preview["applied"] is False
    assert preview["source_checksums"]["raw_results.jsonl"]["sha256"] == before["raw_results.jsonl"]
    assert not (tmp_path / "campaigns" / "new").exists()
    result = campaign.migrate_legacy_run("old", "new", runs_dir=runs, campaigns_root=tmp_path / "campaigns", apply=True)
    paths = campaign.resolve_paths("new", campaigns_root=tmp_path / "campaigns")
    assert result["applied"] and result["source_immutability_verified"] and result["destination_valid"]
    assert (paths.primary_raw_results.read_text() == (source / "raw_results.jsonl").read_text())
    assert (paths.reports_dir / "report.html").exists()
    assert (paths.recovery_dir / "recovery_results.jsonl").exists()
    assert paths.judge_results.exists()
    provenance = json.loads((paths.plan_dir / "migration_provenance.json").read_text())
    assert provenance["migration_policy_version"] == "rc20-legacy-copy-1"
    assert provenance["source_immutability_verified"] is True
    assert provenance["original_model_identities"]["m"]["digest"] == "legacy_digest_unavailable"
    assert provenance["task_ids_and_hashes"]["exact"] == "legacy_task_hash_unavailable"
    readiness = json.loads(paths.readiness_json.read_text())
    assert readiness["readiness"] == "diagnostic_only"
    assert json.loads((paths.primary_dir / "model_identities.json").read_text())["m"]["digest"] == "legacy_digest_unavailable"
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()} == before
    assert not (tmp_path / "rankings" / "master_raw.jsonl").exists()


def test_migration_refuses_repeat_malformed_traversal_symlink_and_failure(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    source = runs / "old"
    source.mkdir(parents=True)
    (source / "raw_results.jsonl").write_text('{"score":1}\n')
    campaign.migrate_legacy_run("old", "new", runs_dir=runs, campaigns_root=tmp_path / "campaigns", apply=True)
    with pytest.raises(campaign.CampaignError, match="already exists"):
        campaign.migrate_legacy_run("old", "new", runs_dir=runs, campaigns_root=tmp_path / "campaigns", apply=True)
    with pytest.raises(campaign.CampaignError):
        campaign.migrate_legacy_run("../old", "x", runs_dir=runs, campaigns_root=tmp_path / "campaigns")
    bad = runs / "bad"
    bad.mkdir()
    (bad / "raw_results.jsonl").write_text("not-json\n")
    with pytest.raises(campaign.CampaignError):
        campaign.migrate_legacy_run("bad", "bad-target", runs_dir=runs, campaigns_root=tmp_path / "campaigns", apply=True)
    escaped = runs / "escaped"
    escaped.mkdir()
    (escaped / "raw_results.jsonl").symlink_to(source / "raw_results.jsonl")
    with pytest.raises(campaign.CampaignError, match="symlink"):
        campaign.migrate_legacy_run("escaped", "escaped-target", runs_dir=runs, campaigns_root=tmp_path / "campaigns")
    failing = runs / "failing"
    failing.mkdir()
    (failing / "raw_results.jsonl").write_text('{"score":1}\n')
    original = campaign.shutil.copy2
    monkeypatch.setattr(campaign.shutil, "copy2", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected copy failure")))
    with pytest.raises(OSError):
        campaign.migrate_legacy_run("failing", "failing-target", runs_dir=runs, campaigns_root=tmp_path / "campaigns", apply=True)
    assert not (tmp_path / "campaigns" / "failing-target").exists()
    monkeypatch.setattr(campaign.shutil, "copy2", original)


def test_migration_paths_and_root_artifact_hygiene(tmp_path):
    runs = tmp_path / "runs"
    source = runs / "old"
    source.mkdir(parents=True)
    (source / "raw_results.jsonl").write_text('{"score":1}\n')
    result = campaign.migrate_legacy_run("old", "new", runs_dir=runs, campaigns_root=tmp_path / "campaigns", apply=True)
    paths = campaign.resolve_paths("new", campaigns_root=tmp_path / "campaigns")
    assert result["destination_path"] == str(paths.root)
    assert paths.root.resolve().is_relative_to((tmp_path / "campaigns").resolve())
    root_files = [item for item in tmp_path.iterdir() if item.is_file()]
    assert root_files == []
    assert not (tmp_path / "rankings-separate").exists()
    assert campaign.verify_package(paths)


def test_campaign_mode_commands_have_no_root_artifact_leakage(tmp_path, monkeypatch):
    """Every campaign-mode command keeps outputs below campaigns/<id>/.

    Commands that require a live model or an adoption confirmation are still
    invoked through their normal parser; their expected bounded errors must not
    create legacy root outputs.
    """
    from llm_modelbench.cli import main

    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs" / "legacy"
    runs.mkdir(parents=True)
    (runs / "raw_results.jsonl").write_text('{"model":"m","task":"exact","score":1}\n')
    for argv in (
        ["campaign", "plan", "--campaign-id", "plan", "--mock", "--tasks", "json_extract", "--level", "short"],
        ["campaign", "status", "plan"],
        ["campaign", "run", "--campaign-id", "run", "--mock", "--tasks", "json_extract", "--level", "short", "--samples", "1"],
        ["campaign", "migrate-legacy", "--run-id", "legacy", "--campaign-id", "migrated", "--dry-run"],
        ["rankings", "--adopt", "plan", "--dry-run"],
    ):
        try:
            main(argv)
        except (SystemExit, ValueError, campaign.CampaignError):
            pass
    # Create a valid temporary campaign for package/clean/status and an
    # interrupted one for the explicit resume parser.
    _fixture(tmp_path, campaign_id="packaged", campaigns_root=tmp_path / "campaigns")
    main(["campaign", "package", "packaged"])
    main(["campaign", "clean", "packaged", "--dry-run"])
    interrupted, manifest = campaign.create_campaign("interrupted", models=["m"], campaigns_root=tmp_path / "campaigns")
    manifest = campaign.transition(interrupted, manifest, "planned")
    manifest = campaign.transition(interrupted, manifest, "generating")
    campaign.transition(interrupted, manifest, "interrupted")
    main(["campaign", "resume", "interrupted"])
    forbidden = {"acceptance_artifacts", "overnight_logs", "rankings-separate", "report-package", "report_package"}
    for name in forbidden:
        assert not (tmp_path / name).exists(), name
    root_archives = [item for item in tmp_path.iterdir() if item.is_file() and item.suffix.lower() in {".zip", ".tar", ".tgz"}]
    assert root_archives == []
    assert all(item.parent.name == "campaigns" or item.parent.parent.name == "campaigns" or item.name in {"runs"}
               for item in (tmp_path / "campaigns").iterdir())
