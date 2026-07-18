import json
import os
import warnings
import zipfile
import pytest
from llm_modelbench import campaign


def fixture(tmp_path, *, origin="primary", complete_refs=False):
    paths, manifest = campaign.create_campaign("pkg", models=["x"], campaigns_root=tmp_path / "campaigns")
    paths.plan_json.write_text(json.dumps({"task_hashes": {"exact": "h"}}))
    paths.inventory_json.write_text(json.dumps([{"name": "x", "digest": "d"}]))
    paths.capabilities_json.write_text(json.dumps({"x": {"supported_families": ["text"]}}))
    paths.primary_raw_results.write_text('{"model":"x","task":"exact","score":100}\n')
    paths.primary_run_validity.write_text('{"status":"valid"}')
    (paths.primary_dir / "model_identities.json").write_text('{"x":{"digest":"d"}}')
    effective={"model":"x","task":"exact","task_hash":"h","result_origin":origin,"terminal_disposition":"scored"}
    if origin == "recovered": effective["recovery_child_id"]="child"
    if origin == "judged": effective["judge_row_hash"]="judge-row"
    paths.effective_rows.write_text(json.dumps(effective)+'\n')
    if complete_refs and origin == "recovered":
        paths.recovery_plan.write_text('{"actions":[]}'); paths.recovery_result.write_text('{"status":"complete"}')
        paths.recovery_attempts.write_text('{"child_run_id":"child"}\n')
        child=paths.recovery_children_dir/"child"; child.mkdir(); (child/"attempt.json").write_text('{}')
    if complete_refs and origin == "judged":
        (paths.judge_dir/"judge_selection.json").write_text('{"judge":{"digest":"jd"}}')
        paths.judge_results.write_text('{"source_row_hash":"judge-row","judge_digest":"jd"}\n')
        paths.judge_summary.write_text('{"judged":1}')
    (paths.reports_dir/"readiness.json").write_text('{"readiness":"ready_for_adoption"}')
    (paths.reports_dir/"readiness.md").write_text('# Ready\n')
    (paths.reports_dir/"report.html").write_text('report')
    (paths.primary_dir/"report.html").write_text('report')
    (paths.candidate_rankings_dir/"master_raw.jsonl").write_text('{"model":"x","task":"exact","score":100}\n')
    (paths.candidate_rankings_dir/"master_summary.json").write_text('[]')
    for state in ("planned","generating","packaged"):
        manifest=campaign.transition(paths,manifest,state)
    campaign.package_campaign(paths)
    return paths


def rewrite(paths, *, remove=None, change=None, extra=None, duplicate=None):
    package=paths.packages_dir/f"{paths.campaign_id}-review.zip"
    with zipfile.ZipFile(package) as z:
        members=[(i,z.read(i.filename)) for i in z.infolist() if i.filename != remove]
    temp=package.with_suffix('.new')
    with zipfile.ZipFile(temp,'w') as z:
        for info,data in members:
            if change and info.filename==change[0]: data=change[1]
            z.writestr(info,data)
        if extra: z.writestr(extra[0],extra[1])
        if duplicate:
            data=next(data for info,data in members if info.filename==duplicate)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning); z.writestr(duplicate,data)
    os.replace(temp,package)
    paths.checksums_json.write_text(json.dumps({"package":campaign._sha256(package),"size":package.stat().st_size}))


def test_complete_package_has_structured_verified_inventory_and_no_duplicate_reports(tmp_path):
    paths=fixture(tmp_path); result=campaign.verify_package_details(paths)
    assert result["valid"] and result["verified_checksum_count"] > 10
    assert result["required_files_valid"] and result["terminal_ledger_valid"] and result["candidate_rankings_valid"]
    with zipfile.ZipFile(paths.packages_dir/"pkg-review.zip") as z:
        names=z.namelist(); assert "package/inventory.json" in names and "package/sha256.json" in names
        assert "reports/report.html" in names and "evidence/primary/report.html" not in names


@pytest.mark.parametrize("missing", ["manifest.json","plan/plan.json","plan/inventory.json","plan/capabilities.json","evidence/primary/raw_results.jsonl","evidence/primary/run_validity.json","evidence/primary/model_identities.json","evidence/effective_rows.jsonl","reports/readiness.json","reports/readiness.md"])
def test_missing_required_member_is_rejected(tmp_path, missing):
    paths=fixture(tmp_path); rewrite(paths,remove=missing)
    result=campaign.verify_package_details(paths)
    assert not result["valid"]


def test_tamper_size_checksum_and_stale_source_are_rejected(tmp_path):
    paths=fixture(tmp_path); rewrite(paths,change=("manifest.json",b"tampered"))
    result=campaign.verify_package_details(paths)
    assert not result["valid"] and any("mismatch" in error for error in result["errors"])
    paths=fixture(tmp_path/"stale"); paths.primary_raw_results.write_text("changed\n")
    assert not campaign.verify_package(paths)
    readiness=json.loads((paths.reports_dir/"readiness.json").read_text())
    assert readiness["readiness"] == "not_ready_manual_items"
    assert "package_verification_failed" in readiness["blockers"]


def test_unlisted_duplicate_and_unsafe_members_are_rejected(tmp_path):
    paths=fixture(tmp_path); rewrite(paths,extra=("unexpected.txt",b"x"))
    assert not campaign.verify_package(paths)
    paths=fixture(tmp_path/"dup"); rewrite(paths,duplicate="manifest.json")
    assert not campaign.verify_package(paths)
    paths=fixture(tmp_path/"traversal"); rewrite(paths,extra=("../escape",b"x"))
    assert not campaign.verify_package(paths)
    paths=fixture(tmp_path/"absolute"); rewrite(paths,extra=("/absolute",b"x"))
    assert not campaign.verify_package(paths)


def test_symlink_and_malformed_internal_metadata_are_rejected(tmp_path):
    paths=fixture(tmp_path); package=paths.packages_dir/"pkg-review.zip"
    with zipfile.ZipFile(package,"a") as z:
        info=zipfile.ZipInfo("link"); info.create_system=3; info.external_attr=(0o120777 << 16); z.writestr(info,b"target")
    paths.checksums_json.write_text(json.dumps({"package":campaign._sha256(package),"size":package.stat().st_size}))
    assert not campaign.verify_package(paths)
    paths=fixture(tmp_path/"badsha"); rewrite(paths,change=("package/sha256.json",b"{"))
    assert not campaign.verify_package(paths)
    paths=fixture(tmp_path/"badinv"); rewrite(paths,change=("package/inventory.json",b"[]"))
    assert not campaign.verify_package(paths)


def test_recovery_and_judge_references_require_evidence(tmp_path):
    paths=fixture(tmp_path,origin="recovered")
    result=campaign.verify_package_details(paths)
    assert not result["valid"] and not result["recovery_references_valid"]
    paths=fixture(tmp_path/"judge",origin="judged")
    result=campaign.verify_package_details(paths)
    assert not result["valid"] and not result["judge_references_valid"]


@pytest.mark.parametrize("origin", ["recovered", "judged"])
def test_complete_recovery_and_judge_references_verify(tmp_path, origin):
    paths=fixture(tmp_path,origin=origin,complete_refs=True)
    result=campaign.verify_package_details(paths)
    assert result["valid"]
    assert result["recovery_references_valid"] and result["judge_references_valid"]


def test_atomic_rebuild_preserves_primary_and_one_final_package(tmp_path):
    paths=fixture(tmp_path); before=paths.primary_raw_results.read_bytes()
    campaign.package_campaign(paths)
    assert paths.primary_raw_results.read_bytes()==before
    assert len(list(paths.packages_dir.glob('*.zip')))==1 and campaign.verify_package(paths)


def test_archive_build_failure_preserves_previous_verified_package(tmp_path, monkeypatch):
    paths=fixture(tmp_path); package=paths.packages_dir/"pkg-review.zip"; before=package.read_bytes()
    original=zipfile.ZipFile
    def fail(path, mode="r", *args, **kwargs):
        if mode == "w": raise OSError("injected archive failure")
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(zipfile,"ZipFile",fail)
    with pytest.raises(OSError,match="injected"):
        campaign.package_campaign(paths)
    assert package.read_bytes()==before
    assert not list(paths.packages_dir.glob("*.tmp"))
