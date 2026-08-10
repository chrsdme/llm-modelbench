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


def rewrite_verified(paths, *, remove=None, change=None):
    package = paths.packages_dir / f"{paths.campaign_id}-review.zip"
    with zipfile.ZipFile(package) as z:
        members = {info.filename: z.read(info.filename) for info in z.infolist()}
    if remove:
        members.pop(remove, None)
    if change:
        members[change[0]] = change[1]
    members.pop("package/sha256.json", None)
    members.pop("package/inventory.json", None)
    inventory_files = [
        {
            "path": name,
            "sha256": campaign.hashlib.sha256(content).hexdigest(),
            "size": len(content),
            "role": name.split("/", 1)[0],
        }
        for name, content in sorted(members.items())
    ]
    inventory_bytes = json.dumps(
        {"campaign_id": paths.campaign_id, "files": inventory_files},
        indent=2,
        sort_keys=True,
    ).encode()
    checksum_files = [
        *inventory_files,
        {
            "path": "package/inventory.json",
            "sha256": campaign.hashlib.sha256(inventory_bytes).hexdigest(),
            "size": len(inventory_bytes),
            "role": "package",
        },
    ]
    members["package/inventory.json"] = inventory_bytes
    members["package/sha256.json"] = json.dumps({"files": checksum_files}, indent=2, sort_keys=True).encode()
    temp = package.with_suffix(".verified-new")
    with zipfile.ZipFile(temp, "w") as z:
        for name, content in sorted(members.items()):
            z.writestr(name, content)
    os.replace(temp, package)
    paths.checksums_json.write_text(json.dumps({"package": campaign._sha256(package), "size": package.stat().st_size}))


def supersession_fixture(tmp_path, *, chain_length=1):
    paths = fixture(tmp_path)
    source = {
        "model": "x",
        "model_digest_resolved": "d",
        "task": "exact",
        "task_hash": "h",
        "score": 0,
        "reason": "source",
    }
    replacement_b = {
        "model": "x",
        "model_digest_resolved": "d",
        "task": "exact",
        "task_hash": "h",
        "score": 50 if chain_length == 2 else 100,
        "reason": "replacement b",
    }
    replacement_c = {
        "model": "x",
        "model_digest_resolved": "d",
        "task": "exact",
        "task_hash": "h",
        "score": 100,
        "reason": "replacement c",
    }
    paths.primary_raw_results.write_text(json.dumps(source, sort_keys=True) + "\n")
    first = campaign.record_supersession(
        paths,
        source_campaign_id=paths.campaign_id,
        source_run_id="primary",
        source_row=source,
        replacement_campaign_id=paths.campaign_id,
        replacement_run_id="replacement-b",
        replacement_row=replacement_b,
        reason="replace a with b",
        operator="test",
        tool="pytest",
    )
    second = None
    if chain_length == 2:
        second = campaign.record_supersession(
            paths,
            source_campaign_id=paths.campaign_id,
            source_run_id="replacement-b",
            source_row=replacement_b,
            replacement_campaign_id=paths.campaign_id,
            replacement_run_id="replacement-c",
            replacement_row=replacement_c,
            reason="replace b with c",
            operator="test",
            tool="pytest",
        )
    campaign.write_readiness(paths, [source])
    campaign.package_campaign(paths)
    return paths, first, second


def config_fixture(tmp_path):
    paths = fixture(tmp_path)
    config = campaign.campaign_config_template()
    config["campaign_id"] = paths.campaign_id
    config["models"] = ["x"]
    record = campaign.campaign_config_plan_record(config)
    paths.plan_dir.joinpath("campaign_config.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    manifest = campaign.load_manifest(paths)
    manifest.notes["campaign_config"] = {
        "managed": True,
        "schema_version": campaign.CAMPAIGN_CONFIG_SCHEMA_VERSION,
        "config_signature": record["config_signature"],
        "path": "plan/campaign_config.json",
    }
    campaign.write_manifest(paths, manifest)
    campaign.package_campaign(paths)
    return paths, record


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


def test_stale_package_adoption_remains_fail_closed(tmp_path):
    paths = fixture(tmp_path)
    paths.readiness_json.write_text('{"readiness":"ready_for_adoption"}')
    paths.plan_json.write_text(json.dumps({"task_hashes": {"exact": "changed"}}))
    with pytest.raises(campaign.CampaignError, match="package checksums do not verify"):
        campaign.adopt_campaign(paths, rankings_dir=tmp_path / "rankings", dry_run=True)


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


@pytest.mark.parametrize("chain_length", [1, 2])
def test_superseded_effective_rows_require_valid_packaged_supersession_graph(tmp_path, chain_length):
    paths, _, _ = supersession_fixture(tmp_path, chain_length=chain_length)
    result = campaign.verify_package_details(paths)
    assert result["valid"]
    assert result["supersession_references_valid"]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda paths, first, second: ("evidence/supersessions.jsonl", None), "missing"),
        (lambda paths, first, second: ("evidence/supersessions.jsonl", b"{"), "invalid"),
        (
            lambda paths, first, second: (
                "evidence/supersessions.jsonl",
                (json.dumps({**first, "replacement_row_hash": "tampered"}, sort_keys=True) + "\n").encode(),
            ),
            "invalid",
        ),
        (
            lambda paths, first, second: (
                "evidence/supersessions.jsonl",
                (json.dumps({**first, "replacement": {**first["replacement"], "row_hash": "other"}}, sort_keys=True) + "\n").encode(),
            ),
            "invalid",
        ),
        (
            lambda paths, first, second: (
                "evidence/effective_rows.jsonl",
                (json.dumps({
                    **json.loads(paths.effective_rows.read_text().splitlines()[0]),
                    "supersession": {
                        **json.loads(paths.effective_rows.read_text().splitlines()[0])["supersession"],
                        "terminal_replacement_row_hash": "other",
                    },
                }, sort_keys=True) + "\n").encode(),
            ),
            "inconsistent",
        ),
    ],
)
def test_supersession_package_semantic_tampering_fails(tmp_path, mutator, message):
    paths, first, second = supersession_fixture(tmp_path, chain_length=1)
    target, content = mutator(paths, first, second)
    if content is None:
        rewrite_verified(paths, remove=target)
    else:
        rewrite_verified(paths, change=(target, content))
    result = campaign.verify_package_details(paths)
    assert not result["valid"]
    assert not result["supersession_references_valid"]
    assert any(message in error for error in result["errors"])


def test_supersession_package_rejects_missing_intermediate_fork_and_cycle(tmp_path):
    paths, first, second = supersession_fixture(tmp_path, chain_length=2)
    rewrite_verified(paths, change=("evidence/supersessions.jsonl", (json.dumps(first, sort_keys=True) + "\n").encode()))
    result = campaign.verify_package_details(paths)
    assert not result["valid"] and not result["supersession_references_valid"]
    assert any("inconsistent" in error for error in result["errors"])

    paths, first, _ = supersession_fixture(tmp_path / "fork", chain_length=1)
    replacement = dict(first["replacement_row"])
    replacement["reason"] = "fork"
    fork = campaign._native_supersession_record(
        paths=paths,
        source_campaign_id=paths.campaign_id,
        source_run_id="primary",
        source_row=json.loads(paths.primary_raw_results.read_text().splitlines()[0]),
        replacement_campaign_id=paths.campaign_id,
        replacement_run_id="fork",
        replacement_row=replacement,
        reason="fork",
        operator="test",
        tool="pytest",
    )
    rewrite_verified(
        paths,
        change=("evidence/supersessions.jsonl", (json.dumps(first, sort_keys=True) + "\n" + json.dumps(fork, sort_keys=True) + "\n").encode()),
    )
    result = campaign.verify_package_details(paths)
    assert not result["valid"] and not result["supersession_references_valid"]

    paths, first, _ = supersession_fixture(tmp_path / "cycle", chain_length=1)
    replacement = first["replacement_row"]
    source = json.loads(paths.primary_raw_results.read_text().splitlines()[0])
    cycle = campaign._native_supersession_record(
        paths=paths,
        source_campaign_id=paths.campaign_id,
        source_run_id="replacement-b",
        source_row=replacement,
        replacement_campaign_id=paths.campaign_id,
        replacement_run_id="primary",
        replacement_row=source,
        reason="cycle",
        operator="test",
        tool="pytest",
    )
    rewrite_verified(
        paths,
        change=("evidence/supersessions.jsonl", (json.dumps(first, sort_keys=True) + "\n" + json.dumps(cycle, sort_keys=True) + "\n").encode()),
    )
    result = campaign.verify_package_details(paths)
    assert not result["valid"] and not result["supersession_references_valid"]


def test_config_managed_package_binds_campaign_config_plan(tmp_path):
    paths, _ = config_fixture(tmp_path)
    result = campaign.verify_package_details(paths)
    assert result["valid"]
    assert result["config_references_valid"]


@pytest.mark.parametrize(
    ("target", "content", "message"),
    [
        ("plan/campaign_config.json", None, "missing campaign config plan"),
        ("plan/campaign_config.json", b"{", "malformed campaign config plan"),
    ],
)
def test_config_managed_package_rejects_missing_or_malformed_config_plan(tmp_path, target, content, message):
    paths, _ = config_fixture(tmp_path)
    if content is None:
        rewrite_verified(paths, remove=target)
    else:
        rewrite_verified(paths, change=(target, content))
    result = campaign.verify_package_details(paths)
    assert not result["valid"] and not result["config_references_valid"]
    assert any(message in error for error in result["errors"])


def test_config_managed_package_rejects_modified_config_signature_and_binding_mismatch(tmp_path):
    paths, record = config_fixture(tmp_path)
    changed = json.loads(json.dumps(record))
    changed["config"]["samples"] = 2
    changed["config_signature"] = campaign.campaign_config_signature(changed["config"])
    rewrite_verified(paths, change=("plan/campaign_config.json", json.dumps(changed, sort_keys=True).encode()))
    result = campaign.verify_package_details(paths)
    assert not result["valid"] and not result["config_references_valid"]
    assert any("binding mismatch" in error for error in result["errors"])

    paths, record = config_fixture(tmp_path / "sig")
    bad = dict(record)
    bad["config_signature"] = "wrong"
    rewrite_verified(paths, change=("plan/campaign_config.json", json.dumps(bad, sort_keys=True).encode()))
    result = campaign.verify_package_details(paths)
    assert not result["valid"] and not result["config_references_valid"]
    assert any("signature mismatch" in error for error in result["errors"])

    paths, _ = config_fixture(tmp_path / "manifest")
    manifest = json.loads(paths.manifest.read_text())
    manifest["notes"]["campaign_config"]["config_signature"] = "wrong"
    rewrite_verified(paths, change=("manifest.json", json.dumps(manifest, sort_keys=True).encode()))
    result = campaign.verify_package_details(paths)
    assert not result["valid"] and not result["config_references_valid"]
    assert any("binding mismatch" in error for error in result["errors"])


def test_legacy_package_does_not_infer_config_management_from_stray_file(tmp_path):
    paths = fixture(tmp_path)
    stray = campaign.campaign_config_plan_record({**campaign.campaign_config_template(), "campaign_id": paths.campaign_id, "models": ["x"]})
    paths.plan_dir.joinpath("campaign_config.json").write_text(json.dumps(stray, sort_keys=True))
    campaign.package_campaign(paths)
    result = campaign.verify_package_details(paths)
    assert result["valid"]
    assert result["config_references_valid"]
