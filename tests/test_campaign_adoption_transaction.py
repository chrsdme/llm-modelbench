import argparse
import hashlib
import json
from pathlib import Path
import pytest
from llm_modelbench import campaign, cli, rankings


def fixture(tmp_path, *, signature="sig-new", score=100, effective_hash="h", judge_conflict=False):
    paths, manifest=campaign.create_campaign("adopt_tx",models=["x"],campaigns_root=tmp_path/"campaigns")
    paths.plan_json.write_text(json.dumps({"task_hashes":{"exact":"h"}})); paths.inventory_json.write_text('[{"name":"x","digest":"d"}]')
    paths.capabilities_json.write_text('{"x":{"supported_families":["text"]}}')
    paths.primary_raw_results.write_text('{"model":"x","task":"exact","score":100}\n'); paths.primary_run_validity.write_text('{"status":"valid"}')
    (paths.primary_dir/"model_identities.json").write_text('{"x":{"digest":"d"}}')
    paths.effective_rows.write_text(json.dumps({"model":"x","task":"exact","task_hash":effective_hash,"terminal_disposition":"scored","result_origin":"primary"})+'\n')
    if judge_conflict:
        (paths.judge_dir/"judge_selection.json").write_text('{"judge":{"digest":"d"},"cohort":[{"digest":"d"}]}')
    ready={"readiness":"ready_for_adoption","blockers":[]}; paths.readiness_json.write_text(json.dumps(ready)); (paths.reports_dir/"readiness.json").write_text(json.dumps(ready)); (paths.reports_dir/"readiness.md").write_text('# Ready\n')
    row={"run_id":"primary","_source_signature":signature,"model":"x","model_digest_resolved":"d","task":"exact","task_hash":"h","score":score,"reason":"new","terminal_disposition":"scored","ranking_scope":"separate","canonical_rankings":False}
    paths.candidate_rankings_dir.joinpath("master_raw.jsonl").write_text(json.dumps(row)+'\n'); paths.candidate_rankings_dir.joinpath("master_summary.json").write_text('[]')
    for state in ("planned","generating","packaged"): manifest=campaign.transition(paths,manifest,state)
    campaign.package_campaign(paths); return paths,row


def canonical(tmp_path, row=None):
    out=tmp_path/"rankings"; out.mkdir()
    if row: (out/"master_raw.jsonl").write_text(json.dumps(row)+'\n')
    else: (out/"master_raw.jsonl").write_text('')
    (out/"marker.txt").write_text('before')
    return out


def digest_tree(root):
    h=hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_file(): h.update(str(p.relative_to(root)).encode()+p.read_bytes())
    return h.hexdigest()


def test_detailed_dry_run_is_immutable(tmp_path):
    paths,row=fixture(tmp_path); out=canonical(tmp_path); before=digest_tree(out); candidate=paths.candidate_rankings_dir.joinpath('master_raw.jsonl').read_bytes()
    preview=campaign.adopt_campaign(paths,rankings_dir=out,dry_run=True)
    assert preview["campaign_id"]=="adopt_tx" and preview["package_verified"] and preview["readiness"]=="ready_for_adoption"
    assert preview["changes"][0]["key"]["incoming_signature"]=="sig-new"
    assert preview["changes"][0]["new"]=={"score":100,"reason":"new","disposition":"scored"}
    assert preview["old_coverage"]==0 and preview["new_coverage"]==1 and preview["canonical_scope_conversion"]
    assert digest_tree(out)==before and paths.candidate_rankings_dir.joinpath('master_raw.jsonl').read_bytes()==candidate
    assert not paths.adoption_record.exists() and campaign.load_manifest(paths).state=="packaged"


def test_changed_signature_replaces_and_converts_scope(tmp_path):
    paths,row=fixture(tmp_path); old={**row,"_source_signature":"sig-old","score":0,"reason":"old","ranking_scope":"canonical","canonical_rankings":True}
    out=canonical(tmp_path,old); preview=campaign.adopt_campaign(paths,rankings_dir=out,dry_run=False)
    adopted=json.loads((out/"master_raw.jsonl").read_text().splitlines()[0])
    assert preview["rows_replaced"]==1 and adopted["score"]==100
    assert adopted["ranking_scope"]=="canonical" and adopted["canonical_rankings"] and adopted["campaign_id"]=="adopt_tx"
    assert campaign.load_manifest(paths).state=="accepted" and paths.adoption_record.exists()
    record=json.loads(paths.adoption_record.read_text())
    assert record["transaction_id"] and record["manifest_digest"] and record["canonical_source_after_digest"]
    assert (out/"master_summary.json").exists() and (out/"master_report_data.json").exists()


def test_same_signature_is_explicit_noop_after_acceptance(tmp_path):
    paths,row=fixture(tmp_path); out=canonical(tmp_path); campaign.adopt_campaign(paths,rankings_dir=out,dry_run=False)
    before=digest_tree(out); preview=campaign.adopt_campaign(paths,rankings_dir=out,dry_run=False)
    assert preview["would_be_noop"] and preview["rows_unchanged"]==1 and digest_tree(out)==before


@pytest.mark.parametrize("failure", ["source","rebuild","artifact","replace","record","transition"])
def test_transaction_failures_restore_canonical_and_campaign(tmp_path,monkeypatch,failure):
    paths,_=fixture(tmp_path); out=canonical(tmp_path); before=digest_tree(out)
    if failure=="source":
        original=campaign._atomic_write_text
        monkeypatch.setattr(campaign,"_atomic_write_text",lambda p,t: (_ for _ in ()).throw(OSError("source")) if p.name=="master_raw.jsonl" else original(p,t))
    elif failure=="rebuild": monkeypatch.setattr(rankings,"write_rankings",lambda *a,**k: (_ for _ in ()).throw(RuntimeError("rebuild")))
    elif failure=="artifact": monkeypatch.setattr(rankings,"write_rankings",lambda *a,**k: {})
    elif failure=="replace":
        original=campaign.os.replace
        def replace(src,dst):
            if Path(dst)==out and Path(src).name.startswith('.campaign-adopt-'): raise OSError("replace")
            return original(src,dst)
        monkeypatch.setattr(campaign.os,"replace",replace)
    elif failure=="record":
        original=campaign._atomic_write_text
        monkeypatch.setattr(campaign,"_atomic_write_text",lambda p,t: (_ for _ in ()).throw(OSError("record")) if p==paths.adoption_record else original(p,t))
    else:
        original=campaign.transition
        monkeypatch.setattr(campaign,"transition",lambda p,m,t: (_ for _ in ()).throw(OSError("transition")) if t=="accepted" else original(p,m,t))
    with pytest.raises((RuntimeError,OSError,campaign.CampaignError)): campaign.adopt_campaign(paths,rankings_dir=out,dry_run=False)
    assert digest_tree(out)==before and campaign.load_manifest(paths).state=="packaged" and not paths.adoption_record.exists()


def test_validation_refusals(tmp_path):
    paths,_=fixture(tmp_path); out=canonical(tmp_path)
    data=json.loads(paths.readiness_json.read_text()); data["readiness"]="not_ready_manual_items"; paths.readiness_json.write_text(json.dumps(data))
    with pytest.raises(campaign.CampaignError,match="not ready"): campaign.adopt_campaign(paths,rankings_dir=out)
    paths,_=fixture(tmp_path/"hash",effective_hash="wrong"); out=canonical(tmp_path/"hash")
    with pytest.raises(campaign.CampaignError,match="task hash mismatch"): campaign.adopt_campaign(paths,rankings_dir=out)
    paths,_=fixture(tmp_path/"judge",judge_conflict=True); out=canonical(tmp_path/"judge")
    with pytest.raises(campaign.CampaignError,match="judge digest"): campaign.adopt_campaign(paths,rankings_dir=out)


def test_cli_requires_typed_confirmation_and_has_no_yes_bypass(tmp_path,monkeypatch):
    with pytest.raises(SystemExit): cli.build_parser().parse_args(["rankings","--yes"])
    paths,_=fixture(tmp_path); out=canonical(tmp_path)
    class TTY:
        def isatty(self): return True
    monkeypatch.setattr(campaign,"resolve_paths",lambda *_a,**_k: paths); monkeypatch.setattr(cli.sys,"stdin",TTY()); monkeypatch.setattr("builtins.input",lambda _p:"WRONG")
    args=argparse.Namespace(adopt_campaign="adopt_tx",out=str(out),dry_run=False)
    with pytest.raises(SystemExit,match="cancelled"): cli.cmd_rankings(args,object())
    assert campaign.load_manifest(paths).state=="packaged"


def test_cli_exact_typed_confirmation_succeeds(tmp_path,monkeypatch,capsys):
    paths,_=fixture(tmp_path); out=canonical(tmp_path)
    class TTY:
        def isatty(self): return True
    monkeypatch.setattr(campaign,"resolve_paths",lambda *_a,**_k: paths); monkeypatch.setattr(cli.sys,"stdin",TTY())
    monkeypatch.setattr("builtins.input",lambda _p:"ADOPT adopt_tx")
    cli.cmd_rankings(argparse.Namespace(adopt_campaign="adopt_tx",out=str(out),dry_run=False),object())
    assert campaign.load_manifest(paths).state=="accepted"
