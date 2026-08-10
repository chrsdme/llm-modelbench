import json
from argparse import Namespace

import pytest

from llm_modelbench import campaign, cli


def _write_config(tmp_path, config):
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return path


def _args(path):
    return Namespace(campaign_cmd="execute", campaign_config=str(path), mock=True)


def test_stage4b_execute_config_records_plan_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = campaign.campaign_config_template()
    config["campaign_id"] = "stage4b"
    config["models"] = ["m"]
    config_path = _write_config(tmp_path, config)
    calls = []

    def fake_main(invocation):
        calls.append(invocation)
        paths, manifest = campaign.create_campaign("stage4b", models=["m"], campaigns_root=tmp_path / "campaigns")
        paths.primary_raw_results.write_text(
            json.dumps({"model": "m", "task": "needle", "score": 100, "reason": "ok"}) + "\n",
            encoding="utf-8",
        )
        manifest = campaign.transition(paths, manifest, "planned")
        manifest = campaign.transition(paths, manifest, "generating")
        campaign.transition(paths, manifest, "packaged")

    monkeypatch.setattr(cli, "main", fake_main)
    cli.cmd_campaign(_args(config_path), object())
    paths = campaign.resolve_paths("stage4b", campaigns_root=tmp_path / "campaigns")
    primary_before = paths.primary_raw_results.read_text(encoding="utf-8")
    config_record = json.loads((paths.plan_dir / "campaign_config.json").read_text(encoding="utf-8"))
    assert config_record["config_signature"] == campaign.campaign_config_signature(config)

    cli.cmd_campaign(_args(config_path), object())
    assert len(calls) == 1
    assert paths.primary_raw_results.read_text(encoding="utf-8") == primary_before


def test_stage4b_execute_rejects_changed_config_for_existing_campaign(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = campaign.campaign_config_template()
    config["campaign_id"] = "stage4b_changed"
    paths, manifest = campaign.create_campaign("stage4b_changed", models=["m"], campaigns_root=tmp_path / "campaigns")
    manifest = campaign.transition(paths, manifest, "planned")
    manifest = campaign.transition(paths, manifest, "generating")
    campaign.transition(paths, manifest, "packaged")
    paths.plan_dir.joinpath("campaign_config.json").write_text(
        json.dumps(campaign.campaign_config_plan_record(config), sort_keys=True),
        encoding="utf-8",
    )
    changed = dict(config)
    changed["samples"] = 2
    config_path = _write_config(tmp_path, changed)

    with pytest.raises(SystemExit, match="different config"):
        cli.cmd_campaign(_args(config_path), object())


def test_stage4b_execute_interrupted_state_requires_explicit_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = campaign.campaign_config_template()
    config["campaign_id"] = "stage4b_interrupted"
    paths, manifest = campaign.create_campaign("stage4b_interrupted", models=["m"], campaigns_root=tmp_path / "campaigns")
    manifest = campaign.transition(paths, manifest, "planned")
    manifest = campaign.transition(paths, manifest, "generating")
    campaign.transition(paths, manifest, "interrupted")
    paths.plan_dir.joinpath("campaign_config.json").write_text(
        json.dumps(campaign.campaign_config_plan_record(config), sort_keys=True),
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, config)

    with pytest.raises(SystemExit, match="campaign resume stage4b_interrupted"):
        cli.cmd_campaign(_args(config_path), object())


def test_stage4b_execute_rejects_existing_campaign_without_config_plan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = campaign.campaign_config_template()
    config["campaign_id"] = "stage4b_legacy"
    campaign.create_campaign("stage4b_legacy", models=["m"], campaigns_root=tmp_path / "campaigns")
    config_path = _write_config(tmp_path, config)

    with pytest.raises(SystemExit, match="lacks immutable config plan"):
        cli.cmd_campaign(_args(config_path), object())
