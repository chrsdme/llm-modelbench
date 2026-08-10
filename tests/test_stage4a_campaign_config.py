import json

import pytest

from llm_modelbench import campaign, cli


def test_stage4a_campaign_init_template_round_trips(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "campaign.json"
    cli.main(["campaign", "init", str(target)])

    data = json.loads(target.read_text(encoding="utf-8"))
    validated = campaign.validate_campaign_config(data)
    assert validated["schema_version"] == campaign.CAMPAIGN_CONFIG_SCHEMA_VERSION
    assert validated["campaign_id"] == "my_campaign"
    assert validated["models"] == ["model:tag"]
    assert validated["stop_before_adoption"] is True
    assert "telemetry_policy" not in data
    assert "recovery_policy" not in data
    assert "kv_fallback_policy" not in data


def test_stage4a_campaign_init_refuses_to_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "campaign.json"
    target.write_text("existing\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="already exists"):
        cli.main(["campaign", "init", str(target)])
    assert target.read_text(encoding="utf-8") == "existing\n"


def test_stage4a_valid_minimal_and_full_configs():
    minimal = campaign.campaign_config_template()
    validated = campaign.validate_campaign_config(minimal)
    assert validated["level"] == "full"
    full = dict(minimal)
    full["campaign_id"] = "full_config"
    full["level"] = "short"
    full["samples"] = 3
    full["runtime_policy"] = {"auto": False}
    full["context_needle_policy"] = {"needle_max_ctx": 66560}
    full["executable_scorer_policy"] = {"allow_host_code_execution": True}
    assert campaign.validate_campaign_config(full)["runtime_policy"]["auto"] is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda data: data.update({"unknown": True}), "unknown_campaign_config_key"),
        (lambda data: data.update({"schema_version": 999}), "unsupported_campaign_config_schema"),
        (lambda data: data.update({"id": "other"}), "campaign_config_id_conflict"),
        (lambda data: data.update({"campaign_id": "../bad"}), "invalid campaign id"),
        (lambda data: data.update({"models": "m"}), "campaign_config_models_invalid"),
        (lambda data: data.update({"models": [""]}), "campaign_config_models_invalid"),
        (lambda data: data.update({"level": "deep"}), "campaign_config_level_invalid"),
        (lambda data: data.update({"samples": 0}), "campaign_config_samples_invalid"),
        (lambda data: data.update({"runtime_policy": {"auto": "yes"}}), "runtime_policy_auto_invalid"),
        (lambda data: data.update({"context_needle_policy": {"needle_max_ctx": -1}}), "context_needle_policy_invalid"),
        (lambda data: data.update({"judge_policy": {"enabled": "yes"}}), "judge_policy_enabled_invalid"),
        (lambda data: data.update({"judge_policy": {"enabled": True}}), "judge_policy_enabled_requires_stage4b"),
        (lambda data: data.update({"stop_before_adoption": False}), "campaign_config_must_stop_before_adoption"),
        (
            lambda data: data.update({"executable_scorer_policy": {"allow_host_code_execution": "yes"}}),
            "executable_scorer_policy_invalid",
        ),
        (lambda data: data.update({"telemetry_policy": {"enabled": True}}), "unknown_campaign_config_key"),
    ],
)
def test_stage4a_campaign_config_rejects_malformed_values(mutator, message):
    data = campaign.campaign_config_template()
    mutator(data)
    with pytest.raises(campaign.CampaignError, match=message):
        campaign.validate_campaign_config(data)


def test_stage4a_execute_rejects_invalid_config_before_running(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = campaign.campaign_config_template()
    config["telemetry_policy"] = {"enabled": True}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(SystemExit, match="unknown_campaign_config_key"):
        cli.main(["campaign", "execute", "--config", str(path), "--mock"])
    assert not (tmp_path / "campaigns").exists()
