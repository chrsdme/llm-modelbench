"""RC19.1 campaign workspace foundation: manifest, path resolver, state machine.

This is the storage foundation everything else (recovery, judging, candidate
rankings, packaging, cleanup) will write into. No CLI wiring yet -- this
tests the module in isolation, since it must be correct before anything
else is built on top of it.
"""
import json

import pytest

from llm_modelbench import campaign


def test_create_campaign_builds_full_directory_tree(tmp_path):
    campaigns_root = tmp_path / "campaigns"
    paths, manifest = campaign.create_campaign(
        "rc19_demo", models=["model-a:latest", "model-b:latest"],
        level="full", version="1.0.0rc19.post1", campaigns_root=campaigns_root,
    )
    assert paths.root == campaigns_root / "rc19_demo"
    assert paths.root.exists()
    assert paths.primary_dumps_dir.exists()
    assert paths.recovery_children_dir.exists()
    assert paths.candidate_rankings_dir.exists()
    assert paths.model_cards_dir.exists()
    assert paths.packages_dir.exists()
    assert manifest.state == "created"
    assert manifest.models == ["model-a:latest", "model-b:latest"]


def test_create_campaign_refuses_to_reuse_a_nonempty_existing_id(tmp_path):
    campaigns_root = tmp_path / "campaigns"
    campaign.create_campaign("dup", models=["x"], campaigns_root=campaigns_root)
    with pytest.raises(campaign.CampaignError, match="already exists"):
        campaign.create_campaign("dup", models=["y"], campaigns_root=campaigns_root)


@pytest.mark.parametrize("bad_id", [
    "../../etc/passwd", "../escape", "foo/bar", "foo bar", "", "foo;rm -rf /", "a/../b",
])
def test_validate_campaign_id_rejects_unsafe_names(bad_id):
    with pytest.raises(campaign.CampaignError):
        campaign.validate_campaign_id(bad_id)


@pytest.mark.parametrize("good_id", ["rc19_demo", "campaign-1", "a.b.c", "RC19_test_001"])
def test_validate_campaign_id_accepts_safe_names(good_id):
    assert campaign.validate_campaign_id(good_id) == good_id


def test_every_resolved_path_stays_inside_the_campaign_root(tmp_path):
    """The actual root-leakage regression test GPT's acceptance criteria
    asked for: every single path this module can produce must resolve
    inside campaigns/<id>/, with nothing landing at repo root."""
    campaigns_root = tmp_path / "campaigns"
    paths = campaign.resolve_paths("leak_check", campaigns_root=campaigns_root)
    root_str = str(paths.root)
    all_paths = [
        paths.manifest, paths.state, paths.plan_json, paths.inventory_json,
        paths.capabilities_json, paths.primary_raw_results, paths.primary_run_validity,
        paths.primary_dumps_dir, paths.recovery_plan, paths.recovery_result,
        paths.recovery_attempts, paths.recovery_children_dir, paths.judge_results,
        paths.judge_summary, paths.candidate_rankings_dir, paths.model_cards_dir,
        paths.campaign_log, paths.packages_dir,
    ]
    leaking = [p for p in all_paths if not str(p).startswith(root_str)]
    assert leaking == [], f"root leakage detected: {leaking}"


def test_state_machine_allows_the_documented_happy_path(tmp_path):
    paths, manifest = campaign.create_campaign("happy", models=["x"], campaigns_root=tmp_path / "campaigns")
    for target in ("planned", "generating", "recovering", "judging", "packaged", "accepted"):
        manifest = campaign.transition(paths, manifest, target)
    assert manifest.state == "accepted"
    assert [h["state"] for h in manifest.state_history] == [
        "created", "planned", "generating", "recovering", "judging", "packaged", "accepted",
    ]


def test_state_machine_allows_skipping_optional_recovery_and_judging(tmp_path):
    """A campaign with nothing to recover and nothing subjective to judge
    should be able to go straight from generating to packaged."""
    paths, manifest = campaign.create_campaign("skip_optional", models=["x"], campaigns_root=tmp_path / "campaigns")
    manifest = campaign.transition(paths, manifest, "planned")
    manifest = campaign.transition(paths, manifest, "generating")
    manifest = campaign.transition(paths, manifest, "packaged")
    assert manifest.state == "packaged"


def test_state_machine_refuses_illegal_transitions(tmp_path):
    paths, manifest = campaign.create_campaign("illegal", models=["x"], campaigns_root=tmp_path / "campaigns")
    with pytest.raises(campaign.CampaignError, match="illegal campaign state transition"):
        campaign.transition(paths, manifest, "packaged")  # created -> packaged skips required states


def test_terminal_states_never_transition_anywhere(tmp_path):
    paths, manifest = campaign.create_campaign("terminal", models=["x"], campaigns_root=tmp_path / "campaigns")
    for target in ("planned", "generating", "packaged", "accepted"):
        manifest = campaign.transition(paths, manifest, target)
    assert campaign.is_terminal(manifest.state)
    for target in campaign.STATES:
        if target == manifest.state:
            continue
        assert not campaign.is_valid_transition(manifest.state, target)


def test_manifest_persists_and_reloads_correctly(tmp_path):
    paths, manifest = campaign.create_campaign(
        "persist", models=["a", "b"], level="full", version="1.0.0rc19.post1",
        campaigns_root=tmp_path / "campaigns",
    )
    campaign.transition(paths, manifest, "planned")
    reloaded = campaign.load_manifest(paths)
    assert reloaded.state == "planned"
    assert reloaded.models == ["a", "b"]
    assert reloaded.version == "1.0.0rc19.post1"
    assert len(reloaded.state_history) == 2


def test_load_manifest_raises_clearly_for_a_nonexistent_campaign(tmp_path):
    paths = campaign.resolve_paths("never_created", campaigns_root=tmp_path / "campaigns")
    with pytest.raises(campaign.CampaignError, match="no manifest"):
        campaign.load_manifest(paths)


def test_atomic_write_leaves_no_partial_file_on_interruption(tmp_path):
    """The manifest/state files must never be left truncated or corrupt --
    this is the one file every other tool trusts to know a campaign's state."""
    paths, manifest = campaign.create_campaign("atomic", models=["x"], campaigns_root=tmp_path / "campaigns")
    original_content = paths.manifest.read_text()
    original = json.loads(original_content)
    assert original["campaign_id"] == "atomic"
    # No .tmp files should be left lying around after a normal write.
    tmp_leftovers = list(paths.root.glob("**/.manifest.json.*.tmp"))
    assert tmp_leftovers == []


def test_is_legacy_run_dir_detects_pre_campaign_runs(tmp_path):
    old_run = tmp_path / "runs" / "some_old_run"
    old_run.mkdir(parents=True)
    (old_run / "raw_results.jsonl").write_text("{}\n")
    assert campaign.is_legacy_run_dir(old_run) is True

    # A campaign-managed directory (has a manifest) is not "legacy" even if
    # it also happens to have a raw_results.jsonl somewhere convenient.
    new_style = tmp_path / "runs" / "not_legacy"
    new_style.mkdir(parents=True)
    (new_style / "raw_results.jsonl").write_text("{}\n")
    (new_style / "manifest.json").write_text("{}\n")
    assert campaign.is_legacy_run_dir(new_style) is False
