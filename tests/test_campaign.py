"""RC19.1 campaign workspace foundation tests."""

import json
from pathlib import Path

import pytest

from llm_modelbench import campaign


def test_create_campaign_builds_full_directory_tree(tmp_path):
    campaigns_root = tmp_path / "campaigns"
    paths, manifest = campaign.create_campaign(
        "rc19_demo",
        models=["model-a:latest", "model-b:latest"],
        level="full",
        version="1.0.0rc19.post1",
        campaigns_root=campaigns_root,
    )
    assert paths.root == campaigns_root / "rc19_demo"
    assert paths.root.exists()
    assert paths.primary_dumps_dir.exists()
    assert paths.recovery_children_dir.exists()
    assert paths.candidate_rankings_dir.exists()
    assert paths.model_cards_dir.exists()
    assert paths.packages_dir.exists()
    assert not hasattr(paths, "state")
    assert manifest.state == "created"
    assert manifest.resume_state is None
    assert manifest.models == ["model-a:latest", "model-b:latest"]


def test_create_campaign_refuses_to_reuse_a_nonempty_existing_id(tmp_path):
    campaigns_root = tmp_path / "campaigns"
    campaign.create_campaign(
        "dup",
        models=["x"],
        campaigns_root=campaigns_root,
    )
    with pytest.raises(campaign.CampaignError, match="already exists"):
        campaign.create_campaign(
            "dup",
            models=["y"],
            campaigns_root=campaigns_root,
        )


@pytest.mark.parametrize(
    "bad_id",
    [
        "../../etc/passwd",
        "../escape",
        "foo/bar",
        "foo bar",
        "",
        "foo;rm -rf /",
        "a/../b",
    ],
)
def test_validate_campaign_id_rejects_unsafe_names(bad_id):
    with pytest.raises(campaign.CampaignError):
        campaign.validate_campaign_id(bad_id)


@pytest.mark.parametrize(
    "good_id",
    ["rc19_demo", "campaign-1", "a.b.c", "RC19_test_001"],
)
def test_validate_campaign_id_accepts_safe_names(good_id):
    assert campaign.validate_campaign_id(good_id) == good_id


def test_every_resolved_path_stays_inside_the_campaign_root(tmp_path):
    campaigns_root = tmp_path / "campaigns"
    paths = campaign.resolve_paths(
        "leak_check",
        campaigns_root=campaigns_root,
    )
    root = paths.root.resolve()
    all_paths = [
        paths.manifest,
        paths.plan_json,
        paths.inventory_json,
        paths.capabilities_json,
        paths.primary_raw_results,
        paths.primary_run_validity,
        paths.primary_dumps_dir,
        paths.recovery_plan,
        paths.recovery_result,
        paths.recovery_attempts,
        paths.recovery_children_dir,
        paths.judge_results,
        paths.judge_summary,
        paths.candidate_rankings_dir,
        paths.model_cards_dir,
        paths.campaign_log,
        paths.packages_dir,
    ]
    for path in all_paths:
        assert path.resolve().is_relative_to(root), (
            f"root leakage detected: {path}"
        )


def test_state_machine_allows_the_documented_happy_path(tmp_path):
    paths, manifest = campaign.create_campaign(
        "happy",
        models=["x"],
        campaigns_root=tmp_path / "campaigns",
    )
    for target in (
        "planned",
        "generating",
        "recovering",
        "judging",
        "packaged",
        "accepted",
    ):
        manifest = campaign.transition(paths, manifest, target)
    assert manifest.state == "accepted"
    assert manifest.resume_state is None
    assert [entry["state"] for entry in manifest.state_history] == [
        "created",
        "planned",
        "generating",
        "recovering",
        "judging",
        "packaged",
        "accepted",
    ]


def test_state_machine_allows_skipping_optional_recovery_and_judging(tmp_path):
    paths, manifest = campaign.create_campaign(
        "skip_optional",
        models=["x"],
        campaigns_root=tmp_path / "campaigns",
    )
    manifest = campaign.transition(paths, manifest, "planned")
    manifest = campaign.transition(paths, manifest, "generating")
    manifest = campaign.transition(paths, manifest, "packaged")
    assert manifest.state == "packaged"


def test_state_machine_refuses_illegal_transitions(tmp_path):
    paths, manifest = campaign.create_campaign(
        "illegal",
        models=["x"],
        campaigns_root=tmp_path / "campaigns",
    )
    with pytest.raises(
        campaign.CampaignError,
        match="illegal campaign state transition",
    ):
        campaign.transition(paths, manifest, "packaged")


def test_publication_terminal_states_never_transition_anywhere(tmp_path):
    paths, manifest = campaign.create_campaign(
        "terminal",
        models=["x"],
        campaigns_root=tmp_path / "campaigns",
    )
    for target in ("planned", "generating", "packaged", "accepted"):
        manifest = campaign.transition(paths, manifest, target)
    assert campaign.is_terminal(manifest.state)
    for target in campaign.STATES:
        if target == manifest.state:
            continue
        assert not campaign.is_valid_transition(manifest.state, target)


@pytest.mark.parametrize(
    "phase",
    ["generating", "recovering", "judging"],
)
def test_interrupted_campaign_resumes_only_its_recorded_phase(
    tmp_path,
    phase,
):
    paths, manifest = campaign.create_campaign(
        f"resume_{phase}",
        models=["x"],
        campaigns_root=tmp_path / "campaigns",
    )
    manifest = campaign.transition(paths, manifest, "planned")
    manifest = campaign.transition(paths, manifest, "generating")
    if phase == "recovering":
        manifest = campaign.transition(paths, manifest, "recovering")
    elif phase == "judging":
        manifest = campaign.transition(paths, manifest, "judging")

    manifest = campaign.transition(paths, manifest, "interrupted")
    assert manifest.state == "interrupted"
    assert manifest.resume_state == phase
    assert manifest.state_history[-1]["resume_state"] == phase

    manifest = campaign.transition(paths, manifest, phase)
    assert manifest.state == phase
    assert manifest.resume_state is None


def test_interrupted_campaign_refuses_wrong_resume_phase(tmp_path):
    paths, manifest = campaign.create_campaign(
        "wrong_resume",
        models=["x"],
        campaigns_root=tmp_path / "campaigns",
    )
    manifest = campaign.transition(paths, manifest, "planned")
    manifest = campaign.transition(paths, manifest, "generating")
    manifest = campaign.transition(paths, manifest, "interrupted")

    with pytest.raises(
        campaign.CampaignError,
        match="must resume 'generating'",
    ):
        campaign.transition(paths, manifest, "judging")


def test_interrupted_campaign_may_be_marked_failed(tmp_path):
    paths, manifest = campaign.create_campaign(
        "interrupt_fail",
        models=["x"],
        campaigns_root=tmp_path / "campaigns",
    )
    manifest = campaign.transition(paths, manifest, "planned")
    manifest = campaign.transition(paths, manifest, "generating")
    manifest = campaign.transition(paths, manifest, "interrupted")
    manifest = campaign.transition(paths, manifest, "failed")
    assert manifest.state == "failed"
    assert manifest.resume_state is None


def test_failed_campaign_cannot_be_accepted(tmp_path):
    paths, manifest = campaign.create_campaign(
        "failed_not_accepted",
        models=["x"],
        campaigns_root=tmp_path / "campaigns",
    )
    manifest = campaign.transition(paths, manifest, "failed")
    assert not campaign.is_terminal(manifest.state)
    with pytest.raises(
        campaign.CampaignError,
        match="illegal campaign state transition",
    ):
        campaign.transition(paths, manifest, "accepted")


def test_failed_campaign_can_be_rejected(tmp_path):
    paths, manifest = campaign.create_campaign(
        "failed_rejected",
        models=["x"],
        campaigns_root=tmp_path / "campaigns",
    )
    manifest = campaign.transition(paths, manifest, "failed")
    manifest = campaign.transition(paths, manifest, "rejected")
    assert manifest.state == "rejected"
    assert campaign.is_terminal(manifest.state)


def test_manifest_persists_and_reloads_correctly(tmp_path):
    paths, manifest = campaign.create_campaign(
        "persist",
        models=["a", "b"],
        level="full",
        version="1.0.0rc19.post1",
        campaigns_root=tmp_path / "campaigns",
    )
    campaign.transition(paths, manifest, "planned")
    reloaded = campaign.load_manifest(paths)
    assert reloaded.state == "planned"
    assert reloaded.resume_state is None
    assert reloaded.models == ["a", "b"]
    assert reloaded.version == "1.0.0rc19.post1"
    assert len(reloaded.state_history) == 2


def test_load_manifest_raises_clearly_for_a_nonexistent_campaign(tmp_path):
    paths = campaign.resolve_paths(
        "never_created",
        campaigns_root=tmp_path / "campaigns",
    )
    with pytest.raises(campaign.CampaignError, match="no manifest"):
        campaign.load_manifest(paths)


def test_load_manifest_rejects_interrupted_state_without_resume_state(
    tmp_path,
):
    paths, _ = campaign.create_campaign(
        "bad_interrupted",
        models=["x"],
        campaigns_root=tmp_path / "campaigns",
    )
    data = json.loads(paths.manifest.read_text())
    data["state"] = "interrupted"
    data["resume_state"] = None
    paths.manifest.write_text(json.dumps(data))

    with pytest.raises(
        campaign.CampaignError,
        match="must record a valid resume_state",
    ):
        campaign.load_manifest(paths)


def test_atomic_write_leaves_no_temp_file_after_success(tmp_path):
    paths, _ = campaign.create_campaign(
        "atomic",
        models=["x"],
        campaigns_root=tmp_path / "campaigns",
    )
    original = json.loads(paths.manifest.read_text())
    assert original["campaign_id"] == "atomic"
    assert list(paths.root.glob("**/.manifest.json.*.tmp")) == []


def test_atomic_write_failure_preserves_existing_manifest(
    tmp_path,
    monkeypatch,
):
    paths, manifest = campaign.create_campaign(
        "atomic_failure",
        models=["x"],
        campaigns_root=tmp_path / "campaigns",
    )
    original_text = paths.manifest.read_text()

    original_replace = Path.replace

    def fail_manifest_replace(self, target):
        if self.name.startswith(".manifest.json.") and self.suffix == ".tmp":
            raise OSError("injected replace failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_manifest_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        campaign.transition(paths, manifest, "planned")

    assert paths.manifest.read_text() == original_text
    assert manifest.state == "created"
    assert list(paths.root.glob("**/.manifest.json.*.tmp")) == []


def test_is_legacy_run_dir_detects_pre_campaign_runs(tmp_path):
    old_run = tmp_path / "runs" / "some_old_run"
    old_run.mkdir(parents=True)
    (old_run / "raw_results.jsonl").write_text("{}\n")
    assert campaign.is_legacy_run_dir(old_run) is True

    new_style = tmp_path / "runs" / "not_legacy"
    new_style.mkdir(parents=True)
    (new_style / "raw_results.jsonl").write_text("{}\n")
    (new_style / "manifest.json").write_text("{}\n")
    assert campaign.is_legacy_run_dir(new_style) is False
