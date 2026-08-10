import json
from pathlib import Path

import pytest

from llm_modelbench import campaign, cli


def _row(name, *, score=None, ctx=65536):
    row = {
        "model": "m",
        "model_digest_resolved": "digest-m",
        "task": "needle",
        "task_hash": "needle-h",
        "reason": name,
        "context": ctx,
        "needle_max_ctx": ctx,
    }
    if score is None:
        row["error_kind"] = "harness_error"
    else:
        row["score"] = score
    return row


def _paths(tmp_path):
    paths, _ = campaign.create_campaign("stage3b", models=["m"], campaigns_root=tmp_path / "campaigns")
    return paths


def _replacement_campaign(tmp_path, row, *, campaign_id="replacement", run_id="synthetic-catchup"):
    paths, _ = campaign.create_campaign(campaign_id, models=["m"], campaigns_root=tmp_path / "campaigns")
    if run_id == "primary":
        destination = paths.primary_raw_results
    else:
        destination = paths.evidence_dir / run_id / "raw_results.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_stage3b_cli_dry_run_validates_without_appending(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    paths = _paths(tmp_path)
    source = _row("historical 65536 harness ceiling")
    replacement = _row("corrected 66560 needle ceiling", score=100, ctx=66560)
    paths.primary_raw_results.write_text(json.dumps(source, sort_keys=True) + "\n", encoding="utf-8")
    replacement_path = tmp_path / "replacement.json"
    _write_json(replacement_path, replacement)

    cli.main([
        "campaign", "supersede",
        "--campaign-id", paths.campaign_id,
        "--source-row-hash", campaign._primary_row_hash(source),
        "--replacement-run-id", "synthetic-catchup-66560",
        "--replacement-row", str(replacement_path),
        "--replacement-row-hash", campaign._primary_row_hash(replacement),
        "--reason", "synthetic 65536 -> 66560 needle correction",
        "--operator", "test",
        "--dry-run",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["would_record"]["replacement_row_hash"] == campaign._primary_row_hash(replacement)
    assert not paths.supersessions.exists()


def test_stage3b_cli_rejects_source_row_not_in_immutable_primary_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = _paths(tmp_path)
    source = _row("real source")
    replacement = _row("replacement", score=100, ctx=66560)
    paths.primary_raw_results.write_text(json.dumps(source, sort_keys=True) + "\n", encoding="utf-8")
    arbitrary = tmp_path / "arbitrary-source.json"
    replacement_path = tmp_path / "replacement.json"
    _write_json(arbitrary, _row("arbitrary mutable source"))
    _write_json(replacement_path, replacement)

    with pytest.raises(SystemExit, match="source_row_hash_not_found"):
        cli.main([
            "campaign", "supersede",
            "--campaign-id", paths.campaign_id,
            "--source-row", str(arbitrary),
            "--replacement-run-id", "synthetic-catchup",
            "--replacement-row", str(replacement_path),
            "--reason", "must anchor source",
        ])


def test_stage3b_effective_ledger_resolves_synthetic_needle_correction_transitively(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = _paths(tmp_path)
    source = _row("historical 65536 harness ceiling")
    replacement_b = _row("first corrected row", score=0, ctx=66560)
    replacement_c = _row("final corrected 66560 row", score=100, ctx=66560)
    paths.primary_raw_results.write_text(json.dumps(source, sort_keys=True) + "\n", encoding="utf-8")
    primary_before = paths.primary_raw_results.read_bytes()
    replacement_paths = _replacement_campaign(
        tmp_path,
        replacement_b,
        campaign_id="replacement-b",
        run_id="synthetic-catchup-b",
    )
    replacement_before = (replacement_paths.evidence_dir / "synthetic-catchup-b" / "raw_results.jsonl").read_bytes()
    replacement_path = tmp_path / "replacement-b.json"
    _write_json(replacement_path, replacement_b)

    cli.main([
        "campaign", "supersede",
        "--campaign-id", paths.campaign_id,
        "--source-row-hash", campaign._primary_row_hash(source),
        "--replacement-campaign-id", replacement_paths.campaign_id,
        "--replacement-run-id", "synthetic-catchup-b",
        "--replacement-row", str(replacement_path),
        "--replacement-row-hash", campaign._primary_row_hash(replacement_b),
        "--reason", "synthetic 65536 -> 66560 needle correction",
        "--operator", "test",
    ])
    campaign.record_supersession(
        paths,
        source_campaign_id=replacement_paths.campaign_id,
        source_run_id="synthetic-catchup-b",
        source_row=replacement_b,
        replacement_campaign_id=paths.campaign_id,
        replacement_run_id="synthetic-catchup-c",
        replacement_row=replacement_c,
        reason="synthetic catch-up final correction",
        operator="test",
        tool="pytest",
    )

    summary = campaign.write_readiness(paths, [source])
    assert summary["readiness"] == "ready_for_adoption"
    assert paths.primary_raw_results.read_bytes() == primary_before
    assert (replacement_paths.evidence_dir / "synthetic-catchup-b" / "raw_results.jsonl").read_bytes() == replacement_before
    effective = json.loads(paths.effective_rows.read_text(encoding="utf-8").splitlines()[0])
    assert effective["result_origin"] == "superseded"
    assert effective["effective_score"] == 100
    assert effective["effective_reason"] == "final corrected 66560 row"
    assert effective["terminal_disposition"] == "scored"
    assert effective["supersession"]["terminal_replacement_row_hash"] == campaign._primary_row_hash(replacement_c)
    assert effective["supersession"]["chain_length"] == 2
    assert [edge["replacement_run_id"] for edge in effective["supersession"]["chain"]] == [
        "synthetic-catchup-b",
        "synthetic-catchup-c",
    ]
    assert effective["provenance"]["supersession_chain_length"] == 2


def test_stage3b_cli_rejects_replacement_hash_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = _paths(tmp_path)
    source = _row("source")
    replacement = _row("replacement", score=100, ctx=66560)
    paths.primary_raw_results.write_text(json.dumps(source, sort_keys=True) + "\n", encoding="utf-8")
    replacement_path = tmp_path / "replacement.json"
    _write_json(replacement_path, replacement)

    with pytest.raises(SystemExit, match="replacement_row_hash_mismatch"):
        cli.main([
            "campaign", "supersede",
            "--campaign-id", paths.campaign_id,
            "--source-row-hash", campaign._primary_row_hash(source),
            "--replacement-run-id", "synthetic-catchup",
            "--replacement-row", str(replacement_path),
            "--replacement-row-hash", "not-the-row-hash",
            "--reason", "reject mismatch",
            "--dry-run",
        ])


def test_stage3b_apply_requires_stored_replacement_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = _paths(tmp_path)
    source = _row("source")
    replacement = _row("replacement", score=100, ctx=66560)
    paths.primary_raw_results.write_text(json.dumps(source, sort_keys=True) + "\n", encoding="utf-8")
    source_before = paths.primary_raw_results.read_bytes()
    replacement_path = tmp_path / "replacement.json"
    _write_json(replacement_path, replacement)

    with pytest.raises(SystemExit, match="replacement_campaign_not_found"):
        cli.main([
            "campaign", "supersede",
            "--campaign-id", paths.campaign_id,
            "--source-row-hash", campaign._primary_row_hash(source),
            "--replacement-campaign-id", "absent-replacement",
            "--replacement-run-id", "synthetic-catchup",
            "--replacement-row", str(replacement_path),
            "--replacement-row-hash", campaign._primary_row_hash(replacement),
            "--reason", "must anchor replacement",
        ])

    assert not paths.supersessions.exists()
    assert paths.primary_raw_results.read_bytes() == source_before


def test_stage3b_apply_rejects_missing_replacement_evidence_location(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = _paths(tmp_path)
    replacement_paths, _ = campaign.create_campaign("replacement-missing-run", models=["m"], campaigns_root=tmp_path / "campaigns")
    source = _row("source")
    replacement = _row("replacement", score=100, ctx=66560)
    paths.primary_raw_results.write_text(json.dumps(source, sort_keys=True) + "\n", encoding="utf-8")
    replacement_path = tmp_path / "replacement.json"
    _write_json(replacement_path, replacement)

    with pytest.raises(SystemExit, match="replacement_evidence_location_not_found"):
        cli.main([
            "campaign", "supersede",
            "--campaign-id", paths.campaign_id,
            "--source-row-hash", campaign._primary_row_hash(source),
            "--replacement-campaign-id", replacement_paths.campaign_id,
            "--replacement-run-id", "synthetic-catchup",
            "--replacement-row", str(replacement_path),
            "--replacement-row-hash", campaign._primary_row_hash(replacement),
            "--reason", "missing evidence location",
        ])
    assert not paths.supersessions.exists()


def test_stage3b_apply_rejects_wrong_or_duplicate_replacement_hash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = _paths(tmp_path)
    source = _row("source")
    replacement = _row("replacement", score=100, ctx=66560)
    paths.primary_raw_results.write_text(json.dumps(source, sort_keys=True) + "\n", encoding="utf-8")
    replacement_paths = _replacement_campaign(tmp_path, replacement)
    replacement_path = tmp_path / "replacement.json"
    _write_json(replacement_path, replacement)

    with pytest.raises(SystemExit, match="replacement_row_hash_not_found"):
        cli.main([
            "campaign", "supersede",
            "--campaign-id", paths.campaign_id,
            "--source-row-hash", campaign._primary_row_hash(source),
            "--replacement-campaign-id", replacement_paths.campaign_id,
            "--replacement-run-id", "synthetic-catchup",
            "--replacement-row", str(replacement_path),
            "--replacement-row-hash", campaign._primary_row_hash(_row("other", score=100, ctx=66560)),
            "--reason", "wrong stored hash",
        ])

    stored = replacement_paths.evidence_dir / "synthetic-catchup" / "raw_results.jsonl"
    stored.write_text(stored.read_text(encoding="utf-8") * 2, encoding="utf-8")
    with pytest.raises(SystemExit, match="ambiguous_replacement_row_hash"):
        cli.main([
            "campaign", "supersede",
            "--campaign-id", paths.campaign_id,
            "--source-row-hash", campaign._primary_row_hash(source),
            "--replacement-campaign-id", replacement_paths.campaign_id,
            "--replacement-run-id", "synthetic-catchup",
            "--replacement-row", str(replacement_path),
            "--replacement-row-hash", campaign._primary_row_hash(replacement),
            "--reason", "duplicate stored hash",
        ])
    assert not paths.supersessions.exists()


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("task", "replacement_task_contradicts_source"),
        ("model", "replacement_model_contradicts_source"),
        ("model_digest_resolved", "replacement_model_digest_contradicts_source"),
    ],
)
def test_stage3b_apply_rejects_replacement_provenance_contradictions(tmp_path, monkeypatch, field, message):
    monkeypatch.chdir(tmp_path)
    paths = _paths(tmp_path)
    source = _row("source")
    wrong_row = _row("wrong provenance", score=100, ctx=66560)
    wrong_row[field] = "other"
    paths.primary_raw_results.write_text(json.dumps(source, sort_keys=True) + "\n", encoding="utf-8")
    wrong_paths = _replacement_campaign(tmp_path, wrong_row, campaign_id=f"replacement-wrong-{field.replace('_', '-')}")
    wrong_path = tmp_path / f"wrong-{field}.json"
    _write_json(wrong_path, wrong_row)

    with pytest.raises(SystemExit, match=message):
        cli.main([
            "campaign", "supersede",
            "--campaign-id", paths.campaign_id,
            "--source-row-hash", campaign._primary_row_hash(source),
            "--replacement-campaign-id", wrong_paths.campaign_id,
            "--replacement-run-id", "synthetic-catchup",
            "--replacement-row", str(wrong_path),
            "--replacement-row-hash", campaign._primary_row_hash(wrong_row),
            "--reason", "wrong provenance",
        ])

    assert not paths.supersessions.exists()


def test_stage3b_apply_rejects_preview_contradiction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = _paths(tmp_path)
    source = _row("source")
    paths.primary_raw_results.write_text(json.dumps(source, sort_keys=True) + "\n", encoding="utf-8")

    replacement = _row("stored replacement", score=100, ctx=66560)
    replacement_paths = _replacement_campaign(tmp_path, replacement, campaign_id="replacement-preview")
    preview = dict(replacement)
    preview["reason"] = "mutated preview"
    preview_path = tmp_path / "preview.json"
    _write_json(preview_path, preview)

    with pytest.raises(SystemExit, match="replacement_preview_contradicts_stored_evidence"):
        cli.main([
            "campaign", "supersede",
            "--campaign-id", paths.campaign_id,
            "--source-row-hash", campaign._primary_row_hash(source),
            "--replacement-campaign-id", replacement_paths.campaign_id,
            "--replacement-run-id", "synthetic-catchup",
            "--replacement-row", str(preview_path),
            "--replacement-row-hash", campaign._primary_row_hash(replacement),
            "--reason", "preview must match stored",
        ])
    assert not paths.supersessions.exists()


def test_stage3b_apply_uses_stored_replacement_row_and_preserves_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = _paths(tmp_path)
    source = _row("source")
    replacement = _row("stored replacement", score=100, ctx=66560)
    paths.primary_raw_results.write_text(json.dumps(source, sort_keys=True) + "\n", encoding="utf-8")
    source_before = paths.primary_raw_results.read_bytes()
    replacement_paths = _replacement_campaign(tmp_path, replacement)
    replacement_file = replacement_paths.evidence_dir / "synthetic-catchup" / "raw_results.jsonl"
    replacement_before = replacement_file.read_bytes()
    replacement_path = tmp_path / "replacement.json"
    _write_json(replacement_path, replacement)

    cli.main([
        "campaign", "supersede",
        "--campaign-id", paths.campaign_id,
        "--source-row-hash", campaign._primary_row_hash(source),
        "--replacement-campaign-id", replacement_paths.campaign_id,
        "--replacement-run-id", "synthetic-catchup",
        "--replacement-row", str(replacement_path),
        "--replacement-row-hash", campaign._primary_row_hash(replacement),
        "--reason", "anchored replacement",
        "--operator", "test",
    ])

    edge = json.loads(paths.supersessions.read_text(encoding="utf-8"))
    assert edge["replacement_campaign_id"] == replacement_paths.campaign_id
    assert edge["replacement_row"] == replacement
    assert paths.primary_raw_results.read_bytes() == source_before
    assert replacement_file.read_bytes() == replacement_before
