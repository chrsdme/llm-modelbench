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
    replacement_path = tmp_path / "replacement-b.json"
    _write_json(replacement_path, replacement_b)

    cli.main([
        "campaign", "supersede",
        "--campaign-id", paths.campaign_id,
        "--source-row-hash", campaign._primary_row_hash(source),
        "--replacement-run-id", "synthetic-catchup-b",
        "--replacement-row", str(replacement_path),
        "--reason", "synthetic 65536 -> 66560 needle correction",
        "--operator", "test",
    ])
    campaign.record_supersession(
        paths,
        source_campaign_id=paths.campaign_id,
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
        ])
