import copy
import json

import pytest

from llm_modelbench import campaign


def _row(name, *, task="needle", model="m", digest="d", score=None):
    row = {"model": model, "model_digest_resolved": digest, "task": task, "reason": name}
    if score is None:
        row["error_kind"] = "harness_error"
    else:
        row["score"] = score
    return row


def _campaign(tmp_path, name="c"):
    return campaign.create_campaign(name, models=["m"], campaigns_root=tmp_path / "campaigns")[0]


def _native(paths, source, replacement, *, source_run="primary", replacement_run="replacement",
            source_campaign=None, replacement_campaign=None, recorded_at="2026-08-10T00:00:00+00:00",
            reason="synthetic correction", operator="test"):
    return campaign._native_supersession_record(
        paths=paths,
        source_campaign_id=source_campaign or paths.campaign_id,
        source_run_id=source_run,
        source_row=source,
        replacement_campaign_id=replacement_campaign or paths.campaign_id,
        replacement_run_id=replacement_run,
        replacement_row=replacement,
        reason=reason,
        operator=operator,
        tool="pytest",
        recorded_at=recorded_at,
    )


def _write_records(paths, records):
    paths.supersessions.parent.mkdir(parents=True, exist_ok=True)
    paths.supersessions.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in records), encoding="utf-8")


def test_stage3a_valid_native_record_and_duplicate_are_idempotent(tmp_path):
    paths = _campaign(tmp_path)
    source = _row("a")
    replacement = _row("b", score=100)

    first = campaign.record_supersession(
        paths,
        source_campaign_id=paths.campaign_id,
        source_row=source,
        replacement_run_id="catchup",
        replacement_row=replacement,
        reason="synthetic correction",
        operator="test",
        tool="pytest",
    )
    second = campaign.record_supersession(
        paths,
        source_campaign_id=paths.campaign_id,
        source_row=source,
        replacement_run_id="catchup",
        replacement_row=replacement,
        reason="synthetic correction",
        operator="test",
        tool="pytest",
    )

    assert first["schema_version"] == campaign.SUPERSESSION_SCHEMA_VERSION
    assert first["supersession_id"] == second["supersession_id"]
    assert len(paths.supersessions.read_text().splitlines()) == 1
    graph = campaign.load_supersession_graph(paths)
    assert graph["valid"] is True
    assert len(graph["edges"]) == 1


def test_stage3a_schema_dispatch_is_strict(tmp_path):
    paths = _campaign(tmp_path)
    source = _row("a")
    replacement = _row("b", score=100)
    native = _native(paths, source, replacement)
    assert campaign.validate_supersession_record(native)["format"] == "native"

    legacy = {
        "source_campaign_id": "legacy-c",
        "source_row_hash": "legacy-source",
        "replacement_run_id": "legacy-run",
        "replacement_row_hash": campaign._primary_row_hash(replacement),
        "replacement_row": replacement,
        "reason": "legacy correction",
    }
    assert campaign.validate_supersession_record(legacy)["format"] == "legacy"

    for value in (1, 3, 999, -1, "two", [], {}, True):
        bad = copy.deepcopy(legacy)
        bad["schema_version"] = value
        with pytest.raises(campaign.CampaignError, match="unsupported_supersession_schema"):
            campaign.validate_supersession_record(bad)


def test_stage3a_row_hash_and_provenance_contradictions_fail(tmp_path):
    paths = _campaign(tmp_path)
    source = _row("a", task="needle", model="m", digest="d")
    replacement = _row("b", task="needle", model="m", digest="d", score=100)
    record = _native(paths, source, replacement)

    bad_source = copy.deepcopy(record)
    bad_source["source"]["row_hash"] = "not-the-source-hash"
    with pytest.raises(campaign.CampaignError, match="source_row_hash_mismatch"):
        campaign.validate_supersession_record(bad_source, source_row=source, replacement_row=replacement)

    bad_replacement = copy.deepcopy(record)
    bad_replacement["replacement"]["row_hash"] = "not-the-replacement-hash"
    with pytest.raises(campaign.CampaignError, match="replacement_row_hash_mismatch"):
        campaign.validate_supersession_record(bad_replacement, source_row=source, replacement_row=replacement)

    bad_source_task = copy.deepcopy(record)
    bad_source_task["source"]["task"] = "other"
    with pytest.raises(campaign.CampaignError, match="source_task_mismatch"):
        campaign.validate_supersession_record(bad_source_task, source_row=source, replacement_row=replacement)

    bad_replacement_digest = copy.deepcopy(record)
    bad_replacement_digest["replacement"]["model_digest"] = "other"
    with pytest.raises(campaign.CampaignError, match="replacement_model_digest_mismatch"):
        campaign.validate_supersession_record(bad_replacement_digest, source_row=source, replacement_row=replacement)

    bad_embedded = copy.deepcopy(record)
    bad_embedded["replacement_row"] = _row("different", score=0)
    with pytest.raises(campaign.CampaignError, match="replacement_row_hash_mismatch"):
        campaign.validate_supersession_record(bad_embedded)


def test_stage3a_required_native_fields_and_aliases_are_enforced(tmp_path):
    paths = _campaign(tmp_path)
    source = _row("a")
    replacement = _row("b", score=100)
    valid = _native(paths, source, replacement)
    assert campaign.validate_supersession_record(valid)["format"] == "native"

    missing_id = copy.deepcopy(valid)
    missing_id.pop("supersession_id")
    with pytest.raises(campaign.CampaignError, match="supersession_id_missing"):
        campaign.validate_supersession_record(missing_id)

    missing_source_hash = copy.deepcopy(valid)
    missing_source_hash["source"]["row_hash"] = ""
    missing_source_hash["supersession_id"] = campaign.supersession_semantic_hash(missing_source_hash)
    with pytest.raises(campaign.CampaignError, match="source_row_hash_missing"):
        campaign.validate_supersession_record(missing_source_hash)

    missing_replacement_hash = copy.deepcopy(valid)
    missing_replacement_hash["replacement"]["row_hash"] = ""
    missing_replacement_hash["supersession_id"] = campaign.supersession_semantic_hash(missing_replacement_hash)
    with pytest.raises(campaign.CampaignError, match="replacement_row_hash_missing"):
        campaign.validate_supersession_record(missing_replacement_hash)

    missing_provenance = copy.deepcopy(valid)
    missing_provenance["source"]["campaign_id"] = ""
    missing_provenance["supersession_id"] = campaign.supersession_semantic_hash(missing_provenance)
    with pytest.raises(campaign.CampaignError, match="source_campaign_id_missing"):
        campaign.validate_supersession_record(missing_provenance)

    bad_alias = copy.deepcopy(valid)
    bad_alias["source_row_hash"] = "contradictory"
    with pytest.raises(campaign.CampaignError, match="source_row_hash_alias_mismatch"):
        campaign.validate_supersession_record(bad_alias)

    bad_replacement_alias = copy.deepcopy(valid)
    bad_replacement_alias["replacement_run_id"] = "contradictory"
    with pytest.raises(campaign.CampaignError, match="replacement_run_id_alias_mismatch"):
        campaign.validate_supersession_record(bad_replacement_alias)


def test_stage3a_native_active_false_fails_but_legacy_active_false_is_honored(tmp_path):
    paths = _campaign(tmp_path)
    source = _row("a")
    replacement = _row("b", score=100)
    native = _native(paths, source, replacement)
    native["active"] = True
    _write_records(paths, [native])
    assert len(campaign.load_supersession_graph(paths)["edges"]) == 1

    native["active"] = False
    _write_records(paths, [native])
    with pytest.raises(campaign.CampaignError, match="invalid_native_supersession_state"):
        campaign.load_supersession_graph(paths)

    legacy_paths = _campaign(tmp_path, "legacy-inactive")
    legacy = {
        "active": False,
        "source_campaign_id": "legacy-c",
        "source_row_hash": "legacy-source",
        "replacement_run_id": "legacy-run",
        "replacement_row_hash": campaign._primary_row_hash(replacement),
        "replacement_row": replacement,
        "reason": "legacy correction",
    }
    _write_records(legacy_paths, [legacy])
    legacy_graph = campaign.load_supersession_graph(legacy_paths)
    assert legacy_graph["valid"] is True
    assert legacy_graph["edges"] == []


def test_stage3a_semantic_hash_is_canonical_and_stable(tmp_path):
    paths = _campaign(tmp_path)
    source = _row("a")
    replacement = _row("b", score=100)
    record = _native(paths, source, replacement, recorded_at="one")
    reordered = json.loads(json.dumps(record, sort_keys=True))
    changed_time = copy.deepcopy(record)
    changed_time["recorded_at"] = "two"
    changed_source = copy.deepcopy(record)
    changed_source["source"]["row_hash"] = campaign._primary_row_hash(_row("other"))
    changed_replacement = copy.deepcopy(record)
    changed_replacement["replacement"]["row_hash"] = campaign._primary_row_hash(_row("other", score=0))
    changed_provenance = copy.deepcopy(record)
    changed_provenance["replacement"]["campaign_id"] = "other-campaign"

    assert campaign.supersession_semantic_hash(record) == campaign.supersession_semantic_hash(reordered)
    assert campaign.supersession_semantic_hash(record) == campaign.supersession_semantic_hash(changed_time)
    assert campaign.supersession_semantic_hash(record) != campaign.supersession_semantic_hash(changed_source)
    assert campaign.supersession_semantic_hash(record) != campaign.supersession_semantic_hash(changed_replacement)
    assert campaign.supersession_semantic_hash(record) != campaign.supersession_semantic_hash(changed_provenance)


def test_stage3a_same_successor_multi_edge_evidence_is_deterministic(tmp_path):
    paths = _campaign(tmp_path)
    a = _row("a")
    b = _row("b", score=100)
    e1 = _native(paths, a, b, reason="reason one", operator="operator-one")
    e2 = _native(paths, a, b, reason="reason two", operator="operator-two")
    assert e1["supersession_id"] != e2["supersession_id"]

    graph = campaign.build_supersession_graph([
        campaign.validate_supersession_record(e1),
        campaign.validate_supersession_record(e2),
        campaign.validate_supersession_record(copy.deepcopy(e1)),
    ])
    reversed_graph = campaign.build_supersession_graph([
        campaign.validate_supersession_record(e2),
        campaign.validate_supersession_record(e1),
    ])
    assert graph["valid"] is True
    assert json.dumps(graph, sort_keys=True, default=str) == json.dumps(reversed_graph, sort_keys=True, default=str)
    source_key = graph["edges"][0]["source_key"]
    replacement_key = graph["edges"][0]["replacement_key"]
    supporting = graph["by_source"][source_key][replacement_key]
    assert [edge["supersession_id"] for edge in supporting] == sorted([e1["supersession_id"], e2["supersession_id"]])
    resolution = campaign.resolve_supersession_chain(graph, graph["edges"][0]["source"])
    assert [edge["supersession_id"] for edge in resolution["chain"]] == sorted([e1["supersession_id"], e2["supersession_id"]])


def test_stage3a_chain_resolution_preserves_edges_and_is_order_independent(tmp_path):
    paths = _campaign(tmp_path)
    a = _row("a")
    b = _row("b", score=0)
    c = _row("c", score=100)
    ab = _native(paths, a, b, source_run="primary", replacement_run="b")
    bc = _native(paths, b, c, source_run="b", replacement_run="c")
    _write_records(paths, [ab, bc])

    graph = campaign.load_supersession_graph(paths)
    resolution = campaign.resolve_supersession_chain(graph, graph["edges"][0]["source"])
    assert resolution["valid"] is True
    assert resolution["terminal"]["row_hash"] == campaign._primary_row_hash(c)
    assert [edge["replacement"]["row_hash"] for edge in resolution["chain"]] == [
        campaign._primary_row_hash(b),
        campaign._primary_row_hash(c),
    ]

    reversed_paths = _campaign(tmp_path, "reverse")
    _write_records(reversed_paths, [bc, ab])
    reversed_graph = campaign.load_supersession_graph(reversed_paths)
    reversed_resolution = campaign.resolve_supersession_chain(reversed_graph, graph["edges"][0]["source"])
    assert reversed_resolution["terminal"] == resolution["terminal"]
    assert [edge["supersession_id"] for edge in reversed_resolution["chain"]] == [
        edge["supersession_id"] for edge in resolution["chain"]
    ]


def test_stage3a_cycles_and_forks_fail_closed(tmp_path):
    paths = _campaign(tmp_path)
    a = _row("a")
    b = _row("b", score=0)
    c = _row("c", score=100)

    self_edge = _native(paths, a, a)
    self_edge["replacement"] = dict(self_edge["source"])
    self_edge["replacement_campaign_id"] = self_edge["replacement"]["campaign_id"]
    self_edge["replacement_run_id"] = self_edge["replacement"]["run_id"]
    self_edge["replacement_row_hash"] = self_edge["replacement"]["row_hash"]
    self_edge["supersession_id"] = campaign.supersession_semantic_hash(self_edge)
    with pytest.raises(campaign.CampaignError, match="cyclic_supersession"):
        campaign.validate_supersession_record(self_edge)

    ab = _native(paths, a, b, source_run="a", replacement_run="b")
    ba = _native(paths, b, a, source_run="b", replacement_run="a")
    graph = campaign.build_supersession_graph([
        campaign.validate_supersession_record(ab),
        campaign.validate_supersession_record(ba),
    ])
    assert graph["valid"] is False
    assert graph["errors"][0]["reason"] == "cyclic_supersession"

    bc = _native(paths, b, c, source_run="b", replacement_run="c")
    ca = _native(paths, c, a, source_run="c", replacement_run="a")
    graph = campaign.build_supersession_graph([
        campaign.validate_supersession_record(ab),
        campaign.validate_supersession_record(bc),
        campaign.validate_supersession_record(ca),
    ])
    assert graph["valid"] is False
    assert graph["errors"][0]["reason"] == "cyclic_supersession"

    ac = _native(paths, a, c, source_run="a", replacement_run="c")
    graph = campaign.build_supersession_graph([
        campaign.validate_supersession_record(ab),
        campaign.validate_supersession_record(ac),
        campaign.validate_supersession_record(copy.deepcopy(ab)),
    ])
    assert graph["valid"] is False
    assert graph["errors"][0]["reason"] == "ambiguous_supersession_fork"


def test_stage3a_malformed_jsonl_and_legacy_compatibility(tmp_path):
    paths = _campaign(tmp_path)
    replacement = _row("legacy replacement", score=100)
    legacy = {
        "source_campaign_id": "legacy-c",
        "source_row_hash": "legacy-source",
        "replacement_run_id": "legacy-run",
        "replacement_row_hash": campaign._primary_row_hash(replacement),
        "replacement_row": replacement,
        "reason": "legacy correction",
    }
    _write_records(paths, [legacy])
    graph = campaign.load_supersession_graph(paths)
    assert graph["valid"] is True
    assert graph["edges"][0]["format"] == "legacy"
    assert graph["edges"][0]["record"]["legacy_compatibility"] == "explicit"

    insufficient = _campaign(tmp_path, "insufficient")
    _write_records(insufficient, [{"source_row_hash": "a", "replacement_row_hash": "b"}])
    with pytest.raises(campaign.CampaignError, match="insufficient_legacy_supersession"):
        campaign.load_supersession_graph(insufficient)

    malformed = _campaign(tmp_path, "malformed")
    malformed.supersessions.parent.mkdir(parents=True, exist_ok=True)
    malformed.supersessions.write_text('{"ok": true}\n{"broken"\n', encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="malformed_supersession_ledger"):
        campaign.load_supersession_graph(malformed)


def test_stage3a_atomic_failure_does_not_report_success_or_mutate_rows(tmp_path, monkeypatch):
    paths = _campaign(tmp_path)
    source = _row("a")
    replacement = _row("b", score=100)
    source_before = copy.deepcopy(source)
    replacement_before = copy.deepcopy(replacement)
    original_write = campaign._atomic_write_text

    def fail_supersessions(path, text):
        if path == paths.supersessions:
            raise OSError("injected supersession write failure")
        return original_write(path, text)

    monkeypatch.setattr(campaign, "_atomic_write_text", fail_supersessions)
    with pytest.raises(OSError, match="injected supersession write failure"):
        campaign.record_supersession(
            paths,
            source_campaign_id=paths.campaign_id,
            source_row=source,
            replacement_run_id="catchup",
            replacement_row=replacement,
            reason="synthetic correction",
        )
    assert not paths.supersessions.exists()
    assert source == source_before
    assert replacement == replacement_before


def test_stage3a_failed_second_atomic_append_preserves_existing_ledger(tmp_path, monkeypatch):
    paths = _campaign(tmp_path)
    a = _row("a")
    b = _row("b", score=0)
    c = _row("c", score=100)
    campaign.record_supersession(
        paths,
        source_campaign_id=paths.campaign_id,
        source_row=a,
        replacement_run_id="b",
        replacement_row=b,
        reason="first correction",
        operator="test",
    )
    original_bytes = paths.supersessions.read_bytes()
    b_before = copy.deepcopy(b)
    c_before = copy.deepcopy(c)
    original_write = campaign._atomic_write_text

    def fail_supersessions(path, text):
        if path == paths.supersessions:
            raise OSError("injected second write failure")
        return original_write(path, text)

    monkeypatch.setattr(campaign, "_atomic_write_text", fail_supersessions)
    with pytest.raises(OSError, match="injected second write failure"):
        campaign.record_supersession(
            paths,
            source_campaign_id=paths.campaign_id,
            source_row=b,
            replacement_run_id="c",
            replacement_row=c,
            reason="second correction",
            operator="test",
        )
    assert paths.supersessions.read_bytes() == original_bytes
    graph = campaign.load_supersession_graph(paths)
    assert len(graph["edges"]) == 1
    assert graph["edges"][0]["source"]["row_hash"] == campaign._primary_row_hash(a)
    assert graph["edges"][0]["replacement"]["row_hash"] == campaign._primary_row_hash(b)
    assert b == b_before
    assert c == c_before


def test_stage3a_repeated_graph_load_is_deterministic(tmp_path):
    paths = _campaign(tmp_path)
    a = _row("a")
    b = _row("b", score=0)
    c = _row("c", score=100)
    _write_records(paths, [
        _native(paths, a, b, source_run="a", replacement_run="b"),
        _native(paths, b, c, source_run="b", replacement_run="c"),
    ])

    first = campaign.load_supersession_graph(paths)
    second = campaign.load_supersession_graph(paths)
    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(second, sort_keys=True, default=str)
