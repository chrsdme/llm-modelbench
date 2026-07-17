import json
import time
from llm_modelbench import rankings


def _write_run(runs_dir, run_id, level, rows, identities=None):
    run = runs_dir / run_id
    run.mkdir()
    (run / "raw_results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": level}))
    (run / "model_identities.json").write_text(json.dumps(identities or {}))
    return run


def test_write_rankings_produces_all_three_outputs(tmp_path):
    runs_dir = tmp_path / "runs"; runs_dir.mkdir()
    _write_run(runs_dir, "s1", "smoke",
               [{"model": "m", "task": "a", "category": "c", "score": 100.0,
                 "timestamp": "2026-01-01T00:00:00Z"}],
               identities={"m": {"digest": "d1"}})
    out_dir = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out_dir, html_template="<html>__MASTER_SUMMARY_JSON__</html>")
    assert (out_dir / "master_raw.jsonl").exists()
    assert (out_dir / "master_summary.json").exists()
    assert (out_dir / "master_report.html").exists()
    assert '"d1"' in (out_dir / "master_report.html").read_text()


def test_every_row_gets_a_unique_6_char_import_tag(tmp_path):
    runs_dir = tmp_path / "runs"; runs_dir.mkdir()
    _write_run(runs_dir, "s1", "smoke",
               [{"model": "m1", "task": "a", "category": "c", "score": 100.0, "timestamp": "2026-01-01T00:00:00Z"},
                {"model": "m2", "task": "a", "category": "c", "score": 90.0, "timestamp": "2026-01-01T00:00:00Z"}],
               identities={"m1": {"digest": "d1"}, "m2": {"digest": "d2"}})
    out_dir = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out_dir)
    rows = rankings.load_accumulated(out_dir)
    tags = [r["import_tag"] for r in rows]
    assert len(tags) == 2
    assert len(set(tags)) == 2  # unique
    assert all(len(t) == 6 and t.isalnum() for t in tags)


def test_import_tags_are_stable_across_repeated_updates(tmp_path):
    runs_dir = tmp_path / "runs"; runs_dir.mkdir()
    _write_run(runs_dir, "s1", "smoke",
               [{"model": "m", "task": "a", "category": "c", "score": 100.0, "timestamp": "2026-01-01T00:00:00Z"}],
               identities={"m": {"digest": "d1"}})
    out_dir = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out_dir)
    tag1 = rankings.load_accumulated(out_dir)[0]["import_tag"]
    rankings.write_rankings(runs_dir, out_dir)  # re-run, nothing changed on disk
    rows = rankings.load_accumulated(out_dir)
    assert len(rows) == 1  # not duplicated
    assert rows[0]["import_tag"] == tag1  # same tag, not regenerated


def test_deleting_a_run_directory_does_not_remove_it_from_the_database(tmp_path):
    """The core behavior this design exists for: an underperforming model's
    run directory can be deleted from runs/ without losing the historical
    evidence of why it was pruned."""
    runs_dir = tmp_path / "runs"; runs_dir.mkdir()
    _write_run(runs_dir, "devstral_run", "short",
               [{"model": "devstral:24b", "task": "py_anagram", "category": "coding_python",
                 "score": 40.0, "timestamp": "2026-01-01T00:00:00Z"}],
               identities={"devstral:24b": {"digest": "d-devstral"}})
    out_dir = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out_dir)

    # the model underperformed, operator deletes its run directory entirely
    import shutil
    shutil.rmtree(runs_dir / "devstral_run")

    # a later run for something unrelated triggers another rankings update
    _write_run(runs_dir, "other_run", "short",
               [{"model": "other:7b", "task": "py_anagram", "category": "coding_python",
                 "score": 90.0, "timestamp": "2026-01-02T00:00:00Z"}],
               identities={"other:7b": {"digest": "d-other"}})
    rankings.write_rankings(runs_dir, out_dir)

    summary = json.loads((out_dir / "master_summary.json").read_text())
    digests = {m["digest"] for m in summary}
    assert "d-devstral" in digests, "deleted run's model must still be in the database"
    assert "d-other" in digests
    devstral_entry = next(m for m in summary if m["digest"] == "d-devstral")
    assert devstral_entry["overall_mean_score"] == 40.0


def test_a_run_that_changes_on_disk_after_import_gets_reimported(tmp_path):
    runs_dir = tmp_path / "runs"; runs_dir.mkdir()
    run = _write_run(runs_dir, "s1", "smoke",
                      [{"model": "m", "task": "a", "category": "c", "score": 0.0,
                        "error_kind": "thinking_only", "timestamp": "2026-01-01T00:00:00Z"}],
                      identities={"m": {"digest": "d1"}})
    out_dir = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out_dir)

    time.sleep(0.05)
    (run / "raw_results.jsonl").write_text(json.dumps(
        {"model": "m", "task": "a", "category": "c", "score": 100.0,
         "error_kind": None, "timestamp": "2026-01-01T00:05:00Z"}) + "\n")
    rankings.write_rankings(runs_dir, out_dir)

    rows = rankings.load_accumulated(out_dir)
    assert len(rows) == 1
    assert rows[0]["score"] == 100.0


def test_rank_for_output_prefers_higher_level_across_accumulated_history(tmp_path):
    runs_dir = tmp_path / "runs"; runs_dir.mkdir()
    _write_run(runs_dir, "smoke_run", "smoke",
               [{"model": "m", "task": "t1", "category": "c", "score": 0.0,
                 "error_kind": "thinking_only", "timestamp": "2026-01-01T00:00:00Z"}],
               identities={"m": {"digest": "d1"}})
    out_dir = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out_dir)

    _write_run(runs_dir, "short_run", "short",
               [{"model": "m", "task": "t1", "category": "c", "score": 100.0,
                 "error_kind": None, "timestamp": "2026-01-02T00:00:00Z"}],
               identities={"m": {"digest": "d1"}})
    rankings.write_rankings(runs_dir, out_dir)

    # both runs are still in the raw database (accumulated, nothing thrown away)
    rows = rankings.load_accumulated(out_dir)
    assert len(rows) == 2

    # but the summary ranks off the better (short-level) result only
    summary = json.loads((out_dir / "master_summary.json").read_text())
    m = next(x for x in summary if x["digest"] == "d1")
    assert m["overall_mean_score"] == 100.0


def test_retrieval_embed_failure_gets_harness_error_not_a_bare_zero():
    """Regression guard: a failed embed() call (right-length list of empty
    vectors, the real OllamaClient failure shape) must be distinguishable from
    genuine 0.0 retrieval quality. Confirmed against real data: two models
    (llama-embed-nemotron-8b, KaLM-Reranker-V1-Large) hit this exact case in
    a real run, both landed as bare 0.0/error_kind=None before this fix."""
    from llm_modelbench import runner
    from llm_modelbench.tasks import TASKS
    from llm_modelbench.config import Config

    class FailEmbedClient:
        def embed(self, model, texts):
            return [[] for _ in texts]

    task = next(t for t in TASKS if t.id == "ret_ukdocs")
    result = runner._run_once(FailEmbedClient(), Config(), "broken-embedder", task)
    assert result["score"] == 0.0
    assert result["error_kind"] == "harness_error"
    assert "embed failed" in result["reason"]


def test_kv_scalar_bytes_recognizes_q4_0_instead_of_falling_through_to_f16():
    """Regression guard: q4_0/q4_1 previously fell through to the f16 default
    (2 bytes/scalar) while still being labeled correctly, silently overstating
    real VRAM cost and causing the needle pre-flight check to skip attempts
    that would likely fit. Confirmed against a real run where this happened."""
    import os
    from llm_modelbench.runner import _kv_scalar_bytes
    old = os.environ.get("OLLAMA_KV_CACHE_TYPE")
    try:
        os.environ["OLLAMA_KV_CACHE_TYPE"] = "q4_0"
        scalar, label = _kv_scalar_bytes()
        assert scalar == 0.5
        assert label == "q4_0"
        os.environ["OLLAMA_KV_CACHE_TYPE"] = "q8_0"
        assert _kv_scalar_bytes() == (1.0, "q8_0")
        os.environ["OLLAMA_KV_CACHE_TYPE"] = "f16"
        assert _kv_scalar_bytes() == (2.0, "f16")
    finally:
        if old is None:
            os.environ.pop("OLLAMA_KV_CACHE_TYPE", None)
        else:
            os.environ["OLLAMA_KV_CACHE_TYPE"] = old


def test_fully_tested_reflects_cumulative_levels_not_three_separate_runs(tmp_path):
    """Regression guard: full is cumulative and already runs everything short/smoke
    would run, so a model tested directly at --level full (a legitimate, documented
    workflow for models with existing confidence) must count as fully tested. The
    old {"smoke","short","full"}.issubset(...) check required three separate
    recorded levels and was False for every real model in the fleet, including
    ones genuinely given the full battery."""
    runs_dir = tmp_path / "runs"; runs_dir.mkdir()
    _write_run(runs_dir, "full_only_run", "full",
               [{"model": "m", "task": "py_anagram", "category": "coding_python", "score": 100.0,
                 "timestamp": "2026-01-01T00:00:00Z"},
                {"model": "m", "task": "needle", "category": "long_context", "score": 100.0,
                 "timestamp": "2026-01-01T00:05:00Z"}],
               identities={"m": {"digest": "d1"}})
    out_dir = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out_dir)
    summary = json.loads((out_dir / "master_summary.json").read_text())
    entry = next(x for x in summary if x["digest"] == "d1")
    assert entry["levels_seen"] == ["full"]
    assert entry["fully_tested"] is True
    assert entry["total_wall_seconds"] is not None


def test_fully_tested_is_still_false_for_genuinely_partial_coverage(tmp_path):
    """The fix must not become vacuously True for everything -- a model only ever
    run at smoke or short still correctly shows fully_tested False."""
    runs_dir = tmp_path / "runs"; runs_dir.mkdir()
    _write_run(runs_dir, "smoke_only", "smoke",
               [{"model": "m", "task": "py_anagram", "category": "coding_python", "score": 100.0,
                 "timestamp": "2026-01-01T00:00:00Z"}],
               identities={"m": {"digest": "d-smoke"}})
    _write_run(runs_dir, "short_only", "short",
               [{"model": "n", "task": "py_anagram", "category": "coding_python", "score": 100.0,
                 "timestamp": "2026-01-01T00:00:00Z"}],
               identities={"n": {"digest": "d-short"}})
    out_dir = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out_dir)
    summary = json.loads((out_dir / "master_summary.json").read_text())
    smoke_entry = next(x for x in summary if x["digest"] == "d-smoke")
    short_entry = next(x for x in summary if x["digest"] == "d-short")
    assert smoke_entry["fully_tested"] is False
    assert smoke_entry["total_wall_seconds"] is None
    assert short_entry["fully_tested"] is False
    assert short_entry["total_wall_seconds"] is None


def test_overall_mean_score_excludes_gate_task_inflation_like_real_ornith_case(tmp_path):
    """Regression guard, reproduces the real Ornith-1.0-9B-MTP case found in
    production rankings data: a thin, smoke-only sample dominated by the
    project's own difficulty=0.0 gate tasks (txt_sort, txt_emails, json_extract,
    git_commit) must not average up to a perfect 100.0 that outranks models with
    real full-battery coverage. txt_sort is a real difficulty=0.0 task in the
    live TASKS registry; py_anagram is a real difficulty=1.2 coding_python task.
    A flat mean of [100, 100, 100, 100, 50] is 90.0. The gate-excluded,
    category-weighted composite must reflect only the one real task's score."""
    runs_dir = tmp_path / "runs"; runs_dir.mkdir()
    _write_run(runs_dir, "smoke_thin", "smoke",
               [{"model": "m", "task": "txt_sort", "category": "text_ops", "score": 100.0,
                 "timestamp": "2026-01-01T00:00:00Z"},
                {"model": "m", "task": "txt_emails", "category": "text_ops", "score": 100.0,
                 "timestamp": "2026-01-01T00:00:01Z"},
                {"model": "m", "task": "json_extract", "category": "knowledge_base", "score": 100.0,
                 "timestamp": "2026-01-01T00:00:02Z"},
                {"model": "m", "task": "git_commit", "category": "git", "score": 100.0,
                 "timestamp": "2026-01-01T00:00:03Z"},
                {"model": "m", "task": "py_anagram", "category": "coding_python", "score": 50.0,
                 "timestamp": "2026-01-01T00:00:04Z"}],
               identities={"m": {"digest": "d-thin"}})
    out_dir = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out_dir)
    summary = json.loads((out_dir / "master_summary.json").read_text())
    entry = next(x for x in summary if x["digest"] == "d-thin")
    flat_mean = (100.0 * 4 + 50.0) / 5
    assert flat_mean == 90.0  # sanity-check the trap this test is guarding against
    assert entry["overall_mean_score"] == 50.0
    assert entry["overall_mean_score"] != flat_mean


def test_overall_mean_score_still_weights_across_multiple_real_categories(tmp_path):
    """Not just gate-exclusion: a model with two real, weighted categories must
    get a properly weighted composite, not a flat mean across them either."""
    runs_dir = tmp_path / "runs"; runs_dir.mkdir()
    _write_run(runs_dir, "two_cat", "short",
               [{"model": "m", "task": "py_anagram", "category": "coding_python", "score": 100.0,
                 "timestamp": "2026-01-01T00:00:00Z"},
                {"model": "m", "task": "web_nav", "category": "coding_web", "score": 0.0,
                 "timestamp": "2026-01-01T00:00:01Z"}],
               identities={"m": {"digest": "d-two"}})
    out_dir = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out_dir)
    summary = json.loads((out_dir / "master_summary.json").read_text())
    entry = next(x for x in summary if x["digest"] == "d-two")
    from llm_modelbench.config import DEFAULT_WEIGHTS
    w_py, w_web = DEFAULT_WEIGHTS["coding_python"], DEFAULT_WEIGHTS["coding_web"]
    expected = round((100.0 * w_py + 0.0 * w_web) / (w_py + w_web), 2)
    flat_mean = 50.0
    assert expected != flat_mean  # weights for these two categories are genuinely unequal
    assert entry["overall_mean_score"] == expected


def test_model_card_payload_preserves_all_historical_attempts_and_identity(tmp_path):
    runs_dir = tmp_path / "runs"; runs_dir.mkdir()
    identity = {"m": {"digest": "d1", "size": 5_000_000_000,
                      "parameter_size": "8.0B", "quantization_level": "Q4_K_M",
                      "family": "llama", "families": ["llama"]}}
    _write_run(runs_dir, "old", "smoke",
               [{"model": "m", "task": "py_anagram", "category": "coding_python", "family": "text",
                 "score": 20.0, "task_hash": rankings._CURRENT_HASHES["py_anagram"],
                 "timestamp": "2026-01-01T00:00:00Z"}], identities=identity)
    _write_run(runs_dir, "new", "short",
               [{"model": "m", "task": "py_anagram", "category": "coding_python", "family": "text",
                 "score": 90.0, "task_hash": rankings._CURRENT_HASHES["py_anagram"],
                 "timestamp": "2026-01-02T00:00:00Z"}], identities=identity)
    out = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out, html_template="<script>const DATA=__MASTER_SUMMARY_JSON__</script>")
    entry = json.loads((out / "master_summary.json").read_text())[0]
    assert entry["history_count"] == 2
    assert [h["score"] for h in entry["history"]] == [90.0, 20.0]
    assert sum(bool(h["used_for_current_ranking"]) for h in entry["history"]) == 1
    assert entry["parameter_size"] == "8.0B"
    assert entry["quantization_level"] == "Q4_K_M"
    assert entry["architecture_family"] == "llama"
    html = (out / "master_report.html").read_text()
    assert "history_count" in html and "Q4_K_M" in html


def test_report_payload_has_top_category_class_and_multimodal_sections(tmp_path):
    runs_dir = tmp_path / "runs"; runs_dir.mkdir()
    _write_run(runs_dir, "r1", "short",
               [{"model": "vl", "task": "ocr_invoice", "category": "ocr", "family": "vision",
                 "score": 100.0, "task_hash": rankings._CURRENT_HASHES["ocr_invoice"],
                 "class": "vision", "timestamp": "2026-01-01T00:00:00Z"}],
               identities={"vl": {"digest": "d-vl", "parameter_size": "7B", "quantization_level": "Q4_K_M"}})
    out = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out)
    payload = json.loads((out / "master_report_data.json").read_text())
    assert payload["top_by_category"]["ocr"][0]["model"] == "vl"
    assert payload["top_by_class"]["vision"][0]["model"] == "vl"
    assert payload["multimodal"][0]["model"] == "vl"


def test_provisional_status_explains_incomplete_current_scope(tmp_path):
    runs_dir = tmp_path / "runs"; runs_dir.mkdir()
    _write_run(runs_dir, "r1", "smoke",
               [{"model": "m", "task": "py_anagram", "category": "coding_python", "family": "text",
                 "score": 100.0, "task_hash": rankings._CURRENT_HASHES["py_anagram"],
                 "timestamp": "2026-01-01T00:00:00Z"}], identities={"m": {"digest": "d1"}})
    out = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out)
    entry = json.loads((out / "master_summary.json").read_text())[0]
    assert entry["quality_status"] == "provisional"
    assert entry["missing_quality_tasks"]
    assert any("full-level" in reason for reason in entry["quality_status_reasons"])


def test_embedding_only_specialist_is_not_cross_class_overall_ranked(tmp_path):
    runs_dir = tmp_path / "runs"; runs_dir.mkdir()
    _write_run(runs_dir, "embed", "full",
               [{"model": "embed", "task": "ret_ukdocs", "category": "retrieval", "family": "embedding",
                 "score": 100.0, "task_hash": rankings._CURRENT_HASHES["ret_ukdocs"],
                 "timestamp": "2026-01-01T00:00:00Z"}], identities={"embed": {"digest": "d-embed"}})
    out = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, out)
    entry = json.loads((out / "master_summary.json").read_text())[0]
    assert entry["overall_comparable"] is False
    assert entry["overall_rank"] is None
    assert entry["overall_mean_score"] == 100.0


def test_a_timed_out_reprobe_with_no_score_never_supersedes_an_earlier_valid_judged_result(monkeypatch):
    """Regression guard for a real incident: R1-Coder-DARE-7B's canonical run
    got kb_taxonomy/wr_rag genuinely judged (70.0/50.0), then a --think off
    reprobe pass timed out partway through, leaving those same tasks at
    score=None ("raw only, judge off") with a LATER timestamp. Since both
    rows share the same current task_hash, and _canonical_run_score() doesn't
    treat a think-only override as diagnostic, the tie fell through to raw
    timestamp, letting the incomplete timeout row silently displace the
    valid, already-judged one. A row's validity must outrank its recency."""
    monkeypatch.setattr(rankings, "_CURRENT_HASHES", {"wr_rag": "CURRENT"})
    canonical = {
        "model_digest_resolved": "d1", "task": "wr_rag", "task_hash": "CURRENT",
        "level": "full", "timestamp": "2026-07-15T02:43:21+00:00",
        "run_id": "overnight_v2_..._latest", "score": 50.0,
        "run_configuration": {"ctx_override": None, "num_predict_override": None,
                               "task_regex": None, "think": "auto"},
    }
    timed_out_reprobe = {
        "model_digest_resolved": "d1", "task": "wr_rag", "task_hash": "CURRENT",
        "level": "full", "timestamp": "2026-07-15T03:13:24+00:00",
        "run_id": "overnight_v2_..._latest_thinkoff", "score": None,
        "run_configuration": {"ctx_override": None, "num_predict_override": None,
                               "task_regex": None, "think": None},
    }
    result = rankings.rank_for_output([canonical, timed_out_reprobe])
    assert len(result) == 1
    assert result[0]["score"] == 50.0
    assert result[0]["run_id"] == "overnight_v2_..._latest"


def test_a_genuinely_later_valid_result_still_wins_over_an_earlier_valid_one(monkeypatch):
    """Guard against overcorrecting: when both rows DO have a valid score,
    recency must still decide, a real fresh reprobe result should still be
    able to supersede an older one."""
    monkeypatch.setattr(rankings, "_CURRENT_HASHES", {"wr_rag": "CURRENT"})
    older = {
        "model_digest_resolved": "d1", "task": "wr_rag", "task_hash": "CURRENT",
        "level": "full", "timestamp": "2026-07-15T02:43:21+00:00",
        "run_id": "old", "score": 50.0,
        "run_configuration": {"ctx_override": None, "num_predict_override": None, "task_regex": None},
    }
    newer_valid = {
        "model_digest_resolved": "d1", "task": "wr_rag", "task_hash": "CURRENT",
        "level": "full", "timestamp": "2026-07-15T03:13:24+00:00",
        "run_id": "new", "score": 85.0,
        "run_configuration": {"ctx_override": None, "num_predict_override": None, "task_regex": None},
    }
    result = rankings.rank_for_output([older, newer_valid])
    assert result[0]["score"] == 85.0


def test_valid_repair_result_supersedes_canonical_error_score(monkeypatch):
    """A canonical model-error row often carries numeric 0.0.  That is not a
    valid quality result and must not block a later bounded repair result."""
    monkeypatch.setattr(rankings, "_CURRENT_HASHES", {"py_anagram": "CURRENT"})
    failed = {
        "model_digest_resolved": "d1", "task": "py_anagram", "task_hash": "CURRENT",
        "level": "full", "timestamp": "2026-07-15T01:00:00Z", "run_id": "canonical",
        "score": 0.0, "error_kind": "thinking_only",
        "run_configuration": {"think": "auto"},
    }
    repaired = {
        "model_digest_resolved": "d1", "task": "py_anagram", "task_hash": "CURRENT",
        "level": "full", "timestamp": "2026-07-15T02:00:00Z", "run_id": "repair",
        "score": 100.0, "error_kind": None, "repair_kind": "retry_generation",
        "run_configuration": {"think": "off", "num_predict_override": 4096},
    }
    result = rankings.rank_for_output([failed, repaired])
    assert result[0]["run_id"] == "repair"
    assert result[0]["score"] == 100.0
