import json

from llm_modelbench.simulate import load_rows, simulate


def test_simulate_only_replays_estimated_environment_skips(tmp_path):
    rows = [
        {"model": "environment", "needle_skipped": [
            {"skip_class": "environment", "size": 32000, "estimated_total_gb": 18.0, "vram_budget_gb": 12.0},
            {"skip_class": "operator", "size": 64000, "estimated_total_gb": 30.0},
            {"skip_class": "environment", "size": 64000},
        ]},
    ]
    (tmp_path / "raw_results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))

    results = simulate(load_rows(tmp_path), 24.0)

    assert len(results) == 1
    assert results[0]["model"] == "environment"
    assert results[0]["would_fit_at_simulated_budget"] is True
