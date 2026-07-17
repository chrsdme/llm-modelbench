from llm_modelbench.report import _html


def test_html_report_is_self_contained_and_has_csp(tmp_path):
    path = tmp_path / "report.html"
    leaderboard = [{
        "model": "m", "class": "text", "quality": 80.0, "tok_s": 12.0,
        "offload": 0.0, "value_per_gb": 10.0, "score_blended": 80.0,
        "size_gb": 8.0, "err": 0, "completion_rate": 1.0,
        "categories": {"reasoning": 80.0},
    }]
    _html(path, leaderboard, {"reasoning": [("m", 80.0)]}, [], object(), {})
    text = path.read_text()
    assert "Content-Security-Policy" in text
    assert "https://" not in text
    assert "cdn.jsdelivr" not in text
    assert "Top 5 category matrix" in text
