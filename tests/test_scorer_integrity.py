from llm_modelbench import scoring, judge
from llm_modelbench.progress import classify_row
from llm_modelbench.aggregate import aggregate


def test_web_nav_accepts_spacing_and_reports_missing():
    ok = """<nav></nav><style>nav { display : flex } @media (max-width: 600 px){ nav { flex-direction: column; } }</style>"""
    score, reason = scoring.score_web_nav(ok, {})
    assert score == 100.0
    bad_score, bad_reason = scoring.score_web_nav("<div>menu</div>", {})
    assert bad_score < 100.0
    assert "missing:" in bad_reason


def test_empty_output_is_error_event():
    row = {"model":"m", "task":"t", "score":None, "reason":"ERROR_EMPTY_OUTPUT: model returned zero visible characters", "error_kind":"empty_output"}
    ev = classify_row(row)
    assert ev and ev["kind"] == "ERROR"


def test_raw_subjective_not_aggregate_error():
    rows = [{"model":"m", "task":"wr_rag", "category":"tech_writing", "score":None, "reason":"raw only, judge off: 100 chars"}]
    lb, _ = aggregate(rows, {"tech_writing": 1.0}, {"wr_rag": 1.0})
    assert lb[0]["err"] == 0


def test_judge_json_parse_and_invalid_no_fake_50():
    score, reason = judge._parse_score('{"score": 88, "confidence": 0.7, "verdict": "good"}')
    assert score == 88
    assert "judge_json" in reason
    bad_score, bad_reason = judge._parse_score('criterion 1 is okay but no json')
    assert bad_score is None
    assert bad_reason.startswith('judge_error')
