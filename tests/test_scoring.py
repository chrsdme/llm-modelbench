"""Pytest wrapper. Runs the offline self-test and a few direct scorer checks.
Run: pytest -q
"""
from llm_modelbench import scoring, selftest
from llm_modelbench.aggregate import aggregate


def test_selftest_passes():
    assert selftest.run() == 0


def test_ocr_edit_distance():
    assert scoring.score_ocr("hello world", {"reference": "hello world"})[0] == 100.0
    partial = scoring.score_ocr("hello wodld", {"reference": "hello world"})[0]
    assert 0 < partial < 100


def test_quality_is_not_speed_contaminated():
    rows = [
        {"model": "slow", "task": "t", "category": "coding_python", "score": 100.0, "tps": 3.0},
        {"model": "fast", "task": "t", "category": "coding_python", "score": 100.0, "tps": 90.0},
    ]
    lb, _ = aggregate(rows, {"coding_python": 1.0}, {"t": 1.0})
    q = {r["model"]: r["quality"] for r in lb}
    assert q["slow"] == q["fast"]


def test_json_schema_accepts_fenced_valid_json():
    clean = scoring.score_json_schema('{"a":1,"b":2}', {"required_keys": ["a", "b"]})[0]
    fenced = scoring.score_json_schema('```json\n{"a":1,"b":2}\n```', {"required_keys": ["a", "b"]})[0]
    assert clean == 100.0 and fenced == 100.0


def test_thinking_tags_do_not_break_code_or_exact_scoring():
    resp = '<think>wrong scratchpad</think>```python\ndef dedupe(seq):\n    return list(dict.fromkeys(seq))\n```'
    score, _ = scoring.score_python(resp, {"checks": ["assert dedupe([1,1,2])==[1,2]"]})
    assert score == 100.0
    assert scoring.score_exact('<think>Paris maybe</think>DONE', {"expected": "DONE"})[0] == 100.0

def test_score_exact_strips_only_trailing_sentence_punctuation():
    """Harmless sentence endings are tolerated without restoring substring credit."""
    expected = {"expected": "switch to door 2, 2/3 chance"}

    for response in (
        "switch to door 2, 2/3 chance.",
        "Switch to door 2, 2/3 chance!",
        "  switch to door 2, 2/3 chance...  ",
        "switch to door 2, 2/3 chance…",
        "switch to door 2, 2/3 chance",
    ):
        assert scoring.score_exact(response, expected)[0] == 100.0

    assert scoring.score_exact("stick with door 1, 1/2 chance", expected)[0] == 0.0
    adversarial = (
        "I considered switch to door 2, 2/3 chance, but my final answer "
        "is stick with door 1, 1/2 chance."
    )
    assert scoring.score_exact(adversarial, expected)[0] == 0.0

    # Punctuation that can carry structure or meaning remains significant.
    assert scoring.score_exact("answer,", {"expected": "answer"})[0] == 0.0
    assert scoring.score_exact("answer;", {"expected": "answer"})[0] == 0.0
    assert scoring.score_exact("answer:", {"expected": "answer"})[0] == 0.0
    assert scoring.score_exact("H7-42", {"expected": "H7-42"})[0] == 100.0
    assert scoring.score_exact("H7-42", {"expected": "H7-43"})[0] == 0.0

