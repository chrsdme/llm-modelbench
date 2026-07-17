import pytest

from llm_modelbench import scoring
from llm_modelbench.tasks import TASKS


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("H7-42", "H7-42"),
        ("H7 - 42", "H7-42"),
        ("LOT - 42B", "LOT-42B"),
        ("ZX-7Q/19 .A", "ZX-7Q/19.A"),
    ],
)
def test_exact_code_accepts_only_punctuation_spacing_variants(response, expected):
    assert scoring.score_exact_code(response, {"expected": expected})[0] == 100.0


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("42B", "LOT-42B"),
        ("The Bin value in the row marked HOLD is H7-42.", "H7-42"),
        ("H742", "H7-42"),
        ("LOT 42B", "LOT-42B"),
    ],
)
def test_exact_code_rejects_substrings_prose_and_missing_punctuation(response, expected):
    assert scoring.score_exact_code(response, {"expected": expected})[0] == 0.0


def test_hard_ocr_tasks_use_exact_code_and_legacy_tasks_do_not_change():
    tasks = {task.id: task for task in TASKS}
    hard_ids = {"ocr_receipt_total", "ocr_table_cell", "ocr_form_code", "ocr_noisy_label"}

    assert {tasks[task_id].scorer for task_id in hard_ids} == {"exact_code"}
    assert tasks["ocr_invoice"].scorer == "ocr"
    assert tasks["ocr_noisy"].scorer == "ocr"
    assert tasks["pdf_text"].scorer == "ocr"


@pytest.mark.parametrize(
    ("task_id", "spaced_output", "wrong_output"),
    [
        ("ocr_receipt_total", "18.47", "14.95"),
        ("ocr_table_cell", "H7 - 42", "H7-24"),
        ("ocr_form_code", "ZX-7Q/19 .A", "ZX-7Q/19-A"),
        ("ocr_noisy_label", "LOT - 42B", "LOT-428"),
    ],
)
def test_hard_ocr_metadata_accepts_known_spacing_and_rejects_real_misses(task_id, spaced_output, wrong_output):
    task = next(task for task in TASKS if task.id == task_id)
    scorer = scoring.DETERMINISTIC[task.scorer]

    assert scorer(task.meta["expected"], task.meta)[0] == 100.0
    assert scorer(spaced_output, task.meta)[0] == 100.0
    assert scorer(wrong_output, task.meta)[0] == 0.0
