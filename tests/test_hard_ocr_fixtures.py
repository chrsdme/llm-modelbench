from pathlib import Path

from llm_modelbench import media, scoring
from llm_modelbench.tasks import TASKS


HARD_FIXTURE_IDS = {
    "ocr_receipt_total": ("18.47", "14.95"),
    "ocr_table_cell": ("H7-42", "H7-24"),
    "ocr_form_code": ("ZX-7Q/19.A", "ZX-7Q/19-A"),
    "ocr_noisy_label": ("LOT-42B", "LOT-428"),
}


def _tasks_by_id():
    return {task.id: task for task in TASKS}


def test_hard_ocr_fixture_assets_load_from_repo_root():
    tasks = _tasks_by_id()
    for task_id in HARD_FIXTURE_IDS:
        task = tasks[task_id]
        image_path = task.meta["image_path"]
        payload = media.load_image_file(image_path)

        assert Path(payload["path"]).is_file()
        assert payload["mime_type"] == "image/png"
        assert payload["data"]


def test_hard_ocr_tasks_have_nonzero_exact_answers_and_reject_distractors():
    tasks = _tasks_by_id()
    for task_id, (correct, wrong) in HARD_FIXTURE_IDS.items():
        task = tasks[task_id]
        scorer = scoring.DETERMINISTIC[task.scorer]

        assert task.family == "vision"
        assert task.meta["image_path"]
        assert task.meta["expected"] == correct
        assert task.difficulty > 0.0
        assert scorer(correct + "\n", task.meta)[0] == 100.0
        assert scorer(wrong, task.meta)[0] == 0.0


def test_existing_synthetic_ocr_tasks_remain_without_image_paths():
    tasks = _tasks_by_id()
    for task_id in ("ocr_invoice", "ocr_noisy"):
        assert "reference" in tasks[task_id].meta
        assert "image_path" not in tasks[task_id].meta
