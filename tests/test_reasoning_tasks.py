"""Regression and adversarial tests for the reasoning sanity-check category."""
from llm_modelbench import scoring
from llm_modelbench.config import Config, DEFAULT_WEIGHTS
from llm_modelbench.ollama import MockClient
from llm_modelbench.runner import _run_once
from llm_modelbench.tasks import TASKS, tasks_for

REASONING_TASK_IDS = {
    "reasoning_bridge_crossing",
    "reasoning_poisoned_wine",
    "reasoning_birthday_twins",
    "reasoning_monty_hall",
    "reasoning_wolf_goat_cabbage",
}

CASES = [
    ("reasoning_bridge_crossing", "17 minutes", "14 minutes"),
    ("reasoning_poisoned_wine", "binary-coded prisoner testing", "every prisoner tastes every bottle"),
    ("reasoning_birthday_twins", "1 pair", "23 pairs"),
    ("reasoning_monty_hall", "switch to door 2, 2/3 chance", "stick with door 1, 1/2 chance"),
    ("reasoning_wolf_goat_cabbage", "7 crossings", "9 crossings"),
]


def _task(task_id):
    return next(task for task in TASKS if task.id == task_id)


def _score(task_id, response):
    task = _task(task_id)
    return scoring.DETERMINISTIC[task.scorer](response, task.meta)[0]


def test_all_five_reasoning_tasks_are_registered_with_real_difficulty():
    ids = {task.id for task in TASKS if task.category == "reasoning"}
    assert ids == REASONING_TASK_IDS
    for task_id in REASONING_TASK_IDS:
        task = _task(task_id)
        assert task.family == "text"
        assert task.difficulty > 0.0
        assert task.scorer == "exact"


def test_reasoning_tasks_are_reachable_at_short_and_full_levels():
    short_ids = {task.id for task in tasks_for("short", ["reasoning"], ["text"])}
    full_ids = {task.id for task in tasks_for("full", ["reasoning"], ["text"])}
    assert short_ids == REASONING_TASK_IDS
    assert REASONING_TASK_IDS <= full_ids
    assert tasks_for("smoke", ["reasoning"], ["text"]) == []


def test_reasoning_category_has_a_nonzero_weight():
    assert DEFAULT_WEIGHTS.get("reasoning", 0.0) > 0.0


def test_exact_visible_answer_contract_rejects_mixed_or_missing_answers():
    for task_id, correct, wrong in CASES:
        assert _score(task_id, correct) == 100.0
        assert _score(task_id, wrong) == 0.0
        assert _score(task_id, f"I considered {correct}, but my final answer is {wrong}.") == 0.0
        assert _score(task_id, f"I considered {wrong}.\n{correct}") == 0.0
        assert _score(task_id, f"<think>{correct}</think>{wrong}") == 0.0
        assert _score(task_id, f"<think>{wrong}</think>{correct}") == 100.0
        assert _score(task_id, f"{correct} or {wrong}") == 0.0
        assert _score(task_id, "") == 0.0


def test_mock_run_scores_100_on_all_five_reasoning_tasks():
    cfg = Config()
    client = MockClient()
    for task_id in REASONING_TASK_IDS:
        result = _run_once(client, cfg, "qwen2.5-coder:14b", _task(task_id))
        assert result["score"] == 100.0, (task_id, result)
