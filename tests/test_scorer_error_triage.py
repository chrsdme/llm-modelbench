"""Regression coverage for the V9.5.18 scorer-triage release."""

import pytest

from llm_modelbench import scoring
from llm_modelbench.aggregate import aggregate
from llm_modelbench.tasks import TASKS


def test_empty_fenced_python_block_is_a_non_negative_zero_score():
    score, reason = scoring.score_python("```python\n```", {"checks": ["assert False"]})
    assert score == 0.0
    assert score >= 0.0
    assert "no scorable candidate" in reason


def test_empty_fenced_block_does_not_leak_best_over_blocks_sentinel():
    score, _ = scoring.best_over_blocks(
        "```python\n```", "python", lambda _: (-1.0, "sentinel"), include_raw=False,
    )
    assert score == 0.0
    assert score >= 0.0


def test_web_nav_empty_response_scores_zero():
    score, _ = scoring.score_web_nav("", {})
    assert score == 0.0


def test_web_nav_stripped_thinking_only_scores_zero():
    score, _ = scoring.score_web_nav("<think>unfinished scratchpad", {})
    assert score == 0.0


@pytest.mark.parametrize(
    "response",
    ["```html\n\n```", "```css\n\n```", "```html\n\n```\n```css\n\n```"],
)
def test_web_nav_empty_fenced_blocks_score_zero(response):
    score, _ = scoring.score_web_nav(response, {})
    assert score == 0.0


def test_zero_difficulty_tasks_are_excluded_from_discriminating_quality():
    gate_tasks = [task for task in TASKS if task.difficulty == 0.0]
    gate_ids = {task.id for task in gate_tasks}
    assert {"txt_sort", "txt_emails", "json_extract", "agent_plan", "git_conflict"} <= gate_ids

    rows = [
        {"model": "gate-only", "task": task.id, "category": task.category, "score": 100.0}
        for task in gate_tasks
    ]
    difficulty = {task.id: task.difficulty for task in TASKS}
    leaderboard, per_category = aggregate(rows, {}, difficulty)

    assert leaderboard[0]["quality"] is None
    assert per_category == {}
