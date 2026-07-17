"""Closed outcome primitives for benchmark rows.

The old row format remains JSON-compatible, but aggregation converts each row into one of
these explicit outcomes before doing any math. This prevents harness/operator failures from
quietly becoming leaderboard numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Union


MODEL_FAILURE_KINDS = {"empty_output", "thinking_only", "truncated_no_answer"}
HARNESS_FAILURE_KINDS = {"harness_error"}
NOT_ATTEMPTED_KINDS = {
    "judge_off",
    "needs_judge",
    "operator_cap",
    "partial_coverage",
    "not_attempted",
}


@dataclass(frozen=True)
class Scored:
    """The model answered and the scorer graded it. This is the normal mean input."""

    value: float
    reason: str = ""


@dataclass(frozen=True)
class ModelFailed:
    """The model was asked and failed to deliver. Counts as 0.0 against the model."""

    kind: str
    reason: str = ""

    @property
    def value(self) -> float:
        return 0.0


@dataclass(frozen=True)
class NotAttempted:
    """The measurement was intentionally not taken. It has no numeric value."""

    kind: str
    actor: Literal["operator", "harness", "model"] = "operator"
    reason: str = ""


@dataclass(frozen=True)
class HarnessError:
    """The measurement machinery failed. Quality is unknown, not zero."""

    kind: str
    reason: str = ""


Outcome = Union[Scored, ModelFailed, NotAttempted, HarnessError]


def is_gradable(outcome: Outcome) -> bool:
    return isinstance(outcome, (Scored, ModelFailed))


def numeric_value(outcome: Outcome) -> Optional[float]:
    if isinstance(outcome, Scored):
        return float(outcome.value)
    if isinstance(outcome, ModelFailed):
        return 0.0
    return None


def row_to_outcome(row: Dict[str, Any]) -> Outcome:
    """Map the persisted JSON row shape to an explicit outcome.

    This is intentionally conservative. A numeric score is accepted only when no error_kind
    says the row is a model/harness failure. Missing scores become NotAttempted or
    HarnessError; they never become zero by accident.
    """
    reason = str(row.get("reason") or "")
    kind = str(row.get("error_kind") or "")
    if kind in HARNESS_FAILURE_KINDS or reason.startswith("HARNESS_ERROR"):
        return HarnessError(kind or "harness_error", reason)
    if kind in MODEL_FAILURE_KINDS:
        return ModelFailed(kind, reason)
    if row.get("category") == "long_context" and isinstance(row.get("needle_coverage"), (int, float)) and row.get("needle_coverage") < 1.0:
        return NotAttempted("partial_coverage", "operator", reason)
    score = row.get("score")
    if isinstance(score, (int, float)):
        return Scored(float(score), reason)
    if reason.startswith("raw only, judge off"):
        return NotAttempted("judge_off", "operator", reason)
    if reason == "needs judge" or reason.startswith("judge_error"):
        return NotAttempted("needs_judge", "operator", reason)
    if reason.startswith("no scored needle probes"):
        return NotAttempted("partial_coverage", "harness", reason)
    return NotAttempted("not_attempted", "harness", reason)


def category_score(outcomes: list[Outcome], weights: list[float]) -> tuple[Optional[float], float, Optional[str]]:
    """Return (score, coverage, blocker).

    Only Scored and ModelFailed have numeric values. HarnessError makes the category
    unknown. NotAttempted lowers coverage; partial coverage is not a category score.
    """
    if not outcomes:
        return None, 0.0, "no_outcomes"
    if any(isinstance(o, HarnessError) for o in outcomes):
        coverage = len([o for o in outcomes if is_gradable(o)]) / len(outcomes)
        return None, round(coverage, 4), "harness_error"
    gradable = [(o, weights[i] if i < len(weights) else 1.0) for i, o in enumerate(outcomes) if is_gradable(o)]
    coverage = len(gradable) / len(outcomes)
    if coverage < 1.0:
        return None, round(coverage, 4), "coverage_below_1"
    wsum = sum(w for _, w in gradable)
    if wsum <= 0:
        return None, 1.0, "zero_weight"
    score = sum(float(numeric_value(o) or 0.0) * w for o, w in gradable) / wsum
    return round(score, 2), 1.0, None
