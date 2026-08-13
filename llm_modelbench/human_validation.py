"""Anvil master-plan v2.2 requirement: ``human_validation_status`` ships
now, defaulting to ``not_evaluated`` -- the full blind-grading-and-
correlation programme stays deferred as its own, separate project
dimension. Nothing in generated output (reports, CLI, routing/recommendation
UI) may imply human validation of a ranking or quality claim unless this
field is actually ``validated``.

Schema/type-contract freeze only (Anvil Stage 3.0). No consumer wiring is
introduced in this stage -- see ``local_only/anvil/stage-3.0-schema-freeze.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HumanValidationStatus(str, Enum):
    NOT_EVALUATED = "not_evaluated"
    PROVISIONAL = "provisional"
    VALIDATED = "validated"


class HumanValidationError(ValueError):
    pass


@dataclass(frozen=True)
class HumanCorrelation:
    """Optional; only meaningful once a status is at least ``PROVISIONAL``.
    Not itself a validation status -- a correlation record without a
    corresponding ``VALIDATED``/``PROVISIONAL`` status does not imply
    validation on its own."""

    metric: str
    value: float
    n: int
    rubric_version: str
    evidence_ref: str

    def __post_init__(self) -> None:
        if not self.metric:
            raise HumanValidationError("metric is required")
        if self.n <= 0:
            raise HumanValidationError("n must be positive")
        if not self.rubric_version:
            raise HumanValidationError("rubric_version is required")
        if not self.evidence_ref:
            raise HumanValidationError("evidence_ref is required")
