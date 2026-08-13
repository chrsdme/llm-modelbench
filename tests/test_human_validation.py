"""Anvil Stage 3.0 schema freeze: human_validation_status / HumanCorrelation."""
import pytest

from llm_modelbench.human_validation import (
    HumanCorrelation,
    HumanValidationError,
    HumanValidationStatus,
)


def test_default_expectation_is_not_evaluated():
    """Not a constructor default (HumanValidationStatus is an enum with no
    default), but the master plan's requirement is that any field of this
    type default to NOT_EVALUATED wherever it's used as a field default --
    verify the member exists and is the expected sentinel value."""
    assert HumanValidationStatus.NOT_EVALUATED.value == "not_evaluated"


def test_all_three_statuses_present():
    assert {status.value for status in HumanValidationStatus} == {
        "not_evaluated", "provisional", "validated",
    }


def test_human_correlation_requires_positive_n():
    with pytest.raises(HumanValidationError):
        HumanCorrelation(
            metric="pearson_r", value=0.8, n=0, rubric_version="v1", evidence_ref="ref-1",
        )


def test_human_correlation_requires_all_text_fields():
    with pytest.raises(HumanValidationError):
        HumanCorrelation(metric="", value=0.8, n=10, rubric_version="v1", evidence_ref="ref-1")
    with pytest.raises(HumanValidationError):
        HumanCorrelation(metric="pearson_r", value=0.8, n=10, rubric_version="", evidence_ref="ref-1")
    with pytest.raises(HumanValidationError):
        HumanCorrelation(metric="pearson_r", value=0.8, n=10, rubric_version="v1", evidence_ref="")


def test_human_correlation_constructs_with_valid_fields():
    correlation = HumanCorrelation(
        metric="pearson_r", value=0.82, n=25, rubric_version="rubric-v1", evidence_ref="ref-1",
    )
    assert correlation.value == 0.82
    assert correlation.n == 25
