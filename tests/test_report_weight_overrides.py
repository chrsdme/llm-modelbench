import pytest

from llm_modelbench.weights_override import copy_run_for_override, parse_weight_overrides


DEFAULTS = {"coding_python": 0.2, "agentic_tool": 0.3}


def test_weight_overrides_merge_defaults_and_allow_empty():
    assert parse_weight_overrides(None, DEFAULTS) == DEFAULTS
    assert parse_weight_overrides("coding_python=0.4", DEFAULTS) == {"coding_python": 0.4, "agentic_tool": 0.3}


@pytest.mark.parametrize("spec", ["typo=0.2", "coding_python", "coding_python=bad"])
def test_weight_overrides_reject_unknown_or_malformed_values(spec):
    with pytest.raises(ValueError):
        parse_weight_overrides(spec, DEFAULTS)


def test_override_report_copy_preserves_source_artifacts(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "override"
    source.mkdir()
    (source / "raw_results.jsonl").write_text("immutable")

    copy_run_for_override(source, destination)
    (destination / "report.html").write_text("override")

    assert (source / "raw_results.jsonl").read_text() == "immutable"
    assert not (source / "report.html").exists()
