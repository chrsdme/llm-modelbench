from types import SimpleNamespace

import pytest

from llm_modelbench.cli import _host_code_tasks, _require_host_code_opt_in


def _plan(*tasks):
    return {"active_models": [{"model": "m", "tasks": list(tasks)}]}


def test_host_code_gate_detects_executable_scorers():
    assert _host_code_tasks(_plan("py_anagram", "ocr_form_code", "js_debounce")) == [
        "js_debounce",
        "py_anagram",
    ]


def test_host_code_gate_fails_closed_without_explicit_opt_in():
    with pytest.raises(SystemExit, match="allow-host-code-execution"):
        _require_host_code_opt_in(SimpleNamespace(allow_host_code_execution=False), _plan("py_anagram"))


def test_host_code_gate_allows_non_executable_or_explicit_runs():
    _require_host_code_opt_in(SimpleNamespace(allow_host_code_execution=False), _plan("ocr_form_code"))
    _require_host_code_opt_in(SimpleNamespace(allow_host_code_execution=True), _plan("py_anagram"))
