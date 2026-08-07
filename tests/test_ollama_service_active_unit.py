"""Regression tests for the unprivileged broker client boundary."""
import json
import subprocess

import pytest

from llm_modelbench.ollama_service import BrokerOllamaServiceController, OllamaServiceController, ServiceControlError


def _reply(operation, **extra):
    value = {"protocol": 1, "ok": True, "operation": operation, "transaction": None, "state": "active", "unit": "ollama.service", "kv": None, "observed_kv": None, "error_code": None, "message": ""}
    value.update(extra)
    return json.dumps(value)


def test_compatibility_name_is_safe_broker_facade():
    assert OllamaServiceController is BrokerOllamaServiceController


def test_client_never_sends_display_unit_to_broker():
    calls = []
    def run(argv, **kwargs):
        calls.append(argv); return subprocess.CompletedProcess(argv, 0, _reply("begin", transaction="a" * 32, unit="ollama-real.service"), "")
    controller = OllamaServiceController("untrusted.service", run=run)
    active = controller.verify_owns_live_process()
    assert active.unit == "ollama-real.service"
    assert "untrusted.service" not in calls[0]


def test_interactive_confirmation_is_retained():
    controller = OllamaServiceController(input_fn=lambda _: "NO", isatty_fn=lambda: True)
    with pytest.raises(ServiceControlError, match="declined"):
        controller.confirm("apply", "test")


def test_interactive_confirmation_accepts_keyword():
    OllamaServiceController(input_fn=lambda _: "RESTART", isatty_fn=lambda: True).confirm("apply", "test")
