"""Offline checks for the semantic privileged KV broker boundary."""
import json
import subprocess
from pathlib import Path

import pytest

from llm_modelbench.ollama_service import BrokerOllamaServiceController, ServiceControlError


def _broker_source() -> str:
    return Path("scripts/libexec/llmb-ollama-kv-control").read_text(encoding="utf-8")


def test_broker_exposes_only_closed_kv_protocol_and_no_user_temp_transfer():
    text = _broker_source()
    assert 'KV_VALUES = {"q4_0", "q8_0"}' in text
    assert "--transaction" in text and "--kv" in text and "--port" in text
    assert "_atomic_dropin" in text and "secrets.token_urlsafe" in text
    assert "llmb-ollama-kv-" + "pending-" not in text


@pytest.mark.parametrize("value", ["q6_0", "q8_0\n[Service]", "q8_0;restart", ""])
def test_broker_rejects_nonsemantic_kv_inputs(value):
    text = _broker_source()
    assert "if args.kv not in KV_VALUES" in text
    assert value not in {"q4_0", "q8_0"}


def test_public_sudoers_policy_grants_only_broker():
    text = Path("docs/auto_confirm_sudoers.md").read_text(encoding="utf-8")
    assert "NOPASSWD: /usr/local/libexec/llmb-ollama-kv-control" in text
    for command in ("NOPASSWD: /usr/bin/install", "NOPASSWD: /usr/bin/systemctl", "NOPASSWD: /usr/bin/rm", "NOPASSWD: /usr/bin/cat"):
        assert command not in text


def test_auto_confirm_client_uses_only_sudo_n_broker():
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, json.dumps({"protocol": 1}), "")
    controller = BrokerOllamaServiceController(auto_confirm=True, run=run)
    controller.verify_noninteractive_sudo_ready()
    assert calls == [["sudo", "-n", "/usr/local/libexec/llmb-ollama-kv-control", "version"]]


def test_protocol_mismatch_and_arbitrary_unit_fail_closed():
    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, json.dumps({"protocol": 99}), "")
    controller = BrokerOllamaServiceController("not-accepted.service", auto_confirm=True, run=run)
    with pytest.raises(ServiceControlError, match="incompatible"):
        controller.verify_noninteractive_sudo_ready()


def test_old_reader_and_predictable_pending_file_are_not_tracked():
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    assert "scripts/libexec/llmb-read-kv-env.sh" not in tracked
    marker = "llmb-ollama-kv-" + "pending-"
    assert not any(marker in Path(path).read_text(errors="ignore") for path in tracked if Path(path).is_file())
