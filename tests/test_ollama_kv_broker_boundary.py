"""Static, offline regression checks for the privileged broker's closed API."""
import json
import subprocess
from pathlib import Path

import pytest

from llm_modelbench.ollama_service import BrokerOllamaServiceController, ServiceControlError


def source(): return Path("scripts/libexec/llmb-ollama-kv-control").read_text(encoding="utf-8")


@pytest.mark.parametrize("needle", [
    "#!/usr/bin/python3 -I", "PROTOCOL_VERSION = 1", "KV_VALUES = {\"q4_0\", \"q8_0\"}",
    "SS = \"/usr/bin/ss\"", "SYSTEMCTL = \"/usr/bin/systemctl\"", "SAFE_ENV =",
    "secrets.token_urlsafe", "O_EXCL", "O_NOFOLLOW", "external_modification",
    "owner_mismatch", "transaction_locked", "rollback_failed", "_revalidate", "_assert_hash",
])
def test_broker_has_required_closed_boundary_invariants(needle): assert needle in source()


@pytest.mark.parametrize("forbidden", ["shell=True", "os.system", "#!/usr/bin/env python", "import llm_modelbench", "sys.path"])
def test_broker_has_no_dynamic_or_project_execution(forbidden): assert forbidden not in source()


@pytest.mark.parametrize("needle", [
    "def _caller", "SUDO_UID", "SUDO_USER", "def _lock_path", "O_EXCL",
    "def _rollback", "state[\"state\"] = \"applying\"", "\"state\": \"applied\"",
    "state[\"state\"] = \"rollback_failed\"", "Path(f\"/proc/{pid}/environ\")",
])
def test_broker_security_state_machine_and_owner_binding_are_present(needle): assert needle in source()


@pytest.mark.parametrize("value", ["q6_0", "q8_0\\n[Service]", "q8_0;restart", "", " q4_0"])
def test_kv_input_is_closed_enum(value):
    assert value not in {"q4_0", "q8_0"}
    assert "choices=sorted(KV_VALUES)" in source()


def test_protocol_has_no_path_unit_fragment_or_command_arguments():
    text = source()
    assert "add_argument(\"--port\"" in text and "add_argument(\"--transaction\"" in text and "add_argument(\"--kv\"" in text
    for prohibited in ("--unit", "--path", "--content", "--command", "--environment"):
        assert prohibited not in text


def test_public_sudoers_policy_grants_only_broker():
    text = Path("docs/auto_confirm_sudoers.md").read_text(encoding="utf-8")
    assert "NOPASSWD: /usr/local/libexec/llmb-ollama-kv-control" in text
    for command in ("NOPASSWD: /usr/bin/install", "NOPASSWD: /usr/bin/systemctl", "NOPASSWD: /usr/bin/rm", "NOPASSWD: /usr/bin/cat", "llmb-read-kv-env.sh"):
        assert command not in text


def test_auto_confirm_client_uses_only_sudo_n_broker():
    calls = []
    def run(argv, **kwargs):
        calls.append(argv); return subprocess.CompletedProcess(argv, 0, json.dumps({"protocol": 1, "ok": True, "operation": "version"}), "")
    BrokerOllamaServiceController(auto_confirm=True, run=run).verify_noninteractive_sudo_ready()
    assert calls == [["sudo", "-n", "/usr/local/libexec/llmb-ollama-kv-control", "version"]]


def test_protocol_mismatch_fails_clearly():
    def run(argv, **kwargs): return subprocess.CompletedProcess(argv, 0, json.dumps({"protocol": 99, "ok": True}), "")
    with pytest.raises(ServiceControlError, match="incompatible"):
        BrokerOllamaServiceController(auto_confirm=True, run=run).verify_noninteractive_sudo_ready()


def test_old_reader_and_old_generic_pending_path_are_not_tracked():
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    assert "scripts/libexec/llmb-read-kv-env.sh" not in tracked
    marker = "llmb-ollama-kv-" + "pending-"
    assert not any(marker in Path(path).read_text(errors="ignore") for path in tracked if Path(path).is_file())
