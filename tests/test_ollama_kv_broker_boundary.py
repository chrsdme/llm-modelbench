"""Static, offline regression checks for the privileged broker's closed API."""
import json
import importlib.machinery
import importlib.util
import subprocess
from pathlib import Path

import pytest

from llm_modelbench.ollama_service import BrokerOllamaServiceController, ServiceControlError


def source(): return Path("scripts/libexec/llmb-ollama-kv-control").read_text(encoding="utf-8")


def broker_module():
    loader = importlib.machinery.SourceFileLoader("llmb_kv_broker_test", "scripts/libexec/llmb-ollama-kv-control")
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


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


def test_recover_restores_before_requiring_listener(monkeypatch):
    broker = broker_module()
    token = "t" * 32
    state = {"unit": "ollama.service", "owner": {"uid": "1000", "user": "a", "direct_root": "false"}, "state": "failed", "original": None, "last_hash": None}
    calls = []
    monkeypatch.setattr(broker, "_read_state", lambda value: state)
    monkeypatch.setattr(broker, "_caller", lambda: {"uid": "root", "user": "root", "direct_root": "true"})
    monkeypatch.setattr(broker, "_lock_matches", lambda *args: calls.append("lock"))
    monkeypatch.setattr(broker, "_recovery_target", lambda value: calls.append("target"))
    monkeypatch.setattr(broker, "_assert_hash", lambda value: calls.append("hash"))
    monkeypatch.setattr(broker, "_apply_file", lambda value, data: calls.append("file"))
    monkeypatch.setattr(broker, "_run", lambda *args, **kwargs: calls.append("systemctl") or subprocess.CompletedProcess(args[0], 0, "", ""))
    monkeypatch.setattr(broker, "_revalidate", lambda value: calls.append("listener") or ("ollama.service", 1))
    monkeypatch.setattr(broker, "_write_state", lambda *args: None)
    monkeypatch.setattr(broker, "_finalize", lambda *args: calls.append("finalize"))
    assert broker._recover(token)["restored"] is True
    assert calls.index("file") < calls.index("listener") < calls.index("finalize")


def test_direct_root_cannot_set_another_users_transaction(monkeypatch):
    broker = broker_module()
    state = {"unit": "ollama.service", "owner": {"uid": "1000", "user": "a", "direct_root": "false"}}
    monkeypatch.setattr(broker, "_read_state", lambda token: state)
    monkeypatch.setattr(broker, "_caller", lambda: {"uid": "root", "user": "root", "direct_root": "true"})
    with pytest.raises(broker.BrokerError, match="another sudo identity"):
        broker._set("t" * 32, "q4_0")


@pytest.mark.parametrize("environment", ["OLLAMA_KV_CACHE_TYPE=q4_0", '"OLLAMA_KV_CACHE_TYPE=q8_0"'])
def test_quoted_and_unquoted_systemd_environment_are_parsed(monkeypatch, environment):
    broker = broker_module()
    state = {"unit": "ollama.service", "port": 1}
    monkeypatch.setattr(broker, "_run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, environment, ""))
    monkeypatch.setattr(broker.Path, "read_bytes", lambda _self: b"OLLAMA_KV_CACHE_TYPE=q8_0\0")
    merged, live = broker._observed(state, 1)
    assert merged in {"q4_0", "q8_0"}
    assert live == "q8_0"


@pytest.mark.parametrize("argv", [["unknown"], ["begin", "--port", "0"], ["set", "--transaction", "bad", "--kv", "q6_0"], ["version", "extra"]])
def test_malformed_protocol_is_bounded_json(argv, monkeypatch, capsys):
    broker = broker_module()
    monkeypatch.setattr("sys.argv", ["broker", *argv])
    assert broker.main() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and payload["protocol"] == 1 and payload["error_code"] == "invalid_request"
