"""Regression coverage for active Ollama service discovery and safe KV control."""
from pathlib import Path

import pytest

from llm_modelbench import ollama_service
from llm_modelbench.ollama_service import (
    OllamaServiceController,
    ServiceControlError,
    discover_active_service,
    discover_active_unit,
)


class Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _real_host_shape(argv, **kwargs):
    joined = " ".join(argv)
    if "ss" in argv and "-tlnp" in argv:
        return Result(
            stdout='LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* '
            'users:(("ollama",pid=2119,fd=4))\n'
        )
    if "list-units" in joined:
        return Result(
            stdout=(
                "ollama-gpu0.service loaded active running Ollama GPU0\n"
                "ollama.service loaded activating auto-restart Ollama Service\n"
            )
        )
    if "show" in argv and "ollama-gpu0.service" in argv and "--property=MainPID" in joined:
        return Result(stdout="2119\n")
    if "show" in argv and "ollama.service" in argv and "--property=MainPID" in joined:
        return Result(stdout="0\n")
    return Result()


def test_discover_active_service_finds_real_port_owner():
    active = discover_active_service(run=_real_host_shape)
    assert active.unit == "ollama-gpu0.service"
    assert active.pid == 2119
    assert active.port == 11434
    assert discover_active_unit(run=_real_host_shape) == "ollama-gpu0.service"


def test_discover_active_service_ignores_remote_port_match():
    def fake_run(argv, **kwargs):
        joined = " ".join(argv)
        if "ss" in argv:
            return Result(
                stdout=(
                    'ESTAB 0 0 127.0.0.1:45000 127.0.0.1:11434 users:(("client",pid=88,fd=3))\n'
                    'LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=2119,fd=4))\n'
                )
            )
        if "list-units" in joined:
            return Result(stdout="ollama-gpu0.service loaded active running Ollama\n")
        if "show" in argv:
            return Result(stdout="2119\n")
        return Result()
    assert discover_active_unit(run=fake_run) == "ollama-gpu0.service"


def test_discover_active_service_refuses_no_listener():
    def fake_run(argv, **kwargs):
        if "ss" in argv:
            return Result(stdout="")
        return Result()
    with pytest.raises(ServiceControlError, match="no process"):
        discover_active_service(run=fake_run)


def test_discover_active_service_refuses_no_unit_match():
    def fake_run(argv, **kwargs):
        if "ss" in argv:
            return Result(stdout='LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=9999,fd=4))\n')
        if "list-units" in " ".join(argv):
            return Result(stdout="ollama.service loaded active running Ollama\n")
        if "show" in argv:
            return Result(stdout="1234\n")
        return Result()
    with pytest.raises(ServiceControlError, match="refusing to guess"):
        discover_active_service(run=fake_run)


def test_discover_active_service_refuses_ambiguous_units():
    def fake_run(argv, **kwargs):
        if "ss" in argv:
            return Result(stdout='LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=2119,fd=4))\n')
        if "list-units" in " ".join(argv):
            return Result(stdout=(
                "ollama-a.service loaded active running A\n"
                "ollama-b.service loaded active running B\n"
            ))
        if "show" in argv:
            return Result(stdout="2119\n")
        return Result()
    with pytest.raises(ServiceControlError, match="multiple systemd units"):
        discover_active_service(run=fake_run)


def test_explicit_wrong_unit_is_rejected(monkeypatch):
    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    # sleep_fn is a no-op here: this scenario is genuinely, permanently
    # dormant (not a transient post-restart blip), so it still correctly
    # exhausts the retry window and raises -- it just shouldn't cost 3 real
    # wall-clock seconds in the test suite to prove that.
    controller = OllamaServiceController(
        "ollama.service", run=_real_host_shape, sleep_fn=lambda _: None,
    )
    with pytest.raises(ServiceControlError, match="no running MainPID"):
        controller.verify_owns_live_process()


def test_explicit_matching_unit_is_accepted(monkeypatch):
    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = OllamaServiceController("ollama-gpu0.service", run=_real_host_shape)
    active = controller.verify_owns_live_process()
    assert active.unit == "ollama-gpu0.service"


def test_for_active_service_wires_discovered_unit():
    controller = OllamaServiceController.for_active_service(
        run=_real_host_shape,
        force_password_prompt=False,
    )
    assert controller.unit == "ollama-gpu0.service"
    assert controller.port == 11434


def test_effective_environment_conflict_aborts_before_restart(monkeypatch):
    restart_calls = []

    def fake_run(argv, **kwargs):
        joined = " ".join(argv)
        if "restart" in argv:
            restart_calls.append(argv)
            return Result()
        if "daemon-reload" in argv:
            return Result()
        if "--property=Environment" in joined:
            return Result(stdout="OLLAMA_KV_CACHE_TYPE=q4_0\n")
        if "--property=DropInPaths" in joined:
            return Result(stdout=(
                "/etc/systemd/system/ollama.service.d/zzzz-llmb-repair-kv.conf "
                "/etc/systemd/system/ollama.service.d/zzzzz-override.conf\n"
            ))
        if argv[:2] == ["sudo", "cat"] and argv[-1].endswith("zzzzz-override.conf"):
            return Result(stdout=b'[Service]\nEnvironment="OLLAMA_KV_CACHE_TYPE=q4_0"\n')
        return Result()

    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = OllamaServiceController("ollama.service", run=fake_run)
    with pytest.raises(ServiceControlError, match="zzzzz-override.conf"):
        controller.verify_effective_environment("q8_0")
    assert restart_calls == []


def test_effective_environment_accepts_requested_value(monkeypatch):
    def fake_run(argv, **kwargs):
        if "--property=Environment" in " ".join(argv):
            return Result(stdout="OLLAMA_KV_CACHE_TYPE=q8_0\n")
        return Result()
    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = OllamaServiceController("ollama.service", run=fake_run)
    controller.verify_effective_environment("q8_0")


def test_stale_cuda_uuid_blocks_restart(monkeypatch):
    def fake_run(argv, **kwargs):
        if "--property=Environment" in " ".join(argv):
            return Result(stdout="CUDA_VISIBLE_DEVICES=GPU-old\n")
        if argv and argv[0] == "nvidia-smi":
            return Result(stdout="GPU-current-a\nGPU-current-b\n")
        return Result()
    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = OllamaServiceController("ollama-gpu0.service", run=fake_run)
    with pytest.raises(ServiceControlError, match="stale CUDA_VISIBLE_DEVICES"):
        controller.verify_gpu_binding()


def test_valid_cuda_uuid_is_accepted(monkeypatch):
    def fake_run(argv, **kwargs):
        if "--property=Environment" in " ".join(argv):
            return Result(stdout="CUDA_VISIBLE_DEVICES=GPU-current-a\n")
        if argv and argv[0] == "nvidia-smi":
            return Result(stdout="GPU-current-a\nGPU-current-b\n")
        return Result()
    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = OllamaServiceController("ollama-gpu0.service", run=fake_run)
    assert controller.verify_gpu_binding() is None


def test_set_kv_rechecks_listener_after_restart(monkeypatch, tmp_path):
    state = {"kv": None, "pid": 2119}

    def fake_run(argv, **kwargs):
        joined = " ".join(argv)
        if "ss" in argv and "-H" in argv:
            return Result(stdout=f'LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid={state["pid"]},fd=4))\n')
        if "list-units" in joined:
            return Result(stdout="ollama-gpu0.service loaded active running Ollama\n")
        if "--property=MainPID" in joined:
            return Result(stdout="2119\n")
        if "--property=Environment" in joined:
            suffix = f" OLLAMA_KV_CACHE_TYPE={state['kv']}" if state["kv"] else ""
            return Result(stdout=f"CUDA_VISIBLE_DEVICES=0{suffix}\n")
        if "--property=DropInPaths" in joined:
            return Result(stdout="/etc/systemd/system/ollama-gpu0.service.d/zzzz-llmb-repair-kv.conf\n")
        if argv[:3] == ["sudo", "test", "-e"]:
            return Result(returncode=1)
        if "install" in argv and argv[-1].endswith("zzzz-llmb-repair-kv.conf"):
            content = Path(argv[-2]).read_text()
            state["kv"] = "q8_0" if "q8_0" in content else "q4_0"
            return Result()
        if "restart" in argv:
            state["pid"] = 9999  # a different process takes the port
            return Result()
        if "is-active" in argv:
            return Result(returncode=0)
        return Result()

    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = OllamaServiceController(
        "ollama-gpu0.service", run=fake_run, sleep_fn=lambda _: None,
    )
    with pytest.raises(ServiceControlError, match="does not own the live Ollama process"):
        controller.set_kv_type("q8_0", phase="q8_0")


def test_set_kv_waits_until_active_unit_binds_port(monkeypatch):
    """systemd can report active before Ollama has created its listener."""
    state = {"kv": None, "restarted": False, "post_restart_ss": 0}

    def fake_run(argv, **kwargs):
        joined = " ".join(argv)
        if "ss" in argv and "-H" in argv:
            if not state["restarted"]:
                return Result(stdout='LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=2119,fd=4))\n')
            state["post_restart_ss"] += 1
            if state["post_restart_ss"] < 3:
                return Result(stdout="")
            return Result(stdout='LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=3000,fd=4))\n')
        if "list-units" in joined:
            return Result(stdout="ollama-gpu0.service loaded active running Ollama\n")
        if "--property=MainPID" in joined:
            return Result(stdout=("3000\n" if state["restarted"] else "2119\n"))
        if "--property=Environment" in joined:
            suffix = f" OLLAMA_KV_CACHE_TYPE={state['kv']}" if state["kv"] else ""
            return Result(stdout=f"CUDA_VISIBLE_DEVICES=0{suffix}\n")
        if "--property=DropInPaths" in joined:
            return Result(stdout="/etc/systemd/system/ollama-gpu0.service.d/zzzz-llmb-repair-kv.conf\n")
        if argv[:3] == ["sudo", "test", "-e"]:
            return Result(returncode=1)
        if "install" in argv and argv[-1].endswith("zzzz-llmb-repair-kv.conf"):
            content = Path(argv[-2]).read_text()
            state["kv"] = "q8_0" if "q8_0" in content else "q4_0"
            return Result()
        if "restart" in argv:
            state["restarted"] = True
            return Result()
        if "is-active" in argv:
            return Result(returncode=0)
        if argv[:2] == ["sudo", "sh"]:
            return Result(stdout=f"OLLAMA_KV_CACHE_TYPE={state['kv']}\n")
        return Result()

    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = OllamaServiceController(
        "ollama-gpu0.service", run=fake_run, sleep_fn=lambda _: None,
        kv_read_helper_exists_fn=lambda p: False,
    )
    phase = controller.set_kv_type("q8_0", phase="q8_0")
    assert phase.verified is True
    assert state["post_restart_ss"] == 3


def test_restore_waits_until_listener_returns(monkeypatch):
    """The error-recovery restart uses the same socket-readiness barrier."""
    state = {"restart_count": 0, "ss_after_restore": 0}

    def fake_run(argv, **kwargs):
        joined = " ".join(argv)
        if argv[:3] == ["sudo", "test", "-e"]:
            return Result(returncode=1)
        if "restart" in argv:
            state["restart_count"] += 1
            return Result()
        if "is-active" in argv:
            return Result(returncode=0)
        if "--property=MainPID" in joined:
            return Result(stdout="4000\n")
        if "ss" in argv and "-H" in argv:
            state["ss_after_restore"] += 1
            if state["ss_after_restore"] < 2:
                return Result(stdout="")
            return Result(stdout='LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=4000,fd=4))\n')
        if "list-units" in joined:
            return Result(stdout="ollama-gpu0.service loaded active running Ollama\n")
        if argv[:2] == ["sudo", "sh"]:
            return Result(stdout="")
        return Result()

    monkeypatch.setattr(ollama_service.os, "geteuid", lambda: 1000)
    controller = OllamaServiceController(
        "ollama-gpu0.service", run=fake_run, sleep_fn=lambda _: None,
    )
    controller.snapshot = ollama_service.DropInSnapshot(existed=False)
    controller.mutation_started = True
    phase = controller.restore(phase="restore_after_error")
    assert phase.verified is True
    assert state["ss_after_restore"] == 2
    assert controller.mutation_started is False


def test_pre_mutation_ownership_check_tolerates_a_single_transient_blip():
    """The exact gap RC8 missed: verify_owns_live_process() called with no
    restart happening nearby (e.g. cli.py's explicit post-discovery check,
    or the pre-mutation re-check at the top of set_kv_type) used to fail
    hard on one momentary blip. It should now retry briefly and succeed."""
    calls = {"ss": 0}

    def flaky_then_fine(argv, **kwargs):
        joined = " ".join(argv)
        if "ss" in argv and "-tlnp" in argv:
            calls["ss"] += 1
            if calls["ss"] == 1:
                return Result(stdout="")  # one transient blip: nothing found
            return Result(stdout='LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=2119,fd=4))\n')
        if "list-units" in joined:
            return Result(stdout="ollama-gpu0.service loaded active running Ollama GPU0\n")
        if "show" in argv and "--property=MainPID" in joined:
            return Result(stdout="2119\n")
        return Result()

    controller = OllamaServiceController(
        "ollama-gpu0.service", run=flaky_then_fine, sleep_fn=lambda _: None,
    )
    active = controller.verify_owns_live_process(timeout_seconds=3.0)
    assert active.unit == "ollama-gpu0.service"
    assert calls["ss"] == 2, "should have retried exactly once after the blip"


def test_pre_mutation_ownership_check_still_fails_fast_on_real_conflict():
    """A genuine wrong-unit conflict must never be retried away, no matter
    how many times it's checked -- this must fail on the very first attempt."""
    calls = {"attempts": 0}

    def genuinely_wrong_unit(argv, **kwargs):
        joined = " ".join(argv)
        calls["attempts"] += 1 if ("ss" in argv and "-tlnp" in argv) else 0
        if "ss" in argv and "-tlnp" in argv:
            return Result(stdout='LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=9999,fd=4))\n')
        if "list-units" in joined:
            return Result(stdout="ollama-gpu0.service loaded active running Ollama GPU0\n")
        if "show" in argv and "--property=MainPID" in joined:
            return Result(stdout="9999\n")  # the OTHER unit owns the port, not ours
        return Result()

    controller = OllamaServiceController(
        "ollama-gpu1.service", run=genuinely_wrong_unit, sleep_fn=lambda _: None,
    )
    with pytest.raises(ServiceControlError, match="does not own the live Ollama process"):
        controller.verify_owns_live_process(timeout_seconds=3.0)
    assert calls["attempts"] == 1, "a real conflict must fail on the first attempt, never retried"


def test_retry_can_be_disabled_for_call_sites_that_want_single_shot_semantics():
    calls = {"ss": 0}

    def flaky(argv, **kwargs):
        joined = " ".join(argv)
        if "ss" in argv and "-tlnp" in argv:
            calls["ss"] += 1
            return Result(stdout="")
        if "list-units" in joined:
            return Result(stdout="ollama-gpu0.service loaded active running Ollama GPU0\n")
        if "show" in argv and "--property=MainPID" in joined:
            return Result(stdout="2119\n")
        return Result()

    controller = OllamaServiceController(
        "ollama-gpu0.service", run=flaky, sleep_fn=lambda _: None,
    )
    with pytest.raises(ServiceControlError, match="no process with visible ownership"):
        controller.verify_owns_live_process(retry=False)
    assert calls["ss"] == 1, "retry=False must not retry even on a transient-shaped error"
