"""Regression coverage for the simultaneous-live-unit warning.

Additive only: this does not replace or modify RC7's existing
``test_ollama_service_active_unit.py``. It covers one specific, narrow gap
flagged during the RC7 review: discovery should surface a diagnostic warning
when a *second* ollama-pattern unit is genuinely alive at the same time as
the winner, without changing which unit gets selected, and without false-
positives on the real host's actual shape (a merely dormant, crash-looping
unit with no live MainPID).
"""

from llm_modelbench.ollama_service import OllamaServiceController, discover_active_service


class Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_no_warning_when_other_unit_is_merely_dormant():
    """The actual real-host shape: ollama.service exists but MainPID=0
    (crash-looping / auto-restart). This is not a conflict."""

    def fake_run(argv, **kwargs):
        joined = " ".join(argv)
        if "ss" in argv and "-tlnp" in argv:
            return Result(stdout='LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=2119,fd=4))\n')
        if "list-units" in joined:
            return Result(
                stdout=(
                    "  ollama-gpu0.service loaded active     running      Ollama GPU0\n"
                    "\u25cf ollama.service      loaded activating auto-restart Ollama Service\n"
                )
            )
        if "show" in argv and "ollama-gpu0.service" in argv and "--property=MainPID" in joined:
            return Result(stdout="2119\n")
        if "show" in argv and "ollama.service" in argv and "--property=MainPID" in joined:
            return Result(stdout="0\n")
        return Result()

    warnings = []
    active = discover_active_service(run=fake_run, warn_fn=warnings.append)
    assert active.unit == "ollama-gpu0.service"
    assert warnings == []


def test_warns_when_second_unit_is_genuinely_alive():
    """Two ollama-pattern units both have a live MainPID at once. Discovery
    must still resolve correctly (the one that owns the port wins), but must
    not stay silent about the contention risk."""

    def fake_run(argv, **kwargs):
        joined = " ".join(argv)
        if "ss" in argv and "-tlnp" in argv:
            return Result(stdout='LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=2119,fd=4))\n')
        if "list-units" in joined:
            return Result(
                stdout=(
                    "  ollama-gpu0.service loaded active running Ollama GPU0\n"
                    "  ollama-gpu1.service loaded active running Ollama GPU1\n"
                )
            )
        if "show" in argv and "ollama-gpu0.service" in argv and "--property=MainPID" in joined:
            return Result(stdout="2119\n")
        if "show" in argv and "ollama-gpu1.service" in argv and "--property=MainPID" in joined:
            return Result(stdout="3000\n")
        return Result()

    warnings = []
    active = discover_active_service(run=fake_run, warn_fn=warnings.append)
    assert active.unit == "ollama-gpu0.service"
    assert len(warnings) == 1
    assert "ollama-gpu1.service" in warnings[0]


def test_no_warning_without_warn_fn_supplied():
    """warn_fn is optional; omitting it must not raise or change behaviour."""

    def fake_run(argv, **kwargs):
        joined = " ".join(argv)
        if "ss" in argv and "-tlnp" in argv:
            return Result(stdout='LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=2119,fd=4))\n')
        if "list-units" in joined:
            return Result(
                stdout=(
                    "  ollama-gpu0.service loaded active running Ollama GPU0\n"
                    "  ollama-gpu1.service loaded active running Ollama GPU1\n"
                )
            )
        if "show" in argv and "--property=MainPID" in joined and "ollama-gpu0.service" in argv:
            return Result(stdout="2119\n")
        if "show" in argv and "--property=MainPID" in joined and "ollama-gpu1.service" in argv:
            return Result(stdout="3000\n")
        return Result()

    active = discover_active_service(run=fake_run)  # no warn_fn
    assert active.unit == "ollama-gpu0.service"


def test_for_active_service_collects_warnings_onto_controller():
    def fake_run(argv, **kwargs):
        joined = " ".join(argv)
        if "ss" in argv and "-tlnp" in argv:
            return Result(stdout='LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=2119,fd=4))\n')
        if "list-units" in joined:
            return Result(
                stdout=(
                    "  ollama-gpu0.service loaded active running Ollama GPU0\n"
                    "  ollama-gpu1.service loaded active running Ollama GPU1\n"
                )
            )
        if "show" in argv and "ollama-gpu0.service" in argv and "--property=MainPID" in joined:
            return Result(stdout="2119\n")
        if "show" in argv and "ollama-gpu1.service" in argv and "--property=MainPID" in joined:
            return Result(stdout="3000\n")
        return Result()

    received = []
    controller = OllamaServiceController.for_active_service(
        run=fake_run, force_password_prompt=False, warn_fn=received.append,
    )
    assert controller.unit == "ollama-gpu0.service"
    assert controller.discovery_warnings == received
    assert len(received) == 1
    assert "ollama-gpu1.service" in received[0]


def test_for_active_service_leaves_warnings_empty_on_clean_discovery():
    def fake_run(argv, **kwargs):
        joined = " ".join(argv)
        if "ss" in argv and "-tlnp" in argv:
            return Result(stdout='LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=2119,fd=4))\n')
        if "list-units" in joined:
            return Result(stdout="  ollama-gpu0.service loaded active running Ollama GPU0\n")
        if "show" in argv and "--property=MainPID" in joined:
            return Result(stdout="2119\n")
        return Result()

    controller = OllamaServiceController.for_active_service(run=fake_run, force_password_prompt=False)
    assert controller.discovery_warnings == []
