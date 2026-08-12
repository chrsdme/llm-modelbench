"""Anvil Stage 1.3 slice 2: cmd_inventory reuses one GPU detection across
all listed models, and stays read-only -- it must not go through the
composed operational preflight's backend-*selection* step, only detection.
"""
import json

from llm_modelbench import cli
from llm_modelbench.config import Config


def _args(**extra):
    parser = cli.build_parser()
    values = parser.parse_args(["inventory", "--mock", "--json"])
    for key, value in extra.items():
        setattr(values, key, value)
    return values


def test_inventory_detects_gpus_at_most_once_across_multiple_models(monkeypatch, capsys):
    # cmd_inventory does `from .hardware import detect_gpus` locally, so the
    # real interception point is llm_modelbench.hardware, not cli itself.
    import llm_modelbench.hardware as hardware
    calls = []
    monkeypatch.setattr(hardware, "detect_gpus", lambda: calls.append(1) or ())
    cli.cmd_inventory(_args(), Config())
    captured = json.loads(capsys.readouterr().out)
    assert len(captured) > 1  # multiple mock models, proving the loop actually ran multiple times
    assert len(calls) <= 1


def test_inventory_subcommand_never_defines_unattended():
    parser = cli.build_parser()
    args = parser.parse_args(["inventory", "--mock"])
    assert not hasattr(args, "unattended")
