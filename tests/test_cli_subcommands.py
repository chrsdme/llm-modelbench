from llm_modelbench.cli import build_parser


def test_operator_tool_cli_shapes_parse_without_execution():
    parser = build_parser()
    assert parser.parse_args(["simulate", "--run-dir", "run", "--simulate-vram", "24"]).cmd == "simulate"
    assert parser.parse_args(["serve", "--runs-dir", "one", "--runs-dir", "two"]).cmd == "serve"
    assert parser.parse_args(["report", "--run-id", "run", "--weights", "coding_python=0.4"]).weights == "coding_python=0.4"
    assert parser.parse_args(["watch", "--run-id", "run", "--layout", "interactive"]).layout == "interactive"


def _repair_args(tmp_path, *extra):
    parser = build_parser()
    return parser.parse_args([
        "repair", "--run-id", "source_run", "--runs-dir", str(tmp_path),
        "--apply", "--kv-cascade", "--restart-ollama", *extra,
    ])


def _repair_cfg():
    from types import SimpleNamespace
    return SimpleNamespace(
        judge_model="judge", ollama_url="http://127.0.0.1:11434",
        ctx_override=None, think=None,
    )


def test_repair_auto_confirm_implies_apply_confirmation(tmp_path, monkeypatch):
    import pytest
    from types import SimpleNamespace
    from llm_modelbench import cli, repair

    action = SimpleNamespace(kind="retry_needle_guarded")
    plan = SimpleNamespace(
        actions=[action], plan_id="abc123", runs_dir=str(tmp_path),
        to_dict=lambda: {"plan_id": "abc123"},
    )
    monkeypatch.setattr(repair, "build_plan", lambda *a, **k: plan)
    monkeypatch.setattr(repair, "render_plan", lambda p: "plan")

    class StopAfterConfirmation(Exception):
        pass

    observed = {}
    def capture_confirmation(message, *, yes):
        observed["yes"] = yes
        raise StopAfterConfirmation

    monkeypatch.setattr(cli, "_confirm_destructive_compute", capture_confirmation)
    args = _repair_args(tmp_path, "--auto-confirm")
    with pytest.raises(StopAfterConfirmation):
        cli.cmd_repair(args, _repair_cfg())
    assert observed["yes"] is True


def test_repair_with_zero_actions_returns_before_prompt_client_or_sudo(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    from llm_modelbench import cli, repair

    plan = SimpleNamespace(
        actions=[], plan_id="empty123", runs_dir=str(tmp_path),
        to_dict=lambda: {"plan_id": "empty123"},
    )
    monkeypatch.setattr(repair, "build_plan", lambda *a, **k: plan)
    monkeypatch.setattr(repair, "render_plan", lambda p: "plan")
    monkeypatch.setattr(cli, "_client", lambda *a, **k: (_ for _ in ()).throw(AssertionError("client must not be created")))
    monkeypatch.setattr("builtins.input", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not prompt")))

    args = _repair_args(tmp_path, "--auto-confirm")
    cli.cmd_repair(args, _repair_cfg())
    output = capsys.readouterr().out
    assert "nothing to apply" in output
    assert "--force" in output


def test_auto_confirm_requires_restart_cascade(tmp_path):
    import pytest
    from llm_modelbench import cli

    parser = build_parser()
    args = parser.parse_args([
        "repair", "--run-id", "source_run", "--runs-dir", str(tmp_path),
        "--apply", "--auto-confirm",
    ])
    with pytest.raises(SystemExit, match="only valid with --restart-ollama"):
        cli.cmd_repair(args, _repair_cfg())
