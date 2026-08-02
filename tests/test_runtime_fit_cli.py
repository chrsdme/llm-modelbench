from llm_modelbench.cli import build_parser


def test_runtime_fit_cli_is_read_only_and_exposes_safe_arguments():
    args = build_parser().parse_args([
        "runtime-fit", "--model", "fixture", "--runtime-profile", "legacy-ollama",
        "--context", "4096", "--strategy", "layer_split", "--allocation-weights", "GPU-a=1,GPU-b=2", "--json",
    ])
    assert args.cmd == "runtime-fit"
    assert not hasattr(args, "allow_host_code_execution")


def test_runtime_fit_mock_uses_implicit_profile_without_runtime_discovery(monkeypatch):
    from types import SimpleNamespace
    from llm_modelbench import cli
    args = build_parser().parse_args(["runtime-fit", "--model", "fixture", "--mock", "--json"])
    monkeypatch.setattr(cli, "_client", lambda args, cfg: SimpleNamespace(
        model_size_bytes=lambda name: 1, model_info=lambda name: {}, context_length=lambda name: None,
    ))
    monkeypatch.setattr("llm_modelbench.runtime_fit.detect_gpus", lambda: ())
    monkeypatch.setattr("llm_modelbench.runtime_fit.collect_nvidia_gpu_samples", lambda: (_ for _ in ()).throw(AssertionError("not called by injected wrapper")))
    monkeypatch.setattr("llm_modelbench.runtime_fit.collect_runtime_fit", lambda **kwargs: SimpleNamespace(to_dict=lambda: {"schema_version": 1}))
    cli.cmd_runtime_fit(args, SimpleNamespace(ollama_url="http://127.0.0.1:11434"))
