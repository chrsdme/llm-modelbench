"""Anvil Stage 1.3 slice 2: exactly-once GPU detection inside runner.run().

_needle_kv_estimate() is invoked once per probed context depth inside a
single needle task, so a naive per-call detect_gpus() would shell out to
nvidia-smi many times over one run(). These tests prove run() resolves
GPU inventory at most once per invocation regardless of how many needle
probes it triggers internally, and not at all when nothing needs it.
"""
from llm_modelbench.capabilities import CAPABILITY_SCHEMA_VERSION, PROBE_PROTOCOL_VERSION, current_capability_identity
from llm_modelbench.config import Config
from llm_modelbench.ollama import MockClient
from llm_modelbench import runner


MODEL = "qwen2.5-coder:14b"


def _profile(client, model=MODEL):
    identity = current_capability_identity(client, model)
    return {model: {
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "capability_identity": identity,
        "declared_capabilities": ["completion"],
        "supported_families": ["text", "long_context"],
        "measured_supported_families": ["text", "long_context"],
        "measured_capabilities": {"text": {"state": "measured_supported", "route_scored_tasks": True},
                                  "long_context": {"state": "measured_supported", "route_scored_tasks": True}},
    }}


def _run_needle(tmp_path, *, gpu_inventory=None):
    client = MockClient()
    return runner.run(client, Config(fingerprint=False), level="full", out_dir=tmp_path,
                      include=None, exclude=None, skip_offload=False, categories=None,
                      task_ids=["needle"], resume=False, live_ui="off", fingerprint_enabled=False,
                      selected_models=[MODEL], capability_profiles=_profile(client), auto_probe=False,
                      gpu_inventory=gpu_inventory)


def test_supplied_gpu_inventory_means_detect_gpus_is_never_called(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "detect_gpus",
                        lambda: (_ for _ in ()).throw(AssertionError("detect_gpus must not be called when gpu_inventory is supplied")))
    _run_needle(tmp_path, gpu_inventory=())


def test_unsupplied_gpu_inventory_is_detected_at_most_once_across_many_needle_probes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "detect_gpus", lambda: calls.append(1) or ())
    _run_needle(tmp_path, gpu_inventory=None)
    # A needle task probes multiple context depths internally (multiple
    # _needle_kv_estimate calls within one _run_once), so the count must be
    # exactly one: zero would mean detection silently stopped happening,
    # and more than one would mean the per-run cache isn't working.
    assert len(calls) == 1


def test_no_needle_tasks_means_detect_gpus_is_never_called(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "detect_gpus",
                        lambda: (_ for _ in ()).throw(AssertionError("detect_gpus must not be called when no needle/skip-offload path needs it")))
    client = MockClient()
    runner.run(client, Config(fingerprint=False), level="smoke", out_dir=tmp_path,
              include=None, exclude=None, skip_offload=False, categories=None,
              task_ids=["py_anagram"], resume=False, live_ui="off", fingerprint_enabled=False,
              selected_models=[MODEL], capability_profiles=_profile(client), auto_probe=False)
