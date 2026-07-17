import json
import math

import pytest

from llm_modelbench.config import Config, DEFAULT_WEIGHTS


def _write(path, payload):
    path.write_text(json.dumps(payload))
    return str(path)


def test_partial_weight_mapping_merges_over_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr("llm_modelbench.config.detect_gpu", lambda: None)
    monkeypatch.setattr("llm_modelbench.config.suggested_vram_budget_gb", lambda _gpu: 8.0)
    cfg = Config.load(_write(tmp_path / "config.json", {"weights": {"coding_python": 0.4}}))
    assert set(cfg.weights) == set(DEFAULT_WEIGHTS)
    assert cfg.weights["coding_python"] == 0.4
    assert cfg.weights["reasoning"] == DEFAULT_WEIGHTS["reasoning"]


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"unknown": 1}, "unknown config field"),
        ({"weights": {"unknown": 1}}, "unknown weight categories"),
        ({"weights": {"reasoning": -1}}, "finite and non-negative"),
        ({"weights": {"reasoning": math.inf}}, "finite and non-negative"),
        ({"weights": {"reasoning": True}}, "not boolean"),
        ({"ollama_url": "file:///tmp/socket"}, "http:// or https://"),
        ({"ollama_url": "http://user:secret@localhost:11434"}, "must not contain embedded credentials"),
    ],
)
def test_invalid_config_fails_closed(tmp_path, payload, message):
    with pytest.raises(SystemExit, match=message):
        Config.load(_write(tmp_path / "config.json", payload))


def test_example_configs_preserve_current_weight_schema(monkeypatch):
    monkeypatch.setattr("llm_modelbench.config.detect_gpu", lambda: None)
    monkeypatch.setattr("llm_modelbench.config.suggested_vram_budget_gb", lambda _gpu: 8.0)
    json_cfg = Config.load("examples/config.example.json")
    assert json_cfg.weights == DEFAULT_WEIGHTS

    pytest.importorskip("yaml")
    yaml_cfg = Config.load("examples/config.example.yaml")
    assert yaml_cfg.weights == DEFAULT_WEIGHTS
