import json

from llm_modelbench.config import Config
from llm_modelbench.ollama import MockClient
from llm_modelbench.planner import build_plan
from llm_modelbench.grade import export_blind


def test_planner_mock_counts():
    cfg = Config()
    cfg.vram_budget_gb = 12.0
    plan = build_plan(MockClient(), cfg, level="smoke", sample_mode="smart")
    assert plan["models_active"] == 4
    assert plan["tasks_total"] > 0
    assert plan["samples_total"] == plan["tasks_total"]


def test_planner_task_filter_context_only():
    cfg = Config()
    cfg.vram_budget_gb = 12.0
    plan = build_plan(MockClient(), cfg, level="smoke", context_only=True)
    # Mock models all have text/vision/embedding families, but only text/vision get needle where family allows.
    assert plan["tasks_total"] >= 1
    assert all(set(m["tasks"]) <= {"needle"} for m in plan["active_models"])


def test_export_blind_pack(tmp_path):
    d = tmp_path / "run" / "subjective" / "wr_rag"
    d.mkdir(parents=True)
    (d / "model.md").write_text("# wr_rag | model-name\n\nRUBRIC: x\n\n## OUTPUT\nhello")
    pack = export_blind(tmp_path / "run")
    assert pack.exists()
    text = pack.read_text()
    assert "model-name" not in text
    mapping = json.loads((tmp_path / "run" / "subjective" / "blind_mapping.json").read_text())
    assert mapping["M001"]["model"] == "model-name"
