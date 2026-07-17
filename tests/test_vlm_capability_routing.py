from llm_modelbench.classify import classify_model, families_for
from llm_modelbench.config import Config
from llm_modelbench.planner import build_plan


def test_local_vlm_profiles_are_routed_to_vision_tasks():
    names = [
        "hf.co/Jackrong/Qwopus3.5-9B-Coder-GGUF:Q4_K_M",
        "hf.co/unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL",
        "qwen3.5:9b",
        "hf.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF:Q4_K_M",
        "qwen2.5vl:7b",
        "gemma3:12b",
        "openbmb/minicpm-v4.6:latest",
        "gemma3:4b",
    ]

    for name in names:
        assert "vision" in families_for(name), name


def test_text_only_model_is_not_routed_to_vision_tasks():
    assert families_for("qwen2.5-coder:7b") == ["text"]


def test_capabilities_allow_vision_for_boring_name():
    assert families_for("local-model:latest", ["completion", "vision"]) == ["vision", "text"]
    assert classify_model("local-model:latest", ["completion", "vision"]) == "vision"


def test_capabilities_override_gemma_name_heuristic_without_vision():
    name = "hf.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q4_K_M"
    assert families_for(name, ["tools", "completion"]) == ["text", "tools"]
    assert classify_model(name, ["tools", "completion"]) != "vision"


def test_unavailable_capabilities_fall_back_to_vlm_heuristic():
    assert families_for("qwen2.5vl:7b", None) == ["vision", "text"]
    assert families_for("qwen2.5-coder:7b", None) == ["text"]


def test_capabilities_allow_known_gemma3_vision_model():
    assert families_for("gemma3:12b", ["completion", "vision"]) == ["vision", "text"]


def test_planner_uses_capabilities_for_ocr_pdf_routing():
    text_only = "hf.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q4_K_M"
    vision = "gemma3:12b"

    class Client:
        def tags(self):
            return [{"name": text_only, "size": 1}, {"name": vision, "size": 1}]

        def capabilities(self, model):
            return ["completion", "vision"] if model == vision else ["tools", "completion"]

    plan = build_plan(Client(), Config(), level="smoke", categories=["ocr", "pdf"])
    active = {item["model"]: item["tasks"] for item in plan["active_models"]}
    assert active[vision] == ["ocr_invoice"]
    assert text_only not in active


def test_embedding_capability_is_exclusive_even_with_loosely_declared_completion_and_tools():
    """Regression guard for a real overnight-run incident: qwen3-embedding and
    nomic-embed-code both declared ["embedding", "completion", "tools"]-style
    capabilities (Ollama loosely reporting completion/tools alongside a real
    embedding architecture), and the additive union design routed them into
    the full text/tool battery, producing ~28 HTTPError 400 rows each, since
    an embedding architecture cannot actually serve chat completion at all.
    Embedding must stay exclusive regardless of what else is co-declared,
    unlike vision+text+tools+insert which are genuinely compatible together."""
    for caps in (["embedding"], ["embedding", "completion"], ["embedding", "completion", "tools"]):
        assert families_for("qwen3-embedding:latest", caps) == ["embedding"], caps
        assert families_for("hf.co/mradermacher/nomic-embed-code-i1-GGUF:latest", caps) == ["embedding"], caps
        assert classify_model("qwen3-embedding:latest", caps) == "embedding", caps
