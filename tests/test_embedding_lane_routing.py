import pytest

from llm_modelbench.classify import classify_model, families_for
from llm_modelbench.config import Config
from llm_modelbench.planner import build_plan
from llm_modelbench.runner import _score_task
from llm_modelbench.tasks import TASKS


@pytest.mark.parametrize(
    ("name", "capabilities"),
    [
        ("local-vector-model:latest", ["embedding"]),
        ("nomic-embed-text:latest", None),
        ("nomic-embed-text:latest", ["insert"]),
    ],
)
def test_embed_models_route_to_embedding_family(name, capabilities):
    assert classify_model(name, capabilities) == "embedding"
    assert families_for(name, capabilities) == ["embedding"]


def test_completion_capability_overrides_embedding_name_hint():
    name = "nomic-embed-text:latest"
    assert classify_model(name, ["completion"]) != "embedding"
    assert families_for(name, ["completion"]) == ["text"]


@pytest.mark.parametrize(
    ("name", "capabilities"),
    [
        ("local-vector-model:latest", ["embedding"]),
        ("nomic-embed-text:latest", None),
        ("nomic-embed-text:latest", ["insert"]),
        ("nomic-embed-text:latest", ["completion"]),
        ("gemma3:12b", ["completion", "vision"]),
    ],
)
def test_embedding_class_matches_embedding_family(name, capabilities):
    assert (classify_model(name, capabilities) == "embedding") == (
        families_for(name, capabilities) == ["embedding"]
    )


def test_completion_only_gemma4_remains_text_not_vision():
    name = "hf.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q4_K_M"
    assert families_for(name, ["tools", "completion"]) == ["text", "tools"]


def test_planner_schedules_only_retrieval_for_live_embed_model():
    model = "boring-vector-model:latest"

    class Client:
        def tags(self):
            return [{"name": model, "size": 1}]

        def capabilities(self, _model):
            return ["embedding"]

    plan = build_plan(Client(), Config(), level="smoke")
    assert len(plan["active_models"]) == 1
    entry = plan["active_models"][0]
    assert entry["model"] == model
    assert entry["class"] == "embedding"
    assert entry["families"] == ["embedding"]
    assert entry["declared_capabilities"] == ["embedding"]
    assert entry["tasks"] == ["ret_ukdocs"]
    assert entry["tasks_total"] == 1
    assert entry["samples_total"] == 1


@pytest.mark.parametrize("model", ["nomic-embed-text", "mxbai-embed-large", "bge-m3:latest"])
def test_retrieval_embeds_with_model_under_test(model):
    task = next(task for task in TASKS if task.id == "ret_ukdocs")
    calls = []

    class Client:
        def embed(self, embed_model, texts):
            calls.append((embed_model, list(texts)))
            return [[1.0, 0.0] for _ in texts]

    _, reason, _ = _score_task(Client(), Config(embed_model="wrong-fallback"), task, "", model)
    assert calls and calls[0][0] == model
    assert f"embed_model={model}" in reason


def test_ollama_completion_only_report_does_not_block_known_embedding_architectures():
    """Regression guard: llama-embed-nemotron-8b and KaLM-Reranker-V1-Large were
    both misrouted through the general text battery because Ollama reports
    capabilities=["completion"] for both, with no embedding/rerank flag at all.
    Confirmed against real `ollama show` output, not assumed. Both must route
    to the embedding lane despite the reported completion capability."""
    from llm_modelbench.classify import classify_model, families_for
    for name in ("hf.co/mradermacher/llama-embed-nemotron-8b-GGUF:latest",
                 "hf.co/KaLM-Embedding/KaLM-Reranker-V1-Large-Q4_K_M-GGUF:latest"):
        assert classify_model(name, ["completion"]) == "embedding"
        assert families_for(name, ["completion"]) == ["embedding"]


def test_force_embedding_override_does_not_leak_into_unrelated_models():
    from llm_modelbench.classify import classify_model, families_for
    assert classify_model("qwen2.5-coder:14b", ["completion", "tools"]) == "coding"
    assert families_for("qwen2.5-coder:14b", ["completion", "tools"]) == ["text", "tools"]
