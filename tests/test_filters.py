from llm_modelbench.filters import filter_models, filter_tasks, is_context_alias, parse_task_ids, validate_task_ids
from llm_modelbench.tasks import TASKS, tasks_for


def test_context_alias_detection_is_specific():
    assert is_context_alias("hermes3-8b-64k:latest")
    assert is_context_alias("qwen25-coder-14b-64k-exp:latest")
    assert is_context_alias("model-ctx-32768")
    assert not is_context_alias("hf.co/mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF:Q4_K_M")
    assert not is_context_alias("experimental-roleplay-model")


def test_model_filters_base_and_alias_only():
    models = ["hermes3:8b", "hermes3-8b-64k:latest", "qwen2.5-coder:14b", "qwen25-coder-14b-64k-exp:latest"]
    kept, skipped = filter_models(models, family_base_only=True)
    assert kept == ["hermes3:8b", "qwen2.5-coder:14b"]
    assert {s.model for s in skipped} == {"hermes3-8b-64k:latest", "qwen25-coder-14b-64k-exp:latest"}

    kept, skipped = filter_models(models, context_aliases_only=True)
    assert kept == ["hermes3-8b-64k:latest", "qwen25-coder-14b-64k-exp:latest"]
    assert {s.reason for s in skipped} == {"not_context_alias"}


def test_task_filters():
    smoke_text = tasks_for("smoke", None, ["text"])
    ids = parse_task_ids("py_anagram,json_extract")
    selected = filter_tasks(smoke_text, task_ids=ids)
    assert [t.id for t in selected] == ["py_anagram", "json_extract"]

    full_text = tasks_for("full", None, ["text"])
    assert [t.id for t in filter_tasks(full_text, context_only=True)] == ["needle"]
    assert [t.id for t in filter_tasks(full_text, task_regex="needle|context")] == ["needle"]
    assert validate_task_ids(["needle", "not_a_task"], [t.id for t in TASKS]) == ["not_a_task"]
