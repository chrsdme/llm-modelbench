
import pytest

from llm_modelbench.capabilities import interrogate_model
from llm_modelbench.classify import families_for
from llm_modelbench.cli import _confirm_plan, build_parser
from llm_modelbench.config import Config
from llm_modelbench.ollama import MockClient
from llm_modelbench.planner import build_plan
from llm_modelbench.runner import _run_once
from llm_modelbench.selection import SelectorState, parse_models_spec, resolve_exact_models
from llm_modelbench.tasks import TASKS


def test_operator_profile_precedes_partial_ollama_metadata():
    # Qwythos is an explicit fleet override. Completion metadata must not make
    # the configured vision lane unreachable.
    assert families_for("hf.co/DavidAU/Qwythos-9B:Q4_K_M", ["completion"]) == ["vision", "text"]


def test_vision_fallback_covers_real_fleet_names():
    assert "vision" in families_for("InternVL3-8B-Q4_K_M:latest", None)
    assert "vision" in families_for("Garnet-OCR-7B:latest", None)


def test_hybrid_tools_metadata_routes_text_and_native_tools():
    assert families_for("qwen2.5-coder:14b", ["completion", "tools"]) == ["text", "tools"]


def test_vision_probe_does_not_leak_expected_token_into_prompt(monkeypatch):
    seen = {}

    class C:
        def capabilities(self, model):
            return ["completion", "vision"]

        def chat(self, model, prompt, **kwargs):
            seen["prompt"] = prompt
            seen["images"] = kwargs.get("images")
            return {"ok": True, "text": "V7K9Q2" if kwargs.get("images") else "AIW_TEXT_OK"}

        def chat_tools(self, *args, **kwargs):
            return {"ok": False, "error": "not supported", "tool_calls": []}

    monkeypatch.setattr("llm_modelbench.capabilities.media.render_text_png", lambda *a, **k: "base64-image")
    profile = interrogate_model(C(), "unknown-vlm", functional=True)
    assert "vision" in profile["supported_families"]
    assert "v7k9q2" not in seen["prompt"].lower()
    assert seen["images"] == ["base64-image"]


def test_insert_probe_requires_suffix_conditioning_not_generic_completion():
    class C:
        def capabilities(self, model):
            return ["completion", "insert"]

        def chat(self, *args, **kwargs):
            return {"ok": True, "text": "AIW_TEXT_OK"}

        def chat_tools(self, *args, **kwargs):
            return {"ok": False, "tool_calls": []}

        def generate_suffix(self, model, prompt, *, suffix, **kwargs):
            # Same generic output for both unseen suffixes must fail.
            return {"ok": True, "text": "'BLUE'"}

    profile = interrogate_model(C(), "coder", functional=True)
    assert profile["probes"]["insert"]["ok"] is False
    # The endpoint answered a task-equivalent FIM request but failed the tiny
    # assertion. That is model quality, not capability absence, so the real
    # scored task remains routed and records the failure honestly.
    assert "insert" in profile["supported_families"]
    assert profile["probe_states"]["insert"] == "responded_contract_failed"
    assert "functional_response" in profile["sources"]["insert"]


def test_planner_routes_native_tools_and_fim_only_to_capable_model():
    client = MockClient("http://127.0.0.1:11434", 42, 0.0, 10)
    plan = build_plan(client, Config(), level="short", selected_models=["qwen2.5-coder:14b"])
    tasks = plan["active_models"][0]["tasks"]
    assert "agent_native_tool_call" in tasks
    assert "fim_suffix_assertion" in tasks


def test_mock_native_tool_and_fim_tasks_score_successfully():
    client = MockClient("http://127.0.0.1:11434", 42, 0.0, 10)
    by_id = {task.id: task for task in TASKS}
    native = _run_once(client, Config(), "qwen2.5-coder:14b", by_id["agent_native_tool_call"])
    fim = _run_once(client, Config(), "qwen2.5-coder:14b", by_id["fim_suffix_assertion"])
    assert native["score"] == 100.0
    assert fim["score"] == 100.0


def test_semicolon_model_selection_and_exact_resolution():
    assert parse_models_spec("a;b;a") == ["a", "b"]
    assert resolve_exact_models(["QWEN:7B"], ["qwen:7b", "llama:8b"]) == ["qwen:7b"]
    with pytest.raises(ValueError, match="not an installed model"):
        resolve_exact_models(["missing"], ["qwen:7b"])


def test_selector_selects_models_only_state_machine():
    state = SelectorState(["a", "b"], selected={"a", "b"})
    state.handle(" ")
    assert state.selected == {"b"}
    state.handle("j")
    state.handle(" ")
    assert state.selected == set()
    state.handle("a")
    assert state.selected == {"a", "b"}


def test_no_redundant_single_dash_all_alias():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "-all"])
    args = parser.parse_args(["run", "--all", "--mock"])
    assert args.all_models is True
    assert args.auto is True
    assert parser.parse_args(["run", "--mock", "--no-auto-probe"]).auto is False
    assert parser.parse_args(["plan", "--mock"]).auto is False


def test_noninteractive_plan_requires_explicit_yes(monkeypatch):
    class Stdin:
        def isatty(self):
            return False

    monkeypatch.setattr("llm_modelbench.cli.sys.stdin", Stdin())
    with pytest.raises(SystemExit, match="requires --yes"):
        _confirm_plan(type("A", (), {"yes": False, "plan_json": None})(), {"active_models": []})


def test_profile_class_cannot_contradict_embedding_only_route():
    # A class-only name profile must not label an embedding-only runtime as a
    # reasoning model.  The display class and executable family must agree.
    name = "some-ornith-variant:latest"
    assert families_for(name, ["embedding"]) == ["embedding"]
    from llm_modelbench.classify import classify_model
    assert classify_model(name, ["embedding"]) == "embedding"


def test_known_vlm_profiles_survive_partial_completion_metadata():
    expected = {
        "hf.co/atahmih/InternVL3-8B-Q4_K_M-GGUF:latest": "vision",
        "hf.co/mradermacher/Garnet-OCR-7B-0422-i1-GGUF:latest": "vision",
        "hf.co/sillykiwi/VL-1-Coder-Q4_K_M-GGUF:latest": "coding",
    }
    from llm_modelbench.classify import classify_model
    for name, model_class in expected.items():
        assert families_for(name, ["completion"]) == ["vision", "text"]
        assert classify_model(name, ["completion"]) == model_class


def test_auto_vision_probe_uses_name_hint_despite_partial_metadata(monkeypatch):
    seen = {"vision_calls": 0}

    class C:
        def capabilities(self, model):
            return ["completion"]

        def chat(self, model, prompt, **kwargs):
            if kwargs.get("images"):
                seen["vision_calls"] += 1
                return {"ok": True, "text": "V7K9Q2"}
            return {"ok": True, "text": "AIW_TEXT_OK"}

        def chat_tools(self, *args, **kwargs):
            return {"ok": False, "error": "not supported", "tool_calls": []}

    monkeypatch.setattr("llm_modelbench.capabilities.media.render_text_png", lambda *a, **k: "base64-image")
    # Not in MODEL_PROFILES: this proves generic conservative hint discovery,
    # not merely the three explicit fleet overrides above.
    profile = interrogate_model(C(), "future-vision-model:latest", functional=True)
    assert seen["vision_calls"] == 1
    assert profile["probes"]["vision"]["ok"] is True
    assert "vision" in profile["supported_families"]
    assert "functional_probe" in profile["sources"]["vision"]


def test_planner_routes_known_vlm_with_completion_only_metadata():
    model = "hf.co/atahmih/InternVL3-8B-Q4_K_M-GGUF:latest"

    class C:
        def tags(self):
            return [{"name": model, "size": 5_000_000_000}]

        def capabilities(self, name):
            assert name == model
            return ["completion"]

    plan = build_plan(C(), Config(), level="short", categories=["ocr", "pdf"], selected_models=[model])
    assert plan["models_active"] == 1
    active = plan["active_models"][0]
    assert active["families"] == ["vision", "text"]
    assert "ocr_invoice" in active["tasks"]


def test_transient_capability_probe_is_withheld_not_marked_unavailable(monkeypatch):
    class C:
        def capabilities(self, model):
            return ["completion", "vision"]
        def chat(self, model, prompt, **kwargs):
            if kwargs.get("images"):
                raise TimeoutError("temporary vision timeout")
            return {"ok": True, "text": "AIW_TEXT_OK"}
        def chat_tools(self, *args, **kwargs):
            return {"ok": False, "error": "not supported", "tool_calls": []}

    monkeypatch.setattr("llm_modelbench.capabilities.media.render_text_png", lambda *a, **k: "base64-image")
    profile = interrogate_model(C(), "temporary-vlm", functional=True)
    assert profile["probe_states"]["vision"] == "transient_failure"
    assert "vision" not in profile["supported_families"]
    assert "vision" in profile["unverified_families"]
    assert "vision" not in profile["confirmed_unavailable_families"]
