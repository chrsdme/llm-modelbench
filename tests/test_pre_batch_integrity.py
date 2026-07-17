from llm_modelbench import scoring
from llm_modelbench.aggregate import aggregate
from llm_modelbench.config import Config
from llm_modelbench.ollama import MockClient
from llm_modelbench.runner import _run_once, _task_hash, _kv_bytes_per_token
from llm_modelbench.tasks import TASKS


def _task(task_id):
    return next(t for t in TASKS if t.id == task_id)


def test_task_hash_changes_when_prompt_changes():
    t = _task("web_nav")
    h1 = _task_hash(t)
    t2 = type(t)(**{**t.__dict__, "prompt": t.prompt + " extra"})
    assert _task_hash(t2) != h1


def test_web_nav_prompt_explicitly_requires_semantic_nav():
    assert "semantic <nav>" in _task("web_nav").prompt


def test_js_debounce_spread_variable_name_is_accepted():
    code = "function debounce(fn,delay){let t;return (...a)=>{clearTimeout(t);t=setTimeout(()=>fn(...a),delay)}}"
    score, reason = scoring.score_tokens_in_code(code, _task("js_debounce").meta)
    assert score == 100.0, reason


def test_kv_key_length_can_be_derived_from_embedding_and_heads(monkeypatch):
    # This test's expected byte count assumes f16 (2 bytes/scalar). It was
    # silently depending on OLLAMA_KV_CACHE_TYPE being unset in whatever shell
    # runs pytest -- true by coincidence until q4_0/q8_0 got exported for real
    # kv-quant testing, which then correctly changed the computed value and
    # broke this test. Fixed by pinning the env var explicitly, so the test
    # is deterministic regardless of what's exported outside pytest.
    monkeypatch.setenv("OLLAMA_KV_CACHE_TYPE", "f16")

    class C(MockClient):
        def model_info(self, model):
            return {
                "llama.block_count": 32,
                "llama.attention.head_count": 32,
                "llama.attention.head_count_kv": 8,
                "llama.embedding_length": 4096,
            }
    bpt, source = _kv_bytes_per_token(C(), "m")
    assert bpt == 131072
    assert "derived" in source


def test_partial_needle_coverage_excluded_from_quality():
    rows = [{"model": "m", "task": "needle", "category": "long_context", "score": 0.0, "needle_coverage": 0.25}]
    lb, per_cat = aggregate(rows, {"long_context": 1.0}, {"needle": 1.0})
    assert lb[0]["quality"] is None
    assert "long_context" not in per_cat


def test_needle_records_prompt_tokens_and_max_verified_ctx():
    res = _run_once(MockClient(), Config(needle_max_ctx=5000), "qwen2.5-coder:14b", _task("needle"))
    assert res["needle_attempted"]
    assert "prompt_tokens_actual" in res["needle_attempted"][0]
    assert res["max_verified_ctx"] >= 3000
