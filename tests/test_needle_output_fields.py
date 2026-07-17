import json

from llm_modelbench import scoring
from llm_modelbench.aggregate import aggregate
from llm_modelbench.config import Config
from llm_modelbench.ollama import MockClient, OllamaClient
from llm_modelbench.runner import _run_once, _validate_needle_ctx_override
from llm_modelbench.tasks import TASKS


def _task(task_id):
    return next(t for t in TASKS if t.id == task_id)


def test_python_executor_does_not_run_raw_prose_candidate(monkeypatch):
    calls = []

    def fake_run(code, checks, timeout=10):
        calls.append(code)
        return (100.0, "ok") if "def dedupe" in code else (0.0, "bad")

    monkeypatch.setattr(scoring.sandbox, "run_python_checks", fake_run)
    resp = "Here is the answer.\n```python\ndef dedupe(seq):\n    return list(dict.fromkeys(seq))\n```"
    score, reason = scoring.score_python(resp, {"checks": ["assert dedupe([1,1,2])==[1,2]"]})
    assert score == 100.0, reason
    assert len(calls) == 1
    assert calls[0].lstrip().startswith("def dedupe")


def test_best_over_blocks_caps_and_short_circuits():
    calls = []

    def fake_score(code):
        calls.append(code)
        return (100.0, "ok") if "good" in code else (0.0, "bad")

    resp = "```python\nbad1\n```\n```python\ngood\n```\n```python\nbad3\n```\n```python\nbad4\n```"
    score, reason = scoring.best_over_blocks(resp, "python", fake_score, include_raw=False, max_candidates=3)
    assert score == 100.0
    assert len(calls) == 2


def test_all_harness_error_quality_is_null():
    rows = [
        {"model": "m", "task": "a", "category": "coding_python", "score": None, "error_kind": "harness_error", "output_chars": 0},
        {"model": "m", "task": "b", "category": "coding_python", "score": None, "error_kind": "harness_error", "output_chars": 0},
    ]
    lb, _ = aggregate(rows, {"coding_python": 1.0}, {"a": 1.0, "b": 1.0})
    assert lb[0]["quality"] is None
    assert lb[0]["completion_rate"] == 0.0
    assert lb[0]["value_per_gb"] is None


def test_ctx_override_only_refuses_when_zero_needle_probes_fit():
    cfg = Config(ctx_override=5000)
    _validate_needle_ctx_override(cfg, [_task("needle")])
    cfg2 = Config(ctx_override=1000)
    try:
        _validate_needle_ctx_override(cfg2, [_task("needle")])
    except ValueError as exc:
        assert "invalidates all needle probes" in str(exc)
    else:
        raise AssertionError("expected refusal when no probe can fit")


def test_needle_max_ctx_drops_coverage_not_score_denominator_for_operator_skip():
    cfg = Config(needle_max_ctx=5000)
    res = _run_once(MockClient(), cfg, "qwen2.5-coder:14b", _task("needle"))
    assert res["score"] is None
    assert res["needle_coverage"] == 0.25
    assert any(s["reason"] == "needle_max_ctx" for s in res["needle_skipped"])
    rows = [{"model": "m", "task": "needle", "category": "long_context", **res}]
    _, per_cat = aggregate(rows, {"long_context": 1.0}, {"needle": 1.0})
    assert "long_context" not in per_cat


def test_needle_context_capability_skips_score_zero():
    cfg = Config()
    res = _run_once(MockClient(), cfg, "qwen2.5-coder:14b", _task("needle"))
    assert res["score"] == 75.0
    assert res["needle_coverage"] == 1.0
    assert any(s["reason"] == "exceeds_context_length_max" for s in res["needle_skipped"])


def test_needle_kv_budget_skip_drops_coverage_and_aggregate_quality():
    cfg = Config(vram_budget_gb=1.0)
    res = _run_once(MockClient(), cfg, "qwen2.5-coder:14b", _task("needle"))
    assert res["needle_coverage"] < 1.0
    assert any(s["reason"] == "kv_cache_exceeds_vram_budget" for s in res["needle_skipped"])
    rows = [{"model": "m", "task": "needle", "category": "long_context", **res}]
    lb, per_cat = aggregate(rows, {"long_context": 1.0}, {"needle": 1.0})
    assert lb[0]["quality"] is None
    assert "long_context" not in per_cat


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def __iter__(self):
        yield json.dumps({"message": {"content": "ok"}}).encode()
        yield json.dumps({"done": True, "eval_count": 1, "eval_duration": 1_000_000_000, "done_reason": "stop"}).encode()


class _NoThinkClient(OllamaClient):
    def __init__(self):
        super().__init__("http://mock")
        self.payloads = []
    def show(self, model):
        return {"capabilities": [], "model_info": {"general.context_length": 4096}}
    def _post_stream(self, path, payload):
        self.payloads.append(payload)
        return _FakeResponse(payload)


def test_think_false_not_sent_to_non_thinking_model():
    client = _NoThinkClient()
    res = client.chat("plain:latest", "hi", think="off")
    assert res["ok"] is True
    assert res["think_unsupported"] is True
    assert "think" not in client.payloads[0]
