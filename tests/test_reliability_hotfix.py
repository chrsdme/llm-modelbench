import json

from llm_modelbench import scoring, sensitivity
from llm_modelbench.aggregate import aggregate
from llm_modelbench.finals.agentic import manifest
from llm_modelbench.tasks import TASKS


def test_js_debounce_missing_node_is_harness_error(monkeypatch):
    monkeypatch.setattr(scoring.shutil, "which", lambda name: None if name == "node" else "/bin/true")
    score, reason = scoring.score_js_debounce("```javascript\nfunction debounce(){}\n```", {})
    assert score is None
    assert reason.startswith("HARNESS_ERROR: node unavailable")


def test_harness_error_does_not_become_quality_number():
    rows = [
        {"model": "m", "task": "js_debounce", "category": "coding_js", "score": None, "error_kind": "harness_error", "reason": "HARNESS_ERROR: node unavailable"},
    ]
    lb, per_cat = aggregate(rows, {"coding_js": 1.0}, {"js_debounce": 1.0})
    assert lb[0]["quality"] is None
    assert lb[0]["category_ineligible"]["coding_js"].startswith("harness_error")
    assert per_cat == {}


def test_model_failure_counts_as_zero_quality():
    rows = [
        {"model": "m", "task": "web_nav", "category": "coding_web", "score": 0.0, "error_kind": "empty_output", "reason": "ERROR_EMPTY_OUTPUT"},
    ]
    lb, _ = aggregate(rows, {"coding_web": 1.0}, {"web_nav": 1.0})
    assert lb[0]["quality"] == 0.0


def test_web_nav_separate_css_wrong_reason_is_not_no_css():
    answer = """```html
<nav class='topnav'><a>Home</a><a>Docs</a></nav>
```
```css
.topnav a { display: block; }
@media (max-width: 600px) { .topnav a { display: block; } }
```"""
    score, reason = scoring.score_web_nav(answer, {})
    assert score == 20.0
    assert "no CSS" not in reason
    assert "changes layout across 600px" in reason or "not horizontal" in reason


def test_sensitivity_plan_and_report(tmp_path):
    script = sensitivity.plan_commands(
        run_prefix="probe",
        include_regex="m1|m2",
        tasks="web_nav",
        ctx_values="default,4096",
        num_predict_values="512",
    )
    assert "probe_default_np512" in script
    assert "probe_ctx4096_np512" in script
    assert "--ctx 4096" in script
    run_a = tmp_path / "probe_default_np512"
    run_b = tmp_path / "probe_ctx4096_np512"
    run_a.mkdir(); run_b.mkdir()
    (run_a / "summary_meta.json").write_text(json.dumps({"ctx_override": None, "num_predict": 512}))
    (run_b / "summary_meta.json").write_text(json.dumps({"ctx_override": 4096, "num_predict": 512}))
    row_a = {"model": "m", "task": "web_nav", "score": 0.0}
    row_b = {"model": "m", "task": "web_nav", "score": 100.0}
    (run_a / "raw_results.jsonl").write_text(json.dumps(row_a) + "\n")
    (run_b / "raw_results.jsonl").write_text(json.dumps(row_b) + "\n")
    report = sensitivity.report([run_a, run_b])
    assert "Config sensitivity report" in report
    assert "100.0" in report
    assert "wide" in report


def test_agentic_finals_seed_is_active():
    ids = {t.id for t in TASKS}
    mf = manifest()
    assert mf["status"] in {"active_seed", "active_hardened"}
    assert mf["registered_in_tasks"] is True
    for spec in mf["tasks"]:
        assert spec["id"] in ids
