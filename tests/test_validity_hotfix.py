from llm_modelbench import scoring
from llm_modelbench.aggregate import aggregate
from llm_modelbench.config import Config
from llm_modelbench.ollama import MockClient
from llm_modelbench.runner import _run_once, _max_verified_prefix, _measured_kv_slope
from llm_modelbench.tasks import TASKS


def _task(task_id):
    return next(t for t in TASKS if t.id == task_id)


WEB_POSITIVES = [
    "<nav class='n'><a>Home</a></nav><style>.n{display:flex}@media(max-width:600px){.n{flex-direction:column}}</style>",
    "<nav class='n'><a>Home</a></nav><style>.n{display:block}@media(min-width:601px){.n{display:flex;flex-direction:row}}</style>",
    "<nav class='n'><a>Home</a></nav><style>.n{display:grid;grid-auto-flow:column}@media(max-width:37.5em){.n{grid-auto-flow:row}}</style>",
    "```html\n<nav class='n'><a>Home</a></nav>\n```\n```css\n.n{display:flex}@media(max-width:599.98px){.n{display:block!important}}\n```",
    "<nav><ul class='menu'><li><a>Home</a></li></ul></nav><style>.menu{display:flex}.menu a{display:block}@media(max-width:600px){.menu{flex-direction:column}}</style>",
]


def test_web_nav_idiom_stratified_positives_score_100():
    for html in WEB_POSITIVES:
        score, reason = scoring.score_web_nav(html, {})
        assert score == 100.0, reason


def test_web_nav_negative_fixtures_score_below_60():
    negatives = [
        "<nav></nav><style>.sidebar{display:flex}@media(max-width:600px){.footer{flex-direction:column}}</style>",
        "<nav class='n'><a>H</a></nav><style>.n{display:block}@media(max-width:600px){.n{display:block}}</style>",
        "<nav class='n'><a>H</a></nav><style>.n{display:flex}</style>",
        "<nav><a>H</a></nav>",
        "<nav class='n'><a>H</a></nav><style>.missing{display:flex}@media(max-width:600px){.missing{flex-direction:column}}</style>",
        "<div class='navbar'></div><style>.navbar{display:flex}@media(max-width:600px){.navbar{flex-direction:column}}</style>",
        "<nav class='n'><a>H</a></nav><style>.n{display:flex}@media(max-width:1200px){.n{flex-direction:column}}</style>",
    ]
    for html in negatives:
        score, reason = scoring.score_web_nav(html, {})
        assert score < 60.0, (score, reason)


def test_js_debounce_executes_not_token_greps():
    comment = "// setTimeout clearTimeout apply return\nfunction debounce(fn, delay){ return fn }"
    score, reason = scoring.score_js_debounce(comment, {})
    assert score == 0.0, reason
    good = "function debounce(fn,delay){let t;return (...args)=>{clearTimeout(t);t=setTimeout(()=>fn(...args),delay)}}"
    good_score, good_reason = scoring.score_js_debounce(good, {})
    assert good_score == 100.0, good_reason


def test_needle_partial_coverage_has_no_row_score():
    res = _run_once(MockClient(), Config(needle_max_ctx=5000), "qwen2.5-coder:14b", _task("needle"))
    assert res["needle_coverage"] < 1.0
    assert res["score"] is None
    rows = [{"model": "m", "task": "needle", "category": "long_context", **res}]
    lb, per_cat = aggregate(rows, {"long_context": 1.0}, {"needle": 1.0})
    assert lb[0]["quality"] is None
    assert "long_context" not in per_cat


def test_max_verified_ctx_is_unbroken_prefix_and_flags_non_monotonic():
    attempted = [
        {"size": 4000, "found": True, "prompt_tokens_actual": 4010},
        {"size": 16000, "found": False, "prompt_tokens_actual": 15918},
        {"size": 32000, "found": True, "prompt_tokens_actual": 31798},
    ]
    assert _max_verified_prefix(attempted) == (4010, True)


def test_measured_kv_slope_from_two_probe_peaks():
    slope = _measured_kv_slope([(4096, 8000.0), (8192, 8500.0)])
    assert slope is not None and slope > 0


def test_gate_task_difficulty_zero_does_not_create_quality():
    rows = [{"model": "m", "task": "git_commit", "category": "git", "score": 100.0}]
    lb, per_cat = aggregate(rows, {"git": 1.0}, {"git_commit": 0.0})
    assert lb[0]["quality"] is None
    assert "git" not in per_cat
