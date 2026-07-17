
import pytest

from llm_modelbench import scoring
from llm_modelbench.aggregate import aggregate
from llm_modelbench.config import Config
from llm_modelbench.compare import export_review
from llm_modelbench.ollama import MockClient
from llm_modelbench.runner import _dump_raw, _run_once
from llm_modelbench.tasks import TASKS
from llm_modelbench.progress import classify_row


def _task(task_id):
    return next(t for t in TASKS if t.id == task_id)


def test_web_nav_html_css_blocks_both_orders_score_100():
    html = "```html\n<nav class='n'><a>Home</a></nav>\n```"
    css = "```css\n.n{display:flex}@media (max-width:600px){.n{flex-direction:column}}\n```"
    assert scoring.score_web_nav(html + "\n" + css, {})[0] == 100.0
    assert scoring.score_web_nav(css + "\n" + html, {})[0] == 100.0


def test_python_correct_block_with_trailing_example_scores_100():
    resp = """```python
def dedupe(seq):
    return list(dict.fromkeys(seq))
```

```
print(dedupe([1, 1, 2]))
```"""
    score, reason = scoring.score_python(resp, {"checks": ["assert dedupe([1,1,2])==[1,2]"]})
    assert score == 100.0, reason


def test_thinking_draft_then_final_code_scores_100():
    resp = """```thinking
wrong draft
```
```python
def dedupe(seq):
    return list(dict.fromkeys(seq))
```"""
    score, reason = scoring.score_python(resp, {"checks": ["assert dedupe([1,1,2])==[1,2]"]})
    assert score == 100.0, reason


@pytest.mark.parametrize("css", [
    "<nav></nav><style>nav{display:inline-flex}@media (max-width:599.98px){nav{display:block}}</style>",
    "<nav></nav><style>nav{display:grid;grid-auto-flow:column}@media (max-width:37.5em){nav{grid-auto-flow:row}}</style>",
    "<nav></nav><style>nav{display:block}@media (min-width:601px){nav{display:flex;flex-direction:row}}</style>",
])
def test_web_nav_accepts_modern_css_variants(css):
    assert scoring.score_web_nav(css, {})[0] == 100.0


def test_alternative_answers_git_and_js():
    git = scoring.score_contains("git restore --theirs config.json", _task("git_conflict").meta)[0]
    js = scoring.score_tokens_in_code("function debounce(fn,delay){let t;return (...args)=>{clearTimeout(t);t=setTimeout(()=>fn(...args),delay)}}", _task("js_debounce").meta)[0]
    assert git == 100.0
    assert js == 100.0


def test_empty_output_scores_zero_and_affects_quality():
    rows = [
        {"model": "m", "task": "a", "category": "coding_python", "score": 100.0, "tps": 10.0, "output_chars": 10},
        {"model": "m", "task": "b", "category": "coding_python", "score": 0.0, "tps": None, "output_chars": 0, "error_kind": "empty_output"},
    ]
    lb, _ = aggregate(rows, {"coding_python": 1.0}, {"a": 1.0, "b": 1.0})
    assert lb[0]["quality"] == 50.0
    assert lb[0]["completion_rate"] == 0.5
    assert lb[0]["value_per_gb"] is None


def test_classify_row_does_not_false_positive_10_of_10():
    assert classify_row({"model":"m", "task":"t", "score":100.0, "reason":"10/10 lines"}) is None


def test_dump_raw_and_export_review_include_raw(tmp_path):
    task = _task("web_nav")
    raw_path = _dump_raw(tmp_path / "run1", task, "model/name:tag", "hello")
    assert raw_path.exists()
    (tmp_path / "run1" / "scorecard.csv").write_text("x\n")
    (tmp_path / "run1" / "fingerprints.json").write_text("{}")
    pack = export_review([tmp_path / "run1"], tmp_path / "pack.zip")
    import zipfile
    with zipfile.ZipFile(pack) as z:
        names = set(z.namelist())
    assert any("raw/web_nav" in n for n in names)
    assert any(n.endswith("fingerprints.json") for n in names)


def test_needle_records_attempts_and_skips():
    cfg = Config()
    client = MockClient()
    res = _run_once(client, cfg, "qwen2.5-coder:14b", _task("needle"))
    assert "needle_attempted" in res
    assert "needle_skipped" in res
    assert any(item["size"] == 65536 for item in res["needle_skipped"])
    assert res["output_chars"] is None
