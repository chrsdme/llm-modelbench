
from llm_modelbench.runner import _dump, _subjective_raw_reason
from llm_modelbench.tasks import TASKS


def _task(task_id):
    return next(t for t in TASKS if t.id == task_id)


def test_subjective_dump_returns_path_and_reason_is_informative(tmp_path):
    task = _task("kb_taxonomy")
    out = "# Grand Bazaar\nUseful KB entry body."
    path = _dump(tmp_path, task, "model/name:tag", out)
    assert path.exists()
    reason = _subjective_raw_reason(tmp_path, task, "model/name:tag", out, path)
    assert "raw only, judge off" in reason
    assert "chars" in reason
    assert "subjective/kb_taxonomy" in reason


def test_subjective_raw_reason_handles_no_dump(tmp_path):
    task = _task("wr_rag")
    reason = _subjective_raw_reason(tmp_path, task, "m", "hello", None)
    assert reason == "raw only, judge off: 5 chars (not dumped)"
