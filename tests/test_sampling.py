from llm_modelbench.config import Config
from llm_modelbench.runner import _samples_for_task
from llm_modelbench.tasks import TASKS


def _task(task_id):
    return next(t for t in TASKS if t.id == task_id)


def test_smart_sampling_only_multiplies_subjective_tasks():
    cfg = Config(samples=3)
    assert _samples_for_task(_task("py_anagram"), cfg, "smart") == 1
    assert _samples_for_task(_task("json_extract"), cfg, "smart") == 1
    assert _samples_for_task(_task("needle"), cfg, "smart") == 1
    assert _samples_for_task(_task("ret_ukdocs"), cfg, "smart") == 1
    assert _samples_for_task(_task("wr_rag"), cfg, "smart") == 3
    assert _samples_for_task(_task("kb_taxonomy"), cfg, "smart") == 3


def test_all_sampling_keeps_legacy_behavior():
    cfg = Config(samples=3)
    assert _samples_for_task(_task("py_anagram"), cfg, "all") == 3
    assert _samples_for_task(_task("wr_rag"), cfg, "all") == 3


def test_smart_sampling_judge_off_runs_subjective_once():
    cfg = Config(samples=3)
    assert _samples_for_task(_task("wr_rag"), cfg, "smart", judge_mode="off") == 1
    assert _samples_for_task(_task("kb_taxonomy"), cfg, "smart", judge_mode="off") == 1
