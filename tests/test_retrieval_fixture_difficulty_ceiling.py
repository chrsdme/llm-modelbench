"""Privacy-safe retrieval fixtures must not be solvable by keyword overlap alone."""
import re

from llm_modelbench.tasks import TASKS


def _task(task_id):
    return next(task for task in TASKS if task.id == task_id)


def _bow(text):
    return set(re.findall(r"[a-z]+", text.lower()))


def _jaccard(left, right):
    return len(left & right) / len(left | right) if left and right else 0.0


def _trivial_baseline_recall_at_1(task):
    docs, queries = task.meta["docs"], task.meta["queries"]
    doc_bow = {key: _bow(value) for key, value in docs.items()}
    hits = 0
    for query, gold in queries:
        ranked = sorted(docs, key=lambda key: _jaccard(_bow(query), doc_bow[key]), reverse=True)
        hits += ranked[0] == gold
    return hits / len(queries)


def test_services_hard_is_not_solvable_by_keyword_overlap_alone():
    assert _trivial_baseline_recall_at_1(_task("ret_uk_services_hard")) < 1.0


def test_adversarial_is_not_solvable_by_keyword_overlap_alone():
    assert _trivial_baseline_recall_at_1(_task("ret_uk_adversarial")) < 1.0


def test_adversarial_distractors_share_real_words_with_target_or_query():
    task = _task("ret_uk_adversarial")
    docs, queries, cases = task.meta["docs"], task.meta["queries"], task.meta["cases"]
    query_by_target = {target: query for query, target in queries}
    for case in cases:
        words = _bow(docs[case["target"]]) | _bow(query_by_target[case["target"]])
        for distractor in case["distractors"]:
            assert words & _bow(docs[distractor])
