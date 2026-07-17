import re

from llm_modelbench.config import Config
from llm_modelbench.runner import _score_task
from llm_modelbench.tasks import TASKS


def _task(task_id):
    return next(task for task in TASKS if task.id == task_id)


def _words(text):
    return {word for word in re.findall(r"[a-z]{4,}", text.lower()) if word not in {"with", "that", "this", "after", "from", "when"}}


def test_adversarial_cases_have_unique_targets_and_semantic_distractors():
    task = _task("ret_uk_adversarial")
    docs, queries, cases = task.meta["docs"], task.meta["queries"], task.meta["cases"]
    query_by_target = {target: query for query, target in queries}

    assert _task("ret_ukdocs").id == "ret_ukdocs"
    assert _task("ret_uk_services_hard").id == "ret_uk_services_hard"
    assert len(cases) >= 6
    assert len(query_by_target) == len(cases)
    for case in cases:
        target, distractors = case["target"], case["distractors"]
        assert target in docs and query_by_target[target].strip() and docs[target].strip()
        assert len(distractors) >= 3 and len(distractors) == len(set(distractors))
        target_words = _words(docs[target]) | _words(query_by_target[target])
        for distractor in distractors:
            assert distractor in docs and docs[distractor].strip()
            assert len(target_words & _words(docs[distractor])) >= 1


def test_adversarial_retrieval_preserves_model_under_test_provenance():
    task = _task("ret_uk_adversarial")
    doc_ids = list(task.meta["docs"])
    lookup = {text: doc_id for doc_id, text in task.meta["docs"].items()}
    lookup.update({query: target for query, target in task.meta["queries"]})
    calls = []

    class Client:
        def embed(self, model, texts):
            calls.append(model)
            vectors = []
            for text in texts:
                vector = [0.0] * len(doc_ids)
                vector[doc_ids.index(lookup[text])] = 1.0
                vectors.append(vector)
            return vectors

    score, reason, _ = _score_task(Client(), Config(embed_model="wrong"), task, "", "mxbai-embed-large:latest")
    assert score == 100.0
    assert "embed_model=mxbai-embed-large:latest" in reason
    assert calls == ["mxbai-embed-large:latest"]
