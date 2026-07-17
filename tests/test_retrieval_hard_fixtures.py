from llm_modelbench.config import Config
from llm_modelbench.runner import _score_task
from llm_modelbench.tasks import TASKS


def _task(task_id):
    return next(task for task in TASKS if task.id == task_id)


def test_hard_retrieval_fixture_has_unique_reachable_targets_and_distractors():
    task = _task("ret_uk_services_hard")
    docs = task.meta["docs"]
    queries = task.meta["queries"]
    targets = [target for _, target in queries]

    assert task.family == "embedding"
    assert task.difficulty > 0.0
    assert len(targets) == len(set(targets))
    assert set(targets) <= set(docs)
    assert len(docs) >= len(queries) * 2
    assert _task("ret_ukdocs").scorer == "retrieval"


def test_hard_retrieval_scoring_keeps_model_under_test_provenance():
    task = _task("ret_uk_services_hard")
    doc_ids = list(task.meta["docs"])
    by_text = {text: doc_id for doc_id, text in task.meta["docs"].items()}
    by_text.update({query: target for query, target in task.meta["queries"]})
    calls = []

    class Client:
        def embed(self, model, texts):
            calls.append((model, list(texts)))
            vectors = []
            for text in texts:
                vector = [0.0] * len(doc_ids)
                vector[doc_ids.index(by_text[text])] = 1.0
                vectors.append(vector)
            return vectors

    score, reason, _ = _score_task(Client(), Config(embed_model="wrong-fallback"), task, "", "bge-m3:latest")

    assert score == 100.0
    assert "recall@1=1.00" in reason and "mrr=1.00" in reason
    assert "embed_model=bge-m3:latest" in reason
    assert calls[0][0] == "bge-m3:latest"
