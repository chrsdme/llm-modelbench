from llm_modelbench.config import Config
from llm_modelbench.runner import _run_once
from llm_modelbench.tasks import TASKS
import json

def test_retrieval_row_persists_private_case_diagnostics():
    task = next(t for t in TASKS if t.id == "ret_ukdocs")
    ids = list(task.meta["docs"])
    lookup = {text: key for key, text in task.meta["docs"].items()}
    lookup.update({query: target for query, target in task.meta["queries"]})
    class Client:
        def embed(self, model, texts):
            out=[]
            for text in texts:
                v=[0.0]*len(ids); v[ids.index(lookup[text])]=1.0; out.append(v)
            return out
    row = _run_once(Client(), Config(), "embed-model", task)
    case = row["retrieval_cases"][0]
    assert row["embed_model"] == "embed-model"
    assert {"query_index", "target_doc_id", "top1_doc_id", "top3_doc_ids", "target_rank", "margin", "nearest_distractor_doc_id", "pass_at_1", "embed_model"} <= set(case)
    assert "query" not in case and "document" not in case
    loaded = json.loads(json.dumps(row))
    assert len(loaded["retrieval_cases"]) == len(task.meta["queries"])
