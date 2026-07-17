import json
from llm_modelbench import report

def test_retrieval_artifacts_include_partial_and_full_cases(tmp_path):
    rows=[{"model":"old","task":"ret","category":"retrieval","score":80,"reason":"recall@1=0.8; embed_model=old"}, {"model":"new","task":"ret","category":"retrieval","score":90,"reason":"embed_model=new","retrieval_cases":[{"query_index":0,"gold_doc_id":"a","top1_doc_id":"a","top3_doc_ids":["a","b"],"target_rank":1,"margin":.2}]}]
    report._retrieval_diagnostics(tmp_path, rows)
    data=json.loads((tmp_path/"retrieval_diagnostics.json").read_text())
    assert data[0]["embed_model"] == "old" and "unavailable" in data[0]["note"]
    assert data[1]["cases"][0]["top3_doc_ids"] == ["a","b"]
    assert "margin=0.2" in (tmp_path/"retrieval_diagnostics.md").read_text()
