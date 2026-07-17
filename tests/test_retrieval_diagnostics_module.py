from llm_modelbench.retrieval_diagnostics import diagnostics

def test_diagnostics_known_ranking_and_margin():
    meta={"docs":{"a":"a","b":"b","c":"c"},"queries":[("q","b")]}
    vectors=[[1,0],[.8,.2],[0,1],[.9,.1]]
    row=diagnostics(lambda _: vectors,meta,"embed")[0]
    assert row["top3_doc_ids"] == ["a","b","c"] and row["target_rank"] == 2
    assert row["nearest_distractor_doc_id"] == "a" and row["embed_model"] == "embed"
