from llm_modelbench.coverage import pending_categories_for_model, update_ledger_from_run
from llm_modelbench.tasks import TASKS

def test_coverage_digest_and_staleness():
    ledger = update_ledger_from_run({}, raw_rows=[{"model":"old", "task":"ret_ukdocs", "score":100}], identities={"old":{"digest":"d"}}, tasks=[t for t in TASKS if t.id=="ret_ukdocs"], benchmark_version="x", out_dir="r", timestamp="1")
    assert pending_categories_for_model(ledger,"d",["embedding"],[t for t in TASKS if t.id=="ret_ukdocs"]) == []
    assert pending_categories_for_model(ledger,"d",["embedding"],TASKS) == ["retrieval"]
    assert pending_categories_for_model(ledger,None,["embedding"],TASKS) == ["retrieval"]
