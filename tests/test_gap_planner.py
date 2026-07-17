from llm_modelbench.gap_planner import gap_report
from llm_modelbench.classify import classify_model, families_for
from llm_modelbench.tasks import TASKS

def test_gap_report_is_advisory_and_digest_based():
    class C:
        def tags(self): return [{"name":"m", "digest":"d"}]
        def capabilities(self, m): return ["embedding"]
    assert gap_report(C(),{},TASKS,classify_model,families_for)["m"] == ["retrieval"]

def test_gap_report_reads_digest_from_tags_row_not_show():
    class C:
        def tags(self): return [{"name": "m", "digest": "d"}]
        def capabilities(self, m): return ["embedding"]
    retrieval_ids = [t.id for t in TASKS if t.category == "retrieval"]
    ledger = {"d": {"categories": {"retrieval": {"task_ids_covered": retrieval_ids}}}}
    assert gap_report(C(), ledger, TASKS, classify_model, families_for) == {}
