from llm_modelbench.reliability import cell_summary, empirical_noise_band, category_summary
from llm_modelbench.compare import repeatability_report
import json

def test_reliability_statuses_and_noise_band():
    stable=cell_summary([{"score":80,"reason":"ok"},{"score":80,"reason":"ok"}])
    moving=cell_summary([{"score":80,"reason":"ok"},{"score":85,"reason":"ok"}])
    assert stable["status"] == "stable" and stable["range"] == 0
    assert moving["status"] == "moving" and moving["range"] == 5
    assert cell_summary([{"score":80,"reason":"a"},{"score":80,"reason":"b"}])["status"] == "reason-moving"
    assert cell_summary([{"score":80}])["status"] == "insufficient-repeats"
    assert cell_summary([{"score":80},None])["status"] == "missing"
    assert empirical_noise_band([stable,moving]) == 5
    assert empirical_noise_band([]) is None
    assert category_summary([("x",[{"score":1},{"score":2}])])["x"]["max_range"] == 1

def test_repeat_report_surfaces_noise_band_and_insufficient_evidence(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"; a.mkdir(); b.mkdir()
    (a / "raw_results.jsonl").write_text(json.dumps({"model":"m","task":"t","score":80,"reason":"ok"})+"\n")
    (b / "raw_results.jsonl").write_text(json.dumps({"model":"m","task":"t","score":83,"reason":"ok"})+"\n")
    text = repeatability_report([a,b])
    assert "empirical noise band" in text and "3.0" in text
    assert "moving" in text

def test_repeat_report_counts_missing_cells_without_calling_them_stable(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"; a.mkdir(); b.mkdir()
    (a / "raw_results.jsonl").write_text(json.dumps({"model":"m","task":"only-a","score":80,"reason":"ok"})+"\n")
    (b / "raw_results.jsonl").write_text("")
    text = repeatability_report([a,b])
    assert "| `m` | `only-a` | 80, null | n/a | none | missing |" in text
    assert "missing=1" in text and "stable" not in text.split("Summary:")[0]
