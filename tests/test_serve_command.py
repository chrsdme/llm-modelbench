import json

from llm_modelbench.serve import build_index, endpoint_response, route, serve


def _write_summary(path, rows):
    path.mkdir()
    (path / "summary.json").write_text(json.dumps(rows))


def test_build_index_unions_categories_and_route_orders_tied_band(tmp_path):
    coding, agentic = tmp_path / "coding", tmp_path / "agentic"
    _write_summary(coding, [
        {"model": "small-fast", "size_gb": 4, "tok_s": 30, "categories": {"coding_python": 90}},
        {"model": "large-slow", "size_gb": 8, "tok_s": 10, "categories": {"coding_python": 89.7}},
    ])
    _write_summary(agentic, [{"model": "small-fast", "size_gb": 4, "tok_s": 31, "categories": {"agentic_tool": 80}}])
    index = build_index([coding, agentic])

    assert index["models"]["small-fast"]["categories"] == {"coding_python": 90, "agentic_tool": 80}
    routed = route(index, "coding_python", 8)
    assert [row["model"] for row in routed["tied_band"]] == ["small-fast", "large-slow"]
    assert routed["recommended"] == "small-fast"
    assert "no model" in route(index, "ocr", 8)["note"]


def test_read_only_endpoint_dispatch_exposes_health_models_and_routing(tmp_path):
    run = tmp_path / "run"
    _write_summary(run, [{"model": "m", "size_gb": 2, "tok_s": 9, "categories": {"ocr": 77}}])
    index = build_index([run])
    assert endpoint_response(index, "/health")[0]["read_only"] is True
    assert endpoint_response(index, "/models")[0]["models"][0]["model"] == "m"
    assert endpoint_response(index, "/routing?use_case=ocr&vram=4")[0]["recommended"] == "m"
    assert endpoint_response(index, "/routing")[1] == 400


def test_missing_summaries_are_reported_as_degraded(tmp_path):
    index = build_index([tmp_path / "missing"])
    health, code = endpoint_response(index, "/health")
    assert code == 200
    assert health["status"] == "degraded"
    assert health["valid_runs"] == 0
    assert health["runs_loaded"][0]["error"]


def test_serve_refuses_remote_binding_without_explicit_opt_in(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="allow-remote"):
        serve([tmp_path / "missing"], "0.0.0.0", 8756)


def test_serve_refuses_empty_index_by_default(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="no valid run summaries"):
        serve([tmp_path / "missing"], "127.0.0.1", 8756)
