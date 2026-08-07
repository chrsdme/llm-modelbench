from llm_modelbench import doctor, report


def test_report_header_renders_optional_telemetry_without_placement_claims():
    lines = report._header_lines({"runtime_telemetry": {
        "backend": "ollama", "endpoint_available": False,
        "declared_gpu_uuids": ["GPU-declared"], "observed_gpu_uuids": ["GPU-observed"],
    }})
    text = "\n".join(lines)
    assert "GPU-declared" in text and "GPU-observed" in text
    assert "does not prove layer" in text


def test_report_old_artifact_without_telemetry_stays_unchanged():
    text = "\n".join(report._header_lines({}))
    assert "Runtime telemetry:" not in text


def test_doctor_renders_idle_and_partial_telemetry_conservatively():
    text = doctor.render({"gpu": {}, "gpus": [], "runtime_telemetry": {
        "endpoint_available": False, "physical_inventory": [],
        "schema_version": 1,
        "gpu_processes": {"successful": True},
        "process_discovery": {"socket_evidence_complete": False},
    }})
    assert "compute-query=supported/idle" in text
    assert "socket-evidence=partial" in text
    assert "schema=readable" in text


def test_doctor_distinguishes_failed_compute_query():
    text = doctor.render({"gpu": {}, "gpus": [], "runtime_telemetry": {
        "gpu_processes": {"successful": False, "errors": [{"state": "failed"}]},
        "process_discovery": {},
    }})
    assert "compute-query=failed" in text


def test_report_handles_missing_malformed_and_future_optional_artifacts(tmp_path):
    assert report._load_runtime_telemetry(tmp_path / "runtime_telemetry.json") == {}
    path = tmp_path / "runtime_telemetry.json"
    path.write_text("not json")
    assert report._load_runtime_telemetry(path)["status"] == "unavailable"
    path.write_text('{"schema_version": 99}')
    assert "unsupported" in report._load_runtime_telemetry(path)["warning"]
