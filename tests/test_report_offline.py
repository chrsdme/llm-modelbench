import csv
import json

from llm_modelbench.config import Config
from llm_modelbench.report import _html, build
from llm_modelbench.runtime_identity import (
    RuntimeExecutionSettings,
    RuntimeIdentity,
    RuntimeModelIdentity,
    write_runtime_identity_artifact,
)


def test_html_report_is_self_contained_and_has_csp(tmp_path):
    path = tmp_path / "report.html"
    leaderboard = [{
        "model": "m", "class": "text", "quality": 80.0, "tok_s": 12.0,
        "offload": 0.0, "value_per_gb": 10.0, "score_blended": 80.0,
        "size_gb": 8.0, "err": 0, "completion_rate": 1.0,
        "categories": {"reasoning": 80.0},
    }]
    _html(path, leaderboard, {"reasoning": [("m", 80.0)]}, [], object(), {})
    text = path.read_text()
    assert "Content-Security-Policy" in text
    assert "https://" not in text
    assert "cdn.jsdelivr" not in text
    assert "Top 5 category matrix" in text


def test_mixed_runtime_provenance_is_additive_and_never_injected_into_legacy(tmp_path):
    uuid_a = "GPU-00000000-0000-0000-0000-0000000000a1"
    uuid_b = "GPU-00000000-0000-0000-0000-0000000000b2"
    identity = RuntimeIdentity(
        backend="llama_cpp", adapter_identity="fixture", endpoint="http://127.0.0.1:8081",
        profile_name="fixture-profile", profile_provenance="test", profile_schema_version=1,
        server_version="fixture-build", model=RuntimeModelIdentity("current", "current", "sha256:current", provenance="test"),
        physical_gpu_uuids=(uuid_a, uuid_b), declared_device_order=(uuid_a, uuid_b),
        execution=RuntimeExecutionSettings("layer_split", {uuid_a: 3, uuid_b: 1}, context_size=4096),
        evidence_provenance="test",
    )
    variant = RuntimeIdentity(
        backend="llama_cpp", adapter_identity="fixture", endpoint="http://127.0.0.1:8081",
        profile_name="fixture-profile", profile_provenance="test", profile_schema_version=1,
        server_version="fixture-build", model=RuntimeModelIdentity("current", "current", "sha256:current", provenance="test"),
        physical_gpu_uuids=(uuid_a, uuid_b), declared_device_order=(uuid_a, uuid_b),
        execution=RuntimeExecutionSettings("layer_split", {uuid_a: 3, uuid_b: 1}, context_size=8192),
        evidence_provenance="test",
    )
    rows = [
        {"model": "current", "task": "py_good", "task_hash": "same", "category": "coding", "score": 50,
         "class": "general", "size_gb": 1, "tps": 1, "runtime_identity_schema_version": 1,
         "runtime_identity_hash": identity.identity_hash, "runtime_variant_id": identity.identity_hash,
         "backend": "llama_cpp", "runtime_profile": "fixture-profile", "model_artifact_digest": "sha256:current",
         "physical_gpu_uuids": [uuid_a, uuid_b], "declared_device_order": [uuid_a, uuid_b],
         "execution_strategy": "layer_split", "allocation_weights": {uuid_a: 3, uuid_b: 1}, "context_size": 4096},
        {"model": "current", "task": "py_good", "task_hash": "same", "category": "coding", "score": 50,
         "class": "general", "size_gb": 1, "tps": 1, "runtime_identity_schema_version": 1,
         "runtime_identity_hash": variant.identity_hash, "runtime_variant_id": variant.identity_hash,
         "backend": "llama_cpp", "runtime_profile": "fixture-profile", "model_artifact_digest": "sha256:current",
         "physical_gpu_uuids": [uuid_a, uuid_b], "declared_device_order": [uuid_a, uuid_b],
         "execution_strategy": "layer_split", "allocation_weights": {uuid_a: 3, uuid_b: 1}, "context_size": 8192},
        {"model": "legacy", "task": "py_good", "task_hash": "legacy", "category": "coding", "score": 50,
         "class": "general", "size_gb": 1, "tps": 1},
    ]
    (tmp_path / "raw_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_runtime_identity_artifact(tmp_path / "runtime_identity.json", identity)
    (tmp_path / "runtime_telemetry.json").write_text(json.dumps({"schema_version": 1, "status": "unavailable"}))
    (tmp_path / "runtime_fit.json").write_text(json.dumps({"status": "conditional_fit"}))

    build(tmp_path, Config())

    summary = json.loads((tmp_path / "summary.json").read_text())
    current = next(row for row in summary if row["model"] == "current")
    legacy = next(row for row in summary if row["model"] == "legacy")
    assert current["runtime_variant_count"] == 2
    assert current["runtime_identity_artifact"]["status"] == "available"
    assert legacy["runtime_identity_state"] == "legacy_unknown"
    assert legacy["runtime_identity_artifact"]["status"] == "legacy_unknown"
    assert all(not item.get("runtime_identity_hash") for item in legacy["runtime_variants"])
    meta = json.loads((tmp_path / "summary_meta.json").read_text())["runtime_provenance"]
    assert meta["runtime_variant_counts_by_model"] == {"current": 2, "legacy": 1}
    assert meta["runtime_fit_state"] == {"status": "conditional_fit", "advisory": True}
    with (tmp_path / "scorecard.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert {"runtime_identity_state", "runtime_variant_count", "runtime_backends", "runtime_profiles", "runtime_identity_hashes"} <= set(csv_rows[0])
    assert next(row for row in csv_rows if row["model"] == "legacy")["runtime_identity_state"] == "legacy_unknown"
