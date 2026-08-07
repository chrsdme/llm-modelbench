import pytest

from llm_modelbench.runtime_identity import (
    RuntimeExecutionSettings, RuntimeIdentity, RuntimeModelIdentity,
    compare_runtime_identities,
)


U1 = "GPU-00000000-0000-0000-0000-000000000001"
U2 = "GPU-00000000-0000-0000-0000-000000000002"


def _identity(**changes):
    value = dict(backend="llama_cpp", adapter_identity="LlamaCppBackendAdapter", endpoint="http://127.0.0.1:8081",
                 profile_name="external", profile_provenance="configured", profile_schema_version=1, server_version="build-a",
                 model=RuntimeModelIdentity("m", "m", "sha256:one", provenance="fixture"), physical_gpu_uuids=(U1, U2),
                 declared_device_order=(U1, U2), execution=RuntimeExecutionSettings("layer_split", {U1: 3, U2: 1}, context_size=4096),
                 evidence_provenance="fixture")
    value.update(changes)
    return RuntimeIdentity(**value)


def test_identity_hash_is_deterministic_and_runtime_material_changes_refuse():
    frozen = _identity()
    assert frozen.identity_hash == _identity().identity_hash
    assert compare_runtime_identities(frozen, _identity()).compatible
    changes = (("backend", "ollama"), ("endpoint", "http://127.0.0.1:11434"),
               ("server_version", "build-b"), ("execution", RuntimeExecutionSettings("layer_split", {U1: 1, U2: 3}, context_size=8192)))
    for field, replacement in changes:
        changed = _identity(**{field: replacement})
        assert not compare_runtime_identities(frozen, changed).compatible
    changed = _identity(physical_gpu_uuids=(U1,), declared_device_order=(U1,),
                        execution=RuntimeExecutionSettings("single_device", context_size=4096))
    assert not compare_runtime_identities(frozen, changed).compatible


def test_identity_rejects_secret_endpoint_malformed_uuid_and_positional_weights():
    with pytest.raises(ValueError, match="credential"):
        _identity(endpoint="http://token@127.0.0.1:8081")
    with pytest.raises(ValueError, match="canonical"):
        _identity(physical_gpu_uuids=("CUDA0",))
    with pytest.raises(ValueError):
        _identity(execution=RuntimeExecutionSettings("layer_split", ((U1, 1), (U1, 1))))


def test_legacy_identity_fails_closed():
    result = compare_runtime_identities(None, _identity())
    assert not result.compatible
    assert result.mismatches[0].code == "legacy_runtime_identity_missing"


def test_material_execution_and_digest_mismatches_have_specific_codes():
    frozen = _identity()
    checks = [
        (_identity(model=RuntimeModelIdentity("m", "m", "sha256:two", provenance="fixture")), "model_artifact_changed"),
        (_identity(execution=RuntimeExecutionSettings("layer_split", {U1: 3, U2: 1}, context_size=4096, batch_size=2)), "batch_size_changed"),
        (_identity(execution=RuntimeExecutionSettings("layer_split", {U1: 3, U2: 1}, context_size=4096, micro_batch_size=2)), "micro_batch_size_changed"),
        (_identity(execution=RuntimeExecutionSettings("layer_split", {U1: 3, U2: 1}, context_size=4096, kv_cache_type="q8")), "kv_cache_type_changed"),
        (_identity(execution=RuntimeExecutionSettings("layer_split", {U1: 3, U2: 1}, context_size=4096, allow_cpu_spill=True)), "spill_policy_changed"),
    ]
    for changed, code in checks:
        assert code in [item.code for item in compare_runtime_identities(frozen, changed).mismatches]
