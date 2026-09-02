"""Anvil Stage 3.2D-1 -- deterministic stable RuntimeProfileIdentity factory."""
from llm_modelbench.identity import (
    GPU_PLACEMENT_POLICY, ModelArtifactIdentity, RuntimeInstanceIdentity,
    RuntimeProfileIdentity, resolve_runtime_profile_identity,
)
from llm_modelbench.runtime_identity import RuntimeExecutionSettings


def _settings(**kw) -> RuntimeExecutionSettings:
    return RuntimeExecutionSettings(**kw)


# --- stable across restarts, sensitive to the recipe --------------------------

def test_materially_identical_config_yields_the_same_stable_key():
    a = resolve_runtime_profile_identity(backend="llama_cpp", backend_version="b4000",
                                         execution_settings=_settings(context_size=8192, kv_cache_type="q8_0"))
    b = resolve_runtime_profile_identity(backend="llama_cpp", backend_version="b4000",
                                         execution_settings=_settings(context_size=8192, kv_cache_type="q8_0"))
    assert a.stable_key() == b.stable_key()


def test_material_runtime_config_change_changes_the_key():
    base = resolve_runtime_profile_identity(backend="llama_cpp", execution_settings=_settings(context_size=8192))
    ctx = resolve_runtime_profile_identity(backend="llama_cpp", execution_settings=_settings(context_size=16384))
    kv = resolve_runtime_profile_identity(backend="llama_cpp",
                                          execution_settings=_settings(context_size=8192, kv_cache_type="q4_0"))
    strat = resolve_runtime_profile_identity(backend="llama_cpp",
                                             execution_settings=_settings(context_size=8192, strategy="layer_split"))
    assert base.stable_key() != ctx.stable_key()
    assert base.stable_key() != kv.stable_key()
    assert base.stable_key() != strat.stable_key()


def test_backend_version_change_changes_the_key():
    a = resolve_runtime_profile_identity(backend="ollama", backend_version="0.5.0")
    b = resolve_runtime_profile_identity(backend="ollama", backend_version="0.6.0")
    assert a.stable_key() != b.stable_key()


# --- separation: instance facts must not move the profile key -----------------

def test_instance_facts_do_not_change_the_profile_key():
    profile = resolve_runtime_profile_identity(backend="llama_cpp", backend_version="b4000",
                                               execution_settings=_settings(context_size=8192))
    art = ModelArtifactIdentity(artifact_set_id="a1", primary_sha256="a1")
    inst_a = RuntimeInstanceIdentity(profile=profile, endpoint="http://127.0.0.1:8080",
                                     process_id=4127, started_at="2026-09-02T10:00:00Z",
                                     gpu_uuid_assignment=("GPU-00000000-0000-0000-0000-000000000001",),
                                     loaded_artifact=art)
    inst_b = RuntimeInstanceIdentity(profile=profile, endpoint="http://127.0.0.1:9090",
                                     process_id=8192, started_at="2026-09-02T18:00:00Z",
                                     gpu_uuid_assignment=("GPU-00000000-0000-0000-0000-000000000002",),
                                     loaded_artifact=art)
    # endpoint + PID + started_at + GPU-UUID-assignment all differ ...
    assert inst_a.profile.stable_key() == inst_b.profile.stable_key()
    # ... but the concrete instances remain distinguishable
    assert inst_a.instance_key() != inst_b.instance_key()


def test_placement_class_change_alone_does_not_affect_profile_identity():
    # `full_gpu` / `multi_gpu` / `ram_spill` is per-execution evidence, not a
    # RuntimeProfileIdentity field -- it cannot enter the stable key at all.
    profile = resolve_runtime_profile_identity(backend="llama_cpp", execution_settings=_settings(context_size=8192))
    assert "ram_spill" not in profile.stable_key()  # not derivable from it
    key_fields = (profile.backend, profile.backend_version, profile.protocol_version,
                  profile.template_hash, profile.runtime_configuration_hash,
                  profile.gpu_policy, profile.feature_flags)
    assert "full_gpu" not in str(key_fields) and "ram_spill" not in str(key_fields)


def test_spill_permission_does_not_change_the_reusable_profile_key():
    # Same benchmark recipe, one run with --allow-ram-spill and one without:
    # the reusable runtime recipe is identical; only an execution-time operator
    # permission differed. That permission is identity-bearing in
    # RuntimeIdentity.identity_hash (Stage 3.2C-2b), NOT here (§9).
    permitted = resolve_runtime_profile_identity(
        backend="llama_cpp", execution_settings=_settings(context_size=8192, allow_cpu_spill=True))
    forbidden = resolve_runtime_profile_identity(
        backend="llama_cpp", execution_settings=_settings(context_size=8192, allow_cpu_spill=None))
    assert permitted.stable_key() == forbidden.stable_key()


def test_allocation_weights_do_not_enter_the_profile_key():
    # allocation_weights are keyed by physical GPU UUID (an environment fact,
    # §8). `strategy` carries the recipe-level split choice; the weights do not.
    u1 = "GPU-00000000-0000-0000-0000-000000000001"
    u2 = "GPU-00000000-0000-0000-0000-000000000002"
    a = resolve_runtime_profile_identity(backend="llama_cpp", execution_settings=_settings(
        strategy="layer_split", allocation_weights={u1: 1.0, u2: 3.0}))
    b = resolve_runtime_profile_identity(backend="llama_cpp", execution_settings=_settings(
        strategy="layer_split", allocation_weights={u1: 2.0, u2: 2.0}))
    assert a.stable_key() == b.stable_key()


# --- the D-1 improvement over the current minimal adapter key ----------------

def test_resolved_key_is_richer_than_the_backend_only_adapter_key():
    minimal = RuntimeProfileIdentity(backend="llama_cpp", backend_version="b4000")
    resolved = resolve_runtime_profile_identity(backend="llama_cpp", backend_version="b4000",
                                                execution_settings=_settings(context_size=8192))
    assert minimal.stable_key() != resolved.stable_key()
    assert resolved.gpu_policy == GPU_PLACEMENT_POLICY
    assert resolved.runtime_configuration_hash is not None


def test_honest_none_for_unsourced_fields():
    resolved = resolve_runtime_profile_identity(backend="ollama")
    assert resolved.protocol_version is None
    assert resolved.template_hash is None
    assert resolved.feature_flags == ()
