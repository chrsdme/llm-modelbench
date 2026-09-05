"""Anvil Stage 3B.4 corrective (third) -- pre-spawn KV-cache estimate wiring.

Follows the second corrective (210796f, "Fail closed managed llama context
load"), which made the resolver refuse ``FIT_UNKNOWN`` for owned managed
placement whenever ``kv_cache_bytes`` is ``None``. This module supplies that
estimate from the resolved local GGUF artifact's own header -- never from a
live backend, never invented when no concrete context is configured.

No real GPU / Ollama / llama.cpp / inference. GGUF bytes are built in-memory
via the same minimal writer as ``test_gguf_metadata.py``.
"""
from __future__ import annotations

import struct

import pytest

from llm_modelbench import cli
from llm_modelbench import runtime_materialisation as rm
from llm_modelbench.config import Config
from llm_modelbench.gguf_metadata import KV_CACHE_DTYPE_BYTES, PARALLEL_SEQUENCES
from llm_modelbench.llama_server_materialisation import build_llama_server_command
from llm_modelbench.runtime_profiles import RuntimeCandidate, RuntimeProfile

_T_UINT32 = 4
_T_STRING = 8


def _kv_string(key: str, value: str) -> bytes:
    key_b = key.encode("utf-8")
    val_b = value.encode("utf-8")
    return (
        struct.pack("<Q", len(key_b)) + key_b
        + struct.pack("<I", _T_STRING)
        + struct.pack("<Q", len(val_b)) + val_b
    )


def _kv_u32(key: str, value: int) -> bytes:
    key_b = key.encode("utf-8")
    return (
        struct.pack("<Q", len(key_b)) + key_b
        + struct.pack("<I", _T_UINT32)
        + struct.pack("<I", value)
    )


def _valid_llama_gguf_bytes(*, context_length=8192) -> bytes:
    """A minimal, spec-conforming llama-architecture GGUF header: 32 layers,
    32 heads, 4096 embedding (-> head_dimension 128 derived), no explicit
    key/value length (exercises the same derivation path as a real
    llama.cpp-family GGUF that omits them)."""
    pairs = [
        _kv_string("general.architecture", "llama"),
        _kv_u32("llama.block_count", 32),
        _kv_u32("llama.attention.head_count", 32),
        _kv_u32("llama.embedding_length", 4096),
        _kv_u32("llama.context_length", context_length),
    ]
    header = (
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(pairs))
    )
    return header + b"".join(pairs)


@pytest.fixture()
def valid_gguf(tmp_path):
    path = tmp_path / "bench-model.gguf"
    path.write_bytes(_valid_llama_gguf_bytes())
    return str(path)


# ===========================================================================
# B. KV estimate wiring (the helper function directly)
# ===========================================================================
def test_kv_estimate_none_when_no_resolved_path():
    kv, reason = cli._managed_kv_cache_estimate(None, 4096)
    assert kv is None
    assert reason == "no_resolved_gguf_artifact"


def test_kv_estimate_none_when_no_concrete_context_never_invents_a_default(valid_gguf):
    kv, reason = cli._managed_kv_cache_estimate(valid_gguf, None)
    assert kv is None
    assert reason == "no_concrete_requested_context"


def test_kv_estimate_resolves_from_real_gguf_header_for_the_exact_context(valid_gguf):
    kv, reason = cli._managed_kv_cache_estimate(valid_gguf, 4096)
    assert kv is not None
    assert reason == "derived_kv_cache_bytes"
    # 2 * layers(32) * kv_heads(32) * head_dim(128) * ctx(4096) * dtype(2) * seq(1)
    expected = 2 * 32 * 32 * 128 * 4096 * KV_CACHE_DTYPE_BYTES * PARALLEL_SEQUENCES
    assert kv == expected


def test_kv_estimate_fails_closed_for_malformed_gguf(tmp_path):
    path = tmp_path / "broken.gguf"
    path.write_bytes(b"NOT A GGUF FILE")
    kv, reason = cli._managed_kv_cache_estimate(str(path), 4096)
    assert kv is None
    assert reason == "gguf_metadata_not_a_gguf_file"


def test_kv_estimate_fails_closed_when_context_exceeds_model_maximum(tmp_path):
    path = tmp_path / "small-context.gguf"
    path.write_bytes(_valid_llama_gguf_bytes(context_length=2048))
    kv, reason = cli._managed_kv_cache_estimate(str(path), 4096)
    assert kv is None
    assert reason == "gguf_metadata_resolved_but_requested_context_exceeds_model_maximum"


# ===========================================================================
# C. CLI/product path: ctx_override + gguf_artifacts reaches managed planning
# ===========================================================================
class _FakePreflight:
    def __init__(self, endpoint):
        self.blocked = False
        self.blocker = None
        self.selected_candidate = RuntimeCandidate(
            profile=RuntimeProfile(name="p", backend="llama_cpp", endpoint=endpoint, provenance="configured"),
            health="healthy", source=("saved_profile",), detail="fx",
        )
        self.candidates = [self.selected_candidate]
        self.gpu_inventory = []
        self.topology = object()


def _wire_cli(monkeypatch, *, models, cfg, capture):
    import llm_modelbench.runtime_profiles as rp
    import llm_modelbench.hardware as hw
    monkeypatch.setattr(cli, "load_profiles", lambda store: ([], None))
    monkeypatch.setattr(
        cli, "resolve_operational_preflight",
        lambda *a, **k: _FakePreflight("http://127.0.0.1:8081"),
    )
    monkeypatch.setattr(rp, "discover_backend_executables", lambda **k: None)
    monkeypatch.setattr(cli, "_llama_server_executable_path", lambda be: "/opt/llama-server")
    monkeypatch.setattr(hw, "host_memory_snapshot", lambda: {})

    def _capture_ram(**kwargs):
        capture.update(kwargs)
        return rm.RuntimeMaterialisationOutcome(
            ok=False, backend=kwargs.get("selected_backend") or "", resolution_status="fx",
            refusal_reason="stub", artifact_resolution=kwargs.get("artifact_resolution"),
        )

    monkeypatch.setattr(rm, "production_seams", lambda **k: object())
    monkeypatch.setattr(rm, "resolve_and_materialise_runtime", _capture_ram)

    class _Args:
        pass
    args = _Args()
    args.models = models
    args.unattended = False
    args.runtime_profile = None
    args.runtime_profiles_file = None
    return args


def test_cli_with_ctx_override_and_valid_gguf_reaches_managed_kv_estimate(monkeypatch, valid_gguf):
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": valid_gguf}
    cfg.ctx_override = 4096
    cap = {}
    args = _wire_cli(monkeypatch, models="bench-model", cfg=cfg, capture=cap)
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert cap["requested_context"] == 4096
    assert cap["kv_cache_bytes"] is not None
    expected = 2 * 32 * 32 * 128 * 4096 * KV_CACHE_DTYPE_BYTES * PARALLEL_SEQUENCES
    assert cap["kv_cache_bytes"] == expected
    ar = cap["artifact_resolution"]
    assert ar["kv_cache_bytes"] == expected
    assert ar["kv_cache_bytes_source"] == "derived_kv_cache_bytes"


def test_cli_without_ctx_override_still_fails_closed_before_spawn(monkeypatch, valid_gguf):
    """No context configured -> no KV estimate is invented; the resolver
    (210796f) is the one that turns this into FIT_UNKNOWN, but the CLI seam
    must not paper over it by fabricating a context."""
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": valid_gguf}
    cap = {}
    args = _wire_cli(monkeypatch, models="bench-model", cfg=cfg, capture=cap)
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert cap["requested_context"] is None
    assert cap["kv_cache_bytes"] is None
    ar = cap["artifact_resolution"]
    assert ar["kv_cache_bytes"] is None
    assert ar["kv_cache_bytes_source"] == "no_concrete_requested_context"


def test_cli_invalid_gguf_metadata_fails_before_spawn_with_evidence(monkeypatch, tmp_path):
    bad_path = tmp_path / "bad.gguf"
    bad_path.write_bytes(b"NOT A GGUF FILE")
    cfg = Config()
    cfg.gguf_artifacts = {"bench-model": str(bad_path)}
    cfg.ctx_override = 4096
    cap = {}
    args = _wire_cli(monkeypatch, models="bench-model", cfg=cfg, capture=cap)
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert cap["kv_cache_bytes"] is None
    ar = cap["artifact_resolution"]
    assert ar["kv_cache_bytes"] is None
    assert ar["kv_cache_bytes_source"] == "gguf_metadata_not_a_gguf_file"


def test_reuse_only_path_never_computes_a_kv_estimate(monkeypatch):
    """Ollama / no-artifact llama_cpp is reuse-only -- no fit gate, so no KV
    estimate should even be attempted (nothing to place)."""
    cfg = Config()  # nothing configured
    cap = {}
    args = _wire_cli(monkeypatch, models="bench-model", cfg=cfg, capture=cap)
    cli._resolve_and_materialise_for_run(args, cfg, inventory=[])
    assert cap["kv_cache_bytes"] is None
    ar = cap["artifact_resolution"]
    assert ar["workload_estimate_status"] == "not_required_reuse_only"


# ===========================================================================
# fit and launch must use the same context (chained resolver -> argv test)
# ===========================================================================
def test_resolver_fit_context_matches_the_argv_ctx_size(valid_gguf):
    """A single chained check: the resolver's accepted requested_context is
    exactly what build_llama_server_command emits as --ctx-size -- the class
    of disagreement 210796f was written to prevent."""
    from llm_modelbench.hardware import GPUDevice
    from llm_modelbench.runtime_resolution import resolve_runtime
    from llm_modelbench.topology_budget import topology_from_inventory

    kv_cache_bytes, reason = cli._managed_kv_cache_estimate(valid_gguf, 4096)
    assert reason == "derived_kv_cache_bytes"

    device = GPUDevice(0, "GPU-00000000-1111-2222-3333-444444444444", "0000:01:00.0",
                        "A", 24000.0, None, None)
    candidate = RuntimeCandidate(
        profile=RuntimeProfile(name="llama-local", backend="llama_cpp",
                               endpoint="http://127.0.0.1:8081", provenance="configured"),
        health="unreachable", source=("saved_profile",), detail="fx",
    )
    res = resolve_runtime(
        selected_backend="llama_cpp",
        discovered_candidates=[candidate],
        topology=topology_from_inventory((device,)),
        host_meminfo={"ram_total_mb": 128000, "ram_available_mb": 64000,
                      "swap_free_mb": 0, "swap_used_mb": 0},
        weight_bytes=1 * 1024 * 1024 * 1024,
        kv_cache_bytes=kv_cache_bytes,
        requested_context=4096,
        backend_executables=[type("Exe", (), {"backend": "llama_cpp", "state": "installed"})()],
    )
    assert res.resolved is not None
    assert res.resolved.requested_context == 4096

    cmd, status, _detail = build_llama_server_command(
        _fake_request(res.resolved), executable_path="/opt/llama-server",
        model_path=valid_gguf, hardware_inventory=(device,),
    )
    assert status is None
    assert cmd.argv[cmd.argv.index("--ctx-size") + 1] == "4096"


def _fake_request(resolved):
    from llm_modelbench.runtime_lifecycle import MaterialisationRequest
    from llm_modelbench.runtime_resolution import RuntimeResolution, RuntimeResolutionStatus
    resolution = RuntimeResolution(
        status=RuntimeResolutionStatus.RESOLVED, reason="resolved", detail="fx",
        resolved=resolved,
    )
    return MaterialisationRequest.from_resolution(resolution)
