"""Anvil Stage 3.2D-2 -- per-run protocol/runtime/model binding + evidence.

Ties together the D-0 deterministic ``BenchmarkProtocol`` builder and the
D-1 stable ``RuntimeProfileIdentity`` factory into a validated
``BenchmarkRuntimeBinding`` per model, and serializes it for evidence.

The binding is a *reference*: it carries identity keys, never embedded
objects. Placement class (``full_gpu`` / ``multi_gpu`` / ``ram_spill``),
physical GPU UUIDs, endpoint and live telemetry stay on the per-execution
runtime-identity / row evidence (amendment §15.2.1) -- a canonical
model+protocol binding never erases or normalizes those.

No persistent registry: the protocol is derived for the concrete run and
persisted alongside its evidence (``benchmark_bindings.json``), same as
``runtime_identity.json``.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

from .benchmark_policy import build_benchmark_protocol
from .benchmark_protocol import BenchmarkProtocol, BenchmarkRuntimeBinding, bind_runtime_to_protocol
from .config import Config
from .identity import ModelArtifactIdentity, resolve_runtime_profile_identity
from .runtime_identity import RuntimeExecutionSettings
from .tasks import Task


def _execution_settings_from_config(cfg: Config) -> RuntimeExecutionSettings:
    """The recipe-level runtime settings ModelBench actually resolves for a
    run. ``context_size`` is the one operator adaptation; ``allow_cpu_spill``
    is passed through only as a permission and is deliberately NOT part of
    the stable profile identity (identity.resolve_runtime_profile_identity
    excludes it -- §9)."""
    return RuntimeExecutionSettings(
        context_size=getattr(cfg, "ctx_override", None),
        allow_cpu_spill=True if getattr(cfg, "allow_ram_spill", False) else None,
    )


def _allowed_adaptations_used(cfg: Config) -> Tuple[str, ...]:
    """Operator-level context adaptation only. The needle/long-context path
    resolves a per-depth ``wanted_ctx`` internally, but that is task-intrinsic
    protocol behaviour (the depths are in ``context_sizes`` -> already in
    ``prompt_semantics_hash``), not a runtime adaptation of this binding."""
    return ("context_size",) if getattr(cfg, "ctx_override", None) else ()


def build_model_binding(
    *,
    model_artifact_identity: ModelArtifactIdentity,
    selected_tasks: Iterable[Task],
    cfg: Config,
    backend: str,
    backend_version: Optional[str],
) -> Tuple[BenchmarkProtocol, BenchmarkRuntimeBinding]:
    """Deterministic protocol + validated binding for one model's actual
    (resume-independent) task set."""
    tasks = list(selected_tasks)
    protocol = build_benchmark_protocol(tasks, cfg)
    profile_identity = resolve_runtime_profile_identity(
        backend=backend,
        backend_version=backend_version,
        execution_settings=_execution_settings_from_config(cfg),
    )
    binding = bind_runtime_to_protocol(
        protocol,
        model_artifact_identity=model_artifact_identity,
        runtime_profile_identity_key=profile_identity.stable_key(),
        allowed_adaptations_used=_allowed_adaptations_used(cfg),
        provenance="anvil_stage3.2d_deterministic_resolution",
    )
    return protocol, binding


def protocol_to_dict(protocol: BenchmarkProtocol) -> Dict[str, Any]:
    return {
        "protocol_id": protocol.protocol_id,
        "version": protocol.version,
        "identity_key": protocol.identity_key(),
        "task_ids": list(protocol.task_ids),
        "prompt_semantics_hash": protocol.prompt_semantics_hash,
        "sampling_policy_hash": protocol.sampling_policy_hash,
        "output_budget_policy_hash": protocol.output_budget_policy_hash,
        "scorer_versions": [list(pair) for pair in protocol.scorer_versions],
        "allowed_adaptations": list(protocol.allowed_adaptations),
    }


def binding_to_dict(binding: BenchmarkRuntimeBinding) -> Dict[str, Any]:
    return {
        "binding_key": binding.binding_key(),
        "model_artifact_set_id": binding.model_artifact_identity.artifact_set_id,
        "benchmark_protocol_identity_key": binding.benchmark_protocol_identity_key,
        "runtime_profile_identity_key": binding.runtime_profile_identity_key,
        "allowed_adaptations_used": list(binding.allowed_adaptations_used),
        "provenance": binding.provenance,
    }


def row_binding_reference(binding: BenchmarkRuntimeBinding, protocol: BenchmarkProtocol) -> Dict[str, Any]:
    """Small additive per-row reference -- mirrors
    ``runtime_identity.row_identity_reference``. Absence on a legacy row is a
    valid "no binding recorded" state, never an error."""
    return {
        "benchmark_binding_key": binding.binding_key(),
        "benchmark_protocol_identity_key": binding.benchmark_protocol_identity_key,
        "benchmark_protocol_id": protocol.protocol_id,
        "benchmark_protocol_version": protocol.version,
    }
