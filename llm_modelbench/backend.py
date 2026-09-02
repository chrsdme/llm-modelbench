"""Backend-neutral inference-client contract for the RC21 migration, extended
in Anvil Stage 1 toward the plan's ``RuntimeBackend`` contract
(ANVIL_MASTER_PLAN.md v2.2, Stage 1.2).

The protocol intentionally mirrors only ModelBench's existing client surface.
Ollama HTTP behavior remains in :mod:`llm_modelbench.ollama`; adapters here
delegate to it unchanged while making backend-level support explicit.

Stage 1 additions are deliberately additive, matching Stage 0's discipline:
``InferenceClient`` is used throughout the codebase only as a type
annotation (confirmed by inspection -- no ``isinstance(..., InferenceClient)``
runtime check exists anywhere), so new Protocol methods cannot break an
existing caller even before every implementer grows them. The per-model
metadata method was renamed from ``capabilities()`` to
``capability_hints()`` throughout the codebase (Stage 1.2), to close off a
"someone writes `if 'vision' in backend.capability_hints(model)` and treats
declared metadata as execution authority" regression risk. The name change
is metadata-only: the underlying Ollama/llama.cpp JSON payload field is
still literally ``"capabilities"`` on the wire and is left alone -- only the
Python method that surfaces it was renamed.

Scope note (Stage 1.6): this contract is NVIDIA-only in practice today
(nothing here is NVIDIA-specific by construction, but no other backend is
implemented) and Ollama/llama.cpp-only by backend. It is the actual
extension point non-goals like "leave room for other backends/GPU vendors
later" depend on -- see ``hardware.py``'s own scope documentation
(``detect_gpu()``'s docstring already frames the scalar path as a
"compatibility projection" for exactly this reason) for the hardware side
of the same story.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Protocol, runtime_checkable

from .identity import RuntimeProfileIdentity, resolve_runtime_profile_identity


class CapabilityStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class BackendCapability(str, Enum):
    INVENTORY = "inventory"
    VERSION = "version"
    MODEL_METADATA = "model_metadata"
    CHAT = "chat"
    NATIVE_TOOLS = "native_tools"
    SUFFIX_GENERATION = "suffix_generation"
    EMBEDDINGS = "embeddings"
    LOADED_MODEL_STATS = "loaded_model_stats"
    OFFLOAD_FRACTION = "offload_fraction"
    MODEL_UNLOAD = "model_unload"
    FLUSH_ALL = "flush_all"
    OLLAMA_SERVICE_REPAIR = "ollama_service_repair"
    OLLAMA_KV_REPAIR = "ollama_kv_repair"
    HEALTH_CHECK = "health_check"
    # NOTE (Anvil Stage 3.2): the generic managed-process-lifecycle
    # capabilities (START_MODEL / STOP_MODEL / LOAD_MODEL /
    # MANAGED_RUNTIME_LAUNCH / MANAGED_RUNTIME_STOP / MODEL_SWITCH) were
    # removed here. They were scaffolding for the rejected persistent
    # runtime-management architecture (ANVIL_ARCHITECTURE_AMENDMENT.md
    # 2026-09-01) and had zero production callers. Future Stage 3B
    # llama.cpp execution is a narrowly campaign-owned ephemeral
    # child-process runner/context manager, not a generic backend method;
    # Ollama stays an endpoint client; sequential GGUF execution is
    # start/run/stop/start-next, not a switch_model() operation.


@dataclass(frozen=True)
class BackendIdentity:
    """Static adapter identity; runtime-profile identity is a later-stage concern."""

    backend: str
    implementation: str
    endpoint: Optional[str] = None


@dataclass(frozen=True)
class BackendCapabilities:
    """Backend-level operation availability, separate from per-model metadata."""

    states: Mapping[BackendCapability, CapabilityStatus]

    def state(self, capability: BackendCapability) -> CapabilityStatus:
        return self.states.get(capability, CapabilityStatus.UNKNOWN)

    def supports(self, capability: BackendCapability) -> bool:
        return self.state(capability) is CapabilityStatus.SUPPORTED


class BackendCapabilityError(RuntimeError):
    """Raised before an explicitly unsupported backend operation is attempted."""


@runtime_checkable
class InferenceClient(Protocol):
    """The complete client method surface currently consumed by ModelBench."""

    def backend_identity(self) -> BackendIdentity: ...
    def backend_capabilities(self) -> BackendCapabilities: ...
    def tags(self) -> List[Dict[str, Any]]: ...
    def version(self) -> Optional[str]: ...
    def show(self, model: str) -> Dict[str, Any]: ...
    def capability_hints(self, model: str) -> List[str]: ...
    def supports_thinking(self, model: str) -> bool: ...
    def model_info(self, model: str) -> Dict[str, Any]: ...
    def model_size_bytes(self, model: str) -> Optional[int]: ...
    def context_length(self, model: str) -> Optional[int]: ...
    def chat(self, model: str, prompt: str, **kwargs: Any) -> Dict[str, Any]: ...
    def chat_tools(self, model: str, prompt: str, **kwargs: Any) -> Dict[str, Any]: ...
    def generate_suffix(self, model: str, prompt: str, **kwargs: Any) -> Dict[str, Any]: ...
    def embed(self, model: str, texts: List[str]) -> List[List[float]]: ...
    def loaded_model_stats(self, model: str) -> Optional[Dict[str, Any]]: ...
    def offload_fraction(self, model: str, exact: bool = True) -> Optional[float]: ...
    def unload(self, model: str) -> None: ...
    def flush_all(self) -> None: ...

    # -- Anvil Stage 1 additions (ANVIL_MASTER_PLAN.md v2.2, Stage 1.2) --
    def health(self) -> bool: ...
    def runtime_profile_identity(self) -> RuntimeProfileIdentity: ...


def require_capability(client: InferenceClient, capability: BackendCapability) -> None:
    """Fail closed before an adapter-declared unsupported operation."""
    state = client.backend_capabilities().state(capability)
    if state is CapabilityStatus.SUPPORTED:
        return
    identity = client.backend_identity()
    raise BackendCapabilityError(
        f"backend {identity.backend!r} does not support {capability.value!r}: {state.value}"
    )


def supports_capability(client: object, capability: BackendCapability) -> bool:
    """Respect adapter capabilities while retaining legacy direct-client compatibility."""
    report = getattr(client, "backend_capabilities", None)
    if not callable(report):
        return True
    return report().supports(capability)


_OLLAMA_CAPABILITIES = BackendCapabilities({
    BackendCapability.INVENTORY: CapabilityStatus.SUPPORTED,
    BackendCapability.VERSION: CapabilityStatus.SUPPORTED,
    BackendCapability.MODEL_METADATA: CapabilityStatus.SUPPORTED,
    BackendCapability.CHAT: CapabilityStatus.SUPPORTED,
    BackendCapability.NATIVE_TOOLS: CapabilityStatus.SUPPORTED,
    BackendCapability.SUFFIX_GENERATION: CapabilityStatus.SUPPORTED,
    BackendCapability.EMBEDDINGS: CapabilityStatus.SUPPORTED,
    BackendCapability.LOADED_MODEL_STATS: CapabilityStatus.SUPPORTED,
    BackendCapability.OFFLOAD_FRACTION: CapabilityStatus.SUPPORTED,
    BackendCapability.MODEL_UNLOAD: CapabilityStatus.SUPPORTED,
    BackendCapability.FLUSH_ALL: CapabilityStatus.SUPPORTED,
    BackendCapability.OLLAMA_SERVICE_REPAIR: CapabilityStatus.SUPPORTED,
    BackendCapability.OLLAMA_KV_REPAIR: CapabilityStatus.SUPPORTED,
    BackendCapability.HEALTH_CHECK: CapabilityStatus.SUPPORTED,
})

_MOCK_CAPABILITIES = BackendCapabilities({
    **_OLLAMA_CAPABILITIES.states,
    BackendCapability.OLLAMA_SERVICE_REPAIR: CapabilityStatus.UNSUPPORTED,
    BackendCapability.OLLAMA_KV_REPAIR: CapabilityStatus.UNSUPPORTED,
})


class OllamaBackendAdapter:
    """Delegate the neutral protocol to the established ``OllamaClient`` API."""

    def __init__(self, client: Any):
        self.client = client

    def backend_identity(self) -> BackendIdentity:
        return BackendIdentity("ollama", "ollama", getattr(self.client, "base", None))

    def backend_capabilities(self) -> BackendCapabilities:
        return _OLLAMA_CAPABILITIES

    def tags(self) -> List[Dict[str, Any]]:
        return self.client.tags()

    def version(self) -> Optional[str]:
        return self.client.version()

    def show(self, model: str) -> Dict[str, Any]:
        return self.client.show(model)

    def capability_hints(self, model: str) -> List[str]:
        return self.client.capability_hints(model)

    def supports_thinking(self, model: str) -> bool:
        return self.client.supports_thinking(model)

    def model_info(self, model: str) -> Dict[str, Any]:
        return self.client.model_info(model)

    def model_size_bytes(self, model: str) -> Optional[int]:
        return self.client.model_size_bytes(model)

    def context_length(self, model: str) -> Optional[int]:
        return self.client.context_length(model)

    def chat(self, model: str, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return self.client.chat(model, prompt, **kwargs)

    def chat_tools(self, model: str, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return self.client.chat_tools(model, prompt, **kwargs)

    def generate_suffix(self, model: str, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return self.client.generate_suffix(model, prompt, **kwargs)

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        return self.client.embed(model, texts)

    def loaded_model_stats(self, model: str) -> Optional[Dict[str, Any]]:
        return self.client.loaded_model_stats(model)

    def offload_fraction(self, model: str, exact: bool = True) -> Optional[float]:
        return self.client.offload_fraction(model, exact=exact)

    def unload(self, model: str) -> None:
        self.client.unload(model)

    def flush_all(self) -> None:
        self.client.flush_all()

    def health(self) -> bool:
        try:
            return self.client.version() is not None
        except Exception:
            return False

    def runtime_profile_identity(self) -> RuntimeProfileIdentity:
        """Only backend + version are known at this adapter layer. Anvil
        Stage 3.4B: this supplies those two facts to the shared
        ``identity.resolve_runtime_profile_identity`` factory rather than
        constructing a ``RuntimeProfileIdentity`` with its own semantics --
        one derivation, several fact-suppliers (§2, §11). No resolved
        ``RuntimeExecutionSettings`` recipe exists here, so
        ``execution_settings`` is ``None`` and the factory leaves
        ``runtime_configuration_hash`` / ``gpu_policy`` honestly ``None``.
        template_hash/protocol_version population is real later-stage work
        (capability probing must run first to derive a template hash)."""
        try:
            backend_version = self.client.version()
        except Exception:
            backend_version = None
        return resolve_runtime_profile_identity(
            backend=self.backend_identity().backend,
            backend_version=backend_version,
            execution_settings=None,
        )


class MockBackendAdapter(OllamaBackendAdapter):
    """Offline adapter retaining deterministic ``MockClient`` behavior."""

    def backend_identity(self) -> BackendIdentity:
        return BackendIdentity("mock", "llm-modelbench-mock")

    def backend_capabilities(self) -> BackendCapabilities:
        return _MOCK_CAPABILITIES
