"""Immutable, backend-neutral RC21 runtime execution identity.

The constructors in this module are deliberately pure.  Collection is opt-in
and limited to already selected clients, profiles, and inventory evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit


RUNTIME_IDENTITY_SCHEMA_VERSION = 1
_UUID = re.compile(r"^GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$")
_MAX_TEXT = 512


def _text(value: object, field: str, *, allow_none: bool = False) -> Optional[str]:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return " ".join(value.split())[:_MAX_TEXT]


def _uuid(value: object) -> str:
    text = _text(value, "physical GPU UUID")
    if not _UUID.fullmatch(text):
        raise ValueError("physical GPU UUID must be canonical NVIDIA GPU-UUID")
    return text


def normalize_endpoint(value: object) -> Optional[str]:
    if value is None:
        return None
    text = _text(value, "runtime endpoint")
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("runtime endpoint must be a credential-free http(s) origin")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")


def _nonnegative(value: object, field: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer or None")
    return value


def _weights(value: object, uuids: Tuple[str, ...]) -> Tuple[Tuple[str, float], ...]:
    if value is None:
        return ()
    pairs = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    normalized = []
    for key, weight in pairs:
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) or float(weight) <= 0:
            raise ValueError("allocation weights must be finite positive numbers")
        normalized.append((_uuid(key), float(weight)))
    if len({key for key, _ in normalized}) != len(normalized):
        raise ValueError("allocation weights must not duplicate physical GPU UUIDs")
    if normalized and {key for key, _ in normalized} != set(uuids):
        raise ValueError("allocation weights must cover exactly declared physical GPU UUIDs")
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class RuntimeModelIdentity:
    display_name: str
    backend_model_id: Optional[str] = None
    artifact_digest: Optional[str] = None
    artifact_path: Optional[str] = None
    provenance: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "display_name", _text(self.display_name, "display model name"))
        for field in ("backend_model_id", "artifact_digest", "artifact_path"):
            object.__setattr__(self, field, _text(getattr(self, field), field, allow_none=True))
        object.__setattr__(self, "provenance", _text(self.provenance, "model identity provenance"))

    def to_dict(self) -> dict:
        return {"display_name": self.display_name, "backend_model_id": self.backend_model_id,
                "artifact_digest": self.artifact_digest, "artifact_path": self.artifact_path,
                "provenance": self.provenance}


@dataclass(frozen=True)
class RuntimeExecutionSettings:
    strategy: Optional[str] = None
    allocation_weights: object = ()
    context_size: Optional[int] = None
    batch_size: Optional[int] = None
    micro_batch_size: Optional[int] = None
    kv_cache_type: Optional[str] = None
    parallel_sequences: Optional[int] = None
    allow_cpu_spill: Optional[bool] = None
    offload_layers: Optional[int] = None

    def normalized(self, uuids: Tuple[str, ...]) -> dict:
        if self.strategy not in {None, "single_device", "layer_split", "tensor_split"}:
            raise ValueError("unsupported runtime execution strategy")
        if self.strategy in {"layer_split", "tensor_split"} and len(uuids) < 2:
            raise ValueError("multi-device strategy requires at least two physical GPU UUIDs")
        if self.allow_cpu_spill is not None and not isinstance(self.allow_cpu_spill, bool):
            raise ValueError("allow_cpu_spill must be bool or None")
        return {"strategy": self.strategy, "allocation_weights": {key: weight for key, weight in _weights(self.allocation_weights, uuids)},
                "context_size": _nonnegative(self.context_size, "context_size"), "batch_size": _nonnegative(self.batch_size, "batch_size"),
                "micro_batch_size": _nonnegative(self.micro_batch_size, "micro_batch_size"), "kv_cache_type": _text(self.kv_cache_type, "kv_cache_type", allow_none=True),
                "parallel_sequences": _nonnegative(self.parallel_sequences, "parallel_sequences"), "allow_cpu_spill": self.allow_cpu_spill,
                "offload_layers": _nonnegative(self.offload_layers, "offload_layers")}


@dataclass(frozen=True)
class RuntimeIdentity:
    backend: str
    adapter_identity: str
    endpoint: Optional[str]
    profile_name: Optional[str]
    profile_provenance: Optional[str]
    profile_schema_version: Optional[int]
    server_version: Optional[str]
    model: RuntimeModelIdentity
    physical_gpu_uuids: Tuple[str, ...] = ()
    pci_bus_ids: Tuple[Tuple[str, str], ...] = ()
    declared_device_order: Tuple[str, ...] = ()
    execution: RuntimeExecutionSettings = RuntimeExecutionSettings()
    evidence_provenance: str = "collected"
    schema_version: int = RUNTIME_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_IDENTITY_SCHEMA_VERSION:
            raise ValueError("unsupported runtime identity schema")
        if self.backend not in {"ollama", "llama_cpp"}:
            raise ValueError("runtime backend must be ollama or llama_cpp")
        object.__setattr__(self, "adapter_identity", _text(self.adapter_identity, "backend adapter identity"))
        object.__setattr__(self, "endpoint", normalize_endpoint(self.endpoint))
        for field in ("profile_name", "profile_provenance", "server_version"):
            object.__setattr__(self, field, _text(getattr(self, field), field, allow_none=True))
        object.__setattr__(self, "profile_schema_version", _nonnegative(self.profile_schema_version, "profile_schema_version"))
        object.__setattr__(self, "evidence_provenance", _text(self.evidence_provenance, "identity provenance"))
        uuids = tuple(_uuid(value) for value in self.physical_gpu_uuids)
        if len(uuids) != len(set(uuids)):
            raise ValueError("physical GPU UUIDs must be unique")
        order = tuple(_uuid(value) for value in self.declared_device_order)
        if order and set(order) != set(uuids):
            raise ValueError("declared device order must cover exactly physical GPU UUIDs")
        object.__setattr__(self, "physical_gpu_uuids", tuple(sorted(uuids)))
        object.__setattr__(self, "declared_device_order", order or tuple(sorted(uuids)))
        pci = tuple(sorted((_uuid(key), _text(value, "PCI bus ID")) for key, value in self.pci_bus_ids))
        if {key for key, _ in pci} - set(uuids):
            raise ValueError("PCI evidence may only support declared physical GPU UUIDs")
        object.__setattr__(self, "pci_bus_ids", pci)
        if not isinstance(self.model, RuntimeModelIdentity):
            raise TypeError("runtime identity requires RuntimeModelIdentity")
        if not isinstance(self.execution, RuntimeExecutionSettings):
            raise TypeError("runtime identity requires RuntimeExecutionSettings")
        self.execution.normalized(self.physical_gpu_uuids)

    def compatibility_dict(self) -> dict:
        return {"schema_version": self.schema_version, "backend": self.backend, "adapter_identity": self.adapter_identity,
                "endpoint": self.endpoint, "profile_name": self.profile_name, "profile_provenance": self.profile_provenance,
                "profile_schema_version": self.profile_schema_version, "server_version": self.server_version,
                "model": self.model.to_dict(), "physical_gpu_uuids": list(self.physical_gpu_uuids),
                "declared_device_order": list(self.declared_device_order), "execution": self.execution.normalized(self.physical_gpu_uuids)}

    @property
    def identity_hash(self) -> str:
        payload = json.dumps(self.compatibility_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def completeness(self) -> str:
        required = (self.endpoint, self.profile_name, self.model.backend_model_id, self.model.artifact_digest)
        return "complete" if all(required) else "partial"

    def to_dict(self) -> dict:
        return {**self.compatibility_dict(), "pci_bus_ids": {key: value for key, value in self.pci_bus_ids},
                "identity_hash": self.identity_hash, "completeness": self.completeness,
                "evidence_provenance": self.evidence_provenance}

    @classmethod
    def from_dict(cls, value: object) -> "RuntimeIdentity":
        """Strict, bounded reconstruction for resume gates (never trust JSON directly)."""
        if not isinstance(value, Mapping) or len(value) > 32:
            raise ValueError("runtime identity must be a bounded object")
        if value.get("schema_version") != RUNTIME_IDENTITY_SCHEMA_VERSION:
            raise ValueError("unsupported runtime identity schema")
        model=value.get("model"); execution=value.get("execution")
        if not isinstance(model, Mapping) or not isinstance(execution, Mapping):
            raise ValueError("runtime identity nested fields are malformed")
        if len(model)>8 or len(execution)>16:
            raise ValueError("runtime identity nested object too large")
        pci=value.get("pci_bus_ids") or {}
        if not isinstance(pci, Mapping): raise ValueError("PCI identity evidence malformed")
        return cls(backend=value.get("backend"), adapter_identity=value.get("adapter_identity"), endpoint=value.get("endpoint"),
            profile_name=value.get("profile_name"), profile_provenance=value.get("profile_provenance"), profile_schema_version=value.get("profile_schema_version"),
            server_version=value.get("server_version"), model=RuntimeModelIdentity(**dict(model)), physical_gpu_uuids=tuple(value.get("physical_gpu_uuids") or ()),
            pci_bus_ids=tuple(pci.items()), declared_device_order=tuple(value.get("declared_device_order") or ()), execution=RuntimeExecutionSettings(**dict(execution)),
            evidence_provenance=value.get("evidence_provenance", "frozen"), schema_version=value.get("schema_version"))


@dataclass(frozen=True)
class RuntimeIdentityMismatch:
    code: str
    detail: str

    def to_dict(self) -> dict:
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class RuntimeIdentityCompatibility:
    compatible: bool
    mismatches: Tuple[RuntimeIdentityMismatch, ...] = ()

    def to_dict(self) -> dict:
        return {"compatible": self.compatible, "mismatches": [item.to_dict() for item in self.mismatches]}


def compare_runtime_identities(frozen: Optional[RuntimeIdentity], current: Optional[RuntimeIdentity]) -> RuntimeIdentityCompatibility:
    if frozen is None:
        return RuntimeIdentityCompatibility(False, (RuntimeIdentityMismatch("legacy_runtime_identity_missing", "existing campaign lacks a reconstructable runtime identity; create a new campaign"),))
    if current is None:
        return RuntimeIdentityCompatibility(False, (RuntimeIdentityMismatch("current_runtime_identity_missing", "current runtime identity is unavailable"),))
    left, right = frozen.compatibility_dict(), current.compatibility_dict()
    labels = {"backend":"backend_changed", "adapter_identity":"adapter_changed", "endpoint":"endpoint_changed", "profile_name":"profile_changed", "profile_provenance":"profile_provenance_changed", "profile_schema_version":"profile_schema_changed", "server_version":"server_version_changed", "physical_gpu_uuids":"physical_gpu_uuids_changed", "declared_device_order":"device_order_changed"}
    mismatches=[]
    for field, code in labels.items():
        if left[field] != right[field]:
            mismatches.append(RuntimeIdentityMismatch(code, f"frozen {field} differs from current identity"))
    for field, code in {"artifact_digest": "model_artifact_changed", "backend_model_id": "backend_model_changed"}.items():
        if left["model"].get(field) != right["model"].get(field):
            mismatches.append(RuntimeIdentityMismatch(code, f"frozen model {field} differs from current identity"))
    for field, code in {"strategy":"strategy_changed", "allocation_weights":"allocation_weights_changed", "context_size":"context_changed", "batch_size":"batch_size_changed", "micro_batch_size":"micro_batch_size_changed", "kv_cache_type":"kv_cache_type_changed", "parallel_sequences":"parallel_sequences_changed", "allow_cpu_spill":"spill_policy_changed", "offload_layers":"offload_policy_changed"}.items():
        if left["execution"].get(field) != right["execution"].get(field):
            mismatches.append(RuntimeIdentityMismatch(code, f"frozen execution {field} differs from current identity"))
    return RuntimeIdentityCompatibility(not mismatches, tuple(mismatches))


def runtime_variant_id(identity: RuntimeIdentity) -> str:
    """Full identity hash is authoritative; this is a stable grouping alias."""
    return identity.identity_hash


def row_identity_reference(identity: RuntimeIdentity) -> dict:
    """Small authoritative row reference for a run-level identity artifact."""
    value = identity.to_dict()
    return {"runtime_identity_schema_version": value["schema_version"], "runtime_identity_hash": value["identity_hash"],
            "runtime_variant_id": value["identity_hash"], "backend": value["backend"],
            "runtime_profile": value["profile_name"], "model_artifact_digest": value["model"]["artifact_digest"]}


def write_runtime_identity_artifact(path: Path, identity: RuntimeIdentity) -> dict:
    """Atomically write finite canonical evidence, returning its row reference."""
    value = identity.to_dict()
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True); raise
    return row_identity_reference(identity)

def write_runtime_identity_map_artifact(path: Path, identities: Mapping[str, RuntimeIdentity]) -> None:
    if not identities or len(identities) > 512:
        raise ValueError("runtime identity map must contain 1..512 models")
    normalized = {}
    for name, identity in identities.items():
        if not isinstance(name, str) or not name.strip() or len(name) > _MAX_TEXT or not isinstance(identity, RuntimeIdentity):
            raise ValueError("runtime identity map is malformed")
        normalized[name] = identity.to_dict()
    payload={"schema_version": RUNTIME_IDENTITY_SCHEMA_VERSION, "identities": dict(sorted(normalized.items()))}
    text=json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n"; path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=str(path.parent),prefix=f".{path.name}.",suffix=".tmp")
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as handle: handle.write(text); handle.flush(); os.fsync(handle.fileno())
        Path(tmp).replace(path)
    except BaseException: Path(tmp).unlink(missing_ok=True); raise

def validate_resume_runtime_identities(out_dir: Path, current_identities: Mapping[str, RuntimeIdentity], selected_models: Iterable[str]) -> RuntimeIdentityCompatibility:
    """Read-only fail-closed resume gate shared by CLI and runner."""
    models=tuple(selected_models)
    if len(models)!=len(set(models)): raise ValueError("runtime identity mismatch: runtime_identity_model_unexpected")
    if any(not isinstance(current_identities.get(model), RuntimeIdentity) for model in models):
        raise ValueError("runtime identity mismatch: current_runtime_identity_missing")
    artifact=load_runtime_identity_artifact(Path(out_dir)/"runtime_identity.json")
    if artifact.get("status") == "legacy_unknown": raise ValueError("runtime identity mismatch: legacy_runtime_identity_missing")
    if artifact.get("status") != "available": raise ValueError("runtime identity mismatch: runtime_identity_artifact_unavailable")
    raw=artifact.get("identities")
    if raw is None:
        if len(models)!=1: raise ValueError("runtime identity mismatch: runtime_identity_model_missing")
        raw={models[0]:artifact.get("identity")}
    codes=[]
    for model in models:
        item=raw.get(model)
        try: frozen=RuntimeIdentity.from_dict(item) if item else None
        except Exception: raise ValueError("runtime identity mismatch: runtime_identity_artifact_unavailable")
        if item and item.get("identity_hash") != frozen.identity_hash: raise ValueError("runtime identity mismatch: runtime_identity_artifact_unavailable")
        codes.extend(x.code for x in compare_runtime_identities(frozen,current_identities[model]).mismatches)
    extra=set(raw)-set(models)
    if extra:
        # Extra identities are safe only if raw rows do not belong to them.
        rows=Path(out_dir,"raw_results.jsonl")
        if rows.exists() and any(json.loads(x).get("model") in extra for x in rows.read_text().splitlines() if x.strip()): codes.append("runtime_identity_model_unexpected")
    if codes: raise ValueError("runtime identity mismatch: "+", ".join(sorted(set(codes))))
    return RuntimeIdentityCompatibility(True)

def validate_frozen_runtime_identity_map(frozen_values: object, current_identities: Mapping[str, RuntimeIdentity], selected_models: Iterable[str]) -> RuntimeIdentityCompatibility:
    """Read-only campaign/runner mapping comparison with the standard codes."""
    if not isinstance(frozen_values, Mapping): raise ValueError("runtime identity mismatch: legacy_runtime_identity_missing")
    models=tuple(selected_models); codes=[]
    for model in models:
        value=frozen_values.get(model)
        if not isinstance(current_identities.get(model), RuntimeIdentity): codes.append("current_runtime_identity_missing"); continue
        if value is None: codes.append("runtime_identity_model_missing"); continue
        try:
            frozen=RuntimeIdentity.from_dict(value)
            if value.get("identity_hash") != frozen.identity_hash: raise ValueError("hash")
        except Exception: codes.append("runtime_identity_artifact_unavailable"); continue
        codes.extend(x.code for x in compare_runtime_identities(frozen,current_identities[model]).mismatches)
    if set(frozen_values)-set(models): codes.append("runtime_identity_model_unexpected")
    if codes: raise ValueError("runtime identity mismatch: "+", ".join(sorted(set(codes))))
    return RuntimeIdentityCompatibility(True)


def load_runtime_identity_artifact(path: Path) -> dict:
    """Bounded compatibility reader; reports never invent missing identity."""
    try:
        if Path(path).stat().st_size > 4 * 1024 * 1024: raise ValueError("identity artifact too large")
        value=json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict): raise ValueError("identity artifact malformed")
        if "identities" in value:
            identities=value.get("identities")
            if not isinstance(identities, Mapping) or len(identities)>512: raise ValueError("identity map malformed")
            for key, item in identities.items():
                if not isinstance(key,str) or len(key)>_MAX_TEXT: raise ValueError("identity map key malformed")
                RuntimeIdentity.from_dict(item)
            return {"status":"available", "identities":identities}
        RuntimeIdentity.from_dict(value)
        return {"status":"available", "identity":value}
    except FileNotFoundError: return {"status":"legacy_unknown", "warning":"runtime identity artifact missing"}
    except Exception: return {"status":"unavailable", "warning":"runtime identity artifact unreadable or unsupported"}


def collect_runtime_identity(*, client: object, profile: object, model_name: str,
                             model_row: Optional[Mapping[str, Any]] = None,
                             inventory: Iterable[object] = (), config: object = None,
                             execution: Optional[RuntimeExecutionSettings] = None) -> RuntimeIdentity:
    """Collect bounded evidence from an already selected runtime.

    This helper starts nothing and performs no discovery.  Any client metadata
    call is an existing backend read-only interface; callers select when it is
    appropriate to make those bounded reads.
    """
    backend = _text(getattr(profile, "backend", None), "runtime backend")
    endpoint = getattr(profile, "endpoint", None)
    profile_name = getattr(profile, "name", None)
    provenance = getattr(profile, "provenance", None)
    row = dict(model_row or {})
    details = row.get("details") if isinstance(row.get("details"), Mapping) else {}
    digest = row.get("digest") or row.get("id") or row.get("model_id") or details.get("digest")
    backend_model_id = row.get("id") or row.get("model") or model_name
    server_version = None
    version = getattr(client, "version", None)
    if callable(version):
        try:
            observed = version()
            server_version = observed if isinstance(observed, str) else None
        except Exception:
            server_version = None
    devices = tuple(inventory)
    uuids = tuple(str(getattr(item, "uuid", "") or "") for item in devices if getattr(item, "uuid", None))
    pci = tuple((str(getattr(item, "uuid")), str(getattr(item, "pci_bus_id"))) for item in devices
                if getattr(item, "uuid", None) and getattr(item, "pci_bus_id", None))
    selected = tuple(getattr(profile, "physical_gpu_uuids", ()) or ())
    if selected:
        uuids = selected
        pci = tuple(item for item in pci if item[0] in set(selected))
    # Anvil Stage 3.2C-2b: the resolved RAM-spill *permission* (not the actual
    # resulting placement) is identity-bearing -- a campaign resumed with spill
    # newly permitted is not automatically equivalent to its original execution
    # (amendment §6; runtime-identity `spill_policy_changed`).  Only an explicit
    # grant is recorded: an absent/false permission stays None, identical to the
    # historical default, so ordinary runs keep a stable identity hash.
    _spill_permitted = bool(getattr(config, "allow_ram_spill", False))
    settings = execution or RuntimeExecutionSettings(
        context_size=getattr(config, "ctx_override", None),
        allow_cpu_spill=True if _spill_permitted else None,
    )
    return RuntimeIdentity(
        backend=backend, adapter_identity=type(client).__name__, endpoint=endpoint,
        profile_name=profile_name, profile_provenance=provenance, profile_schema_version=1,
        server_version=server_version, model=RuntimeModelIdentity(
            model_name, str(backend_model_id) if backend_model_id else None,
            str(digest) if digest else None, None,
            "backend_declared_model_metadata" if digest else "backend_model_name_only_unknown_artifact",
        ), physical_gpu_uuids=uuids, pci_bus_ids=pci, execution=settings,
        evidence_provenance="selected_profile_and_backend_metadata",
    )
