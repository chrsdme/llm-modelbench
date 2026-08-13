"""Anvil Stage 2.7A -- read-only classification of the fleet's existing
capability evidence population.

Scope, per this session's Codex/GPT scoping pass
(``local_only/anvil/codex-advice-pre-stage3.txt``): Stage 2.6 closed the
*authority* question -- no known production path can grant execution,
recovery, or judge eligibility from declared/legacy/identity-incompatible
capability data any more (see
``local_only/anvil/stage-2.6E-authority-audit.md``). Stage 2.7 changes
focus from authority to *evidence population*: for every (model, capability)
cell the fleet cares about, is the currently-stored evidence valid, stale,
ambiguous, or simply missing -- and why?

This module is read-only. It does not reprobe anything (Stage 2.7B/C), does
not mutate any stored ``capability_report.json``, and does not write
anything to disk itself (a caller may choose to serialize its output).

**Reuses the real authority pipeline, does not reinvent it.** Every cell's
CURRENT_VALID / MEASURED_UNSUPPORTED / BACKEND_UNSUPPORTED / NOT_APPLICABLE /
PROBE_INCONCLUSIVE / *_CHANGED / AMBIGUOUS_COMPATIBLE_OBSERVATIONS verdict is
derived by running stored evidence through the *exact* typed pipeline that
already grants live authority
(:func:`~llm_modelbench.capability_evidence_adapter.adapt_legacy_profile_family_to_observation`
-> :func:`~llm_modelbench.capability_projection.project_capability_observation`
-> :func:`~llm_modelbench.capability_projection.decide_capability_from_projection`),
confirmed by reading each of those modules directly (not assumed). This
guarantees the classification report is consistent with what
planner/runner/repair/judge would actually decide today for the same
evidence -- a second, independently-maintained comparison engine could
silently drift from the real one.

**Two-layer bucket derivation.** The typed pipeline above only ever sees
observations it could adapt at all; a legacy profile that fails schema/
protocol/identity preconditions is silently *not* an observation, which is
indistinguishable from "this family was simply never probed" unless the
raw profile is inspected first. So classification runs in two passes:

1. A raw pass over every stored profile dict found for a model, sorting
   each into "structurally adaptable" (identity present, schema/protocol
   current) or not (``LEGACY_SCHEMA`` / ``UNBOUND_IDENTITY``). Only
   structurally-adaptable profiles are handed to the typed pipeline.
2. The typed projection over structurally-adaptable profiles' observations,
   producing SELECTED (with a measured state) / NO_OBSERVATIONS /
   NO_COMPATIBLE_OBSERVATION / AMBIGUOUS_COMPATIBLE_OBSERVATIONS.

When the typed pipeline reports NO_COMPATIBLE_OBSERVATION, the *typed*
compatibility reason (:class:`~llm_modelbench.capability_identity.CapabilityIdentityCompatibilityReason`)
collapses backend and template drift into one ``RUNTIME_PROFILE_CHANGED``
verdict (confirmed by reading ``RuntimeProfileIdentity.stable_key()`` --
it hashes backend, template_hash, and protocol_version together, and the
adapter only ever populates those three from legacy data). The advice
explicitly wants ``BACKEND_CHANGED`` reported as its own bucket, and the
codebase already has a finer-grained (non-authoritative, messaging-only)
answer for exactly that: ``capabilities.capability_identity_compatibility()``,
frozen at Stage 2.6E as "message/observation-field selection ... never to
admit a candidate." This module calls it *only* to sub-classify which
bucket label to attach when the typed pipeline has already decided the
cell is not currently valid -- never to decide validity itself. That
keeps to the same discipline as every other Anvil Stage 2 migration:
legacy dict-shaped logic may narrow a message, never grant authority.

**SUPERSESSION_CONFLICT is listed but structurally unreachable here.** It
only arises from :func:`~llm_modelbench.capability_projection.project_capability_from_ledger`
resolving a ``SUPERSEDES`` chain via ``EvidenceLedger``. Confirmed
unchanged since the Stage 2.6E audit: ``EvidenceLedger`` is never
instantiated in production. This classifier reads plain
``capability_report.json`` files, not a ledger, so no supersession chain
can ever exist to conflict. The bucket is kept in the enum (per the
advice's explicit minimum list) so a later ledger-backed stage can reuse
this same vocabulary without a rename.

**VALID_BUT_OLD is deliberately omitted.** The advice is explicit: only add
it "if there is an actual age/freshness policy. Do not invent staleness
solely from timestamps." No such policy exists anywhere in this codebase
today (confirmed by the same 2.6E audit) -- a compatible, correctly
measured observation is either currently authoritative or it is not;
there is no separate "valid but getting old" state to report honestly.

Not covered by the advice's bucket list, but real: a model whose evidence
predates the client being able to observe it live at all (uninstalled,
renamed, or otherwise absent from ``client.tags()`` right now). The typed
pipeline already has an honest answer for this -- ``current_model_identity``
is simply ``None``, which routes through the same
``CURRENT_IDENTITY_MISSING`` compatibility reason as any other identity
gap. This module reports that case as ``MISSING`` with a distinguishing
reason string rather than inventing a fourteenth bucket, since
operationally the action is the same as any other missing cell: no
current, comparable evidence exists to make a decision from.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .capabilities import (
    CAPABILITY_SCHEMA_VERSION,
    PROBE_PROTOCOL_VERSION,
    capability_identity_compatibility as legacy_capability_identity_compatibility,
    current_capability_identity,
)
from .capability_evidence_adapter import (
    adapt_legacy_profile_family_to_observation,
    typed_identity_from_capability_identity,
)
from .capability_projection import (
    CapabilityDecisionReason,
    CapabilityProjectionStatus,
    decide_capability_from_projection,
    project_capability_from_ledger,
    project_capability_observation,
)
from .classify import FAMILY_ORDER
from .evidence import EvidenceLedger

# `capability_reprobe_execute` imports from this module at top level, so
# `default_ledger_path` is imported lazily inside `classify_fleet()` below
# rather than at module scope, to avoid a circular import.

__all__ = [
    "EvidenceCellStatus",
    "EvidenceCell",
    "FleetEvidenceReport",
    "discover_capability_report_files",
    "load_fleet_evidence",
    "classify_model_capability",
    "classify_fleet",
]


class EvidenceCellStatus(str, Enum):
    CURRENT_VALID = "current_valid"
    MISSING = "missing"
    LEGACY_SCHEMA = "legacy_schema"
    UNBOUND_IDENTITY = "unbound_identity"
    MODEL_IDENTITY_CHANGED = "model_identity_changed"
    RUNTIME_PROFILE_CHANGED = "runtime_profile_changed"
    BACKEND_CHANGED = "backend_changed"
    AMBIGUOUS_COMPATIBLE_OBSERVATIONS = "ambiguous_compatible_observations"
    SUPERSESSION_CONFLICT = "supersession_conflict"
    PROBE_INCONCLUSIVE = "probe_inconclusive"
    MEASURED_UNSUPPORTED = "measured_unsupported"
    BACKEND_UNSUPPORTED = "backend_unsupported"
    NOT_APPLICABLE = "not_applicable"


# Cells in these statuses have current, decision-usable evidence and do not
# need a reprobe. Everything else is either missing evidence or evidence
# that cannot currently be trusted to decide anything.
REPROBE_NOT_REQUIRED = frozenset({
    EvidenceCellStatus.CURRENT_VALID,
    EvidenceCellStatus.MEASURED_UNSUPPORTED,
    EvidenceCellStatus.BACKEND_UNSUPPORTED,
    EvidenceCellStatus.NOT_APPLICABLE,
})


@dataclass(frozen=True)
class EvidenceCell:
    """The classified state of one (model alias, capability family) cell.

    ``model`` is the alias/name the evidence was filed under (legacy
    ``capability_report.json`` files are name-keyed) -- not a durable
    artifact key. Identity comparisons inside this cell's classification
    are digest/artifact-driven regardless; ``model`` only says where the
    evidence was found on disk, per the advice's own framing that names
    are "aliases/display identifiers," not authority.
    """

    model: str
    capability: str
    status: EvidenceCellStatus
    reason: str
    typed_decision_reason: Optional[str] = None
    legacy_compatibility_reason: Optional[str] = None
    stored_profile_count: int = 0
    structurally_adaptable_profile_count: int = 0
    considered_source_paths: Tuple[str, ...] = ()
    current_identity_available: bool = False
    # Anvil Stage 2.7B: surfaced so a reprobe planner can consume this
    # module's classification directly (per the go-ahead advice's "do not
    # reinterpret legacy profiles separately inside the planner") instead
    # of re-deriving identity/evidence-hash material from raw profiles a
    # second time. ``current_*`` describes the live model/runtime this
    # cell was compared against; the evidence-hash fields describe what
    # was actually found in storage, when anything was.
    current_model_artifact_set_id: Optional[str] = None
    current_model_primary_sha256: Optional[str] = None
    current_runtime_profile_stable_key: Optional[str] = None
    current_backend: Optional[str] = None
    selected_evidence_hash: Optional[str] = None
    considered_evidence_hashes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "capability": self.capability,
            "status": self.status.value,
            "reason": self.reason,
            "typed_decision_reason": self.typed_decision_reason,
            "legacy_compatibility_reason": self.legacy_compatibility_reason,
            "stored_profile_count": self.stored_profile_count,
            "structurally_adaptable_profile_count": self.structurally_adaptable_profile_count,
            "considered_source_paths": list(self.considered_source_paths),
            "current_identity_available": self.current_identity_available,
            "current_model_artifact_set_id": self.current_model_artifact_set_id,
            "current_model_primary_sha256": self.current_model_primary_sha256,
            "current_runtime_profile_stable_key": self.current_runtime_profile_stable_key,
            "current_backend": self.current_backend,
            "selected_evidence_hash": self.selected_evidence_hash,
            "considered_evidence_hashes": list(self.considered_evidence_hashes),
            "reprobe_required": self.status not in REPROBE_NOT_REQUIRED,
        }


@dataclass(frozen=True)
class FleetEvidenceReport:
    cells: Tuple[EvidenceCell, ...]
    models_considered: Tuple[str, ...]
    source_files_scanned: Tuple[str, ...]

    def by_status(self) -> Dict[str, int]:
        counts: Dict[str, int] = {status.value: 0 for status in EvidenceCellStatus}
        for cell in self.cells:
            counts[cell.status.value] += 1
        return counts

    def by_capability(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {family: {status.value: 0 for status in EvidenceCellStatus} for family in FAMILY_ORDER}
        for cell in self.cells:
            out.setdefault(cell.capability, {status.value: 0 for status in EvidenceCellStatus})
            out[cell.capability][cell.status.value] += 1
        return out

    def reprobe_required_cells(self) -> Tuple[EvidenceCell, ...]:
        return tuple(cell for cell in self.cells if cell.status not in REPROBE_NOT_REQUIRED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "models_considered": list(self.models_considered),
            "source_files_scanned": list(self.source_files_scanned),
            "cell_count": len(self.cells),
            "by_status": self.by_status(),
            "by_capability": self.by_capability(),
            "cells": [cell.to_dict() for cell in self.cells],
        }


def discover_capability_report_files(*roots: Path) -> List[Path]:
    """Every ``capability_report.json`` reachable under any of ``roots``.

    Ad hoc top-level runs live at ``runs/<run_id>/capability_report.json``.
    Campaign evidence lives nested deeper (``campaigns/<id>/evidence/primary/``,
    ``campaigns/<id>/evidence/recovery/children/<child_id>/`` -- confirmed by
    reading ``campaign.py``'s ``resolve_paths()`` and its recovery-children
    copy allowlist directly). A recursive scan under each root is used
    rather than mirroring every subsystem's own directory topology by hand,
    so this stays correct if that nesting changes later.
    """
    found: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        found.extend(sorted(root.rglob("capability_report.json")))
    return found


def _read_json(path: Path) -> Any:
    try:
        import json
        return json.loads(path.read_text())
    except Exception:
        return None


def load_fleet_evidence(paths: List[Path]) -> Dict[str, List[Tuple[Path, Dict[str, Any]]]]:
    """Group every stored profile dict by the model alias it was filed
    under. One ``capability_report.json`` may contain several models; each
    (path, model) pair contributes exactly one profile entry."""
    by_model: Dict[str, List[Tuple[Path, Dict[str, Any]]]] = {}
    for path in paths:
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        for model, profile in payload.items():
            if not isinstance(profile, dict):
                continue
            by_model.setdefault(str(model), []).append((path, profile))
    return by_model


def _profile_structural_status(profile: Mapping[str, Any]) -> Optional[str]:
    """``None`` if ``profile`` is structurally adaptable (identity present,
    schema/protocol current); otherwise the raw-precondition bucket it
    fails on. Checked before ever calling
    ``adapt_legacy_profile_family_to_observation`` so a profile that fails
    here is never silently indistinguishable from "never probed"."""
    if profile.get("capability_schema_version") != CAPABILITY_SCHEMA_VERSION:
        return EvidenceCellStatus.LEGACY_SCHEMA.value
    probe_protocol_version = profile.get("probe_protocol_version")
    if not probe_protocol_version or not isinstance(probe_protocol_version, str):
        return EvidenceCellStatus.LEGACY_SCHEMA.value
    identity = typed_identity_from_capability_identity(
        profile.get("capability_identity"), protocol_version=probe_protocol_version
    )
    if identity is None:
        return EvidenceCellStatus.UNBOUND_IDENTITY.value
    return None


_LEGACY_REASON_TO_STATUS = {
    "legacy_or_unbound_capability_profile": EvidenceCellStatus.UNBOUND_IDENTITY,
    "capability_schema_version_changed": EvidenceCellStatus.LEGACY_SCHEMA,
    "probe_protocol_version_changed": EvidenceCellStatus.LEGACY_SCHEMA,
    "model_digest_changed": EvidenceCellStatus.MODEL_IDENTITY_CHANGED,
    "backend_changed": EvidenceCellStatus.BACKEND_CHANGED,
    "backend_implementation_changed": EvidenceCellStatus.BACKEND_CHANGED,
    "endpoint_changed": EvidenceCellStatus.RUNTIME_PROFILE_CHANGED,
    "template_config_changed": EvidenceCellStatus.RUNTIME_PROFILE_CHANGED,
}

_DECISION_REASON_TO_STATUS = {
    CapabilityDecisionReason.MEASURED_SUPPORTED: EvidenceCellStatus.CURRENT_VALID,
    CapabilityDecisionReason.MEASURED_UNSUPPORTED: EvidenceCellStatus.MEASURED_UNSUPPORTED,
    CapabilityDecisionReason.BACKEND_UNSUPPORTED: EvidenceCellStatus.BACKEND_UNSUPPORTED,
    CapabilityDecisionReason.NOT_APPLICABLE: EvidenceCellStatus.NOT_APPLICABLE,
    CapabilityDecisionReason.PROBE_INCONCLUSIVE: EvidenceCellStatus.PROBE_INCONCLUSIVE,
}


def classify_model_capability(
    model: str,
    capability: str,
    stored_profiles: List[Tuple[Path, Dict[str, Any]]],
    current_identity: Optional[Mapping[str, Any]],
    *,
    ledger: Optional[EvidenceLedger] = None,
) -> EvidenceCell:
    """Classify one (model alias, capability) cell from every stored
    profile found for ``model`` across the fleet, against ``current_identity``
    (``capabilities.current_capability_identity()``'s dict shape, or
    ``None`` if the model is not currently reachable).

    Anvil Stage 2.9: when ``ledger`` yields a current, identity-compatible
    native observation for this cell (``CapabilityProjectionStatus.SELECTED``),
    that native evidence is authoritative -- the status is derived from it
    directly, regardless of what the legacy ``capability_report.json``-derived
    evidence below would otherwise say (never the reverse; a stale legacy
    positive never overrides a genuine native measurement). If native
    evidence for this cell is itself ambiguous or conflicted
    (``AMBIGUOUS_COMPATIBLE_OBSERVATIONS`` / ``SUPERSESSION_CONFLICT``), that
    status is reported directly -- fails closed rather than falling back to
    a convenient legacy answer. Only when no usable native evidence exists
    for this cell (``NO_OBSERVATIONS`` / ``NO_COMPATIBLE_OBSERVATION``, or no
    ``ledger`` was given at all) does the unchanged legacy-only classification
    below apply, as the Stage 2.6 compatibility fallback for fleet cells not
    yet reprobed under Stage 2.7C. This is what lets the full evidence
    lifecycle close: MISSING -> reprobe planned -> reprobe executed -> native
    observation appended -> next classification reports CURRENT_VALID (or a
    valid terminal negative), not MISSING again."""
    source_paths = tuple(str(path) for path, _ in stored_profiles)

    current_typed = (
        typed_identity_from_capability_identity(current_identity, protocol_version=PROBE_PROTOCOL_VERSION)
        if current_identity is not None else None
    )
    identity_fields = dict(
        current_model_artifact_set_id=current_typed.model_identity.artifact_set_id if current_typed else None,
        current_model_primary_sha256=current_typed.model_identity.primary_sha256 if current_typed else None,
        current_runtime_profile_stable_key=current_typed.runtime_profile_identity.stable_key() if current_typed else None,
        current_backend=current_typed.runtime_profile_identity.backend if current_typed else None,
    )

    native_projection = None
    if ledger is not None and current_typed is not None:
        native_projection = project_capability_from_ledger(
            ledger,
            capability=capability,
            current_model_identity=current_typed.model_identity,
            current_runtime_profile_identity=current_typed.runtime_profile_identity,
            current_probe_protocol_version=PROBE_PROTOCOL_VERSION,
            current_capability_schema_version=CAPABILITY_SCHEMA_VERSION,
            current_template_config_hash=current_typed.template_hash,
            current_endpoint_identity=current_typed.endpoint_identity,
        )

    if native_projection is not None and native_projection.status == CapabilityProjectionStatus.SELECTED:
        native_decision = decide_capability_from_projection(native_projection)
        return EvidenceCell(
            model=model, capability=capability,
            status=_DECISION_REASON_TO_STATUS[native_decision.reason],
            reason=f"selected native EvidenceLedger observation: {native_decision.reason.value}",
            typed_decision_reason=native_decision.reason.value,
            selected_evidence_hash=native_projection.selected_observation.evidence_hash,
            stored_profile_count=len(stored_profiles),
            structurally_adaptable_profile_count=0,
            considered_source_paths=source_paths, current_identity_available=current_identity is not None,
            **identity_fields,
        )

    if native_projection is not None and native_projection.status in (
        CapabilityProjectionStatus.AMBIGUOUS_COMPATIBLE_OBSERVATIONS,
        CapabilityProjectionStatus.SUPERSESSION_CONFLICT,
    ):
        status = (
            EvidenceCellStatus.AMBIGUOUS_COMPATIBLE_OBSERVATIONS
            if native_projection.status == CapabilityProjectionStatus.AMBIGUOUS_COMPATIBLE_OBSERVATIONS
            else EvidenceCellStatus.SUPERSESSION_CONFLICT
        )
        return EvidenceCell(
            model=model, capability=capability, status=status,
            reason=f"native EvidenceLedger evidence for this cell is itself {native_projection.status.value}; failing closed",
            stored_profile_count=len(stored_profiles),
            structurally_adaptable_profile_count=0,
            considered_source_paths=source_paths, current_identity_available=current_identity is not None,
            **identity_fields,
        )

    if not stored_profiles:
        return EvidenceCell(
            model=model, capability=capability, status=EvidenceCellStatus.MISSING,
            reason="no capability_report.json entry found anywhere in the fleet for this model",
            stored_profile_count=0, structurally_adaptable_profile_count=0,
            considered_source_paths=source_paths, current_identity_available=current_identity is not None,
            **identity_fields,
        )

    adaptable: List[Dict[str, Any]] = []
    raw_statuses: List[str] = []
    for _, profile in stored_profiles:
        failure = _profile_structural_status(profile)
        if failure is None:
            adaptable.append(profile)
        else:
            raw_statuses.append(failure)

    if not adaptable:
        # Every stored profile failed a structural precondition. Report the
        # most common failure kind (deterministic tie-break: enum order),
        # rather than an arbitrary "first" profile's failure.
        ordered = [EvidenceCellStatus.LEGACY_SCHEMA.value, EvidenceCellStatus.UNBOUND_IDENTITY.value]
        chosen = next((status for status in ordered if status in raw_statuses), raw_statuses[0])
        return EvidenceCell(
            model=model, capability=capability, status=EvidenceCellStatus(chosen),
            reason=f"{len(stored_profiles)} stored profile(s) found, none structurally adaptable ({chosen})",
            stored_profile_count=len(stored_profiles), structurally_adaptable_profile_count=0,
            considered_source_paths=source_paths, current_identity_available=current_identity is not None,
            **identity_fields,
        )

    observations = [
        obs for obs in (
            adapt_legacy_profile_family_to_observation(profile, capability) for profile in adaptable
        ) if obs is not None
    ]
    considered_evidence_hashes = tuple(sorted({obs.evidence_hash for obs in observations}))

    projection = project_capability_observation(
        observations,
        capability=capability,
        current_model_identity=current_typed.model_identity if current_typed else None,
        current_runtime_profile_identity=current_typed.runtime_profile_identity if current_typed else None,
        current_probe_protocol_version=PROBE_PROTOCOL_VERSION,
        current_capability_schema_version=CAPABILITY_SCHEMA_VERSION,
        current_template_config_hash=current_typed.template_hash if current_typed else None,
        current_endpoint_identity=current_typed.endpoint_identity if current_typed else None,
    )
    decision = decide_capability_from_projection(projection)

    common = dict(
        model=model, capability=capability, stored_profile_count=len(stored_profiles),
        structurally_adaptable_profile_count=len(adaptable),
        considered_source_paths=source_paths, current_identity_available=current_identity is not None,
        considered_evidence_hashes=considered_evidence_hashes,
        **identity_fields,
    )

    if projection.status == CapabilityProjectionStatus.SELECTED:
        status = _DECISION_REASON_TO_STATUS[decision.reason]
        return EvidenceCell(
            status=status, reason=f"selected observation: {decision.reason.value}",
            typed_decision_reason=decision.reason.value,
            selected_evidence_hash=projection.selected_observation.evidence_hash, **common,
        )

    if projection.status == CapabilityProjectionStatus.NO_OBSERVATIONS:
        return EvidenceCell(
            status=EvidenceCellStatus.MISSING,
            reason="structurally adaptable evidence exists for this model, but this capability was never assessed",
            typed_decision_reason=decision.reason.value, **common,
        )

    if projection.status == CapabilityProjectionStatus.AMBIGUOUS_COMPATIBLE_OBSERVATIONS:
        return EvidenceCell(
            status=EvidenceCellStatus.AMBIGUOUS_COMPATIBLE_OBSERVATIONS,
            reason="multiple compatible observations disagree with no supersession relation to resolve them",
            typed_decision_reason=decision.reason.value, **common,
        )

    # NO_COMPATIBLE_OBSERVATION: every candidate observation is identity-
    # incompatible with the current snapshot. Sub-classify *why* using the
    # legacy dict-shaped compatibility reason (messaging-only, per this
    # module's docstring) against whichever stored profile is most
    # informative -- the first adaptable one, since all adaptable profiles
    # already passed the same schema/protocol precondition.
    if current_identity is None:
        return EvidenceCell(
            status=EvidenceCellStatus.MISSING,
            reason="model is not currently reachable (absent from live inventory); cannot compare stored evidence to a current identity",
            typed_decision_reason=decision.reason.value,
            legacy_compatibility_reason="current_capability_identity_missing", **common,
        )

    legacy_reason = "unknown"
    for profile in adaptable:
        verdict = legacy_capability_identity_compatibility(profile, current_identity)
        if not verdict.get("compatible"):
            legacy_reason = str(verdict.get("reason") or "unknown")
            break
    status = _LEGACY_REASON_TO_STATUS.get(legacy_reason, EvidenceCellStatus.MODEL_IDENTITY_CHANGED)
    return EvidenceCell(
        status=status,
        reason=f"identity-incompatible with current snapshot (legacy diagnostic reason: {legacy_reason})",
        typed_decision_reason=decision.reason.value,
        legacy_compatibility_reason=legacy_reason, **common,
    )


def classify_fleet(
    client: Any,
    *,
    runs_dir: Path = Path("runs"),
    campaigns_root: Path = Path("campaigns"),
    models: Optional[List[str]] = None,
    ledger: Optional[EvidenceLedger] = None,
) -> FleetEvidenceReport:
    """Classify every (model, capability) cell across the fleet's stored
    evidence. ``models`` overrides live discovery (``client.tags()``) for
    testability/filtering; when omitted, every currently-installed model
    plus every model with historical evidence but no longer installed is
    considered, so the report stays exhaustive over "the entire current
    capability evidence population" per the advice, not just what's live
    right now.

    Anvil Stage 2.9: ``ledger`` defaults to the same
    ``<runs_dir>/capability_evidence_ledger.jsonl`` convention
    ``reprobe-execute`` writes to (never created/written here) -- pass an
    explicit ``ledger`` to point at a different one. See
    ``classify_model_capability()`` for the native-preferred policy this
    enables."""
    if ledger is None:
        from .capability_reprobe_execute import default_ledger_path  # local: avoid an import cycle

        ledger = EvidenceLedger(default_ledger_path(runs_dir))
    source_files = discover_capability_report_files(runs_dir, campaigns_root)
    evidence_by_model = load_fleet_evidence(source_files)

    if models is not None:
        live_models = list(models)
    else:
        try:
            rows = client.tags() or []
        except Exception:
            rows = []
        live_models = [str(row.get("name") or "") for row in rows if row.get("name")]

    all_models = sorted(set(live_models) | set(evidence_by_model.keys()))

    current_identities: Dict[str, Optional[Dict[str, Any]]] = {}
    for model in all_models:
        if model in live_models:
            try:
                current_identities[model] = current_capability_identity(client, model)
            except Exception:
                current_identities[model] = None
        else:
            current_identities[model] = None

    cells: List[EvidenceCell] = []
    for model in all_models:
        stored = evidence_by_model.get(model, [])
        current_identity = current_identities.get(model)
        for capability in FAMILY_ORDER:
            cells.append(classify_model_capability(model, capability, stored, current_identity, ledger=ledger))

    return FleetEvidenceReport(
        cells=tuple(cells),
        models_considered=tuple(all_models),
        source_files_scanned=tuple(str(path) for path in source_files),
    )
