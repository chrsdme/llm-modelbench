"""Anvil Stage 2.7B -- deterministic, read-only reprobe planning.

Scope, per this session's Codex/GPT go-ahead
(``local_only/anvil/codex-advice-pre-stage3.txt``, ``==== part 2 ====``):
Stage 2.7A classified every (model, capability) cell's current evidence
state. This module answers the next question -- given that classification,
which cells need fresh evidence, and why -- without running any probe,
appending any observation, or mutating any stored evidence. That remains
Stage 2.7C's job.

**Consumes Stage 2.7A directly, does not reinterpret evidence.** Every
:class:`ReprobeAction` is built from one
:class:`~llm_modelbench.capability_evidence_classification.EvidenceCell`
verbatim -- this module never opens a ``capability_report.json``, never
calls the typed authority pipeline, and never re-derives an identity or
evidence hash. The chain stays: authority pipeline -> evidence
classification -> reprobe planning, one consistent pass, not three
independent interpretations of the same underlying data.

**Classification and action are kept as separate fields, never merged.**
An action record carries both ``classification`` (the 2.7A bucket,
verbatim, e.g. ``"model_identity_changed"``) and ``action`` (``"reprobe"``
or ``"no_action"``) as two fields on one record -- never a combined state
like ``"REPROBE_MODEL_IDENTITY_CHANGED"``. The 13 Stage 2.7A buckets
describe evidence truth; ``action`` describes what follows from it. Which
buckets require a reprobe is not re-decided here: it reuses
:data:`~llm_modelbench.capability_evidence_classification.REPROBE_NOT_REQUIRED`
directly, the exact same set Stage 2.7A's own ``EvidenceCell.to_dict()``
already exposes as ``reprobe_required`` -- a second, hand-maintained bucket
list here could silently drift from the first.

**Ambiguity/conflict is never resolved by selection.** For
``AMBIGUOUS_COMPATIBLE_OBSERVATIONS`` and ``SUPERSESSION_CONFLICT`` cells,
this module does not choose a "winning" observation (e.g. newest
timestamp) -- it emits a reprobe action carrying every distinct evidence
hash Stage 2.7A found (``EvidenceCell.considered_evidence_hashes``,
already a hash *set*, not a single selection) and the original conflict
reason, verbatim. Resolving which prior observation was "right" belongs to
the evidence/supersession path (Stage 2.7C+), never to planning.

**Hints stay hints.** This module never reads ``declared_capabilities``,
``capability_hints()``, or any other declared/metadata field -- it only
ever reads a cell's already-computed ``status``/``reason``. A misleading
declared capability therefore cannot suppress a required reprobe or
fabricate a valid one: there is structurally no code path here that could
let it, not merely a policy choice not to.

**Determinism is structural, not incidental.** ``build_reprobe_plan()``
is a pure function of a already-built
:class:`~llm_modelbench.capability_evidence_classification.FleetEvidenceReport`
-- no filesystem access, no client, no wall-clock read, no randomness.
Its output is explicitly re-sorted by ``(model, FAMILY_ORDER index)``
before use, rather than trusting the input report's iteration order, so a
shuffled/reordered input report still produces an identical canonical
plan. :meth:`ReprobePlan.canonical_plan_hash` hashes only content fields
(``json.dumps(..., sort_keys=True)`` over each action's canonical dict) --
no timestamp, no random id, nothing wall-clock-dependent is ever part of
the hashed payload.

**Genuinely read-only.** This module performs no I/O of any kind: no
probes, no ``CapabilityObservation`` writes, no ``capability_report.json``
rewrites, no supersession records, no rankings/profile mutation, no model
pull/delete, no service lifecycle actions. ``plan_fleet_reprobes()`` is
the only function that touches a client at all, and it does so only by
delegating straight to
:func:`~llm_modelbench.capability_evidence_classification.classify_fleet`
(already-audited read-only fleet classification) before calling the pure
planner -- it adds no new I/O of its own.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .capability_evidence_classification import (
    REPROBE_NOT_REQUIRED,
    EvidenceCell,
    FleetEvidenceReport,
    classify_fleet,
)
from .classify import FAMILY_ORDER

__all__ = [
    "ReprobeActionKind",
    "ReprobeAction",
    "ReprobePlan",
    "build_reprobe_plan",
    "plan_fleet_reprobes",
]


class ReprobeActionKind(str, Enum):
    REPROBE = "reprobe"
    NO_ACTION = "no_action"


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ReprobeAction:
    """One (model, capability) plan entry. ``classification`` is the Stage
    2.7A bucket verbatim; ``action`` is what follows from it -- the two are
    never merged into one combined state."""

    model: str
    capability: str
    classification: str
    classification_reason: str
    action: ReprobeActionKind
    action_reason: str
    current_model_artifact_set_id: Optional[str]
    current_model_primary_sha256: Optional[str]
    current_runtime_profile_stable_key: Optional[str]
    current_backend: Optional[str]
    previous_evidence_hash: Optional[str]
    considered_evidence_hashes: Tuple[str, ...]
    typed_decision_reason: Optional[str]
    legacy_compatibility_reason: Optional[str]
    source_paths: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "capability": self.capability,
            "classification": self.classification,
            "classification_reason": self.classification_reason,
            "action": self.action.value,
            "action_reason": self.action_reason,
            "current_model_artifact_set_id": self.current_model_artifact_set_id,
            "current_model_primary_sha256": self.current_model_primary_sha256,
            "current_runtime_profile_stable_key": self.current_runtime_profile_stable_key,
            "current_backend": self.current_backend,
            "previous_evidence_hash": self.previous_evidence_hash,
            "considered_evidence_hashes": list(self.considered_evidence_hashes),
            "typed_decision_reason": self.typed_decision_reason,
            "legacy_compatibility_reason": self.legacy_compatibility_reason,
            "source_paths": list(self.source_paths),
        }


_VALID_NEGATIVE_STATUSES = {"measured_unsupported", "backend_unsupported", "not_applicable"}
_IDENTITY_DRIFT_STATUSES = {"model_identity_changed"}
_RUNTIME_DRIFT_STATUSES = {"backend_changed", "runtime_profile_changed"}


@dataclass(frozen=True)
class ReprobePlan:
    """A deterministic, canonically-ordered plan: one action per classified
    (model, capability) cell, covering every cell Stage 2.7A classified --
    not only the ones that need a reprobe. Filter with :meth:`filtered` for
    a focused view; the plan itself is always the complete record."""

    actions: Tuple[ReprobeAction, ...]
    models_considered: Tuple[str, ...]

    def canonical_plan_hash(self) -> str:
        return _stable_hash([action.to_dict() for action in self.actions])

    def filtered(
        self,
        *,
        model: Optional[str] = None,
        capability: Optional[str] = None,
        backend: Optional[str] = None,
        reason: Optional[str] = None,
        only_required: bool = False,
    ) -> Tuple[ReprobeAction, ...]:
        """A read-only subset of ``self.actions``. Filtering never changes
        whether a cell intrinsically needs a reprobe -- it only narrows
        which already-decided actions are shown."""
        result = self.actions
        if model is not None:
            result = tuple(a for a in result if a.model == model)
        if capability is not None:
            result = tuple(a for a in result if a.capability == capability)
        if backend is not None:
            result = tuple(a for a in result if a.current_backend == backend)
        if reason is not None:
            result = tuple(a for a in result if a.classification == reason)
        if only_required:
            result = tuple(a for a in result if a.action == ReprobeActionKind.REPROBE)
        return result

    def summary(self) -> Dict[str, Any]:
        by_classification: Dict[str, int] = {}
        actions_by_capability: Dict[str, int] = {}
        actions_by_backend: Dict[str, int] = {}
        actions_by_classification: Dict[str, int] = {}
        total_actions = 0
        for action in self.actions:
            by_classification[action.classification] = by_classification.get(action.classification, 0) + 1
            if action.action == ReprobeActionKind.REPROBE:
                total_actions += 1
                actions_by_capability[action.capability] = actions_by_capability.get(action.capability, 0) + 1
                backend_key = action.current_backend or "unknown"
                actions_by_backend[backend_key] = actions_by_backend.get(backend_key, 0) + 1
                actions_by_classification[action.classification] = actions_by_classification.get(action.classification, 0) + 1
        valid_negative = sum(count for status, count in by_classification.items() if status in _VALID_NEGATIVE_STATUSES)
        identity_drift = sum(count for status, count in by_classification.items() if status in _IDENTITY_DRIFT_STATUSES)
        backend_runtime_drift = sum(count for status, count in by_classification.items() if status in _RUNTIME_DRIFT_STATUSES)
        return {
            "total_cells_examined": len(self.actions),
            "total_actions": total_actions,
            "current_valid": by_classification.get("current_valid", 0),
            "valid_negative": valid_negative,
            "missing": by_classification.get("missing", 0),
            "legacy_schema": by_classification.get("legacy_schema", 0),
            "unbound_identity": by_classification.get("unbound_identity", 0),
            "identity_drift": identity_drift,
            "backend_runtime_drift": backend_runtime_drift,
            "ambiguous": by_classification.get("ambiguous_compatible_observations", 0),
            "supersession_conflict": by_classification.get("supersession_conflict", 0),
            "inconclusive": by_classification.get("probe_inconclusive", 0),
            "by_classification": by_classification,
            "actions_by_capability": actions_by_capability,
            "actions_by_backend": actions_by_backend,
            "actions_by_classification": actions_by_classification,
            # Anvil Stage 2.7B item 15: all evidence classified through
            # this stage is capability_evidence_adapter-migrated legacy
            # data -- EvidenceLedger/native CapabilityObservation
            # persistence is confirmed still never instantiated in
            # production (unchanged since the 2.6E audit), so this count
            # is honestly 0 today rather than fabricated per-cell
            # provenance the storage format cannot actually distinguish.
            "native_capability_observation_records": 0,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_hash": self.canonical_plan_hash(),
            "models_considered": list(self.models_considered),
            "summary": self.summary(),
            "actions": [action.to_dict() for action in self.actions],
        }


def _action_for_cell(cell: EvidenceCell) -> ReprobeAction:
    status_value = cell.status.value
    kind = ReprobeActionKind.NO_ACTION if cell.status in REPROBE_NOT_REQUIRED else ReprobeActionKind.REPROBE
    if kind == ReprobeActionKind.REPROBE:
        action_reason = (
            f"current evidence is classified {status_value}, which requires fresh evidence "
            f"before this cell can be trusted for a decision: {cell.reason}"
        )
    else:
        action_reason = (
            f"current evidence is classified {status_value}, a valid terminal state; "
            f"no reprobe required: {cell.reason}"
        )
    return ReprobeAction(
        model=cell.model,
        capability=cell.capability,
        classification=status_value,
        classification_reason=cell.reason,
        action=kind,
        action_reason=action_reason,
        current_model_artifact_set_id=cell.current_model_artifact_set_id,
        current_model_primary_sha256=cell.current_model_primary_sha256,
        current_runtime_profile_stable_key=cell.current_runtime_profile_stable_key,
        current_backend=cell.current_backend,
        previous_evidence_hash=cell.selected_evidence_hash,
        considered_evidence_hashes=cell.considered_evidence_hashes,
        typed_decision_reason=cell.typed_decision_reason,
        legacy_compatibility_reason=cell.legacy_compatibility_reason,
        source_paths=cell.considered_source_paths,
    )


def build_reprobe_plan(report: FleetEvidenceReport) -> ReprobePlan:
    """Pure, deterministic: build a :class:`ReprobePlan` from an
    already-classified :class:`FleetEvidenceReport`. No I/O, no client, no
    filesystem access -- performs no reclassification of any kind, only
    translates each already-decided :class:`EvidenceCell` into one
    :class:`ReprobeAction`.

    Cells are explicitly re-sorted by ``(model, FAMILY_ORDER index)``
    before building actions, rather than trusting ``report.cells``'
    iteration order, so a reordered/shuffled input report still produces
    an identical canonical plan.
    """
    family_rank = {family: index for index, family in enumerate(FAMILY_ORDER)}
    ordered_cells = sorted(
        report.cells,
        key=lambda cell: (cell.model, family_rank.get(cell.capability, len(FAMILY_ORDER))),
    )
    actions = tuple(_action_for_cell(cell) for cell in ordered_cells)
    return ReprobePlan(actions=actions, models_considered=tuple(sorted(report.models_considered)))


def plan_fleet_reprobes(
    client: Any,
    *,
    runs_dir: Any = None,
    campaigns_root: Any = None,
    models: Optional[List[str]] = None,
) -> ReprobePlan:
    """Convenience wrapper: classify the fleet
    (:func:`~llm_modelbench.capability_evidence_classification.classify_fleet`,
    already read-only/audited) then build a plan from the result. Adds no
    I/O beyond what ``classify_fleet`` already performs."""
    from pathlib import Path as _Path
    kwargs: Dict[str, Any] = {}
    if runs_dir is not None:
        kwargs["runs_dir"] = _Path(runs_dir)
    if campaigns_root is not None:
        kwargs["campaigns_root"] = _Path(campaigns_root)
    if models is not None:
        kwargs["models"] = models
    report = classify_fleet(client, **kwargs)
    return build_reprobe_plan(report)
