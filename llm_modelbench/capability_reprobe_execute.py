"""Anvil Stage 2.7C -- reprobe execution + validation.

Consumes a Stage 2.7B :class:`~llm_modelbench.capability_reprobe_plan.ReprobePlan`
and, only for actions the plan marked ``REPROBE``, runs one real functional
probe against the live model, converts the result into a typed
:class:`~llm_modelbench.capability_observation.CapabilityObservation` via
the existing legacy-profile adapter, and appends it to an
:class:`~llm_modelbench.evidence.EvidenceLedger`. This is the first
production write path for ``EvidenceLedger`` anywhere in this codebase
(confirmed by a dedicated research pass before writing this module: every
prior real constructor call was test-only).

Full design rationale, including the two decisions that are not obvious
from the code alone -- how before/after fleet metrics are reported, and
how a pre-existing ambiguous vs. structurally-conflicting ledger state is
handled -- is recorded in ``local_only/anvil/stage-2.7C-execution.md``, not
duplicated here.

Scope, reused verbatim, never reinvented:

- :func:`~llm_modelbench.capabilities.interrogate_model` -- the only place
  that sends a real probe request to a live backend.
- :func:`~llm_modelbench.capability_evidence_adapter.adapt_legacy_profile_family_to_observation`
  -- the only place that converts a legacy-shaped probe result into a typed
  observation.
- :func:`~llm_modelbench.capability_observation.append_capability_observation`
  -- the only ledger write path used.
- :func:`~llm_modelbench.capability_projection.project_capability_from_ledger`
  / :func:`~llm_modelbench.capability_projection.decide_capability_from_projection`
  -- reused both to find a cell's currently-selected native observation
  (for supersession) and to compute its state right after appending.
- :func:`~llm_modelbench.capability_evidence_classification.classify_fleet`
  -- reused, unmodified, to capture the legacy-evidence-axis "before"
  snapshot; this module never reclassifies or replans.

This module never deletes a model, never mutates a stored
``capability_report.json``, never touches a ``NO_ACTION`` cell, and never
runs a probe for any capability other than the one an action explicitly
names.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .capabilities import CAPABILITY_SCHEMA_VERSION, PROBE_PROTOCOL_VERSION, interrogate_model
from .capability_evidence_adapter import adapt_legacy_profile_family_to_observation
from .capability_evidence_classification import (
    EvidenceCellStatus,
    FleetEvidenceReport,
    _DECISION_REASON_TO_STATUS,
    classify_fleet,
)
from .capability_observation import append_capability_observation
from .capability_trust import classify_fresh_capability_trust
from .capability_projection import (
    CapabilityProjection,
    CapabilityProjectionStatus,
    decide_capability_from_projection,
    project_capability_from_ledger,
)
from .capability_reprobe_plan import ReprobeAction, ReprobeActionKind, ReprobePlan
from .classify import FAMILY_ORDER
from .evidence import EvidenceLedger, ProvenanceLink, ProvenanceRelation

__all__ = [
    "ReprobeOutcome",
    "ReprobeExecutionReport",
    "default_ledger_path",
    "execute_reprobe_actions",
    "run_reprobe_execution",
]


def default_ledger_path(runs_dir: Path) -> Path:
    """No prior convention exists anywhere in this repo for where a
    ``CapabilityObservation`` ledger file should live (confirmed by a
    dedicated search: the only ``ledger_path`` reference elsewhere in the
    codebase is an unrelated "coverage ledger" concept). This picks
    ``<runs_dir>/capability_evidence_ledger.jsonl``, consistent with
    ``runs/`` already being the root ``capability-evidence``/
    ``reprobe-plan`` scan for ``capability_report.json`` under."""
    return Path(runs_dir) / "capability_evidence_ledger.jsonl"


def _status_from_projection(projection: CapabilityProjection) -> EvidenceCellStatus:
    """Map a freshly-recomputed ``CapabilityProjection`` to the same
    13-bucket vocabulary Stage 2.7A's legacy-sourced classification uses,
    reusing ``_DECISION_REASON_TO_STATUS`` for the SELECTED case rather
    than re-declaring the reason->bucket mapping a second time."""
    decision = decide_capability_from_projection(projection)
    if projection.status == CapabilityProjectionStatus.SELECTED:
        return _DECISION_REASON_TO_STATUS.get(decision.reason, EvidenceCellStatus.PROBE_INCONCLUSIVE)
    if projection.status == CapabilityProjectionStatus.AMBIGUOUS_COMPATIBLE_OBSERVATIONS:
        return EvidenceCellStatus.AMBIGUOUS_COMPATIBLE_OBSERVATIONS
    if projection.status == CapabilityProjectionStatus.SUPERSESSION_CONFLICT:
        return EvidenceCellStatus.SUPERSESSION_CONFLICT
    return EvidenceCellStatus.MISSING


@dataclass(frozen=True)
class ReprobeOutcome:
    """What actually happened when Stage 2.7C attempted one planned
    ``REPROBE`` action. ``NO_ACTION`` cells never produce an outcome at
    all -- they are filtered out before execution starts, not recorded as
    a no-op outcome, so this list's length is exactly the number of cells
    this run actually touched."""

    model: str
    capability: str
    appended: bool
    skip_reason: Optional[str] = None
    error: Optional[str] = None
    observation_id: Optional[str] = None
    superseded_record_ids: Tuple[str, ...] = ()
    prior_status: Optional[str] = None
    after_status: Optional[str] = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "capability": self.capability,
            "appended": self.appended,
            "skip_reason": self.skip_reason,
            "error": self.error,
            "observation_id": self.observation_id,
            "superseded_record_ids": list(self.superseded_record_ids),
            "prior_status": self.prior_status,
            "after_status": self.after_status,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _execute_one(action: ReprobeAction, client: Any, ledger: EvidenceLedger) -> ReprobeOutcome:
    start = time.monotonic()
    try:
        profile = interrogate_model(client, action.model, functional=True, probe_families=[action.capability])
    except Exception as exc:  # noqa: BLE001 -- a probe failure is a recorded outcome, not a crash
        return ReprobeOutcome(
            model=action.model, capability=action.capability, appended=False,
            error=f"probe raised: {exc}", elapsed_seconds=time.monotonic() - start,
        )

    observation = adapt_legacy_profile_family_to_observation(profile, action.capability)
    if observation is None:
        return ReprobeOutcome(
            model=action.model, capability=action.capability, appended=False,
            skip_reason=(
                "probe result could not be adapted into a typed CapabilityObservation "
                "(current identity still does not satisfy the typed adapter's preconditions)"
            ),
            elapsed_seconds=time.monotonic() - start,
        )

    # Every field project_capability_observation()/project_capability_from_ledger()
    # compares "current" identity against, not just model+runtime identity --
    # omitting template_config_hash/endpoint_identity here would make even the
    # observation just adapted from this exact probe compare as incompatible
    # against itself (confirmed by a smoke test before this fix landed).
    current_identity_kwargs = dict(
        current_model_identity=observation.model_identity,
        current_runtime_profile_identity=observation.runtime_profile_identity,
        current_probe_protocol_version=PROBE_PROTOCOL_VERSION,
        current_capability_schema_version=CAPABILITY_SCHEMA_VERSION,
        current_template_config_hash=observation.template_config_hash,
        current_endpoint_identity=observation.endpoint_identity,
    )

    prior_projection = project_capability_from_ledger(ledger, capability=action.capability, **current_identity_kwargs)
    prior_status = _status_from_projection(prior_projection).value

    if prior_projection.status == CapabilityProjectionStatus.SUPERSESSION_CONFLICT:
        # A resolver-detected structural defect in the ledger's own
        # provenance graph (cycle/missing-source/fork) -- categorically
        # different from "multiple valid but disagreeing measurements".
        # Appending on top without understanding why the graph is
        # malformed would not fix it and could mask the defect, so this
        # action is skipped and flagged for manual review rather than
        # auto-resolved. See stage-2.7C-execution.md decision 2.
        return ReprobeOutcome(
            model=action.model, capability=action.capability, appended=False,
            skip_reason=(
                "pre-existing SUPERSESSION_CONFLICT in the ledger for this cell; "
                "execution does not auto-repair conflicting provenance"
            ),
            prior_status=prior_status, elapsed_seconds=time.monotonic() - start,
        )

    superseded: Tuple[str, ...] = ()
    if prior_projection.status == CapabilityProjectionStatus.SELECTED and prior_projection.selected_record_id:
        superseded = (prior_projection.selected_record_id,)
    elif prior_projection.status == CapabilityProjectionStatus.AMBIGUOUS_COMPATIBLE_OBSERVATIONS:
        # A fresh, authoritative measurement resolves standing ambiguity by
        # explicitly superseding every disagreeing predecessor -- never by
        # picking one on recency (no timestamp is consulted here). See
        # advice item 14 / stage-2.7C-execution.md decision 2.
        superseded = tuple(prior_projection.considered_observation_ids)

    provenance = tuple(ProvenanceLink(ProvenanceRelation.SUPERSEDES, record_id) for record_id in superseded)
    # Anvil Stage 3B.2 (owner's frozen rule): assign the EvidenceTrustClass
    # explicitly at write time. classify_fresh_capability_trust fails closed
    # to UNKNOWN_LEGACY unless the complete current probe contract is
    # explicitly demonstrated by this observation (current probe/schema
    # version, content-addressed model provenance, real runtime identity,
    # committing measured result). Trust is never inferred from the fact
    # that this probe just ran.
    trust_class = classify_fresh_capability_trust(
        observation,
        expected_probe_protocol_version=PROBE_PROTOCOL_VERSION,
        expected_capability_schema_version=CAPABILITY_SCHEMA_VERSION,
    )
    record = append_capability_observation(
        ledger, observation, trust_class=trust_class, provenance=provenance
    )

    after_projection = project_capability_from_ledger(ledger, capability=action.capability, **current_identity_kwargs)
    return ReprobeOutcome(
        model=action.model, capability=action.capability, appended=True,
        observation_id=record.record_id, superseded_record_ids=superseded,
        prior_status=prior_status, after_status=_status_from_projection(after_projection).value,
        elapsed_seconds=time.monotonic() - start,
    )


def execute_reprobe_actions(
    actions: Tuple[ReprobeAction, ...], client: Any, ledger: EvidenceLedger,
) -> Tuple[ReprobeOutcome, ...]:
    """Execute only the ``REPROBE`` actions in ``actions``. ``NO_ACTION``
    entries are never inspected beyond this one filter -- not probed, not
    adapted, not appended. Callers pass an already plan-filtered action
    tuple (:meth:`ReprobePlan.filtered`) when a CLI ``--model``/
    ``--capability``/etc. filter should also narrow what gets executed."""
    return tuple(
        _execute_one(action, client, ledger)
        for action in actions
        if action.action == ReprobeActionKind.REPROBE
    )


@dataclass(frozen=True)
class ReprobeExecutionReport:
    plan_hash: str
    outcomes: Tuple[ReprobeOutcome, ...]
    fleet_before: Dict[str, Any]
    native_evidence_after: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_hash": self.plan_hash,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "fleet_before": self.fleet_before,
            "native_evidence_after": self.native_evidence_after,
        }


def _native_evidence_summary(outcomes: Tuple[ReprobeOutcome, ...]) -> Dict[str, Any]:
    by_capability: Dict[str, Dict[str, int]] = {family: {"appended": 0, "skipped": 0, "errored": 0} for family in FAMILY_ORDER}
    by_after_status: Dict[str, int] = {}
    for outcome in outcomes:
        bucket = by_capability.setdefault(outcome.capability, {"appended": 0, "skipped": 0, "errored": 0})
        if outcome.appended:
            bucket["appended"] += 1
            if outcome.after_status:
                by_after_status[outcome.after_status] = by_after_status.get(outcome.after_status, 0) + 1
        elif outcome.error:
            bucket["errored"] += 1
        else:
            bucket["skipped"] += 1
    return {
        "attempted": len(outcomes),
        "appended": sum(1 for o in outcomes if o.appended),
        "skipped": sum(1 for o in outcomes if not o.appended and not o.error),
        "errored": sum(1 for o in outcomes if o.error),
        "superseded_record_count": sum(len(o.superseded_record_ids) for o in outcomes),
        "by_capability": by_capability,
        "by_after_status": by_after_status,
    }


def run_reprobe_execution(
    plan: ReprobePlan, client: Any, ledger: EvidenceLedger, *,
    runs_dir: Path = Path("runs"), campaigns_root: Path = Path("campaigns"),
    actions: Optional[Tuple[ReprobeAction, ...]] = None,
) -> ReprobeExecutionReport:
    """Execute (a possibly pre-filtered subset of) ``plan``'s actions and
    produce a before/after report. ``fleet_before`` is a :func:`classify_fleet`
    snapshot taken once before any probe in *this* run fires, using this same
    ``ledger`` -- so it already reflects any native evidence accumulated by
    earlier ``reprobe-execute`` runs (Stage 2.9: ``classify_fleet()`` is
    native-ledger-aware, preferring a current compatible native observation
    over the legacy-adapter-derived classification; see
    ``classify_model_capability()``'s docstring for the exact policy). It is
    no longer guaranteed to equal a fresh ``classify_fleet()`` call made
    *after* this run's own execution below, precisely because that execution
    appends new native evidence to this same ledger -- newly-reprobed cells
    are expected to flip from e.g. MISSING to CURRENT_VALID between before
    and after now, which is the Stage 2.9 lifecycle-closure property, not a
    regression of Stage 2.7C's original "legacy axis unchanged" observation
    (that observation is still true of the *legacy* capability_report.json
    axis alone, which this module still never rewrites -- it just is no
    longer the only axis classify_fleet() reports on).
    ``native_evidence_after`` is the genuinely new signal from *this run*
    specifically: per-capability coverage of the ledger evidence this run
    actually wrote. See ``local_only/anvil/stage-2.7C-execution.md`` decision
    1 for why these stay two separate axes rather than one blended number."""
    fleet_before: FleetEvidenceReport = classify_fleet(client, runs_dir=runs_dir, campaigns_root=campaigns_root, ledger=ledger)
    selected_actions = actions if actions is not None else plan.actions
    outcomes = execute_reprobe_actions(selected_actions, client, ledger)
    return ReprobeExecutionReport(
        plan_hash=plan.canonical_plan_hash(),
        outcomes=outcomes,
        fleet_before=fleet_before.to_dict(),
        native_evidence_after=_native_evidence_summary(outcomes),
    )
