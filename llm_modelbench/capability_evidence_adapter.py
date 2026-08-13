"""Anvil Stage 2.6A (phase 1) -- a safe, honest adapter from today's legacy
``interrogate_model()``/``capabilities.py`` profile dict shape into a real
typed :class:`~llm_modelbench.capability_observation.CapabilityObservation`.

This module exists to answer the central open design question flagged
before any planner/runner/recovery call-site migration could start (per
this session's Stage 2.6 continuing advice,
``local_only/anvil/codex-advice-stage2-continuing.txt``): how does live
evidence actually enter the new Stage 2 typed stack, without silently
laundering old, unbound, or merely-declared authority into a supposedly
measured, typed observation?

**Confirmed before writing this adapter, not assumed**: ``EvidenceLedger``
(Stage 0) is never instantiated anywhere in production code today --
only in tests (confirmed by ``git grep`` across ``llm_modelbench/``). So
"read real ``CapabilityObservation`` records from the ledger" is not a
currently viable option; there is nothing in it. The only live evidence
that exists today is the legacy dict-shaped payload
``capabilities.interrogate_model()`` already produces, consumed by
``planner.py``/``runner.py``/``repair.py``/``campaign.py``.

**Why this adapter does not launder old authority into new types**:
inspected ``interrogate_model()``'s actual payload shape directly (not
assumed) and found it already carries genuinely typed-enough evidence
for a faithful translation:
- ``measured_capabilities[family]["state"]`` is already a
  :class:`~llm_modelbench.capabilities.MeasuredCapabilityState` value
  string -- ``MEASURED_SUPPORTED``/``MEASURED_UNSUPPORTED``/
  ``BACKEND_UNSUPPORTED``/``NOT_APPLICABLE``/``PROBE_INCONCLUSIVE`` --
  the exact same enum :class:`CapabilityObservation.result` uses. This
  adapter copies that value verbatim; it never upgrades a metadata-only
  hint into a positive claim. When ``functional=False`` interrogation
  ran (metadata/hints only), ``interrogate_model()`` itself already
  labels every family ``PROBE_INCONCLUSIVE`` -- and that maps onto a
  perfectly honest ``CapabilityObservation`` (evidence exists that says
  "we don't actually know"), not a fabricated one.
- ``capability_identity["model"]`` is built by ``capabilities._model_identity()``
  using the *identical* digest/id/model_id lookup-with-fallback order
  that :meth:`~llm_modelbench.identity.ModelArtifactIdentity.from_ollama_tag_row`
  already implements (Stage 0.4) -- this adapter reuses that factory
  directly on a reshaped row rather than re-deriving digest logic a
  second time.
- ``capability_identity["backend"]``/``["template_config"]`` carry real
  backend/template data, sufficient to build a genuine (if coarser than
  Stage 2.2's fuller model) :class:`~llm_modelbench.identity.RuntimeProfileIdentity`.
  ``template_config`` is itself ``{"available": bool, "hash": str,
  "material": {...}}`` (``capabilities._template_config_identity()``'s
  real shape, confirmed by reading it directly) -- this adapter reuses
  its precomputed ``"hash"`` field verbatim rather than hashing the
  wrapper dict itself, so the resulting ``template_hash`` is the exact
  same value ``capability_identity_compatibility()`` already compares
  today, not a second, adapter-specific hash that wouldn't correlate
  with legacy compatibility decisions at all.

**What this adapter refuses to do, structurally**: returns ``None``
(never a best-effort guess) whenever the profile cannot honestly support
an observation --
schema/protocol version mismatch, missing/malformed ``capability_identity``,
the family was never assessed at all, or an unrecognized state string.
A ``None`` result is not a failure to handle -- it is the correct,
intentional outcome for legacy/unbound evidence, which should
structurally produce ``NO_CURRENT_PROJECTION`` ->
``CAPABILITY_REPROBE_REQUIRED`` by simply having no observation to find,
exactly matching what Stage 2.3 already does for absent evidence.

Migration status (all four live consumers now migrated, as of Anvil Stage
2.6D): ``planner.py`` (2.6A phase 2), ``runner.py`` (2.6B), ``repair.py``
(2.6C), and ``campaign.py``'s judge-selection functions (2.6D) all source
their sole positive execution/recovery/judge-eligibility authority from
this module's ``new_measured_supported_families()``.
``new_measured_supported_families()`` lives here rather than in
``planner.py`` or ``runner.py`` specifically so both can import it
without creating a circular import (``planner.py`` already imports from
``runner.py``); this module has no dependency on either.

**Lifecycle (Anvil Stage 2.6E)**: this module is a *migration bridge* --
validated, already-captured ``interrogate_model()`` profile evidence
translated into typed Stage 2 capability authority. It is explicitly not:
declared-metadata promotion, a generic "bless legacy data" mechanism, or a
permanent parallel capability system meant to coexist with
:mod:`~llm_modelbench.capability_observation`/
:mod:`~llm_modelbench.evidence` indefinitely. It exists only because
production capability evidence is not yet natively produced/persisted as
typed ``CapabilityObservation`` records (``EvidenceLedger`` is still never
instantiated in production, confirmed unchanged since this module was
first built). Its expected demotion/removal point is whenever a later
stage (tracked as Stage 2.7+ legacy-evidence migration, or Stage 3) makes
native ``CapabilityObservation`` production/persistence the normal path --
at that point this adapter's job is done and callers should read real
observations directly instead of translating legacy dicts on every call.

**Long-term architecture (decided at Stage 2.6E closure, not left
implicit)**: each of the four migrated consumers deliberately asks this
module's *capability-only* question (``new_measured_supported_families()``)
rather than being forced through the fuller
:mod:`~llm_modelbench.task_applicability` composition
(capability + environment + operator policy) that only ``planner.py``'s
own family-routing gate ever needed that full shape for. This is an
intentional design choice, not an unfinished migration: "one capability
authority, not necessarily one giant decision object everywhere." A
subsystem that only ever asks a capability question (runner's execution
gate, repair's recovery gate, judge eligibility) should keep asking only
that question through this module; forcing every consumer through
``TaskApplicability`` for uniformity's own sake would add environment/
operator-policy inputs those call sites don't have and don't need. See
``local_only/anvil/stage-2.6E-authority-audit.md`` for the full audit this
decision was made against.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

from .capabilities import CAPABILITY_SCHEMA_VERSION, PROBE_PROTOCOL_VERSION, MeasuredCapabilityState
from .capability_observation import CapabilityObservation
from .capability_projection import decide_capability_from_projection, project_capability_observation
from .classify import FAMILY_ORDER
from .identity import ModelArtifactIdentity, RuntimeProfileIdentity

__all__ = [
    "TypedLegacyIdentity",
    "typed_identity_from_capability_identity",
    "adapt_legacy_profile_family_to_observation",
    "new_measured_supported_families",
]


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:16]


@dataclass(frozen=True)
class TypedLegacyIdentity:
    """The typed identity triple this adapter can honestly derive from one
    legacy ``capability_identity``-shaped dict (the same shape both
    ``interrogate_model()``'s stored ``profile["capability_identity"]``
    and a fresh ``capabilities.current_capability_identity()`` call
    produce). Shared by :func:`adapt_legacy_profile_family_to_observation`
    (stored evidence) and the comparison harness (live "current" identity)
    so the two never drift into two different extraction rules for what
    is, structurally, the same input shape."""

    model_identity: ModelArtifactIdentity
    runtime_profile_identity: RuntimeProfileIdentity
    template_hash: Optional[str]
    endpoint_identity: Optional[str]


def typed_identity_from_capability_identity(
    capability_identity: Any, *, protocol_version: Optional[str] = None
) -> Optional[TypedLegacyIdentity]:
    """Build a :class:`TypedLegacyIdentity` from one legacy
    ``capability_identity`` dict (as produced by
    ``capabilities._model_identity``/``_backend_identity``/
    ``_template_config_identity``), or ``None`` if it's missing required
    material. ``protocol_version`` defaults to the identity dict's own
    ``probe_protocol_version`` field when not given explicitly."""
    if not isinstance(capability_identity, Mapping):
        return None
    model_info = capability_identity.get("model")
    backend_info = capability_identity.get("backend")
    template_config = capability_identity.get("template_config")
    if not isinstance(model_info, Mapping) or not isinstance(backend_info, Mapping):
        return None

    resolved_protocol_version = protocol_version or capability_identity.get("probe_protocol_version")
    if not resolved_protocol_version or not isinstance(resolved_protocol_version, str):
        return None

    # Reuse identity.py's own digest-lookup-with-fallback logic (Stage
    # 0.4) rather than re-deriving it here -- _model_identity()'s output
    # shape is a reshaped but compatible view of an Ollama tags row.
    model_identity = ModelArtifactIdentity.from_ollama_tag_row(
        {
            "name": model_info.get("canonical_name"),
            "digest": model_info.get("digest"),
            "size": model_info.get("size"),
            "details": model_info.get("details") if isinstance(model_info.get("details"), Mapping) else {},
        }
    )

    # capabilities._template_config_identity()'s real shape is
    # {"available": bool, "hash": str, "material": {...}} -- reuse its
    # own precomputed "hash" (the exact value
    # capability_identity_compatibility() already compares) rather than
    # hashing the wrapper dict ourselves, which would produce a
    # different value uncorrelated with legacy compatibility decisions.
    # Fall back to hashing whatever's present only for malformed/older
    # shapes that lack a usable "hash" field.
    if isinstance(template_config, Mapping) and isinstance(template_config.get("hash"), str):
        template_hash = template_config["hash"]
    elif template_config is not None:
        template_hash = _stable_hash(template_config)
    else:
        template_hash = None
    runtime_profile_identity = RuntimeProfileIdentity(
        backend=str(backend_info.get("backend") or "unknown"),
        protocol_version=resolved_protocol_version,
        template_hash=template_hash,
    )
    endpoint_identity = backend_info.get("endpoint")
    return TypedLegacyIdentity(
        model_identity=model_identity,
        runtime_profile_identity=runtime_profile_identity,
        template_hash=template_hash,
        endpoint_identity=str(endpoint_identity) if endpoint_identity else None,
    )


def adapt_legacy_profile_family_to_observation(
    profile: Mapping[str, Any], family: str
) -> Optional[CapabilityObservation]:
    """Translate one family's measured evidence out of a legacy
    ``interrogate_model()``-shaped ``profile`` dict into a real
    :class:`CapabilityObservation`, or return ``None`` if the profile
    cannot honestly support one for this family. Never raises on
    malformed legacy input -- a legacy profile predates this typed
    contract by definition, so "cannot adapt" is the expected, common
    case, not an error."""
    if not isinstance(profile, Mapping):
        return None
    if profile.get("capability_schema_version") != CAPABILITY_SCHEMA_VERSION:
        return None
    probe_protocol_version = profile.get("probe_protocol_version")
    if not probe_protocol_version or not isinstance(probe_protocol_version, str):
        return None

    identity = typed_identity_from_capability_identity(
        profile.get("capability_identity"), protocol_version=probe_protocol_version
    )
    if identity is None:
        return None

    measured = profile.get("measured_capabilities")
    if not isinstance(measured, Mapping):
        return None
    family_measurement = measured.get(family)
    if not isinstance(family_measurement, Mapping):
        return None
    state_value = family_measurement.get("state")
    try:
        result = MeasuredCapabilityState(state_value)
    except ValueError:
        return None

    return CapabilityObservation(
        model_identity=identity.model_identity,
        runtime_profile_identity=identity.runtime_profile_identity,
        capability=family,
        result=result,
        probe_protocol_version=probe_protocol_version,
        capability_schema_version=CAPABILITY_SCHEMA_VERSION,
        template_config_hash=identity.template_hash,
        endpoint_identity=identity.endpoint_identity,
        declared_hint=tuple(profile.get("declared_capabilities") or ()),
    )


def new_measured_supported_families(
    profile: Mapping[str, Any], current_identity: Optional[Mapping[str, Any]]
) -> List[str]:
    """The typed Stage 2.1-2.3 capability-authority replacement for
    ``capabilities.measured_supported_families()``. For each family in
    ``FAMILY_ORDER``, adapts ``profile`` into a ``CapabilityObservation``
    (or finds none), projects it against ``current_identity`` (the same
    dict shape ``capabilities.current_capability_identity()`` produces),
    and includes the family only when the resulting
    ``CapabilityDecision.applicable`` is true.

    Shared by ``planner.py`` (Stage 2.6A phase 2) and ``runner.py``
    (Stage 2.6B) as their execution-authority gate -- built once here so
    both consume the identical adapter/projection/decision pipeline
    rather than two independently-maintained copies.
    """
    current_typed_identity = (
        typed_identity_from_capability_identity(current_identity, protocol_version=PROBE_PROTOCOL_VERSION)
        if current_identity is not None else None
    )
    fams: List[str] = []
    for family in FAMILY_ORDER:
        observation = adapt_legacy_profile_family_to_observation(profile, family)
        projection = project_capability_observation(
            [observation] if observation is not None else [],
            capability=family,
            current_model_identity=current_typed_identity.model_identity if current_typed_identity else None,
            current_runtime_profile_identity=current_typed_identity.runtime_profile_identity if current_typed_identity else None,
            current_probe_protocol_version=PROBE_PROTOCOL_VERSION,
            current_capability_schema_version=CAPABILITY_SCHEMA_VERSION,
            current_template_config_hash=current_typed_identity.template_hash if current_typed_identity else None,
            current_endpoint_identity=current_typed_identity.endpoint_identity if current_typed_identity else None,
        )
        if decide_capability_from_projection(projection).applicable:
            fams.append(family)
    return fams
