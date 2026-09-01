"""Anvil Stage 3.1 -- ``ModelCard``: the one canonical, zero-authority
derived view over a model's current-state evidence.

Per ``local_only/anvil/stage-3-architecture-proposal.md`` (v2) §2.2:
``ModelCard`` consolidates the two prior *generation* structures -- Card A
(``model_cards.py``'s ``build_card()``) and Layer C's per-model rendering
(``rankings_v31.py``) -- into one object. It does not replace structure B
(``rankings.py``'s ``master_summary.json`` projection); every derived field
is sourced read-only from a single named upstream authority and a
``ModelCard`` never re-derives evidence itself.

Capability families in particular are read straight from the model's
``families`` field on the master-summary row -- not recomputed via a second
classifier call -- so that ``ModelCard`` and the master-summary projection
agree by construction (Stage 3.0A already made that row-bound field the
measured-only authority; a second, independent derivation here would just
be a new bypass path of exactly the kind 3.0A closed).

``evidence_trust_class`` is a new field required by the master plan
(line 137-145). It is *projected* from an explicit upstream
``EvidenceTrustClass`` when one exists and otherwise fails closed to
``UNKNOWN_LEGACY`` -- ``ModelCard`` never computes the classification
itself (Stage 3.1 reconciliation ``d5d91af``).

Human validation was removed from Anvil entirely by the 2026-09-01
Architecture Simplification Amendment (§17): ModelBench already has
deterministic scorers, bounded recovery, automated judges, and raw
operator-inspectable evidence, and does not need a formal human-validation
workflow or a permanent ``human_validation_status = not_evaluated`` field
for a workflow that was removed from the product.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from .capabilities import CAPABILITY_SCHEMA_VERSION
from .evidence import EvidenceTrustClass
from .identity import ModelArtifactIdentity
from .model_artifact import ModelArtifact

SCHEMA_VERSION = 2


class ModelCardError(ValueError):
    pass


@dataclass(frozen=True)
class ModelCard:
    """Authoritative fields: none. Every field below is a read-only view
    sourced from a single named upstream authority (see module docstring).
    """

    model: Optional[str]
    digest: Optional[str]
    artifact: Optional[ModelArtifact]
    identity: Dict[str, Any]
    quality: Dict[str, Any]
    limits: Dict[str, Any]
    long_context: Dict[str, Any]
    kv_compatibility: Dict[str, Any]
    evidence_warnings: List[str]
    evidence_trust_class: EvidenceTrustClass
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_trust_class, EvidenceTrustClass):
            raise ModelCardError("evidence_trust_class must be an EvidenceTrustClass")


def _artifact_from_model_row(model: Mapping[str, Any]) -> Optional[ModelArtifact]:
    """The master-summary row's own ``digest`` is already the resolved,
    authoritative artifact digest for that row -- Stage 3.0A's reconciliation
    fixup made row/profile digest agreement a precondition for attaching any
    capability evidence at all, so this reuses that same resolved value
    rather than re-deriving identity independently."""
    digest = model.get("digest")
    if not digest:
        return None
    size = model.get("model_size_bytes")
    return ModelArtifact(
        identity=ModelArtifactIdentity(
            artifact_set_id=str(digest),
            primary_sha256=str(digest),
            size_bytes=int(size) if isinstance(size, (int, float)) else None,
            format="ollama-blob",
            quantization=model.get("quantization_level"),
            source=model.get("display_name"),
        )
    )


def _upstream_evidence_trust_class(model: Mapping[str, Any]) -> Optional[EvidenceTrustClass]:
    """Projects an already-computed, authoritative ``EvidenceTrustClass`` if
    one exists upstream -- ``ModelCard`` must never compute this
    classification itself (see module docstring and ``evidence.py``'s own
    contract: scorer compatibility, task-hash compatibility, known-broken
    capability routing, and calibration/debug-only status all bear on trust,
    not capability-schema freshness alone).

    As of Stage 3.1, a full source-map confirms no production writer sets
    this anywhere in the pipeline: ``EvidenceLedger.append()`` /
    ``LedgerRecord`` both default ``trust_class`` to ``CANONICAL_COMPATIBLE``
    and the repository's one real ``ledger.append()`` call site
    (``capability_observation.append_capability_observation()``) never
    overrides it; no ``evidence_trust_class``/``trust_class`` field is ever
    written into ``master_summary.json`` rows or ``capability_report.json``
    anywhere in ``rankings.py``/``runner.py``/``capabilities.py``. This
    function exists so that a future stage recording a real classification
    is picked up automatically -- without ``ModelCard`` being rewritten --
    rather than because such a value exists today."""
    for source in (model, model.get("capability_profile")):
        if not isinstance(source, Mapping):
            continue
        raw = source.get("evidence_trust_class")
        if raw is None:
            continue
        try:
            return EvidenceTrustClass(raw)
        except ValueError:
            continue  # malformed upstream value -- fails closed, not a crash
    return None


def _evidence_trust_class(model: Mapping[str, Any]) -> EvidenceTrustClass:
    """No authoritative upstream ``EvidenceTrustClass`` exists anywhere in
    this pipeline yet (see :func:`_upstream_evidence_trust_class`), so this
    conservatively defaults to ``UNKNOWN_LEGACY`` rather than inferring a
    stronger class from a proxy signal. Capability-schema-version match is
    deliberately *not* used as a proxy for overall evidence trust -- it says
    nothing about scorer/task-hash compatibility, known-broken capability
    routing, or calibration/debug-only status, so it cannot by itself prove
    ``canonical_compatible`` (schema matches) or ``historical_valid``
    (schema is older). Schema staleness is still surfaced, but as a warning
    (see :func:`_capability_schema_warning`), never as a trust promotion."""
    upstream = _upstream_evidence_trust_class(model)
    if upstream is not None:
        return upstream
    return EvidenceTrustClass.UNKNOWN_LEGACY


def _capability_schema_warning(model: Mapping[str, Any]) -> Optional[str]:
    """A separate concern from :func:`_evidence_trust_class`: a bound
    capability profile on a non-current schema is worth surfacing to a
    reader, but it must never by itself change the card's trust
    classification -- see this module's docstring."""
    profile = model.get("capability_profile")
    if not isinstance(profile, Mapping) or not profile:
        return None
    schema_version = profile.get("capability_schema_version")
    if schema_version == CAPABILITY_SCHEMA_VERSION:
        return None
    return (
        f"Bound capability profile schema version is '{schema_version}', not the "
        f"current '{CAPABILITY_SCHEMA_VERSION}' -- capability-derived fields on this "
        "row may reflect an older probe/classification schema."
    )


def _identity_block(model: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "names_seen": model.get("names_seen") or [],
        "class": model.get("class"),
        "families": model.get("families") or [],
        "parameter_size": model.get("parameter_size"),
        "quantization_level": model.get("quantization_level"),
        "architecture_family": model.get("architecture_family"),
        "model_size_bytes": model.get("model_size_bytes"),
        "size_gb": model.get("size_gb"),
    }


def build_model_card(
    model: Mapping[str, Any],
    *,
    quality: Dict[str, Any],
    limits: Dict[str, Any],
    long_context: Dict[str, Any],
    kv_compatibility: Optional[Dict[str, Any]] = None,
    evidence_warnings: Optional[List[str]] = None,
) -> ModelCard:
    """Assembles a :class:`ModelCard` from already-derived section payloads
    (``quality``/``limits``/``long_context`` keep their existing shapes and
    derivation logic -- see ``model_cards.py``'s ``build_card()``, which is
    now a thin wrapper over this function) plus the two fields Stage 3.1
    adds: ``evidence_trust_class`` (projected, see
    :func:`_evidence_trust_class`)."""
    trust_class = _evidence_trust_class(model)
    warnings = list(evidence_warnings or [])
    schema_warning = _capability_schema_warning(model)
    if schema_warning:
        warnings.append(schema_warning)
    if trust_class != EvidenceTrustClass.CANONICAL_COMPATIBLE:
        warnings.append(
            f"Evidence trust class is '{trust_class.value}', not canonical_compatible -- "
            "no authoritative upstream evidence-trust classification exists for this row "
            "yet, so it is not shown as trusted for current comparison."
        )
    return ModelCard(
        model=model.get("display_name"),
        digest=model.get("digest"),
        artifact=_artifact_from_model_row(model),
        identity=_identity_block(model),
        quality=quality,
        limits=limits,
        long_context=long_context,
        kv_compatibility=kv_compatibility or {},
        evidence_warnings=warnings,
        evidence_trust_class=trust_class,
    )


def model_card_to_dict(card: ModelCard) -> Dict[str, Any]:
    """The existing Card A JSON/Markdown shape, unchanged, plus the new
    Stage 3.1 ``evidence_trust_class`` field. Golden-file compatibility for
    every pre-existing key is a deliberate contract -- see
    ``tests/test_model_card.py``."""
    return {
        "schema_version": card.schema_version,
        "model": card.model,
        "digest": card.digest,
        "identity": dict(card.identity),
        "quality": dict(card.quality),
        "limits": dict(card.limits),
        "long_context": dict(card.long_context),
        "kv_compatibility": dict(card.kv_compatibility),
        "evidence_warnings": list(card.evidence_warnings),
        "evidence_trust_class": card.evidence_trust_class.value,
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _gb_from_mb(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value) / 1024:.2f} GB"


def render_model_card_markdown(card: Dict[str, Any]) -> str:
    """Renders the JSON-shaped dict produced by :func:`model_card_to_dict`
    (Card A's existing Markdown format, extended with a Stage 3.1 evidence
    trust section)."""
    identity = card["identity"]
    quality = card["quality"]
    limits = card["limits"]
    context = card["long_context"]
    kv = card.get("kv_compatibility") or {}
    lines = [
        f"# {card.get('model')}",
        "",
        "## Identity",
        "",
        f"- Digest: `{card.get('digest') or 'n/a'}`",
        f"- Class: `{identity.get('class') or 'n/a'}`",
        f"- Families: {', '.join(identity.get('families') or []) or 'n/a'}",
        f"- Parameters: `{identity.get('parameter_size') or 'n/a'}`",
        f"- Quantization: `{identity.get('quantization_level') or 'n/a'}`",
        f"- Architecture: `{identity.get('architecture_family') or 'n/a'}`",
        f"- Stored size: `{_fmt(identity.get('size_gb'))} GB`",
        "",
        "## Quality and applicability",
        "",
        f"- Status: **{quality.get('status') or 'unknown'}**",
        f"- Overall score: **{_fmt(quality.get('overall_mean_score'))}**",
        f"- Rank / tie band: `{_fmt(quality.get('overall_rank'))}` / `{_fmt(quality.get('tie_band'))}`",
        f"- Coverage: `{_pct(quality.get('coverage_ratio'))}`",
        f"- Capability limited: `{limits.get('capability_limited')}`",
        f"- Measured capability failure: `{limits.get('capability_measured_failure')}`",
        f"- Recovery limited: `{limits.get('recovery_limited')}`",
    ]
    if limits.get("capability_unavailable_tasks"):
        lines.append("- Unavailable tasks: " + ", ".join(limits["capability_unavailable_tasks"]))
    if limits.get("capability_measured_failure_tasks"):
        lines.append("- Measured zero-quality capability tasks: " + ", ".join(limits["capability_measured_failure_tasks"]))
    if limits.get("recovery_exhausted_tasks"):
        lines.append("- Recovery-exhausted tasks: " + ", ".join(limits["recovery_exhausted_tasks"]))

    lines += [
        "",
        "## Long-context operating profile",
        "",
        f"- Target context: `{_fmt(context.get('target_ctx'))}`",
        f"- Operating status: **{context.get('target_status') or 'not_verified'}**",
        f"- Maximum verified effective context: `{_fmt(context.get('max_verified_ctx'))}`",
        f"- Context quality score / coverage: `{_fmt(context.get('score'))}` / `{_pct(context.get('coverage'))}`",
        f"- Target prompt / decode speed: `{_fmt(context.get('target_prompt_tps'))}` / `{_fmt(context.get('target_tps'))}` tok/s",
        f"- Target elapsed time: `{_fmt(context.get('target_elapsed_seconds'))}` seconds",
        f"- Target offload: `{_pct(context.get('target_offload_fraction'))}`",
        f"- Target host-memory delta: `{_gb_from_mb(context.get('target_host_delta_mb'))}`",
        f"- Target swap delta: `{_gb_from_mb(context.get('target_swap_delta_mb'))}`",
        f"- Minimum / median decode speed: `{_fmt(context.get('min_tps'))}` / `{_fmt(context.get('median_tps'))}` tok/s",
        "",
        "| Tier | num_ctx | Result | Prompt tok/s | Decode tok/s | Elapsed | VRAM peak | Host delta | Swap | Offload | Output |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for depth in context.get("depths") or []:
        host = depth.get("ollama_pss_delta_peak_mb")
        if host is None:
            host = depth.get("ollama_rss_delta_peak_mb")
        if host is None:
            host = depth.get("ram_delta_peak_mb")
        result = "pass" if depth.get("found") else (depth.get("error_kind") or "not-run")
        output = "suspect" if depth.get("response_suspect") else ("exact" if depth.get("response_exact") else "n/a")
        lines.append(
            f"| {_fmt(depth.get('size'))} | {_fmt(depth.get('num_ctx'))} | {result} | "
            f"{_fmt(depth.get('prompt_tps'))} | {_fmt(depth.get('tps'))} | "
            f"{_fmt(depth.get('elapsed_seconds'))}s | {_gb_from_mb(depth.get('vram_peak_mb'))} | "
            f"{_gb_from_mb(host)} | {_gb_from_mb(depth.get('swap_delta_peak_mb'))} | "
            f"{_pct(depth.get('offload_fraction'))} | {output} |"
        )

    behavior = context.get("behavior_probe") or {}
    lines += ["", "## 64k behavior probe", ""]
    if behavior:
        lines.extend([
            f"- Operating status: **{behavior.get('operating_status') or 'unknown'}**",
            f"- Effective prompt context: `{_fmt(behavior.get('prompt_eval_count'))}`",
            f"- Prompt / decode speed: `{_fmt(behavior.get('prompt_tps'))}` / `{_fmt(behavior.get('tps'))}` tok/s",
            f"- Exact anchors: `{behavior.get('all_anchors_exact')}`",
            f"- Ordered sequence retained: `{behavior.get('sequence_ok')}`",
            f"- Response repetition ratio: `{_fmt(behavior.get('response_repetition_ratio'))}`",
            f"- Agentic readiness: `{behavior.get('agentic_readiness') or 'not_assessed'}`",
            f"- Scope note: {behavior.get('agentic_readiness_note') or 'Long-horizon agentic reliability was not assessed.'}",
        ])
    else:
        lines.append("No synthetic long-context behavior probe is available.")

    lines += ["", "## KV compatibility", ""]
    if kv:
        lines.append(f"- Preferred KV mode: `{kv.get('preferred_kv_type') or 'n/a'}`")
        lines.append(f"- Current KV supported: `{bool(kv.get('current_kv_supported'))}`")
        lines.append(f"- Avoid quantized KV: `{bool(kv.get('avoid_quantized_kv'))}`")
        for mode, entry in sorted((kv.get("kv_modes") or {}).items()):
            lines.append(
                f"- `{mode}`: `{entry.get('status') or 'unknown'}`"
                + (f" ({', '.join(entry.get('error_kinds') or [])})" if entry.get("error_kinds") else "")
            )
    else:
        lines.append("No build-scoped KV compatibility record is available.")

    if card.get("evidence_warnings"):
        lines += ["", "## Evidence warnings", ""]
        lines.extend(f"- {warning}" for warning in card["evidence_warnings"])

    lines += [
        "",
        "## Evidence trust",
        "",
        f"- Evidence trust class: `{card.get('evidence_trust_class') or 'unknown_legacy'}`",
    ]

    lines += ["", "## Status reasons", ""]
    lines.extend(f"- {reason}" for reason in quality.get("reasons") or ["No reasons recorded."])
    return "\n".join(lines) + "\n"
