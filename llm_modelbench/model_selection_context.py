"""Anvil Stage 3.6 -- ``ModelSelectionContext``: a zero-authority derived
view surfaced *before* a run configuration is finalized, so an operator
re-selecting a previously benchmarked model sees what ModelBench already
knows about it.

Authority: **none**. Every field is a read-only projection sourced from a
single named upstream:

* measured capability families / warnings / native-vs-legacy
  disagreements -- ``planner.build_plan``'s already-resolved
  ``effective_measured_supported_families`` decision (native
  ``EvidenceLedger`` preferred; Stage 2.9 / 3.5 authority). This module
  never re-derives capability support.
* canonical benchmark runtime -- a prior run's immutable
  ``benchmark_bindings.json`` (a real ``BenchmarkRuntimeBinding``
  artifact). Protocol-bound recipe, *not* the fastest observed
  configuration (amendment section 15.2.1). Resolved by a helper that
  never reads throughput rows.
* largest verified context -- the ``max_verified_ctx`` field the runner
  already persists per needle row (``runner._max_verified_prefix``:
  monotonic prefix of successful depths, stops at the first failure;
  skipped depths are never counted). This module reads that field, it
  does not recompute the prefix.
* fastest observed -- ``tps`` on prior benchmark rows, scoped to a single
  ``BenchmarkProtocol`` identity (the comparability boundary -- a
  cross-protocol speed ranking is explicitly disallowed). The winning row
  keeps its own ``benchmark_binding_key`` / ``task``, so a fast run under
  a different runtime profile than the canonical one is surfaced as
  history -- never substituted for the canonical runtime. Resolved by a
  separate helper that never reads binding artifacts.
* lowest VRAM observed -- **unavailable**. ``vram_peak_mb`` is recorded
  only on needle / context-profile probe rows, keyed by context depth and
  not bound to a ``benchmark_binding_key``; it is context-depth telemetry,
  not a per-runtime-profile consumption measurement. Surfaced honestly as
  unavailable rather than faked (Stage 3.6 prompt section 21).

Every historical read is fail-soft: a corrupt or unreadable prior-run
file yields ``unavailable`` for that one observation, never an exception
that would refuse the run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SCHEMA_VERSION = 1

_UNAVAILABLE = "unavailable"
_DEFERRED_CANONICAL = "deferred_pending_protocol_resolution"


@dataclass(frozen=True)
class ModelObservation:
    """One model's prior-knowledge projection. Authoritative fields: none."""

    model: str
    known: bool
    measured_capability_families: List[str]
    capability_warnings: List[str]
    capability_evidence_hash: Optional[str]
    native_legacy_capability_disagreements: List[Dict[str, Any]]
    evidence_trust_class: Optional[str]
    active_protocol_identity: Optional[Dict[str, Any]]
    canonical_benchmark_runtime: Dict[str, Any]
    largest_verified_context: Dict[str, Any]
    fastest_observed: Dict[str, Any]
    lowest_vram_observed: Dict[str, Any]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "known": self.known,
            "measured_capability_families": list(self.measured_capability_families),
            "capability_warnings": list(self.capability_warnings),
            "capability_evidence_hash": self.capability_evidence_hash,
            "native_legacy_capability_disagreements": [
                dict(d) for d in self.native_legacy_capability_disagreements
            ],
            "evidence_trust_class": self.evidence_trust_class,
            "active_protocol_identity": (
                dict(self.active_protocol_identity)
                if self.active_protocol_identity is not None
                else None
            ),
            "canonical_benchmark_runtime": dict(self.canonical_benchmark_runtime),
            "largest_verified_context": dict(self.largest_verified_context),
            "fastest_observed": dict(self.fastest_observed),
            "lowest_vram_observed": dict(self.lowest_vram_observed),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ModelSelectionContext:
    """The full pre-finalization projection over every active model.

    Authoritative fields: none. This object is display/provenance only; it
    is never read back as a decision input.
    """

    schema_version: int
    observations: List[ModelObservation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observations": [obs.to_dict() for obs in self.observations],
        }

    @property
    def by_model(self) -> Dict[str, ModelObservation]:
        return {obs.model: obs for obs in self.observations}


# ---------------------------------------------------------------------------
# fail-soft prior-run readers
# ---------------------------------------------------------------------------


def _iter_run_dirs(runs_dir: Optional[Path]) -> List[Path]:
    if runs_dir is None:
        return []
    try:
        if not runs_dir.exists():
            return []
        return sorted(
            p for p in runs_dir.iterdir()
            if p.is_dir() and (p / "raw_results.jsonl").is_file()
        )
    except OSError:
        return []


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Tolerant line reader -- a truncated final line or a corrupt record is
    skipped, never raised."""
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return []
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(errors="ignore"))
    except (OSError, ValueError):
        return None


def _prior_canonical_runtime(
    model: str,
    runs_dir: Optional[Path],
    *,
    active_protocol_identity_key: Optional[str],
) -> Dict[str, Any]:
    """Resolve the model's canonical benchmark runtime from the *binding
    artifacts* of prior runs. This helper never reads ``raw_results.jsonl``
    throughput rows -- canonical runtime is a protocol-bound recipe, and
    keeping the code path disjoint from the speed scan is what stops a
    future edit from silently substituting "fastest" for "canonical"
    (Stage 3.6 prompt section 20).

    When ``active_protocol_identity_key`` is known, only bindings under a
    matching ``benchmark_protocol_identity_key`` are eligible -- history
    from another protocol is never used to fill the canonical slot for
    this run (section 23). The most recent matching binding wins.

    When the active protocol is *not* resolved at this point in the flow
    (``active_protocol_identity_key is None``), canonical selection is
    **deferred** rather than guessed -- a model can have bindings under
    several protocols and picking one blindly would violate section 10.
    """
    if active_protocol_identity_key is None:
        return {
            "status": _DEFERRED_CANONICAL,
            "reason": "active_protocol_not_resolved_at_plan_time",
        }
    candidates: List[Dict[str, Any]] = []
    for run_dir in _iter_run_dirs(runs_dir):
        payload = _read_json(run_dir / "benchmark_bindings.json")
        if not isinstance(payload, dict):
            continue
        entries: List[Dict[str, Any]] = []
        bindings = payload.get("bindings")
        if isinstance(bindings, dict) and isinstance(bindings.get(model), dict):
            entries.append(bindings[model])
        for entry in payload.get("resume_divergent_bindings") or []:
            if isinstance(entry, dict) and entry.get("model") == model:
                entries.append(entry)
        for entry in entries:
            binding = entry.get("binding") if isinstance(entry.get("binding"), dict) else None
            protocol = entry.get("protocol") if isinstance(entry.get("protocol"), dict) else None
            if not binding:
                continue
            proto_key = binding.get("benchmark_protocol_identity_key")
            if (
                active_protocol_identity_key is not None
                and proto_key is not None
                and proto_key != active_protocol_identity_key
            ):
                continue
            candidates.append({
                "run_id": run_dir.name,
                "binding_key": binding.get("binding_key"),
                "benchmark_protocol_id": (protocol or {}).get("protocol_id"),
                "benchmark_protocol_version": (protocol or {}).get("version"),
                "benchmark_protocol_identity_key": proto_key,
                "runtime_profile_identity_key": binding.get("runtime_profile_identity_key"),
                "allowed_adaptations_used": list(binding.get("allowed_adaptations_used") or []),
                "provenance": binding.get("provenance"),
                "mtime": _safe_mtime(run_dir / "benchmark_bindings.json"),
            })
    if not candidates:
        return {"status": _UNAVAILABLE, "reason": "no_prior_protocol_compatible_binding"}
    best = max(candidates, key=lambda c: (c["mtime"], c["run_id"]))
    best.pop("mtime", None)
    best["status"] = "resolved"
    return best


def _largest_verified_context(model: str, runs_dir: Optional[Path]) -> Dict[str, Any]:
    """Read the runner-persisted ``max_verified_ctx`` off prior needle rows
    and return the largest. The runner is the single deriver of that value
    (``_max_verified_prefix``: successful monotonic prefix, stops at the
    first failed depth); this reads the field, it does not recompute the
    prefix (Stage 3.4B "one derivation, several suppliers").
    """
    best: Optional[int] = None
    best_run: Optional[str] = None
    saw_needle_evidence = False
    for run_dir in _iter_run_dirs(runs_dir):
        for row in _read_jsonl(run_dir / "raw_results.jsonl"):
            if row.get("model") != model:
                continue
            if "max_verified_ctx" not in row and "needle_attempted" not in row:
                continue
            saw_needle_evidence = True
            value = row.get("max_verified_ctx")
            if isinstance(value, (int, float)) and value > 0:
                ivalue = int(value)
                if best is None or ivalue > best:
                    best, best_run = ivalue, run_dir.name
    if best is None:
        return {
            "status": _UNAVAILABLE,
            "reason": (
                "no_verified_needle_depth" if saw_needle_evidence
                else "no_needle_evidence"
            ),
        }
    return {"status": "resolved", "max_verified_ctx": best, "run_id": best_run}


def _fastest_observed(
    model: str,
    runs_dir: Optional[Path],
    *,
    protocol_identity_key: Optional[str],
) -> Dict[str, Any]:
    """Highest ``tps`` on a prior benchmark row for this model, scoped to a
    single ``BenchmarkProtocol`` identity (the comparability boundary --
    section 8), NOT to the canonical binding. The winning row keeps its own
    ``benchmark_binding_key`` / ``task``, so a fastest observation under a
    *different* runtime profile than the canonical one is surfaced as
    exactly that -- history under a comparable protocol -- and is never
    conflated with, or substituted for, the canonical runtime (section 20).

    This helper never reads ``benchmark_bindings.json`` -- it is
    deliberately disjoint from :func:`_prior_canonical_runtime`.

    Without a resolvable protocol identity the comparability scope is
    undefined, so the result is ``unavailable`` rather than a blind
    cross-protocol maximum.
    """
    if not protocol_identity_key:
        return {"status": _UNAVAILABLE, "reason": "no_protocol_scope_for_comparable_throughput"}
    best_tps: Optional[float] = None
    best_row: Optional[Dict[str, Any]] = None
    for run_dir in _iter_run_dirs(runs_dir):
        for row in _read_jsonl(run_dir / "raw_results.jsonl"):
            if row.get("model") != model:
                continue
            if row.get("benchmark_protocol_identity_key") != protocol_identity_key:
                continue
            if row.get("error_kind"):
                continue
            tps = row.get("tps")
            if not isinstance(tps, (int, float)) or tps <= 0:
                continue
            if best_tps is None or tps > best_tps:
                best_tps = float(tps)
                best_row = {
                    "run_id": run_dir.name,
                    "task": row.get("task"),
                    "tps": float(tps),
                    "benchmark_binding_key": row.get("benchmark_binding_key"),
                    "benchmark_protocol_identity_key": protocol_identity_key,
                    "benchmark_protocol_id": row.get("benchmark_protocol_id"),
                    "benchmark_protocol_version": row.get("benchmark_protocol_version"),
                }
    if best_row is None:
        return {"status": _UNAVAILABLE, "reason": "no_throughput_rows_for_protocol"}
    best_row["status"] = "resolved"
    return best_row


def _lowest_vram_observed(model: str) -> Dict[str, Any]:
    """Always unavailable -- see the module docstring and Stage 3.6 prompt
    section 21. ``vram_peak_mb`` is needle-probe context-depth telemetry,
    not a per-runtime-profile consumption measurement, and is not bound to
    a ``benchmark_binding_key``.
    """
    return {
        "status": _UNAVAILABLE,
        "reason": "vram_evidence_is_context_depth_telemetry_not_runtime_profile_scoped",
    }


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# ---------------------------------------------------------------------------
# active-protocol resolution (plan time)
# ---------------------------------------------------------------------------


def _active_protocol_identity(
    model: str,
    *,
    runtime_identity: Any,
    models_row: Mapping[str, Any],
    selected_tasks: Sequence[Any],
    cfg: Any,
    sample_mode: str,
    judge_mode: str,
) -> Optional[Dict[str, Any]]:
    """Build the *active* protocol + binding for this run from the runtime
    identity already collected before plan construction (its ``backend`` /
    ``server_version``). Returns ``None`` when the identity is not
    available at plan time -- callers then leave canonical runtime
    deferred (section 10).
    """
    if runtime_identity is None:
        return None
    try:
        from .benchmark_binding import build_model_binding, protocol_to_dict
        from .identity import ModelArtifactIdentity

        protocol, binding = build_model_binding(
            model_artifact_identity=ModelArtifactIdentity.from_ollama_tag_row(dict(models_row)),
            selected_tasks=list(selected_tasks),
            cfg=cfg,
            backend=getattr(runtime_identity, "backend", None),
            backend_version=getattr(runtime_identity, "server_version", None),
            sample_mode=sample_mode,
            judge_mode=judge_mode,
        )
    except Exception:
        return None
    proto = protocol_to_dict(protocol)
    return {
        "protocol_id": proto.get("protocol_id"),
        "version": proto.get("version"),
        "identity_key": proto.get("identity_key"),
        "benchmark_protocol_identity_key": binding.benchmark_protocol_identity_key,
        "active_binding_key": binding.binding_key(),
    }


# ---------------------------------------------------------------------------
# public builder
# ---------------------------------------------------------------------------


def build_model_selection_context(
    active_models: Sequence[Mapping[str, Any]],
    *,
    runs_dir: Optional[Path] = None,
    runtime_identities: Optional[Mapping[str, Any]] = None,
    models_rows: Optional[Mapping[str, Mapping[str, Any]]] = None,
    cfg: Any = None,
    capability_profiles: Optional[Mapping[str, Mapping[str, Any]]] = None,
    sample_mode: str = "smart",
    judge_mode: str = "off",
    tasks_by_model: Optional[Mapping[str, Sequence[Any]]] = None,
) -> ModelSelectionContext:
    """Assemble the pre-finalization projection.

    ``active_models`` is ``planner.build_plan``'s already-computed
    per-model plan entry list -- the measured capability decision has
    already been made there (native ``EvidenceLedger`` preferred) and is
    consumed by name, never recomputed here.
    """
    runtime_identities = runtime_identities or {}
    models_rows = models_rows or {}
    capability_profiles = capability_profiles or {}
    tasks_by_model = tasks_by_model or {}

    observations: List[ModelObservation] = []
    for entry in active_models:
        model = str(entry.get("model"))
        rid = runtime_identities.get(model)
        profile = capability_profiles.get(model) or {}

        active_protocol = _active_protocol_identity(
            model,
            runtime_identity=rid,
            models_row=models_rows.get(model, {}),
            selected_tasks=tasks_by_model.get(model, []),
            cfg=cfg,
            sample_mode=sample_mode,
            judge_mode=judge_mode,
        )
        active_proto_key = (
            active_protocol.get("benchmark_protocol_identity_key")
            if active_protocol else None
        )

        canonical = _prior_canonical_runtime(
            model, runs_dir, active_protocol_identity_key=active_proto_key
        )
        largest_ctx = _largest_verified_context(model, runs_dir)
        # Fastest is scoped to the *protocol* (the comparability boundary),
        # NOT to the canonical binding -- so a fast historical run under a
        # non-canonical runtime profile is still surfaced as history and is
        # never conflated with the canonical runtime (section 20).
        fastest = _fastest_observed(
            model, runs_dir, protocol_identity_key=active_proto_key
        )
        lowest_vram = _lowest_vram_observed(model)

        families = list(entry.get("families") or [])
        cap_warnings = list(entry.get("capability_warnings") or [])
        disagreements = list(entry.get("native_legacy_capability_evidence_disagreements") or [])

        known = bool(
            families
            or canonical.get("status") == "resolved"
            or largest_ctx.get("status") == "resolved"
            or fastest.get("status") == "resolved"
            or entry.get("capability_evidence_hash")
        )

        warnings: List[str] = []
        if not known:
            warnings.append(
                "No prior authoritative evidence for this model -- measured "
                "capabilities unmeasured, canonical benchmark runtime unresolved, "
                "no history."
            )
        if canonical.get("status") == _DEFERRED_CANONICAL:
            warnings.append(
                "Canonical benchmark runtime deferred: the active BenchmarkProtocol "
                "is not resolved at this point in the flow; protocol-bound history "
                "is shown, canonical selection happens once the protocol is fixed."
            )
        if disagreements:
            warnings.append(
                "Native vs legacy capability-evidence disagreement recorded for "
                f"{len(disagreements)} family/families -- native evidence is authoritative."
            )
        trust = _coerce_trust_class(profile.get("evidence_trust_class"))
        if trust is not None and trust != "canonical_compatible":
            warnings.append(
                f"Evidence trust class is '{trust}', not canonical_compatible -- "
                "historical observations are shown but not treated as trusted for "
                "current comparison."
            )

        observations.append(ModelObservation(
            model=model,
            known=known,
            measured_capability_families=families,
            capability_warnings=cap_warnings,
            capability_evidence_hash=entry.get("capability_evidence_hash"),
            native_legacy_capability_disagreements=disagreements,
            evidence_trust_class=trust,
            active_protocol_identity=active_protocol,
            canonical_benchmark_runtime=canonical,
            largest_verified_context=largest_ctx,
            fastest_observed=fastest,
            lowest_vram_observed=lowest_vram,
            warnings=warnings,
        ))

    return ModelSelectionContext(schema_version=SCHEMA_VERSION, observations=observations)


def _coerce_trust_class(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_model_selection_context(context: Any, *, max_models: int = 40) -> str:
    """Concise operator-facing block for the pre-run plan. Stable
    identifiers are included for diagnostics; full internal hashes are not
    dumped.

    Accepts either a :class:`ModelSelectionContext` or its ``to_dict()``
    payload -- ``planner.render_plan`` reads the plan dict back, so taking
    the dict directly avoids a field-by-field dataclass reconstruction
    that would silently drop any newly added field.
    """
    if isinstance(context, ModelSelectionContext):
        payload = context.to_dict()
    elif isinstance(context, Mapping):
        payload = context
    else:
        return ""
    observations = list(payload.get("observations") or [])
    if not observations:
        return ""
    lines = ["", "Prior model knowledge", "---------------------"]
    for obs in observations[:max_models]:
        known = bool(obs.get("known"))
        lines.append(f"{obs.get('model')}  [{'known' if known else 'new / unmeasured'}]")
        fams = ", ".join(obs.get("measured_capability_families") or []) or "unmeasured"
        lines.append(f"  measured capabilities: {fams}")
        if obs.get("evidence_trust_class"):
            lines.append(f"  evidence trust: {obs.get('evidence_trust_class')}")
        lines.append(
            f"  canonical benchmark runtime: "
            f"{_render_canonical(obs.get('canonical_benchmark_runtime') or {})}"
        )
        proto = obs.get("active_protocol_identity")
        if proto:
            lines.append(
                f"  active protocol: {proto.get('protocol_id')} v{proto.get('version')}"
            )
        lines.append(
            f"  largest verified context: "
            f"{_render_largest_ctx(obs.get('largest_verified_context') or {})}"
        )
        lines.append(f"  fastest observed: {_render_fastest(obs.get('fastest_observed') or {})}")
        lines.append(
            f"  lowest VRAM observed: unavailable "
            f"({(obs.get('lowest_vram_observed') or {}).get('reason')})"
        )
        for warning in obs.get("warnings") or []:
            lines.append(f"  WARN: {warning}")
    if len(observations) > max_models:
        lines.append(f"... {len(observations) - max_models} more models")
    return "\n".join(lines)


def _render_canonical(block: Mapping[str, Any]) -> str:
    status = block.get("status")
    if status == "resolved":
        return (
            f"{block.get('benchmark_protocol_id')} v{block.get('benchmark_protocol_version')} "
            f"profile={_short(block.get('runtime_profile_identity_key'))} "
            f"(from run {block.get('run_id')})"
        )
    if status == _DEFERRED_CANONICAL:
        return "deferred (active protocol not yet resolved)"
    return f"unresolved ({block.get('reason')})"


def _render_largest_ctx(block: Mapping[str, Any]) -> str:
    if block.get("status") == "resolved":
        return f"{block.get('max_verified_ctx')} tokens (run {block.get('run_id')})"
    return f"unavailable ({block.get('reason')})"


def _render_fastest(block: Mapping[str, Any]) -> str:
    if block.get("status") == "resolved":
        return (
            f"{block.get('tps'):.1f} tok/s on {block.get('task')} "
            f"(binding {_short(block.get('benchmark_binding_key'))}, run {block.get('run_id')})"
        )
    return f"unavailable ({block.get('reason')})"


def _short(value: Any) -> str:
    text = str(value or "")
    return text[:12] if len(text) > 12 else (text or "-")
