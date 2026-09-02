"""Anvil Stage 3.2D-0 -- deterministic ``BenchmarkProtocol`` construction.

The protocol identity axes frozen by the owner at Stage 3.2D:

* ``protocol_id = "llm-modelbench-core"`` -- the stable public benchmark
  methodology family. Never Anvil/RC/git/backend/host-specific.
* ``version = "1"`` -- the methodology frozen at Stage 3.2D. A version bump
  is required only when comparability-relevant *semantics* change, not when
  the operator selects a different valid task subset (ordered ``task_ids``
  carry that independently).
* ``allowed_adaptations = ("context_size",)`` -- ModelBench may resolve
  context size to satisfy a task/protocol requirement, subject to model
  capability and environment feasibility. Physical GPU selection, multi-GPU
  placement, RAM spill, backend selection, batch tuning, KV quantization,
  offload/split tuning and any sampling/prompt tuning are **not** protocol
  adaptations -- they are runtime realization or prohibited score tuning.

The three policy hashes are built from small canonical manifests, never from
source files or git revisions, and hashed with
:func:`benchmark_protocol._stable_hash` (the one project hashing idiom --
no second framework).

**Not** :func:`runner._task_hash`: that is a deliberately over-inclusive
resume *cache key* (blind ``meta`` dump, ``difficulty``, ``agentic``,
``judge``) whose job is invalidating stale rows on *any* change. These
manifests are semantic-only and will legitimately disagree with it.

No production call site is wired here -- see Stage 3.2D-2.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .benchmark_protocol import BenchmarkProtocol, _stable_hash
from .config import DEFAULT_WEIGHTS, Config
from .sample_policy import (
    SAMPLE_APPLICABILITY_POLICY_VERSION,
    SAMPLE_COMBINATION,
    samples_for_task,
)
from .tasks import Task

PROTOCOL_ID = "llm-modelbench-core"
PROTOCOL_VERSION = "1"
ALLOWED_ADAPTATIONS: Tuple[str, ...] = ("context_size",)

# Bump ONLY when the canonical aggregation algorithm itself
# (aggregate.aggregate / outcome.category_score) changes comparability
# semantics -- e.g. the coverage-must-be-1.0 rule, the difficulty-weighted
# intra-category mean, the category-weight renormalization, the 2-dp
# rounding, or the agentic_tool decision-score substitution -- while the
# numeric constants (DEFAULT_WEIGHTS, Task.difficulty) stay put. A pure
# numeric-constant change is already caught by the manifest below and needs
# no bump here.
AGGREGATION_CONTRACT_VERSION = 1


class BenchmarkPolicyError(ValueError):
    pass


# --- scorer contract versions -------------------------------------------------
#
# Current ``Task.scorer`` strings are not sufficient identity by themselves.
# This is the smallest explicit scorer-contract-version mapping. Every scorer
# key reachable from a canonical task -- across all four dispatch paths --
# gets an explicit contract version. An unknown/unversioned scorer FAILS
# CLOSED at protocol construction; it is never silently defaulted to 1.
#
# Dispatch paths (verified against runner._score_task / scoring.DETERMINISTIC
# / runner family paths):
#   * scoring.DETERMINISTIC[key]:
#       python web_nav js_debounce lineset contains json_schema filesort
#       agentic_action ocr exact_code exact
#   * runner._score_task explicit branch:      retrieval
#   * runner family-specific paths:            fim (_fim_score),
#                                              native_tool (_native_tool_score)
#   * runner dedicated paths:                  needle (token presence),
#                                              subjective (judge)
#
# A material scoring-semantic change bumps the relevant contract version.
# Refactors / comments / formatting do not. Git hashes are never versions.
SCORER_CONTRACT_VERSIONS: Dict[str, int] = {
    "python": 1,
    "web_nav": 1,
    "js_debounce": 1,
    "lineset": 1,
    "contains": 1,
    "json_schema": 1,
    "filesort": 1,
    "agentic_action": 1,
    "ocr": 1,
    "exact_code": 1,
    "exact": 1,
    "retrieval": 1,
    "fim": 1,
    "native_tool": 1,
    "needle": 1,
    "subjective": 1,
}

# ``meta`` keys that materially determine what the model is asked to do or
# process. Classified by ROLE (does this reach the model?), not by which
# module reads it -- the embedding/retrieval tasks feed ``docs``/``queries``
# straight to the model, so they are prompt-bearing despite being read in
# the scoring layer.
_PROMPT_BEARING_META_KEYS: Tuple[str, ...] = (
    "suffix",        # FIM completion suffix appended to the prompt
    "needle_token",  # long-context needle inserted into the generated haystack
    "context_sizes", # long-context probe depths
    "tools",         # native tool-calling tool definitions
    "image_path",    # vision input image
    "reference",     # vision OCR / rendered-text source
    "noisy",         # vision render mode (changes the rendered image)
    "is_pdf",        # vision input is a PDF page
    "docs",          # retrieval corpus embedded by the model
    "queries",       # retrieval queries embedded by the model
    "cases",         # embedding-diagnostic input cases
)

# Sampling knobs ModelBench does NOT send -- represented with an explicit
# sentinel rather than dropped, so the manifest stays honest about delegation.
_BACKEND_DEFAULT = "backend_default"
_DELEGATED_SAMPLING_KEYS: Tuple[str, ...] = (
    "top_p", "top_k", "min_p", "repeat_penalty", "mirostat",
)

# Bounded-recovery output-budget policy (repair._thinking_retry_profiles):
# canonical recovery re-asks think-off at the original budget, then at
# ``max(original, think_retry_num_predict)``. Recovery can materially raise a
# model's permitted answer budget, so this policy is comparability material.
_RECOVERY_DEFAULT_THINK_RETRY_NUM_PREDICT = 4096
_RECOVERY_SOURCE_NUM_PREDICT_FALLBACK = 2048


def _canonical(value: Any) -> Any:
    """Reduce to plain JSON-friendly primitives -- never hand _stable_hash a
    dataclass or object (its ``default=str`` would fold in a repr)."""
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
        return value
    return str(value)


def _prompt_bearing_meta(meta: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: _canonical(meta[key]) for key in _PROMPT_BEARING_META_KEYS if key in meta}


def resolve_scorer_versions(tasks: Iterable[Task]) -> Tuple[Tuple[str, str], ...]:
    """Per-task ``(task_id, "<scorer_key>@<contract_version>")``.

    Fails closed on any scorer key without an explicit contract version.
    """
    resolved = []
    for task in tasks:
        version = SCORER_CONTRACT_VERSIONS.get(task.scorer)
        if version is None:
            raise BenchmarkPolicyError(
                f"scorer {task.scorer!r} (task {task.id!r}) has no explicit contract "
                "version; add it to SCORER_CONTRACT_VERSIONS -- unknown scorers fail closed"
            )
        resolved.append((task.id, f"{task.scorer}@{version}"))
    return tuple(resolved)


def build_prompt_semantics_manifest(tasks: Iterable[Task]) -> list:
    """Ordered per-task material that determines what the model is asked."""
    manifest = []
    for task in tasks:
        manifest.append({
            "id": task.id,
            "family": task.family,
            "scorer": task.scorer,
            "prompt": task.prompt,
            "agentic": bool(task.agentic),
            "num_predict": int(task.num_predict),
            "meta": _prompt_bearing_meta(task.meta),
        })
    return manifest


def build_sampling_policy_manifest(cfg: Config) -> Dict[str, Any]:
    """Normalized canonical sampling contract ModelBench actually controls."""
    manifest: Dict[str, Any] = {
        "temperature": float(cfg.temperature),
        "seed": int(cfg.seed),
    }
    for key in _DELEGATED_SAMPLING_KEYS:
        manifest[key] = _BACKEND_DEFAULT
    return manifest


def build_output_budget_manifest(tasks: Iterable[Task], cfg: Config) -> Dict[str, Any]:
    """Per-task budgets + benchmark-wide derived ceilings + recovery policy."""
    per_task = {task.id: int(task.num_predict) for task in tasks}
    return {
        "per_task_num_predict": _canonical(per_task),
        "num_predict_override": (
            int(cfg.num_predict_override) if cfg.num_predict_override else None
        ),
        "derived_ceilings": {
            # runner.py: needle_num_predict = max(256, int(_num_predict(cfg, task, 256)))
            "needle_min_num_predict": 256,
            # runner.py: probe path max(1024, int(cfg.num_predict_override or 1024))
            "probe_min_num_predict": 1024,
        },
        "bounded_recovery": {
            # repair._thinking_retry_profiles: [original, max(original, retry_budget)]
            "policy": "think_off_original_then_max_original_retry_budget",
            "think_retry_num_predict": _RECOVERY_DEFAULT_THINK_RETRY_NUM_PREDICT,
            "source_num_predict_fallback": _RECOVERY_SOURCE_NUM_PREDICT_FALLBACK,
        },
    }


def build_aggregation_policy_manifest(
    tasks: Iterable[Task], cfg: Config, *, sample_mode: str, judge_mode: str
) -> Dict[str, Any]:
    """Canonical aggregation policy: everything that can move the canonical
    composite score / ranking from the *same* raw task outputs.

    Category weights: only the categories that materially participate in this
    concrete benchmark. ``aggregate()`` renormalizes the composite over
    exactly the categories that produced a score, so a weight for a category
    that cannot run here cannot change this result -- including all 15 would
    make an unrelated reweight move this identity. This is the smallest
    representation that still guarantees "a weight change capable of changing
    *this* result changes this identity".

    Task difficulty: the raw numeric value per selected task (it is the
    intra-category weight in ``outcome.category_score``) AND the derived
    gate/scored split (``difficulty <= 0`` -> pass/fail gate, contributes no
    positive quality), recorded explicitly so a reader need not re-derive the
    boundary.

    Sample policy: the concrete per-task draw count under this run's
    ``sample_mode`` / ``judge_mode`` (via the shared
    :func:`sample_policy.samples_for_task` -- not reimplemented here), plus
    the applicability-policy version and the numeric-combination contract
    (``aggregate._avg_numeric``).

    ``judge_mode`` is recorded verbatim rather than subsetted the way
    ``category_weights`` is: judging changes scorer *semantics* for
    subjective tasks (judge score vs raw-only), not only the draw count, so
    a ``judge_mode`` change is comparability-relevant wherever a subjective
    or judge task is present. For a task set with none, this is mildly
    over-inclusive (the hash moves though the result cannot) -- accepted as
    the smaller evil than trying to prove judge-irrelevance per task set.

    NOT included -- downstream of the canonical score or comparison-only,
    established by source map: ``regression()`` drop threshold, the
    ``prune_recommendations`` q25/t25 quartiles, ``pareto_frontier``.
    """
    tasks = list(tasks)
    participating = sorted({task.category for task in tasks if task.category})
    category_weights = {c: float(DEFAULT_WEIGHTS.get(c, 0.0)) for c in participating}
    task_difficulty = {task.id: float(task.difficulty) for task in tasks}
    gate_task_ids = tuple(sorted(t.id for t in tasks if float(t.difficulty) <= 0.0))
    sample_counts = {
        task.id: int(samples_for_task(task, cfg, sample_mode, judge_mode)) for task in tasks
    }
    return {
        "aggregation_contract_version": AGGREGATION_CONTRACT_VERSION,
        "category_weights": _canonical(category_weights),
        "task_difficulty": _canonical(task_difficulty),
        "gate_task_ids": list(gate_task_ids),
        "sample_policy": {
            "sample_mode": str(sample_mode),
            "judge_mode": str(judge_mode),
            "requested_samples": max(1, int(getattr(cfg, "samples", 1) or 1)),
            "applicability_policy_version": SAMPLE_APPLICABILITY_POLICY_VERSION,
            "combination": SAMPLE_COMBINATION,
            "per_task_samples": _canonical(sample_counts),
        },
    }


def build_benchmark_protocol(
    selected_tasks: Iterable[Task],
    cfg: Config,
    *,
    sample_mode: str = "smart",
    judge_mode: str = "single",
) -> BenchmarkProtocol:
    """Deterministic production builder over the existing ``BenchmarkProtocol``.

    Same semantic inputs -> same identity key. No global protocol database:
    the protocol is derived for the concrete benchmark and persisted with
    evidence (Stage 3.2D-2).

    ``sample_mode`` / ``judge_mode`` are ``runner.run`` parameters (not
    ``Config`` fields); they are threaded here because they change the
    canonical aggregation (how many draws a task gets, then averaged).
    """
    tasks = list(selected_tasks)
    if not tasks:
        raise BenchmarkPolicyError("build_benchmark_protocol requires at least one task")
    aggregation_policy_hash = _stable_hash(
        "aggregation_policy_v1",
        build_aggregation_policy_manifest(
            tasks, cfg, sample_mode=sample_mode, judge_mode=judge_mode
        ),
    )
    if not aggregation_policy_hash:  # pragma: no cover - _stable_hash never empty
        raise BenchmarkPolicyError("aggregation_policy_hash resolved empty")
    return BenchmarkProtocol(
        protocol_id=PROTOCOL_ID,
        version=PROTOCOL_VERSION,
        task_ids=tuple(task.id for task in tasks),
        prompt_semantics_hash=_stable_hash("prompt_semantics_v1", build_prompt_semantics_manifest(tasks)),
        sampling_policy_hash=_stable_hash("sampling_policy_v1", build_sampling_policy_manifest(cfg)),
        output_budget_policy_hash=_stable_hash("output_budget_policy_v1", build_output_budget_manifest(tasks, cfg)),
        scorer_versions=resolve_scorer_versions(tasks),
        allowed_adaptations=ALLOWED_ADAPTATIONS,
        aggregation_policy_hash=aggregation_policy_hash,
    )


# --- canonical ranking aggregation-policy verifier (Anvil Stage 3.3A) --------
#
# Stage 3.2E made ``BenchmarkProtocol.aggregation_policy_hash`` identity-bearing
# and recorded it per bound run. It is NOT self-enforcing: canonical ranking
# (``rankings.aggregate(canonical_rows, DEFAULT_WEIGHTS, _TASK_DIFFICULTY)``)
# reads the live ``DEFAULT_WEIGHTS`` / ``Task.difficulty`` / sample-policy
# constants and never checks them against any run's recorded hash. Editing a
# ``Task.difficulty`` value or a ``DEFAULT_WEIGHTS`` entry therefore silently
# rewrites every recomputed canonical ranking.
#
# This verifier recomputes the aggregation-policy hash a run WOULD get today --
# same task selection, same sample policy, but the CURRENT module constants --
# and compares it to what the run recorded. It reuses
# ``build_aggregation_policy_manifest`` verbatim (one source of aggregation
# semantics) and the one ``_stable_hash("aggregation_policy_v1", ...)`` idiom
# from ``build_benchmark_protocol`` -- it does not rebuild any of the policy.

# Verdict vocabulary. Kept as plain strings (persisted into master_raw.jsonl
# and the ranking summary), never an Enum -- mirrors the rest of the evidence
# schema.
AGG_VERDICT_VERIFIED = "verified"
AGG_VERDICT_POLICY_DRIFT = "policy_drift"
AGG_VERDICT_UNVERIFIED_LEGACY = "unverified_legacy"
AGG_VERDICT_UNVERIFIED_INCOMPLETE = "unverified_incomplete"


class AggregationPolicyVerdict:
    """Result of checking one run's recorded aggregation-policy hash against
    the hash the same task selection would produce under today's constants."""

    __slots__ = ("verdict", "recorded_hash", "recomputed_hash", "reason", "task_ids")

    def __init__(
        self,
        verdict: str,
        *,
        recorded_hash: str = "",
        recomputed_hash: str = "",
        reason: str = "",
        task_ids: Tuple[str, ...] = (),
    ) -> None:
        self.verdict = verdict
        self.recorded_hash = recorded_hash
        self.recomputed_hash = recomputed_hash
        self.reason = reason
        self.task_ids = task_ids

    @property
    def compatible(self) -> bool:
        """True only when the recorded policy provably matches today's."""
        return self.verdict == AGG_VERDICT_VERIFIED

    def as_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "recorded_aggregation_policy_hash": self.recorded_hash,
            "recomputed_aggregation_policy_hash": self.recomputed_hash,
            "reason": self.reason,
        }


def recompute_aggregation_policy_hash(
    task_ids: Iterable[str],
    *,
    requested_samples: int,
    sample_mode: str,
    judge_mode: str,
    tasks_by_id: Mapping[str, Task],
) -> str:
    """The ``aggregation_policy_hash`` the given task selection would get under
    the CURRENT ``DEFAULT_WEIGHTS`` / ``Task.difficulty`` / sample-policy
    constants. Raises :class:`BenchmarkPolicyError` if any task id is unknown
    to the current suite (the recompute would otherwise silently drop it and
    produce a hash that cannot be compared honestly).
    """
    ordered: List[Task] = []
    for tid in task_ids:
        task = tasks_by_id.get(tid)
        if task is None:
            raise BenchmarkPolicyError(
                f"cannot recompute aggregation policy: task {tid!r} is not in the "
                "current task suite"
            )
        ordered.append(task)
    if not ordered:
        raise BenchmarkPolicyError(
            "cannot recompute aggregation policy: empty task selection"
        )
    cfg_shim = SimpleNamespace(samples=max(1, int(requested_samples or 1)))
    manifest = build_aggregation_policy_manifest(
        ordered, cfg_shim, sample_mode=str(sample_mode), judge_mode=str(judge_mode)
    )
    return _stable_hash("aggregation_policy_v1", manifest)


def verify_recorded_aggregation_policy(
    *,
    recorded_hash: str,
    task_ids: Iterable[str],
    requested_samples: Any,
    sample_mode: Any,
    judge_mode: Any,
    tasks_by_id: Mapping[str, Task],
) -> AggregationPolicyVerdict:
    """Classify one run's recorded aggregation-policy hash.

    * no ``recorded_hash`` -> ``unverified_legacy`` (a pre-Stage-3.2E run, or a
      run whose bindings were not resolved). Canonical ranking still proceeds
      for such rows; a policy is never retrospectively attributed to them.
    * missing ``sample_mode`` / ``judge_mode`` / ``requested_samples``, unknown
      task, or empty selection -> ``unverified_incomplete`` (cannot honestly
      recompute; treated like legacy -- not a drift claim).
    * recorded == recomputed -> ``verified``.
    * recorded != recomputed -> ``policy_drift`` (the live canonical
      aggregation constants have moved since this run was bound).
    """
    task_ids = tuple(task_ids)
    if not recorded_hash:
        return AggregationPolicyVerdict(
            AGG_VERDICT_UNVERIFIED_LEGACY,
            reason="no recorded aggregation_policy_hash (pre-Stage-3.2E run or unresolved binding)",
            task_ids=task_ids,
        )
    if sample_mode in (None, "") or judge_mode in (None, "") or requested_samples in (None, ""):
        return AggregationPolicyVerdict(
            AGG_VERDICT_UNVERIFIED_INCOMPLETE,
            recorded_hash=recorded_hash,
            reason="run configuration missing sample_mode / judge_mode / requested_samples; cannot recompute",
            task_ids=task_ids,
        )
    try:
        recomputed = recompute_aggregation_policy_hash(
            task_ids,
            requested_samples=requested_samples,
            sample_mode=sample_mode,
            judge_mode=judge_mode,
            tasks_by_id=tasks_by_id,
        )
    except BenchmarkPolicyError as exc:
        return AggregationPolicyVerdict(
            AGG_VERDICT_UNVERIFIED_INCOMPLETE,
            recorded_hash=recorded_hash,
            reason=str(exc),
            task_ids=task_ids,
        )
    if recomputed == recorded_hash:
        return AggregationPolicyVerdict(
            AGG_VERDICT_VERIFIED,
            recorded_hash=recorded_hash,
            recomputed_hash=recomputed,
            task_ids=task_ids,
        )
    return AggregationPolicyVerdict(
        AGG_VERDICT_POLICY_DRIFT,
        recorded_hash=recorded_hash,
        recomputed_hash=recomputed,
        reason=(
            "recorded aggregation_policy_hash does not match the hash this task "
            "selection produces under the current canonical weights / task "
            "difficulty / sample policy"
        ),
        task_ids=task_ids,
    )
