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

from typing import Any, Dict, Iterable, Mapping, Tuple

from .benchmark_protocol import BenchmarkProtocol, _stable_hash
from .config import Config
from .tasks import Task

PROTOCOL_ID = "llm-modelbench-core"
PROTOCOL_VERSION = "1"
ALLOWED_ADAPTATIONS: Tuple[str, ...] = ("context_size",)


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


def build_benchmark_protocol(selected_tasks: Iterable[Task], cfg: Config) -> BenchmarkProtocol:
    """Deterministic production builder over the existing ``BenchmarkProtocol``.

    Same semantic inputs -> same identity key. No global protocol database:
    the protocol is derived for the concrete benchmark and persisted with
    evidence (Stage 3.2D-2).
    """
    tasks = list(selected_tasks)
    if not tasks:
        raise BenchmarkPolicyError("build_benchmark_protocol requires at least one task")
    return BenchmarkProtocol(
        protocol_id=PROTOCOL_ID,
        version=PROTOCOL_VERSION,
        task_ids=tuple(task.id for task in tasks),
        prompt_semantics_hash=_stable_hash("prompt_semantics_v1", build_prompt_semantics_manifest(tasks)),
        sampling_policy_hash=_stable_hash("sampling_policy_v1", build_sampling_policy_manifest(cfg)),
        output_budget_policy_hash=_stable_hash("output_budget_policy_v1", build_output_budget_manifest(tasks, cfg)),
        scorer_versions=resolve_scorer_versions(tasks),
        allowed_adaptations=ALLOWED_ADAPTATIONS,
    )
