"""Canonical per-task sample-count policy.

One pure function, shared by the runner (which draws that many samples per
task), the planner (sample-count estimates) and Anvil benchmark-protocol
construction (which must make the sample-count / combination policy
identity-bearing -- Stage 3.2E). Kept in its own module so protocol
construction does not import the runner.

The number of draws a task actually gets is NOT ``cfg.samples`` alone: it
also depends on ``sample_mode`` and, for judge/subjective tasks, on
``judge_mode`` and the task itself. Any code that needs "how many samples
does this task get" MUST call :func:`samples_for_task` rather than reading
``cfg.samples`` directly.
"""
from __future__ import annotations

from typing import Any

# Bump only when the *rule* below changes semantics (a new applicability
# condition, a different default), not when a caller passes different values.
SAMPLE_APPLICABILITY_POLICY_VERSION = 1

# Names the contract of the numeric multi-sample combiner (aggregate._avg_numeric
# / runner._avg_numeric): the per-metric arithmetic mean over the numeric
# samples, 2-dp rounded. Bump if that combiner's semantics change.
SAMPLE_COMBINATION = "arithmetic_mean_numeric_v1"


def samples_for_task(task: Any, cfg: Any, sample_mode: str, judge_mode: str = "single") -> int:
    """Number of draws this task gets under the given run policy.

    ``sample_mode == "all"``  -> every task gets ``cfg.samples``.
    otherwise                 -> judge/subjective tasks get ``cfg.samples``
                                 when judging is on; everything else gets 1.
    """
    requested = max(1, int(getattr(cfg, "samples", 1) or 1))
    if sample_mode == "all":
        return requested
    if judge_mode != "off" and (getattr(task, "scorer", None) == "subjective" or getattr(task, "judge", False)):
        return requested
    return 1
