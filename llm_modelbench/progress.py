"""Progress/status helpers for long operator-facing benchmark runs.

This module is deliberately small and dependency-free. The runner writes a compact
``runs/<run_id>/status.json`` after meaningful events. The watcher reads that file and
adds live hardware telemetry without touching the benchmark process.
"""
from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def seconds_hms(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "unknown"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def pct(done: int, total: int) -> float:
    return round((done / total * 100.0), 1) if total else 0.0


def bar(done: int, total: int, width: int = 52, fill: str = "#") -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    n = max(0, min(width, int(round(width * done / total))))
    return "[" + fill * n + "-" * (width - n) + "]"


def eta_confidence(models_done: int) -> str:
    if models_done < 5:
        return "low"
    if models_done < 10:
        return "medium"
    return "high"


def estimate_remaining(model_durations: List[float], models_remaining: int) -> Dict[str, Any]:
    """Rolling ETA using the last 5 completed model durations.

    Returns a deliberately honest range. Early runs should be treated as low confidence,
    especially when the first models are unusually large/small.
    """
    durations = [d for d in model_durations if d and d > 0]
    if not durations or models_remaining <= 0:
        return {"rolling_seconds": 0 if models_remaining <= 0 else None,
                "low_seconds": None, "high_seconds": None}
    recent = durations[-5:]
    avg = sum(recent) / len(recent)
    rolling = avg * models_remaining
    # Conservative early range. Narrows as more completed models are available.
    spread = 0.35 if len(durations) < 5 else 0.25 if len(durations) < 10 else 0.18
    return {
        "rolling_seconds": round(rolling, 1),
        "low_seconds": round(rolling * (1.0 - spread), 1),
        "high_seconds": round(rolling * (1.0 + spread), 1),
        "avg_model_seconds_last5": round(avg, 1),
    }


def classify_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a failure/warning/anomaly event for a result row, if any."""
    score = row.get("score")
    reason = str(row.get("reason") or "")
    task = row.get("task")
    model = row.get("model")
    event: Optional[Dict[str, Any]] = None
    if row.get("error_kind") in {"empty_output", "thinking_only", "harness_error"} or reason.startswith(("ERROR_EMPTY_OUTPUT", "ERROR_THINKING_ONLY")):
        event = {"kind": "ERROR", "model": model, "task": task, "score": score, "reason": reason}
    elif row.get("warning_kind") == "truncated" or reason.startswith("WARN_TRUNCATED"):
        event = {"kind": "WARN", "model": model, "task": task, "score": score, "reason": reason}
    elif score == 0:
        event = {"kind": "FAIL", "model": model, "task": task, "score": score, "reason": reason}
    elif isinstance(score, (int, float)) and score < 70:
        event = {"kind": "WEAK", "model": model, "task": task, "score": score, "reason": reason}
    # Catch contradictions such as score=100 but reason says 0/4 placed.
    low_detail = bool(re.search(r"(^|[\s,;])0/\d+", reason.lower())) or any(tok in reason.lower() for tok in ("no json", " miss", "failed"))
    high_score = isinstance(score, (int, float)) and score >= 99.9
    if high_score and low_detail:
        event = {"kind": "ANOMALY", "model": model, "task": task, "score": score, "reason": reason}
    return event


class StatusWriter:
    def __init__(self, out_dir: Path, *, run_id: str, level: str, samples: int,
                 model_plan: List[Dict[str, Any]], cfg: Any, gpu: Any,
                 skipped_models: Optional[List[Dict[str, Any]]] = None,
                 filters: Optional[List[str]] = None,
                 sample_mode: str = "smart"):
        self.out_dir = out_dir
        self.path = out_dir / "status.json"
        self.run_id = run_id
        self.level = level
        self.samples = samples
        self.sample_mode = sample_mode
        self.model_plan = model_plan
        self.skipped_models = skipped_models or []
        self.filters = filters or []
        self.cfg = cfg
        self.gpu = gpu
        self.started_monotonic = time.monotonic()
        self.started_at = utc_now()
        self.model_durations: List[float] = []
        self.current_model_started: Optional[float] = None
        self.current_task_started: Optional[float] = None
        self.failures: List[Dict[str, Any]] = []
        self.completed_models: List[Dict[str, Any]] = []
        self.rows_written = 0
        self.tasks_done = 0
        self.samples_done = 0
        self.current: Dict[str, Any] = {}
        self.last_result: Dict[str, Any] = {}
        self.model_rows: List[Dict[str, Any]] = []

    @property
    def models_total(self) -> int:
        return len(self.model_plan)

    @property
    def tasks_total(self) -> int:
        return sum(int(m.get("tasks_total", 0)) for m in self.model_plan)

    @property
    def samples_total(self) -> int:
        # V9.5.5 smart sampling lets deterministic tasks run once while
        # subjective/judged tasks still use the requested sample count.
        return sum(int(m.get("samples_total", int(m.get("tasks_total", 0)) * max(1, self.samples)))
                   for m in self.model_plan)

    def start_run(self) -> None:
        self._write(event="start")

    def start_model(self, index: int, model: str, cls: str, size_gb: float,
                    tasks_total: int, context_length: Optional[int], offload: Any) -> None:
        self.current_model_started = time.monotonic()
        self.model_rows = []
        self.current = {
            "model_index": index,
            "model": model,
            "class": cls,
            "size_gb": size_gb,
            "tasks_total": tasks_total,
            "task_index": 0,
            "task": None,
            "sample_index": None,
            "context_length": context_length,
            "offload_fraction": offload,
            "state": "model_started",
        }
        self._write(event="model_start")

    def start_task(self, task_index: int, task_id: str, samples_for_task: int = 1) -> None:
        self.current_task_started = time.monotonic()
        self.current.update({"task_index": task_index, "task": task_id,
                             "sample_index": 1, "samples_for_task": samples_for_task,
                             "state": "task_running"})
        self._write(event="task_start")

    def update_task_detail(self, **detail: Any) -> None:
        """Publish live intra-task progress, used by multi-depth needle probes.

        Status is evidence, not a dump of stale fields. Starting a new probe
        clears the previous tier's live speed/memory counters, while completed
        probes are retained in a compact history for the context-profile view.
        """
        event = str(detail.get("probe_event") or "task_progress")
        if event in {"needle_calibrating", "needle_calibrated", "needle_probe_planning", "needle_probe_running"}:
            for key in (
                "probe_tps", "probe_prompt_tps", "probe_elapsed_seconds",
                "probe_vram_peak_mb", "probe_ram_delta_peak_mb",
                "probe_ollama_pss_delta_peak_mb", "probe_offload_fraction",
                "probe_reason",
            ):
                self.current.pop(key, None)
        self.current.update(detail)
        if event == "needle_probe_finished":
            history = list(self.current.get("probe_history") or [])
            history.append({
                "probe_index": detail.get("probe_index"),
                "probe_total": detail.get("probe_total"),
                "probe_size": detail.get("probe_size"),
                "probe_num_ctx": detail.get("probe_num_ctx"),
                "probe_state": detail.get("probe_state"),
                "prompt_tps": detail.get("probe_prompt_tps"),
                "tps": detail.get("probe_tps"),
                "elapsed_seconds": detail.get("probe_elapsed_seconds"),
                "vram_peak_mb": detail.get("probe_vram_peak_mb"),
                "ram_delta_peak_mb": detail.get("probe_ram_delta_peak_mb"),
                "ollama_pss_delta_peak_mb": detail.get("probe_ollama_pss_delta_peak_mb"),
                "offload_fraction": detail.get("probe_offload_fraction"),
            })
            self.current["probe_history"] = history[-8:]
        self._write(event=event)

    def finish_task(self, row: Dict[str, Any], samples_used: Optional[int] = None) -> None:
        self.rows_written += 1
        self.tasks_done += 1
        self.samples_done += max(1, int(samples_used if samples_used is not None else row.get("samples_used", self.samples)))
        self.last_result = dict(row)
        self.model_rows.append(dict(row))
        event = classify_row(row)
        if event:
            event["at"] = utc_now()
            self.failures.append(event)
            self.failures = self.failures[-20:]
        self.current.update({"state": "task_finished"})
        self._write(event="task_finish")

    def finish_model(self, model: str) -> None:
        dur = 0.0
        if self.current_model_started is not None:
            dur = time.monotonic() - self.current_model_started
            self.model_durations.append(dur)
        scores = [r.get("score") for r in self.model_rows if isinstance(r.get("score"), (int, float))]
        tps = [r.get("tps") for r in self.model_rows if isinstance(r.get("tps"), (int, float)) and r.get("tps")]
        model_summary = {
            "model": model,
            "duration_seconds": round(dur, 1),
            "quality_avg": round(sum(scores) / len(scores), 2) if scores else None,
            "tps_avg": round(sum(tps) / len(tps), 1) if tps else None,
            "failures": sum(1 for r in self.model_rows if r.get("score") == 0),
            "weak": sum(1 for r in self.model_rows if isinstance(r.get("score"), (int, float)) and 0 < r.get("score") < 70),
        }
        self.completed_models.append(model_summary)
        self.current.update({"state": "model_finished"})
        self._write(event="model_finish")

    def _write(self, *, event: str) -> None:
        elapsed = time.monotonic() - self.started_monotonic
        models_done = len(self.completed_models)
        remaining = max(0, self.models_total - models_done)
        eta = estimate_remaining(self.model_durations, remaining)
        best_quality = None
        fastest = None
        completed = [m for m in self.completed_models if m.get("quality_avg") is not None]
        if completed:
            best_quality = max(completed, key=lambda m: m.get("quality_avg") or -1)
            useful = [m for m in completed if (m.get("quality_avg") or 0) >= 70 and m.get("tps_avg")]
            if useful:
                fastest = max(useful, key=lambda m: m.get("tps_avg") or 0)
        payload = {
            "schema": "llm-modelbench.status.v1.1",
            "version": __version__,
            "status_type": ("context_profile" if bool(getattr(self.cfg, "context_profile_mode", False)) else "benchmark"),
            "event": event,
            "run_id": self.run_id,
            "level": self.level,
            "samples": self.samples,
            "sample_mode": self.sample_mode,
            "started_at": self.started_at,
            "updated_at": utc_now(),
            "elapsed_seconds": round(elapsed, 1),
            "models_total": self.models_total,
            "models_done": models_done,
            "models_skipped": len(self.skipped_models),
            "skipped_models": self.skipped_models[:200],
            "filters": self.filters,
            "tasks_total": self.tasks_total,
            "tasks_done": self.tasks_done,
            "samples_total": self.samples_total,
            "samples_done": self.samples_done,
            "rows_written": self.rows_written,
            "current": self.current,
            "last_result": self.last_result,
            "eta": eta,
            "eta_confidence": eta_confidence(models_done),
            "failed_tasks": self.failures[-12:],
            "completed_models": self.completed_models[-10:],
            "highlights": {"best_quality": best_quality, "fastest_useful": fastest},
            "hardware_config": {
                "gpu_vendor": getattr(self.gpu, "vendor", "none"),
                "gpu_name": getattr(self.gpu, "name", "unknown"),
                "vram_budget_gb": getattr(self.cfg, "vram_budget_gb", None),
                "gpu_pause_temp_c": getattr(self.cfg, "temp_pause_c", None),
                "gpu_resume_temp_c": getattr(self.cfg, "temp_resume_c", None),
                "gpu_driver": getattr(self.gpu, "driver_version", None),
                "ctx_override": getattr(self.cfg, "ctx_override", None),
                "num_predict": getattr(self.cfg, "num_predict_override", None),
                "think": getattr(self.cfg, "think", "auto"),
                "context_profile_mode": bool(getattr(self.cfg, "context_profile_mode", False)),
                "context_profile_target_ctx": getattr(self.cfg, "long_context_target_ctx", None),
                "needle_preflight_mode": getattr(self.cfg, "needle_preflight_mode", "enforce"),
            },
            "profile_phase": (self.current.get("profile_phase") if bool(getattr(self.cfg, "context_profile_mode", False)) else None),
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)
