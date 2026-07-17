"""Benchmark runner. One model resident at a time, always.

Per model: flush VRAM, warm up, probe the real offload fraction, then run its task battery
with per-task telemetry. Coding tasks may use a ReAct retry loop. Each test can be sampled N
times with the median taken for stability. Results append to a JSONL so an interrupted run
resumes exactly where it stopped. A short fixed probe set is run once per model for clone
detection unless the plan is too small for safe fingerprinting. Nothing keeps two models loaded
at once.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import scoring, judge as judge_mod, media, fingerprint, progress, sandbox, __version__
from .filters import (
    describe_filters,
    filter_models,
    filter_tasks,
    validate_task_ids,
)
from .classify import classify_model, size_gb
from .config import Config
from .hardware import Telemetry, ProbeTelemetry, detect_gpu, nvidia_live, host_memory_snapshot
from .tasks import Task, tasks_for, make_needle_prompt, TASKS
from .inline_ui import InlineUI


MODEL_ERROR_KINDS = {"empty_output", "thinking_only"}


def _reason_public(reason: str) -> str:
    """Sanitise scorer details before feeding an agentic retry prompt."""
    reason = str(reason or "")
    reason = re.sub(r",\s*missing:.*", "", reason)
    reason = re.sub(r",\s*expected:.*", "", reason)
    reason = re.sub(r",\s*found:.*", "", reason)
    return reason[:120] or "scorer reported a failing answer"


def _score_task(client, cfg: Config, task: Task, output: str, model: Optional[str] = None) -> tuple:
    if task.scorer in scoring.DETERMINISTIC:
        fn = scoring.DETERMINISTIC[task.scorer]
        score, reason = fn(output, task.meta)
        return score, reason, _reason_public(reason)
    if task.scorer == "retrieval":
        embed_model = model or cfg.embed_model
        score, reason = scoring.score_retrieval(lambda ts: client.embed(embed_model, ts), task.meta)
        reason = f"{reason}; embed_model={embed_model}"
        return score, reason, _reason_public(reason)
    return None, "unknown scorer", "unknown scorer"


def _ctx(cfg: Config) -> Optional[int]:
    v = getattr(cfg, "ctx_override", None)
    try:
        return int(v) if v else None
    except Exception:
        return None


def _num_predict(cfg: Config, task: Task, default: Optional[int] = None) -> int:
    v = getattr(cfg, "num_predict_override", None)
    if v:
        return int(v)
    return int(default if default is not None else task.num_predict)


def _think(cfg: Config) -> str:
    v = str(getattr(cfg, "think", "auto") or "auto").lower()
    return v if v in {"auto", "on", "off"} else "auto"


def _has_numeric_score(value: Any) -> bool:
    return isinstance(value, (int, float))


def _score_cell(value: Any) -> Optional[float]:
    return round(float(value), 2) if _has_numeric_score(value) else None


def _task_hash(task: Task) -> str:
    """Stable task identity used by --resume.

    Prompt, scorer, meta, difficulty, and agentic settings are included so stale rows
    from an older prompt/scorer cannot be reused after an integrity patch.
    """
    payload = {
        "id": task.id,
        "category": task.category,
        "family": task.family,
        "scorer": task.scorer,
        "prompt": task.prompt,
        "meta": task.meta,
        "difficulty": task.difficulty,
        "num_predict": task.num_predict,
        "agentic": task.agentic,
        "judge": task.judge,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _chat(client, cfg: Config, model: str, prompt: str, *, task: Task,
          images: Optional[List[str]] = None, system: Optional[str] = None,
          num_predict: Optional[int] = None, num_ctx: Optional[int] = None,
          think: Optional[str] = None,
          messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return client.chat(
        model,
        prompt,
        images=images,
        system=system,
        messages=messages,
        num_predict=int(num_predict if num_predict is not None else _num_predict(cfg, task)),
        num_ctx=num_ctx if num_ctx is not None else _ctx(cfg),
        think=think if think is not None else _think(cfg),
    )


def _gen_fields(res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tps": res.get("tps"),
        "prompt_tps": res.get("prompt_tps"),
        "ttft_ms": res.get("ttft_ms"),
        "ttft_visible_ms": res.get("ttft_visible_ms"),
        "think_ms": res.get("think_ms"),
        "tokens": res.get("tokens"),
        "eval_count": res.get("eval_count", res.get("tokens")),
        "prompt_eval_count": res.get("prompt_eval_count"),
        "request_elapsed_seconds": res.get("request_elapsed_seconds", res.get("elapsed_seconds")),
        "server_total_duration_ms": res.get("server_total_duration_ms"),
        "server_load_duration_ms": res.get("server_load_duration_ms"),
        "server_prompt_eval_duration_ms": res.get("server_prompt_eval_duration_ms"),
        "server_eval_duration_ms": res.get("server_eval_duration_ms"),
        "done_reason": res.get("done_reason"),
        "itl_p50_ms": res.get("itl_p50_ms"),
        "itl_p95_ms": res.get("itl_p95_ms"),
        "num_predict": res.get("num_predict"),
        "num_ctx_used": res.get("num_ctx"),
        "thinking_chars": res.get("thinking_chars", len(res.get("thinking") or "")),
        "think_sent": res.get("think_sent"),
        "think_unsupported": res.get("think_unsupported"),
        "think_ineffective": bool(str(res.get("think_requested") or "").lower() == "off" and int(res.get("thinking_chars") or len(res.get("thinking") or "") or 0) > 0),
    }


def _harness_error(reason: str, response: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {"score": None, "reason": reason, "output": "", "output_chars": 0,
           "error_kind": "harness_error", "tps": None, "ttft_ms": None, "tokens": 0}
    if response:
        for key in ("http_status", "http_reason", "http_url", "http_error_body"):
            if response.get(key) is not None:
                out[key] = response.get(key)
        if response.get("http_error_body"):
            out["reason"] = f"{reason}; response_body={str(response.get('http_error_body'))[:1000]}"
    return out


def _model_output_error(res: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = res.get("text") or ""
    output_chars = len(text.strip())
    eval_count = int(res.get("eval_count") or res.get("tokens") or 0)
    thinking_chars = int(res.get("thinking_chars") or len(res.get("thinking") or "") or 0)
    base = _gen_fields(res)
    base["output"] = text
    base["output_chars"] = output_chars
    if eval_count == 0:
        base.update({"score": 0.0, "reason": "ERROR_EMPTY_OUTPUT: eval_count == 0; model produced no tokens", "error_kind": "empty_output"})
        return base
    if thinking_chars > 0 and output_chars == 0:
        base.update({"score": 0.0, "reason": "ERROR_THINKING_ONLY: reasoning tokens emitted but no visible answer", "error_kind": "thinking_only"})
        return base
    if output_chars == 0:
        base.update({"score": 0.0, "reason": "ERROR_EMPTY_OUTPUT: model returned zero visible characters", "error_kind": "empty_output"})
        return base
    return None




def _info_int(info: Dict[str, Any], suffixes: List[str]) -> Optional[int]:
    for suffix in suffixes:
        for key, value in (info or {}).items():
            k = str(key).lower()
            if k.endswith(suffix.lower()) or suffix.lower() in k:
                try:
                    return int(value)
                except Exception:
                    pass
    return None


def _kv_scalar_bytes() -> tuple[float, str]:
    """Bytes per KV scalar from Ollama's KV cache env, conservative by default.

    q4_0/q4_1 previously fell through to the f16 default (2 bytes/scalar) while
    still being labeled correctly in kv_estimate_source, silently overstating
    the real VRAM cost of a quantized cache and making the needle pre-flight
    check skip attempts that would likely have fit. Confirmed against a real
    run: OLLAMA_KV_CACHE_TYPE=q4_0 was set, the label read kv=f16, and 32k
    stayed skipped even though q8_0 already wasn't the blocker.
    """
    typ = str(os.environ.get("OLLAMA_KV_CACHE_TYPE") or "f16").strip().lower()
    if typ in {"q8_0", "q8"}:
        return 1.0, typ
    if typ in {"q4_0", "q4_1", "q4"}:
        return 0.5, typ
    if typ in {"f32", "float32"}:
        return 4.0, typ
    return 2.0, typ or "f16"


def _kv_bytes_per_token(client, model: str) -> tuple[Optional[int], str]:
    """Estimate KV-cache bytes per token from Ollama model_info when available.

    GGUF metadata may omit attention.key_length/value_length for llama-style models.
    When possible, derive them from embedding_length // head_count and record that source.
    """
    if not hasattr(client, "model_info"):
        return None, "model_info_unavailable"
    info = client.model_info(model) or {}
    layers = _info_int(info, ["block_count"])
    head_count = _info_int(info, ["attention.head_count", "head_count"])
    kv_heads = _info_int(info, ["attention.head_count_kv", "head_count_kv"]) or head_count
    key_len = _info_int(info, ["attention.key_length", "key_length"])
    value_len = _info_int(info, ["attention.value_length", "value_length"]) or key_len
    source = "metadata"
    if not key_len:
        emb = _info_int(info, ["embedding_length", "n_embd"])
        if emb and head_count:
            key_len = max(1, int(emb // head_count))
            source = "derived_embedding_per_head"
    if not value_len and key_len:
        value_len = key_len
        if source == "metadata":
            source = "derived_value_equals_key"
    if not (layers and kv_heads and key_len and value_len):
        return None, "kv_estimate_unavailable"
    scalar_bytes, kv_type = _kv_scalar_bytes()
    return int(layers * kv_heads * (key_len + value_len) * scalar_bytes), f"{source};kv={kv_type}"


def _needle_kv_estimate(
    client,
    cfg: Config,
    model: str,
    wanted_ctx: int,
    measured: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Estimate memory for a needle depth without confusing VRAM with total memory.

    A measured estimate is accepted only when it carries an explicit anchor and
    ``valid_for_skip``.  Once dynamic offload begins, GPU-VRAM slope alone is
    not a total-memory slope and therefore cannot justify pre-emptively skipping
    a larger context.  Metadata remains available as a low-confidence reference.
    """
    budget_gb = float(getattr(cfg, "vram_budget_gb", 0.0) or 0.0)
    model_size = None
    if hasattr(client, "model_size_bytes"):
        try:
            model_size = client.model_size_bytes(model)
        except Exception:
            model_size = None

    metadata_bpt, metadata_source = _kv_bytes_per_token(client, model)
    metadata_total = (metadata_bpt * int(wanted_ctx) + int(model_size or 0)) if metadata_bpt else None
    metadata_total_gb = round(metadata_total / (1024 ** 3), 3) if metadata_total is not None else None
    metadata_kv_gb = round((metadata_bpt * int(wanted_ctx)) / (1024 ** 3), 3) if metadata_bpt else None

    out: Dict[str, Any] = {
        "wanted_num_ctx": int(wanted_ctx),
        "vram_budget_gb": budget_gb if budget_gb > 0 else None,
        "metadata_estimated_kv_gb": metadata_kv_gb,
        "metadata_estimated_total_gb": metadata_total_gb,
        "metadata_kv_bytes_per_token": metadata_bpt,
        "metadata_kv_estimate_source": metadata_source,
    }

    use_measured = bool(measured and measured.get("valid_for_skip") and measured.get("bytes_per_token"))
    if use_measured:
        bpt = int(measured["bytes_per_token"])
        anchor_ctx = int(measured.get("anchor_ctx") or 0)
        delta_ctx = max(0, int(wanted_ctx) - anchor_ctx)
        anchor_total_mb = float(measured.get("anchor_total_mb") or 0.0)
        anchor_gpu_mb = float(measured.get("anchor_gpu_mb") or 0.0)
        anchor_host_mb = float(measured.get("anchor_host_mb") or 0.0)
        gpu_bpt = measured.get("gpu_bytes_per_token")
        host_bpt = measured.get("host_bytes_per_token")

        projected_total_mb = anchor_total_mb + delta_ctx * bpt / (1024 ** 2)
        observed_floor_mb = float(measured.get("observed_floor_mb") or anchor_total_mb or 0.0)
        projected_total_mb = max(projected_total_mb, anchor_total_mb, observed_floor_mb)

        projected_gpu_mb = None
        if isinstance(gpu_bpt, (int, float)):
            projected_gpu_mb = anchor_gpu_mb + delta_ctx * max(0.0, float(gpu_bpt)) / (1024 ** 2)
            projected_gpu_mb = max(
                projected_gpu_mb,
                anchor_gpu_mb,
                float(measured.get("observed_gpu_floor_mb") or anchor_gpu_mb or 0.0),
            )
        projected_host_mb = None
        if isinstance(host_bpt, (int, float)):
            projected_host_mb = anchor_host_mb + delta_ctx * max(0.0, float(host_bpt)) / (1024 ** 2)
            projected_host_mb = max(projected_host_mb, anchor_host_mb)

        # Components are diagnostic. The monotonic total projection is the
        # authoritative budget comparison and can never be lower than either
        # observed/projected component.
        if projected_gpu_mb is not None:
            projected_total_mb = max(projected_total_mb, projected_gpu_mb)
        if projected_host_mb is not None:
            projected_total_mb = max(projected_total_mb, anchor_gpu_mb + projected_host_mb)

        estimated_total_gb = round(projected_total_mb / 1024.0, 3)
        out.update({
            "estimated_kv_gb": None,
            "estimated_total_gb": estimated_total_gb,
            "estimated_gpu_peak_gb": (round(projected_gpu_mb / 1024.0, 3)
                                      if projected_gpu_mb is not None else None),
            "estimated_host_increment_gb": (round(projected_host_mb / 1024.0, 3)
                                            if projected_host_mb is not None else None),
            "kv_bytes_per_token": bpt,
            "kv_gpu_bytes_per_token": gpu_bpt,
            "kv_host_bytes_per_token": host_bpt,
            "kv_estimate_source": measured.get("source") or "measured_memory_slope",
            "kv_estimate_method": measured.get("method") or "measured_total_resident_slope",
            "kv_estimate_confidence": measured.get("confidence") or "medium",
            "kv_estimate_valid_for_skip": True,
            "kv_estimate_host_memory_signal": measured.get("host_memory_signal"),
            "kv_estimate_anchor_ctx": anchor_ctx or None,
            "kv_estimate_anchor_total_mb": round(anchor_total_mb, 1) if anchor_total_mb else None,
            "kv_estimate_anchor_gpu_mb": round(anchor_gpu_mb, 1) if anchor_gpu_mb else None,
            "kv_estimate_anchor_host_mb": round(anchor_host_mb, 1) if anchor_host_mb else None,
            "kv_estimate_warning": measured.get("warning"),
        })
    else:
        out.update({
            "estimated_kv_gb": metadata_kv_gb,
            "estimated_total_gb": metadata_total_gb,
            "estimated_gpu_peak_gb": None,
            "estimated_host_increment_gb": None,
            "kv_bytes_per_token": metadata_bpt,
            "kv_gpu_bytes_per_token": None,
            "kv_host_bytes_per_token": None,
            "kv_estimate_source": metadata_source,
            "kv_estimate_method": "metadata" if metadata_bpt else "unavailable",
            "kv_estimate_confidence": "low" if metadata_bpt else "unavailable",
            "kv_estimate_valid_for_skip": bool(metadata_bpt),
            "kv_estimate_warning": (measured or {}).get("warning") if measured else None,
        })
        # After offload has begun, metadata describes total theoretical
        # residency but cannot predict how Ollama will split it across VRAM
        # and RAM.  It is informative, but not safe evidence for a hard skip.
        if measured and measured.get("dynamic_offload_observed"):
            out["kv_estimate_valid_for_skip"] = False
            out["kv_estimate_confidence"] = "invalid_for_skip_dynamic_offload"
            out["kv_estimate_warning"] = measured.get("warning") or (
                "dynamic offload changed between successful probes; metadata estimate retained for reference only"
            )

    if out.get("estimated_total_gb") is None or budget_gb <= 0:
        out["kv_exceeds_budget"] = None
    else:
        out["kv_exceeds_budget"] = bool(float(out["estimated_total_gb"]) > budget_gb)
    return out

def _needle_environment_skip(kv: Dict[str, Any], wanted_ctx: int, safe_floor: int = 32768) -> Optional[Dict[str, Any]]:
    """Return a pre-flight skip only when the estimate is valid for that decision.

    Dynamic offload can flatten GPU VRAM while moving memory into host RAM.  In
    that state the estimate remains useful diagnostic context, but is not
    sufficient evidence to call a larger depth impossible.
    """
    valid_for_skip = kv.get("kv_estimate_valid_for_skip")
    if kv.get("kv_exceeds_budget") and valid_for_skip is not False:
        return {"reason": "kv_cache_exceeds_vram_budget", **kv}
    if wanted_ctx > safe_floor and (kv.get("estimated_total_gb") is None or not kv.get("vram_budget_gb")):
        return {"reason": "kv_estimate_unavailable", **kv}
    return None


def _current_vram_used_mb() -> Optional[float]:
    try:
        snap = nvidia_live()
        v = snap.get("vram_used_mb")
        return float(v) if isinstance(v, (int, float)) else None
    except Exception:
        return None


def _normalise_probe_point(point: Any) -> Optional[Dict[str, Any]]:
    if isinstance(point, dict):
        ctx = point.get("num_ctx") or point.get("wanted_num_ctx") or point.get("ctx")
        vram = point.get("vram_peak_mb")
        if not isinstance(ctx, (int, float)) or not isinstance(vram, (int, float)):
            return None

        def number(key: str) -> Optional[float]:
            value = point.get(key)
            return float(value) if isinstance(value, (int, float)) else None

        return {
            "ctx": int(ctx),
            "vram_peak_mb": float(vram),
            "ram_delta_peak_mb": number("ram_delta_peak_mb"),
            "ollama_rss_delta_peak_mb": number("ollama_rss_delta_peak_mb"),
            "ollama_pss_delta_peak_mb": number("ollama_pss_delta_peak_mb"),
            "model_host_bytes": number("model_host_bytes"),
            "offload_fraction": number("offload_fraction"),
        }
    if isinstance(point, (tuple, list)) and len(point) >= 2:
        try:
            return {
                "ctx": int(point[0]), "vram_peak_mb": float(point[1]),
                "ram_delta_peak_mb": None, "ollama_rss_delta_peak_mb": None,
                "ollama_pss_delta_peak_mb": None, "model_host_bytes": None,
                "offload_fraction": 0.0,
            }
        except Exception:
            return None
    return None


def _measured_memory_estimate(points: List[Any], *, offload_tolerance: float = 0.02) -> Optional[Dict[str, Any]]:
    """Build a conservative memory projection from the last two successful probes.

    Only signals that describe where the loaded model actually resides may
    justify a hard pre-flight skip. Process RSS/PSS deltas are valuable
    telemetry, but they include mmap/page-cache/driver effects and can double
    count memory already represented by ``size_vram``. They therefore remain
    diagnostic-only and can never block a larger context tier.

    GPU-only slope is accepted only while both probes are effectively fully
    resident on GPU. Once offload starts or changes, the slope is invalid for
    deciding that a later tier cannot fit; the real controlled probe is the
    stronger evidence.
    """
    pts = [p for p in (_normalise_probe_point(item) for item in points) if p]
    if len(pts) < 2:
        return None
    a, b = pts[-2], pts[-1]
    if b["ctx"] <= a["ctx"]:
        return None

    ctx_delta = b["ctx"] - a["ctx"]
    off_a = a.get("offload_fraction")
    off_b = b.get("offload_fraction")
    offload_changed = (
        isinstance(off_a, (int, float)) and isinstance(off_b, (int, float))
        and abs(float(off_b) - float(off_a)) > offload_tolerance
    )
    dynamic = offload_changed or bool((off_a or 0) > offload_tolerance or (off_b or 0) > offload_tolerance)
    fully_gpu = all(
        isinstance(v, (int, float)) and float(v) <= offload_tolerance
        for v in (off_a, off_b)
    )

    # Ollama /api/ps host-resident bytes are more specific than process RSS/PSS.
    # Even this signal is not projected across a changing offload split, but it
    # provides a defensible observed total-residency floor.
    host_bytes_a = a.get("model_host_bytes")
    host_bytes_b = b.get("model_host_bytes")
    if isinstance(host_bytes_a, (int, float)) and isinstance(host_bytes_b, (int, float)):
        host_a = max(0.0, float(host_bytes_a) / (1024 ** 2))
        host_b = max(0.0, float(host_bytes_b) / (1024 ** 2))
        total_a = float(a["vram_peak_mb"]) + host_a
        total_b = float(b["vram_peak_mb"]) + host_b
        if total_b > total_a:
            total_bpt = max(1, int(((total_b - total_a) * (1024 ** 2)) / ctx_delta))
            gpu_bpt = max(0, int(((float(b["vram_peak_mb"]) - float(a["vram_peak_mb"])) * (1024 ** 2)) / ctx_delta))
            host_bpt = max(0, int(((host_b - host_a) * (1024 ** 2)) / ctx_delta))
            return {
                "bytes_per_token": total_bpt,
                "gpu_bytes_per_token": gpu_bpt,
                "host_bytes_per_token": host_bpt,
                "anchor_ctx": b["ctx"],
                "anchor_gpu_mb": float(b["vram_peak_mb"]),
                "anchor_host_mb": host_b,
                "anchor_total_mb": total_b,
                "observed_gpu_floor_mb": max(float(a["vram_peak_mb"]), float(b["vram_peak_mb"])),
                "observed_floor_mb": max(total_a, total_b),
                "method": "measured_api_ps_resident_slope",
                "source": "gpu_peak_plus_ollama_api_ps_host_bytes",
                "host_memory_signal": "ollama_api_ps_host_bytes",
                "confidence": "medium" if not dynamic else "diagnostic_dynamic_offload",
                "valid_for_skip": not dynamic,
                "dynamic_offload_observed": dynamic,
                "warning": (
                    "dynamic offload changed between probes; observed residency is retained, but projection cannot hard-skip later tiers"
                    if dynamic else None
                ),
            }

    # Before offload starts, a VRAM slope is the only measured signal that does
    # not double-count the loaded model. It is allowed to guide the next tier.
    if fully_gpu and b["vram_peak_mb"] > a["vram_peak_mb"]:
        bpt = max(1, int(((b["vram_peak_mb"] - a["vram_peak_mb"]) * (1024 ** 2)) / ctx_delta))
        return {
            "bytes_per_token": bpt,
            "gpu_bytes_per_token": bpt,
            "host_bytes_per_token": 0,
            "anchor_ctx": b["ctx"],
            "anchor_gpu_mb": b["vram_peak_mb"],
            "anchor_host_mb": 0.0,
            "anchor_total_mb": b["vram_peak_mb"],
            "observed_gpu_floor_mb": max(a["vram_peak_mb"], b["vram_peak_mb"]),
            "observed_floor_mb": max(a["vram_peak_mb"], b["vram_peak_mb"]),
            "method": "measured_vram_slope_no_offload",
            "source": "measured_vram_slope_no_offload",
            "host_memory_signal": "none",
            "confidence": "medium",
            "valid_for_skip": True,
            "dynamic_offload_observed": False,
            "warning": None,
        }

    # PSS/RSS/system RAM remain useful card evidence only. They are deliberately
    # not used as authoritative total-residency slopes because they can include
    # model mappings, page cache and GPU-driver allocations already represented
    # by VRAM telemetry.
    for key, source in (
        ("ollama_pss_delta_peak_mb", "ollama_process_pss_delta"),
        ("ollama_rss_delta_peak_mb", "ollama_process_rss_delta"),
        ("ram_delta_peak_mb", "system_ram_delta"),
    ):
        if isinstance(a.get(key), (int, float)) and isinstance(b.get(key), (int, float)):
            host_a = max(0.0, float(a[key]))
            host_b = max(0.0, float(b[key]))
            total_a = float(a["vram_peak_mb"]) + host_a
            total_b = float(b["vram_peak_mb"]) + host_b
            if total_b > total_a:
                total_bpt = max(1, int(((total_b - total_a) * (1024 ** 2)) / ctx_delta))
                return {
                    "bytes_per_token": total_bpt,
                    "gpu_bytes_per_token": None,
                    "host_bytes_per_token": None,
                    "anchor_ctx": b["ctx"],
                    "anchor_gpu_mb": float(b["vram_peak_mb"]),
                    "anchor_host_mb": host_b,
                    "anchor_total_mb": total_b,
                    "observed_gpu_floor_mb": max(float(a["vram_peak_mb"]), float(b["vram_peak_mb"])),
                    "observed_floor_mb": max(total_a, total_b),
                    "method": "diagnostic_process_or_system_resident_slope",
                    "source": source,
                    "host_memory_signal": source,
                    "confidence": "diagnostic_only_not_de_duplicated",
                    "valid_for_skip": False,
                    "dynamic_offload_observed": dynamic,
                    "warning": "process/system memory may double-count mapped or driver-backed pages; never used for hard context skips",
                }

    return {
        "bytes_per_token": None,
        "gpu_bytes_per_token": None,
        "host_bytes_per_token": None,
        "anchor_ctx": b["ctx"],
        "anchor_gpu_mb": b["vram_peak_mb"],
        "anchor_host_mb": None,
        "anchor_total_mb": b["vram_peak_mb"],
        "observed_gpu_floor_mb": b["vram_peak_mb"],
        "observed_floor_mb": b["vram_peak_mb"],
        "method": "invalid_dynamic_offload",
        "source": "measured_vram_slope_rejected",
        "host_memory_signal": "unavailable",
        "confidence": "invalid_for_skip_dynamic_offload",
        "valid_for_skip": False,
        "dynamic_offload_observed": dynamic,
        "warning": "GPU VRAM slope rejected because offload changed and no de-duplicated host-residency signal was available",
    }

def _measured_kv_slope(points: List[Any]) -> Optional[int]:
    """Backward-compatible scalar view used by older tests/callers."""
    estimate = _measured_memory_estimate(points)
    if not estimate or not estimate.get("valid_for_skip"):
        return None
    value = estimate.get("bytes_per_token")
    return int(value) if isinstance(value, (int, float)) else None

def _max_verified_prefix(attempted: List[Dict[str, Any]]) -> Tuple[Optional[int], bool]:
    max_ctx: Optional[int] = None
    broken = False
    non_mono = False
    for item in sorted(attempted, key=lambda x: int(x.get("size") or 0)):
        found = bool(item.get("found"))
        if found and broken:
            non_mono = True
        if broken:
            continue
        if found:
            ctx = item.get("prompt_tokens_actual") or item.get("prompt_tokens_estimated") or item.get("num_ctx")
            if isinstance(ctx, (int, float)):
                max_ctx = int(ctx)
        else:
            broken = True
    return max_ctx, non_mono


def _needle_prompt_from_cpt(size: int, token: str, chars_per_token: float) -> str:
    return make_needle_prompt(size, token, chars_per_token=chars_per_token)


def _calibrate_needle_cpt(client, cfg: Config, model: str, task: Task, context_max: int) -> tuple[float, str]:
    """Calibrate chars/token for the needle filler using Ollama prompt_eval_count.

    This avoids the old fixed 4 chars/token assumption. If the live tokenizer probe fails,
    use the measured audit fallback for this filler and mark the source so downstream rows
    know the value was not tokenizer-derived.
    """
    sample = make_needle_prompt(2000, task.meta["needle_token"], chars_per_token=6.85)
    try:
        res = _chat(client, cfg, model, sample, task=task, num_predict=1,
                    num_ctx=min(max(4096, len(sample) // 3 + 512), int(context_max or 4096)), think="off")
        pe = int(res.get("prompt_eval_count") or 0)
        if pe > 0:
            return max(1.0, len(sample) / pe), "ollama_prompt_eval_count"
    except Exception:
        pass
    return 6.85, "fallback_audit_filler_cpt"

def _needle_response_diagnostics(text: str, needle_token: str) -> Dict[str, Any]:
    raw = str(text or "")
    stripped = raw.strip()
    token = str(needle_token or "")
    token_re = re.compile(re.escape(token), re.I) if token else None
    occurrences = len(token_re.findall(stripped)) if token_re else 0
    without_token = token_re.sub("", stripped) if token_re else stripped
    non_space_extra = len(re.sub(r"\s+", "", without_token))
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    repeated_line_ratio = 0.0
    if lines:
        counts: Dict[str, int] = {}
        for line in lines:
            key = line.lower()
            counts[key] = counts.get(key, 0) + 1
        repeated_line_ratio = round(max(counts.values()) / len(lines), 3)
    exact = bool(token and stripped.lower() == token.lower())
    return {
        "needle_response_exact": exact,
        "needle_token_occurrences": occurrences,
        "needle_response_extraneous_chars": non_space_extra,
        "needle_response_line_count": len(lines),
        "needle_response_repeated_line_ratio": repeated_line_ratio,
        "needle_response_sha256": hashlib.sha256(raw.encode()).hexdigest() if raw else None,
        "needle_response_suspect": bool(occurrences != 1 or non_space_extra > 24 or repeated_line_ratio > 0.8 and len(lines) > 2),
    }


def _needle_profile_summary(
    attempted: List[Dict[str, Any]], *, target_ctx: int = 64000,
    min_usable_tps: float = 10.0, critical_tps: float = 3.0,
) -> Dict[str, Any]:
    """Summarise long-context operating evidence without changing quality scores.

    The readiness status is diagnostic only.  It separates recall correctness
    from practical operation: decode speed, prefill speed, offload, RAM/swap,
    and suspicious output shape are all surfaced for model cards.
    """
    ok = [p for p in attempted if p.get("found")]
    if not ok:
        return {
            "needle_successful_depths": [],
            "needle_max_requested_size": None,
            "needle_min_tps": None,
            "needle_max_offload_fraction": None,
            "needle_max_ram_delta_mb": None,
            "needle_behavior_suspect": False,
            "needle_target_ctx": int(target_ctx),
            "needle_target_status": "not_verified",
            "needle_target_tps": None,
            "needle_target_prompt_tps": None,
            "needle_slow_depths": [],
            "needle_critical_slow_depths": [],
        }

    tps = [float(p["tps"]) for p in ok if isinstance(p.get("tps"), (int, float))]
    prompt_tps = [float(p["prompt_tps"]) for p in ok if isinstance(p.get("prompt_tps"), (int, float))]
    offload = [float(p["offload_fraction"]) for p in ok if isinstance(p.get("offload_fraction"), (int, float))]
    ram_delta = [float(p["ram_delta_peak_mb"]) for p in ok if isinstance(p.get("ram_delta_peak_mb"), (int, float))]
    rss_delta = [float(p["ollama_rss_delta_peak_mb"]) for p in ok if isinstance(p.get("ollama_rss_delta_peak_mb"), (int, float))]
    pss_delta = [float(p["ollama_pss_delta_peak_mb"]) for p in ok if isinstance(p.get("ollama_pss_delta_peak_mb"), (int, float))]
    swap_delta = [float(p["swap_delta_peak_mb"]) for p in ok if isinstance(p.get("swap_delta_peak_mb"), (int, float))]
    elapsed = [float(p["elapsed_seconds"]) for p in ok if isinstance(p.get("elapsed_seconds"), (int, float))]
    largest = max(ok, key=lambda p: int(p.get("size") or 0))

    target_candidates = [
        p for p in ok
        if int(p.get("prompt_tokens_actual") or p.get("num_ctx") or p.get("size") or 0) >= int(target_ctx)
    ]
    target_probe = min(
        target_candidates,
        key=lambda p: int(p.get("prompt_tokens_actual") or p.get("num_ctx") or p.get("size") or 0),
    ) if target_candidates else None

    target_status = "not_verified"
    if target_probe:
        target_decode_tps = target_probe.get("tps")
        if target_probe.get("needle_response_suspect"):
            target_status = "behavior_warning"
        elif not isinstance(target_decode_tps, (int, float)):
            target_status = "verified_speed_unavailable"
        elif float(target_decode_tps) < float(critical_tps):
            target_status = "impractical_speed"
        elif float(target_decode_tps) < float(min_usable_tps):
            target_status = "slow"
        else:
            target_status = "ready"

    slow_depths = [
        int(p.get("size") or 0) for p in ok
        if isinstance(p.get("tps"), (int, float)) and float(p["tps"]) < float(min_usable_tps)
    ]
    critical_slow_depths = [
        int(p.get("size") or 0) for p in ok
        if isinstance(p.get("tps"), (int, float)) and float(p["tps"]) < float(critical_tps)
    ]

    return {
        "needle_successful_depths": sorted(int(p.get("size") or 0) for p in ok),
        "needle_max_requested_size": int(largest.get("size") or 0) or None,
        "needle_max_depth_tps": largest.get("tps"),
        "needle_max_depth_prompt_tps": largest.get("prompt_tps"),
        "needle_max_depth_ttft_ms": largest.get("ttft_ms"),
        "needle_max_depth_elapsed_seconds": largest.get("elapsed_seconds"),
        "needle_min_tps": round(min(tps), 3) if tps else None,
        "needle_median_tps": round(statistics.median(tps), 3) if tps else None,
        "needle_min_prompt_tps": round(min(prompt_tps), 3) if prompt_tps else None,
        "needle_max_offload_fraction": round(max(offload), 3) if offload else None,
        "needle_max_ram_delta_mb": round(max(ram_delta), 1) if ram_delta else None,
        "needle_max_ollama_rss_delta_mb": round(max(rss_delta), 1) if rss_delta else None,
        "needle_max_ollama_pss_delta_mb": round(max(pss_delta), 1) if pss_delta else None,
        "needle_max_swap_delta_mb": round(max(swap_delta), 1) if swap_delta else None,
        "needle_total_probe_seconds": round(sum(elapsed), 3) if elapsed else None,
        "needle_behavior_suspect": any(bool(p.get("needle_response_suspect")) for p in ok),
        "needle_target_ctx": int(target_ctx),
        "needle_target_min_tps": float(min_usable_tps),
        "needle_target_critical_tps": float(critical_tps),
        "needle_target_status": target_status,
        "needle_target_size": int(target_probe.get("size") or 0) if target_probe else None,
        "needle_target_num_ctx": (target_probe.get("num_ctx") if target_probe else None),
        "needle_target_tps": (target_probe.get("tps") if target_probe else None),
        "needle_target_prompt_tps": (target_probe.get("prompt_tps") if target_probe else None),
        "needle_target_elapsed_seconds": (target_probe.get("elapsed_seconds") if target_probe else None),
        "needle_target_offload_fraction": (target_probe.get("offload_fraction") if target_probe else None),
        "needle_target_ram_delta_mb": (target_probe.get("ram_delta_peak_mb") if target_probe else None),
        "needle_target_ollama_rss_delta_mb": (target_probe.get("ollama_rss_delta_peak_mb") if target_probe else None),
        "needle_target_ollama_pss_delta_mb": (target_probe.get("ollama_pss_delta_peak_mb") if target_probe else None),
        "needle_target_swap_delta_mb": (target_probe.get("swap_delta_peak_mb") if target_probe else None),
        "needle_target_behavior_suspect": bool(target_probe.get("needle_response_suspect")) if target_probe else None,
        "needle_slow_depths": sorted(slow_depths),
        "needle_critical_slow_depths": sorted(critical_slow_depths),
    }


def _native_tool_score(tool_calls: List[Dict[str, Any]], meta: Dict[str, Any]) -> tuple[float, str]:
    expected_tool = meta.get("expected_tool")
    expected_args = meta.get("expected_args") or {}
    for call in tool_calls or []:
        fn = call.get("function") or {}
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if fn.get("name") != expected_tool:
            continue
        matches = sum(1 for key, value in expected_args.items() if args.get(key) == value)
        if not expected_args:
            return 100.0, "native tool selected"
        score = round(100.0 * matches / len(expected_args), 2)
        return score, f"native tool selected; args={matches}/{len(expected_args)}"
    return 0.0, "native tool call missing or wrong tool"


def _fim_score(prefix: str, text: str, meta: Dict[str, Any]) -> tuple[float, str]:
    """Execute the completed prefix plus the held-out suffix assertion.

    Keyword presence is not sufficient evidence of suffix-aware completion.
    The insertion must form valid code and satisfy the unseen assertion.
    """
    insertion = str(text or "").strip()
    fenced = scoring.extract_blocks(insertion, "python", include_raw=False)
    if fenced:
        insertion = fenced[0].strip()
    suffix = str(meta.get("suffix") or "")
    if not insertion or not suffix:
        return 0.0, "fim missing insertion or held-out suffix"
    score, reason = sandbox.run_python_checks(str(prefix) + insertion + suffix, [""], timeout=10)
    return round(score, 2), "fim held-out suffix: " + reason


def _run_once(
    client, cfg: Config, model: str, task: Task,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run a single task once and return score + latency metrics."""
    # Vision image
    images = None
    if task.family == "vision":
        image_path = task.meta.get("image_path")
        if image_path:
            try:
                images = [media.load_image_file(image_path)["data"]]
            except (FileNotFoundError, ValueError) as exc:
                return _harness_error(f"vision image fixture error: {exc}")
        else:
            b64 = media.render_text_png(task.meta["reference"], task.meta.get("noisy", False), cfg.seed)
            if b64 is None:
                return _harness_error("vision skipped (no Pillow)")
            images = [b64]

    # Native structured tool call. The harness validates the proposed call but
    # never executes it.
    if task.scorer == "native_tool":
        if not hasattr(client, "chat_tools"):
            return _harness_error("native tool-call client method unavailable")
        res = client.chat_tools(model, task.prompt, tools=task.meta.get("tools") or [],
                                num_predict=_num_predict(cfg, task), num_ctx=_ctx(cfg), think=_think(cfg))
        if not res.get("ok"):
            return _harness_error(res.get("error", "native tool call failed"), res)
        calls = res.get("tool_calls") or []
        score, reason = _native_tool_score(calls, task.meta)
        output = json.dumps({"tool_calls": calls}, sort_keys=True, default=str)
        out = _gen_fields(res)
        out.update({"score": score, "reason": reason, "output": output,
                    "output_chars": len(output), "native_tool_calls": calls})
        return out

    # Fill-in-the-middle/suffix-conditioned completion.
    if task.scorer == "fim":
        if not hasattr(client, "generate_suffix"):
            return _harness_error("suffix/FIM client method unavailable")
        res = client.generate_suffix(model, task.prompt, suffix=str(task.meta.get("suffix") or ""),
                                     num_predict=_num_predict(cfg, task), num_ctx=_ctx(cfg))
        if not res.get("ok"):
            return _harness_error(res.get("error", "suffix generation failed"), res)
        output = str(res.get("text") or "")
        if not output.strip():
            base = _gen_fields(res)
            base.update({"score": 0.0, "reason": "ERROR_EMPTY_OUTPUT: FIM returned no insertion",
                         "output": output, "output_chars": 0, "error_kind": "empty_output"})
            return base
        score, reason = _fim_score(task.prompt, output, task.meta)
        out = _gen_fields(res)
        out.update({"score": score, "reason": reason, "output": output, "output_chars": len(output)})
        return out

    # Retrieval does not use chat.
    if task.scorer == "retrieval":
        score, reason, _ = _score_task(client, cfg, task, "", model)
        from .retrieval_diagnostics import diagnostics
        cases = diagnostics(lambda texts: client.embed(model, texts), task.meta, model)
        # A failed embed() call and a genuinely poor retrieval score both land
        # at 0.0 from score_retrieval, but they mean very different things: one
        # is a harness/API failure, the other is real evidence about the model.
        # Without this, both were indistinguishable in the data, which is
        # exactly the None-vs-0 conflation this project avoids everywhere else.
        error_kind = "harness_error" if "embed failed" in reason else None
        return {"score": round(score, 2), "reason": reason, "tps": None, "ttft_ms": None,
                "ttft_visible_ms": None, "tokens": None, "output_chars": None,
                "embed_model": model, "retrieval_cases": cases, "error_kind": error_kind}

    # Needle: probe explicit context depths without letting allocation heuristics become scores.
    if task.scorer == "needle":
        context_max = client.context_length(model) or 8000
        hits, reasons, attempted, skipped = [], [], [], []
        excluded_depths = 0
        probe_error_kinds: List[str] = []
        needle_num_predict = max(256, int(_num_predict(cfg, task, 256)))
        if progress_callback:
            progress_callback({
                "probe_event": "needle_calibrating",
                "probe_state": "calibrating tokenizer",
                "probe_index": 0,
                "probe_total": len(task.meta.get("context_sizes") or []),
            })
        cpt, cpt_source = _calibrate_needle_cpt(client, cfg, model, task, context_max)
        if progress_callback:
            progress_callback({
                "probe_event": "needle_calibrated",
                "probe_state": "calibration complete",
                "probe_index": 0,
                "probe_total": len(task.meta.get("context_sizes") or []),
                "needle_chars_per_token": round(float(cpt), 6),
                "needle_cpt_source": cpt_source,
            })
        measured_points: List[Dict[str, Any]] = []
        measured_estimate: Optional[Dict[str, Any]] = None
        context_sizes = list(task.meta["context_sizes"])
        for probe_index, size in enumerate(context_sizes, start=1):
            prompt = _needle_prompt_from_cpt(size, task.meta["needle_token"], cpt)
            # Estimate before issuing the probe. Successful attempts replace this with
            # Ollama's prompt_eval_count, which is the authoritative token count.
            prompt_tokens_est = max(1, int(math.ceil(len(prompt) / max(cpt, 1.0))))
            wanted_ctx = int(prompt_tokens_est + needle_num_predict + 64)
            base_probe = {
                "size": size,
                "prompt_tokens_estimated": prompt_tokens_est,
                "prompt_token_source": cpt_source,
                "needle_chars_per_token": round(float(cpt), 6),
                "wanted_num_ctx": wanted_ctx,
                "num_predict": needle_num_predict,
            }
            if progress_callback:
                progress_callback({
                    "probe_event": "needle_probe_planning",
                    "probe_index": probe_index, "probe_total": len(context_sizes),
                    "probe_size": int(size), "probe_num_ctx": int(wanted_ctx),
                    "probe_state": "planning",
                })
            if getattr(cfg, "needle_max_ctx", None) and wanted_ctx > int(cfg.needle_max_ctx or 0):
                skipped.append({**base_probe, "reason": "needle_max_ctx", "needle_max_ctx": int(cfg.needle_max_ctx or 0), "skip_class": "operator"})
                reasons.append(f"{size//1000}k:skip(needle_max_ctx={int(cfg.needle_max_ctx or 0)})")
                excluded_depths += 1
                if progress_callback:
                    progress_callback({"probe_event": "needle_probe_skipped", "probe_index": probe_index, "probe_total": len(context_sizes), "probe_size": int(size), "probe_num_ctx": int(wanted_ctx), "probe_state": "skipped", "probe_reason": "needle_max_ctx"})
                continue
            if _ctx(cfg) and int(_ctx(cfg) or 0) < wanted_ctx:
                skipped.append({**base_probe, "reason": "ctx_override_too_small", "ctx_override": _ctx(cfg), "skip_class": "operator"})
                reasons.append(f"{size//1000}k:skip(ctx={_ctx(cfg)})")
                excluded_depths += 1
                if progress_callback:
                    progress_callback({"probe_event": "needle_probe_skipped", "probe_index": probe_index, "probe_total": len(context_sizes), "probe_size": int(size), "probe_num_ctx": int(wanted_ctx), "probe_state": "skipped", "probe_reason": "ctx_override_too_small"})
                continue
            if wanted_ctx > context_max:
                skipped.append({**base_probe, "reason": "exceeds_context_length_max", "context_length_max": context_max, "skip_class": "model_capability"})
                hits.append(0.0)
                attempted.append({**base_probe, "found": False, "error_kind": "context_length_capability", "skip_class": "model_capability"})
                reasons.append(f"{size//1000}k:skip(max={context_max})")
                if progress_callback:
                    progress_callback({"probe_event": "needle_probe_skipped", "probe_index": probe_index, "probe_total": len(context_sizes), "probe_size": int(size), "probe_num_ctx": int(wanted_ctx), "probe_state": "skipped", "probe_reason": "exceeds_context_length_max"})
                continue
            kv = _needle_kv_estimate(client, cfg, model, wanted_ctx, measured_estimate)
            env_skip = _needle_environment_skip(kv, wanted_ctx)
            preflight_mode = str(getattr(cfg, "needle_preflight_mode", "enforce") or "enforce").lower()
            preflight_warning = None
            if env_skip and preflight_mode == "advisory":
                preflight_warning = {
                    "reason": env_skip.get("reason"),
                    "estimated_total_gb": env_skip.get("estimated_total_gb"),
                    "vram_budget_gb": env_skip.get("vram_budget_gb"),
                    "estimate_method": env_skip.get("kv_estimate_method"),
                    "estimate_confidence": env_skip.get("kv_estimate_confidence"),
                    "decision": "attempted_under_controlled_profile",
                }
                if progress_callback:
                    progress_callback({
                        "probe_event": "needle_probe_budget_advisory",
                        "probe_index": probe_index,
                        "probe_total": len(context_sizes),
                        "probe_size": int(size),
                        "probe_num_ctx": int(wanted_ctx),
                        "probe_state": "budget advisory; controlled attempt allowed",
                        "probe_reason": env_skip.get("reason"),
                    })
            elif env_skip:
                skipped.append({**base_probe, **env_skip, "skip_class": "environment"})
                reasons.append(f"{size//1000}k:skip({env_skip.get('reason')})")
                excluded_depths += 1
                if progress_callback:
                    progress_callback({"probe_event": "needle_probe_skipped", "probe_index": probe_index, "probe_total": len(context_sizes), "probe_size": int(size), "probe_num_ctx": int(wanted_ctx), "probe_state": "skipped", "probe_reason": env_skip.get("reason")})
                continue

            # A controlled profile may proceed past a conservative estimate, but
            # it never starts another tier if the host is already below the
            # operator's minimum available-RAM floor.
            ram_floor_gb = float(getattr(cfg, "needle_min_available_ram_gb", 2.0) or 0.0)
            host_mem = host_memory_snapshot()
            available_mb = host_mem.get("ram_available_mb")
            if (ram_floor_gb > 0 and isinstance(available_mb, (int, float))
                    and float(available_mb) < ram_floor_gb * 1024.0):
                safety = {
                    "reason": "host_ram_safety_floor",
                    "ram_available_mb": round(float(available_mb), 1),
                    "ram_floor_gb": ram_floor_gb,
                    "skip_class": "environment",
                }
                skipped.append({**base_probe, **kv, **safety})
                reasons.append(f"{size//1000}k:skip(host_ram_safety_floor)")
                excluded_depths += 1
                if progress_callback:
                    progress_callback({
                        "probe_event": "needle_probe_skipped",
                        "probe_index": probe_index,
                        "probe_total": len(context_sizes),
                        "probe_size": int(size),
                        "probe_num_ctx": int(wanted_ctx),
                        "probe_state": "skipped",
                        "probe_reason": "host_ram_safety_floor",
                    })
                continue
            if progress_callback:
                progress_callback({"probe_event": "needle_probe_running", "probe_index": probe_index, "probe_total": len(context_sizes), "probe_size": int(size), "probe_num_ctx": int(wanted_ctx), "probe_state": "running"})
            probe_tel = ProbeTelemetry(interval=0.25)
            probe_tel.start()
            try:
                res = _chat(client, cfg, model, prompt, task=task, num_predict=needle_num_predict, num_ctx=wanted_ctx, think="off")
            except Exception as exc:
                res = {"ok": False, "error": repr(exc)}
            finally:
                probe_hw = probe_tel.stop()
            actual_prompt_tokens = res.get("prompt_eval_count")
            if isinstance(actual_prompt_tokens, (int, float)) and actual_prompt_tokens:
                wanted_ctx = int(actual_prompt_tokens) + needle_num_predict + 64
                kv = _needle_kv_estimate(client, cfg, model, wanted_ctx, measured_estimate)
            err = _model_output_error(res) if res.get("ok") else _harness_error(res.get("error", "failed"), res)
            loaded_stats = client.loaded_model_stats(model) if hasattr(client, "loaded_model_stats") else None
            offload = (loaded_stats or {}).get("offload_fraction")
            if offload is None and hasattr(client, "offload_fraction"):
                offload = client.offload_fraction(model)
            vram_probe_mb = probe_hw.get("vram_peak_mb") or _current_vram_used_mb()
            output_text = str(res.get("text") or "")
            response_diag = _needle_response_diagnostics(output_text, task.meta["needle_token"])
            resident_total_peak_mb = None
            if isinstance(vram_probe_mb, (int, float)):
                resident_total_peak_mb = float(vram_probe_mb)
                if isinstance(probe_hw.get("ram_delta_peak_mb"), (int, float)):
                    resident_total_peak_mb += max(0.0, float(probe_hw["ram_delta_peak_mb"]))
            common = {**base_probe, **kv, **_gen_fields(res), **probe_hw, **response_diag,
                      "num_ctx": wanted_ctx,
                      "prompt_tokens_actual": actual_prompt_tokens,
                      "offload_fraction": offload,
                      "vram_peak_mb": vram_probe_mb,
                      "observed_total_resident_peak_mb": round(resident_total_peak_mb, 1) if resident_total_peak_mb is not None else None,
                      "model_loaded_size_bytes": (loaded_stats or {}).get("size_bytes"),
                      "model_vram_bytes": (loaded_stats or {}).get("size_vram_bytes"),
                      "model_host_bytes": (loaded_stats or {}).get("size_host_bytes"),
                      "model_offloaded_gb": (round(float((loaded_stats or {}).get("size_host_bytes")) / (1024 ** 3), 3)
                                             if isinstance((loaded_stats or {}).get("size_host_bytes"), (int, float)) else None),
                      "loaded_context_length": (loaded_stats or {}).get("context_length"),
                      "think_unsupported": res.get("think_unsupported"),
                      "think_sent": res.get("think_sent"),
                      "think_ineffective": bool(str(res.get("think_requested") or "").lower() == "off" and int(res.get("thinking_chars") or 0) > 0),
                      "thinking_chars": res.get("thinking_chars"),
                      "done_reason": res.get("done_reason"),
                      "preflight_budget_advisory": preflight_warning}
            if progress_callback:
                progress_callback({
                    "probe_event": "needle_probe_finished",
                    "probe_index": probe_index, "probe_total": len(context_sizes),
                    "probe_size": int(size), "probe_num_ctx": int(wanted_ctx),
                    "probe_state": "error" if err else "finished",
                    "probe_tps": res.get("tps"), "probe_prompt_tps": res.get("prompt_tps"),
                    "probe_elapsed_seconds": probe_hw.get("elapsed_seconds"),
                    "probe_vram_peak_mb": vram_probe_mb,
                    "probe_ram_delta_peak_mb": probe_hw.get("ram_delta_peak_mb"),
                    "probe_ollama_pss_delta_peak_mb": probe_hw.get("ollama_pss_delta_peak_mb"),
                    "probe_offload_fraction": offload,
                })
            if err:
                ek = err.get("error_kind") or "harness_error"
                probe_error_kinds.append(ek)
                # Previously only error_kind (a generic label) was kept here;
                # the actual exception detail (err["reason"], which threads
                # back to repr(exc) / HTTP status+body in ollama.py) was
                # computed but discarded before it ever reached
                # raw_results.jsonl. That's what made a real failure
                # undiagnosable without re-running with extra instrumentation.
                detail = {
                    "harness_error_detail": err.get("reason"),
                    "http_status": err.get("http_status"),
                    "http_reason": err.get("http_reason"),
                    "http_error_body": (err.get("http_error_body") or "")[:2000] or None,
                }
                if ek == "harness_error":
                    attempted.append({**common, **detail, "found": False, "error_kind": ek, "skip_class": "environment"})
                    reasons.append(f"{size//1000}k:error({ek},ctx={wanted_ctx})")
                    excluded_depths += 1
                else:
                    hits.append(0.0)
                    attempted.append({**common, **detail, "found": False, "error_kind": ek, "skip_class": "model_behavior"})
                    reasons.append(f"{size//1000}k:fail({ek},ctx={wanted_ctx})")
                continue
            found = task.meta["needle_token"] in output_text
            hits.append(100.0 if found else 0.0)
            attempted.append({**common, "found": bool(found)})
            if found and isinstance(vram_probe_mb, (int, float)):
                measured_points.append({
                    "num_ctx": int(wanted_ctx),
                    "vram_peak_mb": float(vram_probe_mb),
                    "ram_delta_peak_mb": probe_hw.get("ram_delta_peak_mb"),
                    "ollama_rss_delta_peak_mb": probe_hw.get("ollama_rss_delta_peak_mb"),
                    "ollama_pss_delta_peak_mb": probe_hw.get("ollama_pss_delta_peak_mb"),
                    "model_host_bytes": (loaded_stats or {}).get("size_host_bytes"),
                    "offload_fraction": offload,
                })
                candidate_estimate = _measured_memory_estimate(measured_points)
                if candidate_estimate is not None:
                    measured_estimate = candidate_estimate
            reasons.append(f"{size//1000}k:{'ok' if found else 'miss'}(ctx={wanted_ctx})")
        profile_summary = _needle_profile_summary(
            attempted,
            target_ctx=int(getattr(cfg, "long_context_target_ctx", 64000) or 64000),
            min_usable_tps=float(getattr(cfg, "long_context_min_tps", 10.0) or 10.0),
            critical_tps=float(getattr(cfg, "long_context_critical_tps", 3.0) or 3.0),
        )
        total_depths = len(task.meta.get("context_sizes", [])) or 1
        needle_coverage = round((total_depths - excluded_depths) / total_depths, 4)
        max_verified_ctx, non_mono = _max_verified_prefix(attempted)
        if not hits:
            first_detail = next((a.get("harness_error_detail") for a in attempted if a.get("harness_error_detail")), None)
            return {"score": None, "reason": "no scored needle probes; " + " ".join(reasons), "tps": None,
                    "ttft_ms": None, "ttft_visible_ms": None, "tokens": None, "output_chars": None,
                    "needle_attempted": attempted, "needle_skipped": skipped, "needle_coverage": needle_coverage,
                    "max_verified_ctx": max_verified_ctx, "non_monotonic_needle": non_mono,
                    "harness_error_detail": first_detail,
                    "error_kind": "harness_error", "num_predict": needle_num_predict,
                    **profile_summary}
        row_score = None if needle_coverage < 1.0 else round(sum(hits) / len(hits), 2)
        row = {"score": row_score, "reason": " ".join(reasons),
               "tps": None, "ttft_ms": None, "ttft_visible_ms": None, "tokens": None,
               "num_predict": needle_num_predict,
               "output_chars": None, "needle_attempted": attempted, "needle_skipped": skipped,
               "needle_coverage": needle_coverage, "max_verified_ctx": max_verified_ctx,
               "non_monotonic_needle": non_mono, **profile_summary}
        if probe_error_kinds and all(k == "harness_error" for k in probe_error_kinds) and not any(isinstance(h, (int, float)) for h in hits):
            row["error_kind"] = "harness_error"
        return row

    # Subjective.
    if task.scorer == "subjective":
        res = _chat(client, cfg, model, task.prompt, images=images, task=task)
        if not res.get("ok"):
            return _harness_error(res.get("error", "failed"), res)
        err = _model_output_error(res)
        if err:
            return err
        out = _gen_fields(res)
        out.update({"score": None, "reason": "needs judge", "output": res.get("text") or "",
                    "output_chars": len(res.get("text") or "")})
        if res.get("done_reason") == "length":
            out["warning_kind"] = "truncated"
        return out

    # Deterministic (optionally agentic).
    res = _chat(client, cfg, model, task.prompt, images=images, task=task)
    if not res.get("ok"):
        return _harness_error(res.get("error", "failed"), res)
    err = _model_output_error(res)
    if err:
        return err
    output = res.get("text") or ""
    score, reason, reason_public = _score_task(client, cfg, task, output, model)
    recovery = 0.0
    retry_generations = 0
    retry_tokens = 0
    best_res = res
    if task.agentic and _has_numeric_score(score) and score < 100.0:
        best, first = score, score
        best_reason, best_public = reason, reason_public
        for _ in range(cfg.max_reflections):
            follow = (f"Your solution scored {best:.0f}/100 ({best_public}). Fix it and output ONLY "
                      f"the corrected answer.")
            messages = [
                {"role": "user", "content": task.prompt},
                {"role": "assistant", "content": output},
                {"role": "user", "content": follow},
            ]
            r2 = _chat(client, cfg, model, "", task=task, messages=messages)
            retry_generations += 1
            retry_tokens += int(r2.get("tokens") or 0)
            if not r2.get("ok"):
                break
            if _model_output_error(r2):
                continue
            s2, reason2, public2 = _score_task(client, cfg, task, r2.get("text") or "", model)
            if s2 is not None and s2 > best:
                best, output, best_reason, best_public, best_res = s2, r2.get("text") or "", reason2, public2, r2
            if best >= 100.0:
                break
        recovery = round(best - first, 2)
        score, reason = best, best_reason
    out = _gen_fields(best_res)
    out.update({"score": _score_cell(score), "reason": reason,
                "output": output, "output_chars": len(output or ""), "recovery": recovery,
                "retry_generations": retry_generations, "retry_tokens": retry_tokens})
    if task.scorer == "agentic_action":
        detail = scoring.score_agentic_action_details(output, task.meta)
        out["caps_fired"] = detail.get("caps_fired") or []
        out["decision_score"] = detail.get("decision_score")
        out["format_multiplier"] = detail.get("format_multiplier")
        out["format_deviation"] = detail.get("format_deviation")
    if not _has_numeric_score(score) and str(reason or "").startswith("HARNESS_ERROR"):
        out["error_kind"] = "harness_error"
    if best_res.get("done_reason") == "length":
        out["warning_kind"] = "truncated"
        out["reason"] = f"WARN_TRUNCATED: done_reason=length; {reason}"
    return out


def _judge_subjective(client, cfg: Config, task: Task, output: str, mode: str) -> tuple:
    if mode == "panel":
        return judge_mod.judge_panel(client, cfg.judge_model, task.prompt, output, task.rubric,
                                     num_ctx=_ctx(cfg), think=_think(cfg))
    return judge_mod.judge_single(client, cfg.judge_model, task.prompt, output, task.rubric,
                                  num_ctx=_ctx(cfg), think=_think(cfg))


def _samples_for_task(task: Task, cfg: Config, sample_mode: str, judge_mode: str = "single") -> int:
    requested = max(1, int(getattr(cfg, "samples", 1) or 1))
    if sample_mode == "all":
        return requested
    if judge_mode != "off" and (task.scorer == "subjective" or task.judge):
        return requested
    return 1


def _avg_numeric(samples: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [s.get(key) for s in samples if isinstance(s.get(key), (int, float))]
    return round(sum(vals) / len(vals), 2) if vals else None


def _emit(ui: Any, line: str) -> None:
    if ui is not None and getattr(ui, "enabled", False):
        ui.log(line)
    else:
        print(line)


def _subjective_raw_reason(out_dir: Path, task: Task, model: str, output: str, saved_path: Optional[Path]) -> str:
    chars = len(output or "")
    if saved_path is not None:
        try:
            rel = saved_path.relative_to(out_dir)
        except Exception:
            rel = saved_path
        rel_display = str(rel).replace("\\", "/")
        return f"raw only, judge off: {chars} chars -> {rel_display}"
    return f"raw only, judge off: {chars} chars (not dumped)"


def _display_score(score: Any) -> str:
    return "raw" if not _has_numeric_score(score) else str(score)


def _validate_needle_ctx_override(cfg: Config, tasks: List[Task]) -> None:
    ctx = _ctx(cfg)
    if not ctx:
        return
    for task in tasks:
        if task.scorer != "needle":
            continue
        sizes = [int(s + 256 + 64) for s in task.meta.get("context_sizes", [])]
        if sizes and not any(ctx >= needed for needed in sizes):
            raise ValueError(f"--ctx {ctx} invalidates all needle probes; need at least {min(sizes)} or omit --ctx")


def _dump_raw(out_dir: Path, task: Task, model: str, output: str, sample_index: Optional[int] = None) -> Path:
    d = out_dir / "raw" / task.id
    d.mkdir(parents=True, exist_ok=True)
    safe = model.replace("/", "_").replace(":", "_")
    suffix = f".sample{sample_index}" if sample_index is not None else ""
    path = d / f"{safe}{suffix}.txt"
    path.write_text(output or "")
    return path


def run(client, cfg: Config, *, level: str, out_dir: Path,
        include: Optional[str], exclude: Optional[str], skip_offload: bool,
        categories: Optional[List[str]], task_ids: Optional[List[str]] = None,
        task_regex: Optional[str] = None, family_base_only: bool = False,
        context_aliases_only: bool = False, context_only: bool = False,
        resume: bool = True, judge_mode: str = "off",
        dump_subjective: bool = True, dump_raw: bool = True,
        status_interval: float = 5.0, live_ui: str = "off",
        sample_mode: str = "smart", fingerprint_enabled: bool = True,
        selected_models: Optional[List[str]] = None,
        capability_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
        auto_probe: bool = False,
        row_metadata_by_task: Optional[Dict[str, Dict[str, Any]]] = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "raw_results.jsonl"
    gpu = detect_gpu()
    ollama_version = client.version() if hasattr(client, "version") else None

    models_rows = {m.get("name"): m for m in client.tags()}
    models = list(models_rows.keys())
    skipped_models: List[Dict[str, str]] = []
    if selected_models is not None:
        missing = [m for m in selected_models if m not in models_rows]
        if missing:
            raise ValueError("selected model(s) are not installed: " + ", ".join(missing))
        selected_set = set(selected_models)
        skipped_models.extend({"model": m, "reason": "not_selected"} for m in models if m not in selected_set)
        models = [m for m in models if m in selected_set]
    if include:
        rx = re.compile(include, re.I)
        before = list(models)
        models = [m for m in models if rx.search(m)]
        skipped_models.extend({"model": m, "reason": "include_regex_no_match"} for m in before if m not in models)
    if exclude:
        rx = re.compile(exclude, re.I)
        before = list(models)
        models = [m for m in models if not rx.search(m)]
        skipped_models.extend({"model": m, "reason": "exclude_regex_match"} for m in before if m not in models)
    if skip_offload:
        before = list(models)
        models = [m for m in models if size_gb(models_rows[m]) <= cfg.vram_budget_gb]
        skipped_models.extend({"model": m, "reason": "size_exceeds_vram_budget"} for m in before if m not in models)

    models, context_skips = filter_models(
        models,
        family_base_only=family_base_only,
        context_aliases_only=context_aliases_only,
    )
    skipped_models.extend({"model": s.model, "reason": s.reason} for s in context_skips)

    unknown_tasks = validate_task_ids(task_ids, (t.id for t in TASKS))
    if unknown_tasks:
        known = ", ".join(sorted(t.id for t in TASKS))
        raise ValueError(f"unknown task id(s): {', '.join(unknown_tasks)}. Known tasks: {known}")

    done = set()
    existing_rows: List[Dict[str, Any]] = []
    if resume and raw.exists():
        for line in raw.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing_rows.append(r)
                th = r.get("task_hash")
                if th:
                    done.add((r["model"], r["task"], th))

    cats = categories
    task_source_level = "full" if context_only else level
    fingerprints: Dict[str, List[str]] = {}
    profiles = dict(capability_profiles or {})
    if any(model not in profiles for model in models):
        from .capabilities import interrogate_models
        missing_profiles = [model for model in models if model not in profiles]
        profiles.update(interrogate_models(client, missing_profiles, functional=auto_probe))
    model_plan: List[Dict[str, Any]] = []
    active_task_union: List[Task] = []
    for model in models:
        profile = profiles[model]
        capabilities = profile.get("declared_capabilities") or []
        fams = list(profile.get("supported_families") or [])
        all_tasks = filter_tasks(tasks_for(task_source_level, cats, fams), task_ids=task_ids,
                                 task_regex=task_regex, context_only=context_only)
        if not all_tasks:
            skipped_models.append({"model": model, "reason": "no_tasks_after_filter"})
            continue
        active_task_union.extend(all_tasks)
        model_plan.append({
            "model": model,
            "class": classify_model(model, capabilities, fams),
            "size_gb": size_gb(models_rows[model]),
            "families": fams,
            "declared_capabilities": capabilities,
            "capability_evidence_hash": profile.get("evidence_hash"),
            "tasks_total": len(all_tasks),
            "samples_total": sum(_samples_for_task(t, cfg, sample_mode, judge_mode) for t in all_tasks),
        })
    _validate_needle_ctx_override(cfg, active_task_union)

    run_id = out_dir.name
    filter_descriptions = describe_filters(
        task_ids=task_ids,
        task_regex=task_regex,
        family_base_only=family_base_only,
        context_aliases_only=context_aliases_only,
        context_only=context_only,
    )
    active_models = [m["model"] for m in model_plan]
    model_identities = {m: fingerprint.model_identity(m, models_rows.get(m, {})) for m in active_models}
    skipped_models = [s for s in skipped_models if s.get("model") not in active_models or s.get("reason") != "no_tasks_after_filter"]
    (out_dir / "skipped_models.json").write_text(json.dumps(skipped_models, indent=2))
    (out_dir / "model_identities.json").write_text(json.dumps(model_identities, indent=2))
    (out_dir / "capability_report.json").write_text(json.dumps({m: profiles[m] for m in active_models}, indent=2))
    status = progress.StatusWriter(out_dir, run_id=run_id, level=level, samples=max(1, cfg.samples),
                                   model_plan=model_plan, cfg=cfg, gpu=gpu,
                                   skipped_models=skipped_models, filters=filter_descriptions,
                                   sample_mode=sample_mode)
    status.rows_written = len(done)
    status.tasks_done = len(done)
    status.samples_done = sum(int(r.get("samples_used", max(1, cfg.samples))) for r in existing_rows)
    status.start_run()

    inline_layout = {"graph": "bars", "log": "compact"}.get(live_ui, live_ui)
    ui = InlineUI(
        out_dir,
        layout=(inline_layout if inline_layout in {"compact", "full", "bars"} else "compact"),
        enabled=(live_ui != "off"),
        refresh_interval=status_interval,
    )
    if live_ui == "log":
        ui.mode = "log"
    ui.start()

    _emit(ui, f"models={len(model_plan)} active skipped={len(skipped_models)} level={level} judge={judge_mode} seed={cfg.seed} "
          f"vram_budget={cfg.vram_budget_gb}GB gpu={gpu.vendor}")
    _emit(ui, f"sample_mode={sample_mode} requested_samples={max(1, cfg.samples)} num_predict={cfg.num_predict_override or 'task-default'} ctx={_ctx(cfg) or 'server-default'} think={_think(cfg)}")
    if filter_descriptions:
        _emit(ui, "filters=" + ";".join(filter_descriptions))
    if skipped_models:
        reasons = {}
        for item in skipped_models:
            reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
        _emit(ui, "skipped=" + ", ".join(f"{k}:{v}" for k, v in sorted(reasons.items())))
    if model_plan:
        rough = progress.estimate_remaining([15 * 60], len(model_plan))
        if rough.get("rolling_seconds"):
            _emit(ui, f"initial_eta_rough={progress.seconds_hms(rough['rolling_seconds'])} "
                  f"(updates in {out_dir}/status.json after each model)")

    # Fingerprint probes are not useful on tiny/single-task plans and can dominate wall clock.
    min_tasks_for_fingerprint = 2
    auto_skip_fingerprint = bool(model_plan and min(m.get("tasks_total", 0) for m in model_plan) < min_tasks_for_fingerprint)
    fingerprint_skip_reason = None
    if not fingerprint_enabled or not getattr(cfg, "fingerprint", True):
        fingerprint_skip_reason = "disabled_by_operator"
    elif auto_skip_fingerprint:
        fingerprint_skip_reason = "auto_skipped_plan_too_small"

    with raw.open("a") as fh:
        for model_index, model in enumerate(active_models, start=1):
            profile = profiles[model]
            capabilities = profile.get("declared_capabilities") or []
            fams = list(profile.get("supported_families") or [])
            cls = classify_model(model, capabilities, fams)
            sz = size_gb(models_rows[model])
            all_model_tasks = filter_tasks(tasks_for(task_source_level, cats, fams), task_ids=task_ids,
                                           task_regex=task_regex, context_only=context_only)
            model_tasks = [t for t in all_model_tasks if (model, t.id, _task_hash(t)) not in done]
            if not model_tasks:
                continue
            client.flush_all()
            # Warm up through a compatible provider path. Embedding-only models
            # must not be forced through chat, and insert-only models use suffix generation.
            if "text" in fams:
                _ = client.chat(model, "ok", num_predict=1, num_ctx=_ctx(cfg), think=_think(cfg))
            elif "embedding" in fams:
                _ = client.embed(model, ["warmup"])
            elif "insert" in fams and hasattr(client, "generate_suffix"):
                _ = client.generate_suffix(model, "x = ", suffix="\nassert x == 1", num_predict=4, num_ctx=_ctx(cfg))
            context_length_max = client.context_length(model)
            num_ctx_used = _ctx(cfg)
            display_ctx = num_ctx_used or context_length_max
            offload = client.offload_fraction(model, exact=True)
            if offload is None:
                offload = client.offload_fraction(model, exact=False)
            model_started_at = time.perf_counter()
            status.start_model(model_index, model, cls, sz, len(all_model_tasks), display_ctx, offload)
            ctx_s = f" ctx={display_ctx}" if display_ctx else " ctx=?"
            _emit(ui, f"\n=== {model} [{cls}] {sz}GB offload={offload}{ctx_s} : {len(model_tasks)} tasks ===")
            for task_index, task in enumerate(model_tasks, start=1):
                samples_for_task = _samples_for_task(task, cfg, sample_mode, judge_mode)
                status.start_task(task_index, task.id, samples_for_task)
                tel = Telemetry(gpu)
                # temperature safety pause
                t = tel.current_temp()
                while t is not None and t >= cfg.temp_pause_c:
                    _emit(ui, f"  GPU {t}C >= {cfg.temp_pause_c}C, pausing 30s")
                    time.sleep(30); t = tel.current_temp()
                tel.start()
                task_started_at = time.perf_counter()
                samples = [
                    _run_once(
                        client, cfg, model, task,
                        progress_callback=(
                            (lambda detail: status.update_task_detail(**detail))
                            if task.scorer == "needle" else None
                        ),
                    )
                    for _ in range(samples_for_task)
                ]
                task_wall_seconds = round(time.perf_counter() - task_started_at, 3)
                hw = tel.stop()
                base = samples[0]
                score = scoring.median([s.get("score") for s in samples]) if base.get("score") is not None else None
                reason = base.get("reason", "")
                saved_paths: List[Path] = []
                raw_paths: List[Path] = []

                if dump_raw and task.scorer not in {"subjective", "retrieval", "needle"}:
                    for si, sample in enumerate(samples, start=1):
                        if "output" in sample:
                            raw_paths.append(_dump_raw(out_dir, task, model, sample.get("output") or "", sample_index=si if samples_for_task > 1 else None))

                # Subjective judging: judge every sampled output, then median the judge scores.
                if task.scorer == "subjective":
                    judged_scores: List[float] = []
                    judged_reasons: List[str] = []
                    for si, sample in enumerate(samples, start=1):
                        out_text = sample.get("output") or ""
                        if dump_subjective:
                            saved = _dump(out_dir, task, model, out_text, sample_index=si if samples_for_task > 1 else None)
                            saved_paths.append(saved)
                        if sample.get("error_kind") in MODEL_ERROR_KINDS:
                            continue
                        if judge_mode != "off" and out_text:
                            js, jr = _judge_subjective(client, cfg, task, out_text, judge_mode)
                            if isinstance(js, (int, float)):
                                judged_scores.append(float(js))
                            judged_reasons.append(str(jr))
                    if samples and samples[0].get("error_kind") in MODEL_ERROR_KINDS:
                        score = samples[0].get("score", 0.0)
                        reason = samples[0].get("reason", "ERROR_EMPTY_OUTPUT")
                    elif judge_mode != "off":
                        score = scoring.median(judged_scores) if judged_scores else None
                        reason = ("median judge" if len(judged_scores) > 1 else (judged_reasons[0] if judged_reasons else "judge_error: no valid judge score"))
                        if len(judged_scores) > 1:
                            reason += f" samples={','.join(str(round(x, 1)) for x in judged_scores)}"
                    else:
                        score = None
                        reason = _subjective_raw_reason(out_dir, task, model, samples[0].get("output") or "", saved_paths[0] if saved_paths else None)

                row = {
                    "timestamp": progress.utc_now(),
                    "model": model, "task": task.id, "category": task.category,
                    "task_hash": _task_hash(task),
                    "family": task.family, "score": _score_cell(score),
                    "reason": reason, "tps": _avg_numeric(samples, "tps"),
                    "prompt_tps": _avg_numeric(samples, "prompt_tps"),
                    "ttft_ms": _avg_numeric(samples, "ttft_ms"),
                    "ttft_visible_ms": _avg_numeric(samples, "ttft_visible_ms"),
                    "think_ms": _avg_numeric(samples, "think_ms"),
                    "itl_p50_ms": _avg_numeric(samples, "itl_p50_ms"),
                    "itl_p95_ms": _avg_numeric(samples, "itl_p95_ms"),
                    "tokens": _avg_numeric(samples, "tokens"),
                    "eval_count": _avg_numeric(samples, "eval_count"),
                    "prompt_eval_count": _avg_numeric(samples, "prompt_eval_count"),
                    "request_elapsed_seconds": _avg_numeric(samples, "request_elapsed_seconds"),
                    "server_total_duration_ms": _avg_numeric(samples, "server_total_duration_ms"),
                    "server_load_duration_ms": _avg_numeric(samples, "server_load_duration_ms"),
                    "server_prompt_eval_duration_ms": _avg_numeric(samples, "server_prompt_eval_duration_ms"),
                    "server_eval_duration_ms": _avg_numeric(samples, "server_eval_duration_ms"),
                    "done_reason": samples[0].get("done_reason"),
                    "num_predict": samples[0].get("num_predict", _num_predict(cfg, task)),
                    "recovery": _avg_numeric(samples, "recovery"),
                    "retry_generations": _avg_numeric(samples, "retry_generations"),
                    "retry_tokens": _avg_numeric(samples, "retry_tokens"),
                    "samples_used": samples_for_task, "sample_mode": sample_mode,
                    "judge_mode": judge_mode,
                    "offload_fraction": offload, "class": cls, "size_gb": sz,
                    "task_wall_seconds": task_wall_seconds,
                    "model_elapsed_seconds": round(time.perf_counter() - model_started_at, 3),
                    "capabilities_declared": capabilities,
                    "capability_families": fams,
                    "capability_evidence_hash": profile.get("evidence_hash"),
                    "context_length": display_ctx,
                    "context_length_max": context_length_max,
                    "num_ctx_used": samples[0].get("num_ctx_used", num_ctx_used),
                    "ctx_override": _ctx(cfg),
                    "benchmark_version": __version__,
                    "output_chars": samples[0].get("output_chars", len(samples[0].get("output") or "")),
                    "thinking_chars": samples[0].get("thinking_chars"),
                    "think_sent": samples[0].get("think_sent"),
                    "think_unsupported": samples[0].get("think_unsupported"),
                    "think_ineffective": samples[0].get("think_ineffective"),
                    "error_kind": samples[0].get("error_kind"),
                    "http_status": samples[0].get("http_status"),
                    "http_reason": samples[0].get("http_reason"),
                    "http_url": samples[0].get("http_url"),
                    "http_error_body": samples[0].get("http_error_body"),
                    "warning_kind": samples[0].get("warning_kind"),
                    "caps_fired": samples[0].get("caps_fired") if task.scorer == "agentic_action" else None,
                    "decision_score": samples[0].get("decision_score") if task.scorer == "agentic_action" else None,
                    "format_multiplier": samples[0].get("format_multiplier") if task.scorer == "agentic_action" else None,
                    "format_deviation": samples[0].get("format_deviation") if task.scorer == "agentic_action" else None,
                    "subjective_path": (str((saved_paths[0].relative_to(out_dir) if saved_paths else "")) if task.scorer == "subjective" else None),
                    "raw_path": str(raw_paths[0].relative_to(out_dir)) if raw_paths else None,
                    "model_digest": (model_identities.get(model) or {}).get("digest"),
                    "vram_peak_mb": hw["vram_peak_mb"], "power_mean_w": hw["power_mean_w"],
                    "temp_peak_c": hw["temp_peak_c"],
                }
                if row_metadata_by_task and task.id in row_metadata_by_task:
                    # Repair/recovery provenance is additive and written only to the
                    # new child run. Source run evidence remains immutable.
                    row.update(dict(row_metadata_by_task[task.id]))
                if "native_tool_calls" in samples[0]:
                    row["native_tool_calls"] = samples[0].get("native_tool_calls") or []
                if "needle_attempted" in samples[0]:
                    row["needle_attempted"] = samples[0].get("needle_attempted")
                    row["needle_skipped"] = samples[0].get("needle_skipped")
                    row["needle_coverage"] = samples[0].get("needle_coverage")
                    row["max_verified_ctx"] = samples[0].get("max_verified_ctx")
                    row["non_monotonic_needle"] = samples[0].get("non_monotonic_needle")
                    for key in (
                        "needle_successful_depths", "needle_max_requested_size",
                        "needle_max_depth_tps", "needle_max_depth_ttft_ms",
                        "needle_max_depth_elapsed_seconds", "needle_min_tps",
                        "needle_median_tps", "needle_min_prompt_tps",
                        "needle_max_depth_prompt_tps", "needle_max_offload_fraction",
                        "needle_max_ram_delta_mb", "needle_max_ollama_rss_delta_mb",
                        "needle_max_ollama_pss_delta_mb", "needle_max_swap_delta_mb",
                        "needle_total_probe_seconds", "needle_behavior_suspect",
                        "needle_target_ctx", "needle_target_min_tps",
                        "needle_target_critical_tps", "needle_target_status",
                        "needle_target_size", "needle_target_num_ctx",
                        "needle_target_tps", "needle_target_prompt_tps",
                        "needle_target_elapsed_seconds", "needle_target_offload_fraction",
                        "needle_target_ram_delta_mb", "needle_target_ollama_rss_delta_mb",
                        "needle_target_ollama_pss_delta_mb", "needle_target_swap_delta_mb",
                        "needle_target_behavior_suspect", "needle_slow_depths",
                        "needle_critical_slow_depths",
                    ):
                        row[key] = samples[0].get(key)
                if task.scorer == "retrieval":
                    row["embed_model"] = samples[0].get("embed_model")
                    row["retrieval_cases"] = samples[0].get("retrieval_cases") or []
                anomaly = progress.classify_row(row)
                if anomaly and anomaly.get("kind") == "ANOMALY":
                    row["anomaly"] = anomaly["reason"]
                fh.write(json.dumps(row) + "\n"); fh.flush()
                status.finish_task(row, samples_for_task)
                suffix = f" samples={samples_for_task}" if samples_for_task > 1 else ""
                line = f"  {task.id:<14} score={_display_score(row['score'])} tok/s={row['tps']} {row['reason']}{suffix}"
                _emit(ui, line)
                if ui is not None and getattr(ui, "enabled", False):
                    ui.render(force=True)
                    ui.poll_keys()
                    if ui.stop_requested:
                        raise KeyboardInterrupt
            # clone fingerprint probes (text models only)
            if "text" in fams and model not in fingerprints and not fingerprint_skip_reason:
                probe_system = "Answer directly and concisely. Do not include hidden reasoning, chain of thought, or <think> tags."
                outs = [client.chat(model, p, system=probe_system, num_predict=max(1024, int(cfg.num_predict_override or 1024)),
                                    num_ctx=_ctx(cfg), think="off").get("text", "")
                        for p in fingerprint.PROBES]
                fingerprints[model] = outs
            status.finish_model(model)
            client.unload(model)

    (out_dir / "fingerprints.json").write_text(json.dumps(fingerprints, indent=2))
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (out_dir / "filters.json").write_text(json.dumps({
        "filters": filter_descriptions,
        "task_ids": task_ids,
        "task_regex": task_regex,
        "family_base_only": family_base_only,
        "context_aliases_only": context_aliases_only,
        "context_only": context_only,
        "sample_mode": sample_mode,
        "level": level,
        "judge_mode": judge_mode,
        "requested_samples": max(1, cfg.samples),
        "include_regex": include,
        "exclude_regex": exclude,
        "ctx_override": _ctx(cfg),
        "num_predict": cfg.num_predict_override,
        "think": _think(cfg),
        "needle_max_ctx": getattr(cfg, "needle_max_ctx", None),
        "dump_raw": bool(dump_raw),
        "fingerprint_skip_reason": fingerprint_skip_reason,
        "ollama_version": ollama_version,
        "active_models": active_models,
        "selected_models": selected_models,
        "auto_probe": bool(auto_probe),
        "capability_report": "capability_report.json",
        "skipped_models": skipped_models,
        "model_identities": model_identities,
        "repair_row_metadata_tasks": sorted((row_metadata_by_task or {}).keys()),
    }, indent=2))
    if ui is not None:
        ui.close()
    return out_dir


def _dump(out_dir: Path, task: Task, model: str, output: str, sample_index: Optional[int] = None) -> Path:
    d = out_dir / "subjective" / task.id
    d.mkdir(parents=True, exist_ok=True)
    safe = model.replace("/", "_").replace(":", "_")
    suffix = f".sample{sample_index}" if sample_index is not None else ""
    path = d / f"{safe}{suffix}.md"
    path.write_text(
        f"# {task.id} | {model}{' | sample ' + str(sample_index) if sample_index is not None else ''}\n\n"
        f"RUBRIC: {task.rubric}\n\n## PROMPT\n{task.prompt}\n\n## OUTPUT\n{output}")
    return path


def pack_subjective(out_dir: Path) -> None:
    base = out_dir / "subjective"
    if not base.is_dir():
        print("no subjective outputs"); return
    for task in sorted(p.name for p in base.iterdir() if p.is_dir()):
        d = base / task
        chunks = [(d / fn.name).read_text() for fn in sorted(d.iterdir()) if fn.name.endswith(".md")]
        out = base / f"_paste_{task}.md"
        out.write_text(f"# JUDGE THIS TASK: {task}\n\nScore each 0-100 on the same rubric.\n\n---\n\n"
                       + "\n\n---\n\n".join(chunks))
        print("wrote", out)


def assess_run_validity(out_dir: Path) -> Dict[str, Any]:
    """Classify whether a completed run contains usable benchmark evidence.

    Harness failures remain unknown evidence, not model failures. A run with no rows or only
    harness-error rows is invalid and must not be presented as a successful benchmark.
    """
    raw_path = Path(out_dir) / "raw_results.jsonl"
    rows: List[Dict[str, Any]] = []
    if raw_path.exists():
        for line_number, line in enumerate(raw_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = {
                    "error_kind": "harness_error",
                    "reason": f"invalid JSON row at line {line_number}",
                }
            rows.append(row)
    harness_errors = [
        row for row in rows
        if row.get("error_kind") == "harness_error"
        or str(row.get("reason") or "").startswith("HARNESS_ERROR")
    ]
    usable_rows = len(rows) - len(harness_errors)
    if not rows or usable_rows == 0:
        status = "invalid"
    elif harness_errors:
        status = "partial"
    else:
        status = "valid"
    result = {
        "status": status,
        "rows_total": len(rows),
        "usable_rows": usable_rows,
        "harness_error_rows": len(harness_errors),
        "ranking_eligible": status != "invalid",
    }
    (Path(out_dir) / "run_validity.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
