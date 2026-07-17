"""Runtime configuration.

Precedence: built-in defaults < config file (JSON or YAML) < environment variables < CLI
flags. The VRAM budget auto-derives from the detected GPU unless you pin it, so the tool
adapts to whatever machine it runs on instead of assuming one card.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Dict, Optional

from urllib.parse import urlparse

from .hardware import detect_gpu, suggested_vram_budget_gb

# Category weights feed the single composite quality score. They renormalise over whatever
# categories actually ran, so partial suites still produce a fair number. Tune to taste.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "coding_python": 0.18,
    "coding_web": 0.05,
    "coding_js": 0.05,
    "ocr": 0.12,
    "pdf": 0.05,
    "long_context": 0.10,
    "tech_writing": 0.12,
    "knowledge_base": 0.08,
    "text_ops": 0.08,
    "git": 0.07,
    "file_ops": 0.07,
    "agentic": 0.04,
    "agentic_tool": 0.08,
    "retrieval": 0.05,
    "reasoning": 0.05,
}


def _validated_weights(value: Any) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        raise SystemExit("weights must be a mapping of category names to non-negative numbers")
    unknown = sorted(set(value) - set(DEFAULT_WEIGHTS))
    if unknown:
        raise SystemExit(f"unknown weight categories: {', '.join(unknown)}")
    merged = dict(DEFAULT_WEIGHTS)
    for category, raw in value.items():
        if isinstance(raw, bool):
            raise SystemExit(f"weight for {category!r} must be a number, not boolean")
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"weight for {category!r} must be numeric") from exc
        if not math.isfinite(number) or number < 0:
            raise SystemExit(f"weight for {category!r} must be finite and non-negative")
        merged[category] = number
    if not any(weight > 0 for weight in merged.values()):
        raise SystemExit("at least one category weight must be greater than zero")
    return merged


def _validate_ollama_url(value: Any) -> str:
    url = str(value or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SystemExit("ollama_url must be an http:// or https:// URL with a hostname")
    if parsed.username or parsed.password:
        raise SystemExit("ollama_url must not contain embedded credentials")
    return url.rstrip("/")


@dataclass
class Config:
    ollama_url: str = "http://127.0.0.1:11434"
    seed: int = 42
    temperature: float = 0.0
    vram_budget_gb: float = 0.0            # 0 = auto-detect
    judge_model: str = "qwen2.5:14b"
    embed_model: str = "nomic-embed-text"
    samples: int = 1                       # runs per test; median taken where applicable
    max_reflections: int = 2               # agentic ReAct retries
    temp_pause_c: float = 85.0             # pause if GPU hits this (nvidia only)
    temp_resume_c: float = 75.0
    request_timeout: int = 300
    ctx_override: Optional[int] = None     # real field, persisted by asdict()
    num_predict_override: Optional[int] = None
    think: str = "auto"                   # auto | on | off
    dump_raw: bool = True                  # deterministic raw outputs are persisted by default
    fingerprint: bool = True               # clone probes; runner auto-skips tiny plans
    min_report_tasks_per_category: int = 2
    needle_max_ctx: Optional[int] = None       # operator safety cap for needle probes
    long_context_target_ctx: int = 64000       # operating-profile target, not a score gate
    long_context_min_tps: float = 10.0         # preferred decode speed at target context
    long_context_critical_tps: float = 3.0     # below this is generally impractical
    needle_preflight_mode: str = "enforce"    # enforce | advisory (controlled context profiles)
    needle_min_available_ram_gb: float = 2.0   # do not start another tier below this host-RAM floor
    context_profile_mode: bool = False         # watcher/status hint; diagnostic only
    context_profile_behavior_probe: bool = True
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        cfg = cls()
        data: dict = {}
        if path and Path(path).exists():
            raw = Path(path).read_text()
            if path.endswith((".yaml", ".yml")):
                try:
                    import yaml
                    data = yaml.safe_load(raw) or {}
                except ImportError:
                    raise SystemExit("YAML config needs pyyaml: pip install 'llm-modelbench[yaml]'")
            else:
                data = json.loads(raw)
        if not isinstance(data, dict):
            raise SystemExit("config root must be a JSON/YAML object")
        known_fields = {item.name for item in fields(cls)}
        unknown_fields = sorted(set(data) - known_fields)
        if unknown_fields:
            raise SystemExit(f"unknown config field(s): {', '.join(unknown_fields)}")
        for key, value in data.items():
            if key == "weights":
                cfg.weights = _validated_weights(value)
            else:
                setattr(cfg, key, value)
        env_map = {
            "LLM_MODELBENCH_OLLAMA_URL": ("ollama_url", str),
            "LLM_MODELBENCH_JUDGE_MODEL": ("judge_model", str),
            "LLM_MODELBENCH_EMBED_MODEL": ("embed_model", str),
            "LLM_MODELBENCH_VRAM_BUDGET_GB": ("vram_budget_gb", float),
            "LLM_MODELBENCH_SEED": ("seed", int),
            "LLM_MODELBENCH_CTX": ("ctx_override", int),
            "LLM_MODELBENCH_NUM_CTX": ("ctx_override", int),
            "LLM_MODELBENCH_NUM_PREDICT": ("num_predict_override", int),
            "LLM_MODELBENCH_THINK": ("think", str),
            "LLM_MODELBENCH_NEEDLE_MAX_CTX": ("needle_max_ctx", int),
            "LLM_MODELBENCH_LONG_CONTEXT_TARGET_CTX": ("long_context_target_ctx", int),
            "LLM_MODELBENCH_LONG_CONTEXT_MIN_TPS": ("long_context_min_tps", float),
            "LLM_MODELBENCH_LONG_CONTEXT_CRITICAL_TPS": ("long_context_critical_tps", float),
        }
        for env, (attr, cast) in env_map.items():
            if os.environ.get(env):
                setattr(cfg, attr, cast(os.environ[env]))
        if cfg.think not in {"auto", "on", "off"}:
            raise SystemExit("think must be one of: auto, on, off")
        cfg.ollama_url = _validate_ollama_url(cfg.ollama_url)
        cfg.weights = _validated_weights(cfg.weights)
        # auto VRAM budget
        if not cfg.vram_budget_gb:
            cfg.vram_budget_gb = suggested_vram_budget_gb(detect_gpu())
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
