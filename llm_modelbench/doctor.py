"""Environment preflight checks for LLM ModelBench."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from . import __version__
from .hardware import detect_gpu, live_snapshot


def _run(cmd: List[str], timeout: int = 5) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def _url_json(url: str, timeout: float = 2.5) -> Dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # nosec B310
            return json.loads(r.read().decode())
    except Exception as exc:
        return {"error": repr(exc)}


def collect(cfg: Any) -> Dict[str, Any]:
    import llm_modelbench
    exe = shutil.which("llm-modelbench")
    disk = shutil.disk_usage(os.getcwd())
    gpu = detect_gpu()
    hw, _ = live_snapshot(None)
    base = cfg.ollama_url.rstrip("/")
    version = _url_json(base + "/api/version")
    tags = _url_json(base + "/api/tags")
    ps = _url_json(base + "/api/ps")
    return {
        "llm_version": __version__,
        "python": sys.version.split()[0],
        "sys_executable": sys.executable,
        "imported_from": str(Path(llm_modelbench.__file__).resolve()),
        "entrypoint": exe,
        "cwd": os.getcwd(),
        "venv": os.environ.get("VIRTUAL_ENV"),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "ollama_url": cfg.ollama_url,
        "ollama_version": version.get("version"),
        "ollama_error": version.get("error") or tags.get("error"),
        "ollama_model_count": len(tags.get("models", []) or []),
        "ollama_loaded_count": len(ps.get("models", []) or []) if isinstance(ps.get("models"), list) else 0,
        "nvidia_smi": shutil.which("nvidia-smi"),
        "nvidia_smi_query": _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.free,memory.total", "--format=csv,noheader,nounits"]) if shutil.which("nvidia-smi") else None,
        "node": shutil.which("node"),
        "node_version": _run(["node", "--version"]) if shutil.which("node") else None,
        "gpu": gpu.__dict__,
        "hardware_live": hw,
        "disk_free_gb": round(disk.free / 1024**3, 1),
        "disk_total_gb": round(disk.total / 1024**3, 1),
    }


def render(data: Dict[str, Any]) -> str:
    ok_ollama = "ok" if data.get("ollama_version") else "FAIL"
    ok_gpu = "ok" if data.get("nvidia_smi") or data.get("gpu", {}).get("vendor") != "unknown" else "warn"
    warn_shadow = ""
    imported = data.get("imported_from") or ""
    entry = data.get("entrypoint") or ""
    if entry and "/.venv/" not in entry and data.get("venv"):
        warn_shadow = "WARNING: entrypoint is outside the active venv. Check PATH/shadowing."
    lines = [
        "LLM ModelBench doctor",
        "====================",
        f"LLM version:       {data.get('llm_version')}",
        f"Python:            {data.get('python')}  {data.get('sys_executable')}",
        f"Imported from:     {imported}",
        f"Entrypoint:        {entry}",
        f"Venv:              {data.get('venv') or 'none'}",
        f"PYTHONPATH:        {data.get('pythonpath') or 'empty'}",
    ]
    if warn_shadow:
        lines.append(warn_shadow)
    lines += [
        "",
        f"Ollama:            {ok_ollama}  {data.get('ollama_url')}  version={data.get('ollama_version') or 'n/a'}",
        f"Models installed:  {data.get('ollama_model_count')}",
        f"Models loaded:     {data.get('ollama_loaded_count')}",
    ]
    if data.get("ollama_error"):
        lines.append(f"Ollama error:      {data.get('ollama_error')}")
    gpu = data.get("gpu") or {}
    lines += [
        "",
        f"GPU:               {ok_gpu}  {gpu.get('vendor')}  {gpu.get('name')}  VRAM={gpu.get('total_vram_gb')}GB",
        f"nvidia-smi:        {data.get('nvidia_smi') or 'not found'}",
        f"Node.js:           {'ok' if data.get('node') else 'WARN'}  {data.get('node') or 'not found'}  version={data.get('node_version') or 'n/a'}",
    ]
    if not data.get("node"):
        lines.append("Node warning:      js_debounce will be marked HarnessError/skipped from quality instead of scoring models 0.0.")
    if data.get("nvidia_smi_query"):
        lines.append(f"nvidia-smi query:  {data.get('nvidia_smi_query')}")
    hw = data.get("hardware_live") or {}
    if hw:
        lines.append(f"Live telemetry:    GPU {hw.get('gpu_temp_c','n/a')}C, VRAM {hw.get('vram_used_mb','n/a')}/{hw.get('vram_total_mb','n/a')} MiB, RAM {hw.get('ram_used_pct','n/a')}%")
    lines += [
        "",
        f"Disk free:         {data.get('disk_free_gb')}GB / {data.get('disk_total_gb')}GB",
    ]
    return "\n".join(lines)
