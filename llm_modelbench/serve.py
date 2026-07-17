"""Read-only REST helpers over already-computed ``summary.json`` artifacts."""
from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

TIE_BAND_EPSILON = 0.5


def _load_run(run_dir: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    path = run_dir / "summary.json"
    if not path.exists():
        return [], "summary.json not found"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [], f"could not read summary.json: {exc}"
    if not isinstance(payload, list):
        return [], "summary.json root must be a list"
    rows = [row for row in payload if isinstance(row, dict)]
    if not rows:
        return [], "summary.json contains no model rows"
    return rows, None


def build_index(run_dirs: Iterable[Path]) -> Dict[str, Any]:
    models: Dict[str, Dict[str, Any]] = {}
    loaded = []
    for run_dir in run_dirs:
        rows, error = _load_run(run_dir)
        loaded.append({
            "run_dir": str(run_dir),
            "models": len(rows),
            "loaded_at": time.time(),
            "status": "error" if error else "loaded",
            "error": error,
        })
        for row in rows:
            name = row.get("model")
            if not name:
                continue
            target = models.setdefault(name, {"model": name, "size_gb": None, "tok_s": None, "categories": {}})
            for key in ("size_gb", "tok_s"):
                if row.get(key) is not None:
                    target[key] = row[key]
            target["categories"].update(row.get("categories") or {})
    return {
        "models": models,
        "loaded": loaded,
        "valid_runs": sum(1 for item in loaded if item["status"] == "loaded"),
    }


def route(index: Dict[str, Any], use_case: str, max_vram_gb: Optional[float]) -> Dict[str, Any]:
    candidates = []
    for model in index.get("models", {}).values():
        quality = (model.get("categories") or {}).get(use_case)
        if not isinstance(quality, (int, float)):
            continue
        size = model.get("size_gb")
        if max_vram_gb is not None and (not isinstance(size, (int, float)) or size > max_vram_gb):
            continue
        candidates.append({**model, "quality": quality})
    if not candidates:
        return {"use_case": use_case, "vram_limit_gb": max_vram_gb, "tied_band": [],
                "note": "no model has data for this category within the given VRAM limit"}
    best = max(item["quality"] for item in candidates)
    band = [item for item in candidates if item["quality"] >= best - TIE_BAND_EPSILON]
    band.sort(key=lambda item: (item.get("size_gb") if isinstance(item.get("size_gb"), (int, float)) else float("inf"),
                                -(item.get("tok_s") or 0)))
    return {"use_case": use_case, "vram_limit_gb": max_vram_gb, "tied_band_quality": round(best, 2),
            "tied_band": [{key: item.get(key) for key in ("model", "quality", "size_gb", "tok_s")} for item in band],
            "recommended": band[0]["model"],
            "note": "tied ceiling band; ordered by VRAM then speed" if len(band) > 1 else None}


def endpoint_response(index: Dict[str, Any], path: str) -> Tuple[Dict[str, Any], int]:
    """Pure endpoint dispatch used by the HTTP wrapper and offline tests."""
    parsed = urlparse(path)
    query = parse_qs(parsed.query)
    if parsed.path == "/health":
        errors = [item for item in index["loaded"] if item.get("status") == "error"]
        return {
            "status": "degraded" if errors else "ok",
            "read_only": True,
            "valid_runs": index.get("valid_runs", 0),
            "runs_loaded": index["loaded"],
        }, 200
    if parsed.path == "/models":
        return {"models": list(index["models"].values())}, 200
    if parsed.path == "/routing":
        use_case = (query.get("use_case") or [None])[0]
        if not use_case:
            return {"error": "use_case query param is required"}, 400
        try:
            vram = float((query.get("vram") or [None])[0]) if query.get("vram") else None
        except ValueError:
            return {"error": "vram must be numeric"}, 400
        return route(index, use_case, vram), 200
    return {"error": "not found", "endpoints": ["/health", "/models", "/routing"]}, 404


def make_handler(index: Dict[str, Any]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _fmt, *_args):
            pass

        def _json(self, payload: Dict[str, Any], code: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            payload, code = endpoint_response(index, self.path)
            self._json(payload, code)
    return Handler


def serve(
    run_dirs: Iterable[Path], host: str, port: int, *,
    allow_remote: bool = False, allow_empty: bool = False,
) -> None:
    loopback_hosts = {"127.0.0.1", "localhost", "::1"}
    if host not in loopback_hosts and not allow_remote:
        raise ValueError("non-loopback binding requires --allow-remote")
    if not 1 <= int(port) <= 65535:
        raise ValueError("port must be between 1 and 65535")
    index = build_index(run_dirs)
    if not index.get("valid_runs") and not allow_empty:
        details = "; ".join(
            f"{item['run_dir']}: {item.get('error') or 'no rows'}" for item in index["loaded"]
        )
        raise ValueError(f"no valid run summaries loaded ({details})")
    HTTPServer((host, port), make_handler(index)).serve_forever()
