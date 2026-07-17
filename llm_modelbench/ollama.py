"""Ollama HTTP client, standard library only.

Streaming chat captures real latency metrics: time to first token, inter-token latency
(p50/p95), and generation tokens/sec from eval_count/eval_duration. Also exposes the
/api/ps offload fraction so we can tell when a model has spilled into system RAM.

MockClient implements the same surface with deterministic canned answers, so the whole
pipeline (scoring, aggregation, reporting) can run and be tested without a live Ollama.
"""
from __future__ import annotations

import json
import statistics
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional


def _exception_payload(exc: Exception) -> Dict[str, Any]:
    """Preserve HTTP status and response body instead of reducing failures to repr()."""
    payload: Dict[str, Any] = {"error": repr(exc)}
    if isinstance(exc, urllib.error.HTTPError):
        payload["http_status"] = getattr(exc, "code", None)
        payload["http_reason"] = str(getattr(exc, "reason", "") or "")
        payload["http_url"] = str(getattr(exc, "url", "") or "")
        try:
            body = exc.read()
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="replace")
            payload["http_error_body"] = str(body or "")[:8192]
        except Exception:
            payload["http_error_body"] = ""
    return payload

class OllamaClient:
    def __init__(self, base: str = "http://127.0.0.1:11434", seed: int = 42,
                 temperature: float = 0.0, timeout: int = 300):
        from urllib.parse import urlsplit

        parsed = urlsplit(str(base))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Ollama base URL must be an http(s) URL without embedded credentials")
        self.base = str(base).rstrip("/")
        self.seed = seed
        self.temperature = temperature
        self.timeout = timeout
        self._show_cache: Dict[str, Dict[str, Any]] = {}

    # ---- low level
    def _post_stream(self, path: str, payload: dict):
        req = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=self.timeout)  # nosec B310

    def _post(self, path: str, payload: dict, timeout: Optional[int] = None) -> dict:
        req = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:  # nosec B310
            return json.loads(r.read().decode())

    def _get(self, path: str, timeout: int = 30) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:  # nosec B310
            return json.loads(r.read().decode())

    # ---- discovery
    def tags(self) -> List[Dict[str, Any]]:
        try:
            return self._get("/api/tags").get("models", [])
        except Exception:
            return []

    def version(self) -> Optional[str]:
        try:
            return str(self._get("/api/version", timeout=10).get("version") or "") or None
        except Exception:
            return None

    def show(self, model: str) -> Dict[str, Any]:
        if model in self._show_cache:
            return self._show_cache[model]
        try:
            info = self._post("/api/show", {"model": model}, timeout=30)
        except Exception:
            info = {}
        self._show_cache[model] = info
        return info

    def capabilities(self, model: str) -> List[str]:
        info = self.show(model)
        caps = info.get("capabilities") or []
        return [str(c).lower() for c in caps if c is not None]

    def supports_thinking(self, model: str) -> bool:
        caps = self.capabilities(model)
        return any(c in {"thinking", "think"} or "thinking" in c for c in caps)

    def model_info(self, model: str) -> Dict[str, Any]:
        return dict((self.show(model).get("model_info") or {}))

    def model_size_bytes(self, model: str) -> Optional[int]:
        try:
            for m in self.tags():
                if m.get("name") == model:
                    size = m.get("size")
                    return int(size) if size else None
        except Exception:
            pass
        return None

    def context_length(self, model: str) -> Optional[int]:
        info = self.show(model)
        for k, v in (info.get("model_info") or {}).items():
            if k.endswith("context_length"):
                try:
                    return int(v)
                except Exception:
                    pass
        return None

    # ---- generation with streaming metrics
    def chat(self, model: str, prompt: str, *, images: Optional[List[str]] = None,
             system: Optional[str] = None, num_predict: int = 1024,
             num_ctx: Optional[int] = None, think: str = "auto",
             messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if messages is not None:
            messages = [dict(m) for m in messages]
            if system:
                messages.insert(0, {"role": "system", "content": system})
        else:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            user: Dict[str, Any] = {"role": "user", "content": prompt}
            if images:
                user["images"] = images
            messages.append(user)
        payload = {
            "model": model, "messages": messages, "stream": True,
            "options": {"temperature": self.temperature, "seed": self.seed,
                        "num_predict": int(num_predict)},
        }
        if num_ctx:
            payload["options"]["num_ctx"] = int(num_ctx)
        think_requested = think in {"on", "off"}
        think_sent = False
        think_unsupported = False
        if think_requested:
            if self.supports_thinking(model):
                payload["think"] = (think == "on")
                think_sent = True
            else:
                # Do not send top-level think to non-thinking models. Some Ollama builds reject it.
                think_unsupported = True
        t0 = time.perf_counter()
        text = ""
        thinking = ""
        any_times: List[float] = []
        visible_times: List[float] = []
        stats: Dict[str, Any] = {}
        try:
            with self._post_stream("/api/chat", payload) as resp:
                for line in resp:
                    if not line:
                        continue
                    d = json.loads(line)
                    msg = d.get("message") or {}
                    c = msg.get("content")
                    th = msg.get("thinking")
                    now = time.perf_counter()
                    if th:
                        thinking += th
                        any_times.append(now)
                    if c:
                        text += c
                        any_times.append(now)
                        visible_times.append(now)
                    if d.get("done"):
                        stats = d
                        break
        except urllib.error.HTTPError as exc:
            if think_sent and 400 <= getattr(exc, "code", 0) < 500:
                payload.pop("think", None)
                think_sent = False
                think_unsupported = True
                text = ""
                thinking = ""
                any_times = []
                visible_times = []
                stats = {}
                try:
                    with self._post_stream("/api/chat", payload) as resp:
                        for line in resp:
                            if not line:
                                continue
                            d = json.loads(line)
                            msg = d.get("message") or {}
                            c = msg.get("content")
                            th = msg.get("thinking")
                            now = time.perf_counter()
                            if th:
                                thinking += th
                                any_times.append(now)
                            if c:
                                text += c
                                any_times.append(now)
                                visible_times.append(now)
                            if d.get("done"):
                                stats = d
                                break
                except Exception as exc2:
                    return {"ok": False, **_exception_payload(exc2), "text": "", "thinking": "",
                            "ttft_ms": None, "ttft_visible_ms": None, "think_ms": None,
                            "tps": None, "itl_p50_ms": None, "itl_p95_ms": None, "tokens": 0,
                            "eval_count": 0, "prompt_eval_count": None, "done_reason": None,
                            "num_predict": num_predict, "num_ctx": num_ctx, "think_sent": False,
                            "think_unsupported": think_unsupported, "think_requested": think}
            else:
                return {"ok": False, **_exception_payload(exc), "text": "", "thinking": "",
                        "ttft_ms": None, "ttft_visible_ms": None, "think_ms": None,
                        "tps": None, "itl_p50_ms": None, "itl_p95_ms": None, "tokens": 0,
                        "eval_count": 0, "prompt_eval_count": None, "done_reason": None,
                        "num_predict": num_predict, "num_ctx": num_ctx, "think_sent": think_sent,
                        "think_unsupported": think_unsupported, "think_requested": think}
        except Exception as exc:
            return {"ok": False, **_exception_payload(exc), "text": "", "thinking": "",
                    "ttft_ms": None, "ttft_visible_ms": None, "think_ms": None,
                    "tps": None, "itl_p50_ms": None, "itl_p95_ms": None, "tokens": 0,
                    "eval_count": 0, "prompt_eval_count": None, "done_reason": None,
                    "num_predict": num_predict, "num_ctx": num_ctx, "think_sent": think_sent,
                    "think_unsupported": think_unsupported, "think_requested": think}
        itls = [(visible_times[i + 1] - visible_times[i]) * 1000
                for i in range(len(visible_times) - 1)]
        eval_seconds = float(stats.get("eval_duration") or 0) / 1e9
        prompt_seconds = float(stats.get("prompt_eval_duration") or 0) / 1e9
        total_seconds = float(stats.get("total_duration") or 0) / 1e9
        load_seconds = float(stats.get("load_duration") or 0) / 1e9
        eval_count = int(stats.get("eval_count") or 0)
        prompt_eval_count = stats.get("prompt_eval_count")
        wall_seconds = time.perf_counter() - t0
        p95_idx = int(0.95 * (len(itls) - 1)) if itls else 0
        return {
            "ok": True, "text": text, "thinking": thinking,
            "ttft_ms": round((any_times[0] - t0) * 1000, 1) if any_times else None,
            "ttft_visible_ms": round((visible_times[0] - t0) * 1000, 1) if visible_times else None,
            "think_ms": round((visible_times[0] - any_times[0]) * 1000, 1) if any_times and visible_times and visible_times[0] >= any_times[0] else (None if not thinking else 0.0),
            "tps": round((eval_count / eval_seconds), 2) if eval_seconds else None,
            "prompt_tps": (round(float(prompt_eval_count) / prompt_seconds, 2)
                           if isinstance(prompt_eval_count, (int, float)) and prompt_seconds else None),
            "itl_p50_ms": round(statistics.median(itls), 1) if itls else None,
            "itl_p95_ms": round(sorted(itls)[p95_idx], 1) if itls else None,
            "tokens": eval_count,
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "request_elapsed_seconds": round(wall_seconds, 3),
            "server_total_duration_ms": round(total_seconds * 1000.0, 3) if total_seconds else None,
            "server_load_duration_ms": round(load_seconds * 1000.0, 3) if load_seconds else None,
            "server_prompt_eval_duration_ms": round(prompt_seconds * 1000.0, 3) if prompt_seconds else None,
            "server_eval_duration_ms": round(eval_seconds * 1000.0, 3) if eval_seconds else None,
            "done_reason": stats.get("done_reason"),
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "thinking_chars": len(thinking or ""),
            "think_sent": think_sent,
            "think_unsupported": think_unsupported,
            "think_requested": think,
        }

    def chat_tools(self, model: str, prompt: str, *, tools: List[Dict[str, Any]],
                   system: Optional[str] = None, num_predict: int = 512,
                   num_ctx: Optional[int] = None, think: str = "auto") -> Dict[str, Any]:
        """Call Ollama's native structured tool interface without executing tools."""
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed,
                        "num_predict": int(num_predict)},
        }
        if num_ctx:
            payload["options"]["num_ctx"] = int(num_ctx)
        if think in {"on", "off"} and self.supports_thinking(model):
            payload["think"] = think == "on"
        started = time.perf_counter()
        try:
            data = self._post("/api/chat", payload)
        except Exception as exc:
            return {"ok": False, **_exception_payload(exc), "text": "", "tool_calls": [],
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "num_predict": num_predict, "num_ctx": num_ctx}
        message = data.get("message") or {}
        eval_count = int(data.get("eval_count") or 0)
        eval_duration = float(data.get("eval_duration") or 0) / 1e9
        prompt_duration = float(data.get("prompt_eval_duration") or 0) / 1e9
        prompt_count = data.get("prompt_eval_count")
        return {
            "ok": True,
            "text": str(message.get("content") or ""),
            "thinking": str(message.get("thinking") or ""),
            "tool_calls": list(message.get("tool_calls") or []),
            "tokens": eval_count,
            "eval_count": eval_count,
            "prompt_eval_count": prompt_count,
            "tps": round(eval_count / eval_duration, 2) if eval_duration else None,
            "prompt_tps": (round(float(prompt_count) / prompt_duration, 2)
                           if isinstance(prompt_count, (int, float)) and prompt_duration else None),
            "server_total_duration_ms": round(float(data.get("total_duration") or 0) / 1e6, 3) or None,
            "server_load_duration_ms": round(float(data.get("load_duration") or 0) / 1e6, 3) or None,
            "server_prompt_eval_duration_ms": round(prompt_duration * 1000.0, 3) if prompt_duration else None,
            "server_eval_duration_ms": round(eval_duration * 1000.0, 3) if eval_duration else None,
            "done_reason": data.get("done_reason"),
            "request_elapsed_seconds": round(time.perf_counter() - started, 3),
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        }

    def generate_suffix(self, model: str, prompt: str, *, suffix: str,
                        num_predict: int = 256, num_ctx: Optional[int] = None) -> Dict[str, Any]:
        """Use Ollama's suffix/FIM generation path."""
        payload: Dict[str, Any] = {
            "model": model, "prompt": prompt, "suffix": suffix, "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed,
                        "num_predict": int(num_predict)},
        }
        if num_ctx:
            payload["options"]["num_ctx"] = int(num_ctx)
        started = time.perf_counter()
        try:
            data = self._post("/api/generate", payload)
        except Exception as exc:
            return {"ok": False, **_exception_payload(exc), "text": "",
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "num_predict": num_predict, "num_ctx": num_ctx}
        eval_count = int(data.get("eval_count") or 0)
        eval_duration = float(data.get("eval_duration") or 0) / 1e9
        prompt_duration = float(data.get("prompt_eval_duration") or 0) / 1e9
        prompt_count = data.get("prompt_eval_count")
        return {
            "ok": True,
            "text": str(data.get("response") or ""),
            "tokens": eval_count,
            "eval_count": eval_count,
            "prompt_eval_count": prompt_count,
            "tps": round(eval_count / eval_duration, 2) if eval_duration else None,
            "prompt_tps": (round(float(prompt_count) / prompt_duration, 2)
                           if isinstance(prompt_count, (int, float)) and prompt_duration else None),
            "server_total_duration_ms": round(float(data.get("total_duration") or 0) / 1e6, 3) or None,
            "server_load_duration_ms": round(float(data.get("load_duration") or 0) / 1e6, 3) or None,
            "server_prompt_eval_duration_ms": round(prompt_duration * 1000.0, 3) if prompt_duration else None,
            "server_eval_duration_ms": round(eval_duration * 1000.0, 3) if eval_duration else None,
            "done_reason": data.get("done_reason"),
            "request_elapsed_seconds": round(time.perf_counter() - started, 3),
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        }

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        try:
            data = self._post("/api/embed", {"model": model, "input": texts})
            if data.get("embeddings"):
                return data["embeddings"]
        except Exception:
            pass
        out = []
        for t in texts:
            try:
                out.append(self._post("/api/embeddings", {"model": model, "prompt": t})["embedding"])
            except Exception:
                out.append([])
        return out

    # ---- vram management (one model at a time)
    def loaded_model_stats(self, model: str) -> Optional[Dict[str, Any]]:
        """Return exact /api/ps residency data for one loaded model.

        ``size_vram`` is the portion resident on GPU.  The difference between
        ``size`` and ``size_vram`` is reported as model bytes resident outside
        VRAM; it is useful model-card evidence but is not a complete accounting
        of KV cache, allocator workspace, or host process memory.
        """
        try:
            target_digest = None
            for t in self.tags():
                if t.get("name") == model:
                    target_digest = t.get("digest")
                    break
            for m in self._get("/api/ps").get("models", []):
                name = m.get("name", "")
                digest = m.get("digest")
                if name != model and not (target_digest and digest == target_digest):
                    continue
                size = int(m.get("size") or 0)
                size_vram = int(m.get("size_vram") or 0)
                host = max(0, size - size_vram) if size else None
                return {
                    "name": name or model,
                    "digest": digest or target_digest,
                    "size_bytes": size or None,
                    "size_vram_bytes": size_vram if size else None,
                    "size_host_bytes": host,
                    "offload_fraction": round(1.0 - size_vram / size, 3) if size else None,
                    "context_length": m.get("context_length"),
                    "expires_at": m.get("expires_at"),
                }
        except Exception:
            pass
        return None

    def offload_fraction(self, model: str, exact: bool = True) -> Optional[float]:
        """Return offload fraction for the exact loaded model.

        Older code used loose prefix matching for fallback, so llama3:8b could match
        llama3.1:8b and qwen2.5:14b could match qwen2.5-coder:14b. V9.5.11
        refuses ambiguous prefix matches; exact=False now only permits exact name or
        exact digest matches when Ollama exposes a digest.
        """
        try:
            target_digest = None
            try:
                for t in self.tags():
                    if t.get("name") == model:
                        target_digest = t.get("digest")
                        break
            except Exception:
                target_digest = None
            for m in self._get("/api/ps").get("models", []):
                name = m.get("name", "")
                digest = m.get("digest")
                match = (name == model) or (bool(target_digest) and digest == target_digest)
                if match:
                    size, vram = m.get("size") or 0, m.get("size_vram") or 0
                    if size:
                        return round(1.0 - vram / size, 3)
        except Exception:
            pass
        return None

    def unload(self, model: str) -> None:
        try:
            self._post("/api/chat", {"model": model, "messages": [], "keep_alive": 0}, timeout=30)
        except Exception:
            pass

    def flush_all(self) -> None:
        try:
            for m in self._get("/api/ps").get("models", []):
                self.unload(m.get("name", ""))
        except Exception:
            pass


class MockClient(OllamaClient):
    """Deterministic offline stand-in. Returns plausible answers keyed off prompt content so
    the full pipeline can run and be tested without Ollama. Used by `run --mock`."""

    _MODELS = [
        {"name": "qwen2.5-coder:14b", "size": 9_000_000_000, "digest": "mock-qwen25coder14b"},
        {"name": "llama3.1:8b", "size": 4_900_000_000, "digest": "mock-llama318b"},
        {"name": "qwen2.5-vl:7b", "size": 6_000_000_000, "digest": "mock-qwen25vl7b"},
        {"name": "nomic-embed-text:latest", "size": 274_000_000, "digest": "mock-nomicembed"},
    ]

    def tags(self):
        return list(self._MODELS)

    def version(self):
        return "mock"

    def capabilities(self, model):
        n = str(model).lower()
        if "nomic-embed" in n:
            return ["embedding"]
        if "vl" in n:
            return ["completion", "vision"]
        if "coder" in n:
            return ["completion", "tools", "insert"]
        return ["completion"]

    def show(self, model):
        return {"model_info": {"general.context_length": 32768,
                               "llama.block_count": 32,
                               "llama.attention.head_count_kv": 8,
                               "llama.attention.key_length": 128,
                               "llama.attention.value_length": 128},
                "capabilities": []}

    def model_info(self, model):
        return dict(self.show(model).get("model_info") or {})

    def supports_thinking(self, model):
        return False

    def model_size_bytes(self, model):
        for m in self._MODELS:
            if m.get("name") == model:
                return int(m.get("size") or 0)
        return None

    def context_length(self, model):
        return 32768

    def chat(self, model, prompt, *, images=None, system=None, num_predict=1024, num_ctx=None, think="auto", messages=None):
        if messages is not None and not prompt:
            prompt = "\n".join(str(m.get("content") or "") for m in messages)
        if images and "short code shown in the image" in str(prompt).lower():
            text = "V7K9Q2"
        elif "AIW_TEXT_OK" in str(prompt):
            text = "AIW_TEXT_OK"
        else:
            text = self._answer(prompt)
        tokens = max(1, len(text) // 4) if text else 0
        prompt_tokens = max(1, int(len(prompt or "") / 6.85))
        return {"ok": True, "text": text, "thinking": "", "ttft_ms": 120.0 if text else None,
                "ttft_visible_ms": 120.0 if text else None, "think_ms": 0.0, "tps": 42.0 if text else None,
                "itl_p50_ms": 22.0 if text else None, "itl_p95_ms": 40.0 if text else None,
                "tokens": tokens, "eval_count": tokens, "prompt_eval_count": prompt_tokens, "done_reason": "stop",
                "num_predict": num_predict, "num_ctx": num_ctx, "thinking_chars": 0,
                "think_sent": False, "think_unsupported": think in {"on", "off"}, "think_requested": think}

    def chat_tools(self, model, prompt, *, tools, system=None, num_predict=512, num_ctx=None, think="auto"):
        tool = (tools[0].get("function") or {}).get("name") if tools else None
        if tool == "lookup_weather":
            args = {"city": "Paris", "units": "celsius"}
        else:
            args = {}
        return {"ok": True, "text": "", "tool_calls": [{"function": {"name": tool, "arguments": args}}],
                "tokens": 8, "eval_count": 8, "prompt_eval_count": 20, "tps": 42.0,
                "done_reason": "stop", "num_predict": num_predict, "num_ctx": num_ctx}

    def generate_suffix(self, model, prompt, *, suffix, num_predict=256, num_ctx=None):
        if "normalize_status" in prompt:
            text = "value.strip().lower()"
        else:
            expected = "BLUE" if "BLUE" in suffix else ("RED" if "RED" in suffix else "completed")
            text = repr(expected)
        return {"ok": True, "text": text, "tokens": 4, "eval_count": 4,
                "prompt_eval_count": 12, "tps": 42.0, "done_reason": "stop",
                "num_predict": num_predict, "num_ctx": num_ctx}

    def embed(self, model, texts):
        # cheap deterministic pseudo-embeddings: bag-of-chars vector
        vecs = []
        for t in texts:
            v = [0.0] * 32
            for ch in t.lower():
                v[ord(ch) % 32] += 1.0
            vecs.append(v)
        return vecs

    def loaded_model_stats(self, model):
        size = self.model_size_bytes(model) or 0
        return {"name": model, "digest": None, "size_bytes": size or None,
                "size_vram_bytes": size or None, "size_host_bytes": 0 if size else None,
                "offload_fraction": 0.0, "context_length": 32768, "expires_at": None}

    def offload_fraction(self, model, exact=True):
        return 0.0

    def _answer(self, prompt: str) -> str:
        p = prompt.lower()
        if "return only valid json" in p and "score" in p and ("grading a model answer" in p or "persona:" in p):
            return '{"score": 88, "confidence": 0.9, "verdict": "mock judge: accurate and usable"}'
        if "group_anagrams" in p:
            return ("```python\ndef group_anagrams(words):\n"
                    "    from collections import defaultdict\n"
                    "    d = defaultdict(list)\n"
                    "    for w in words: d[''.join(sorted(w))].append(w)\n"
                    "    return list(d.values())\n```")
        if "dedupe" in p:
            return ("```python\ndef dedupe(seq):\n    seen=set(); out=[]\n"
                    "    for x in seq:\n        if x not in seen: seen.add(x); out.append(x)\n"
                    "    return out\n```")
        if "parse_csv" in p:
            return ("```python\ndef parse_csv(text):\n    lines=[l for l in text.splitlines() if l]\n"
                    "    if not lines: return []\n    h=lines[0].split(',')\n"
                    "    return [dict(zip(h,l.split(','))) for l in lines[1:]]\n```")
        if "moves" in p and ".txt" in p:
            return ("```python\nimport os, shutil\nfor f in os.listdir('.'):\n"
                    "    if f.endswith('.txt'): os.makedirs('text', exist_ok=True); shutil.move(f,'text/'+f)\n"
                    "    elif f.endswith('.md'): os.makedirs('docs', exist_ok=True); shutil.move(f,'docs/'+f)\n```")
        if "debounce" in p:
            return ("```javascript\nfunction debounce(fn, delay){let t;return function(...a){"
                    "clearTimeout(t);t=setTimeout(()=>fn.apply(this,a),delay);};}\n```")
        if "nav" in p and "flex" in p:
            return ("```html\n<nav class='topnav'><a href='#'>Home</a><a href='#'>Docs</a></nav>\n```\n"
                    "```css\n.topnav{display:flex;gap:1rem}@media (max-width:600px){.topnav{flex-direction:column}}\n```")
        if "incident acknowledged" in p:
            return '{"tool":"send_email","args":{"to":"ops@example.com","subject":"Incident acknowledged"}}'
        if "backup complete" in p and "ops@example.com" in p:
            return '{"tool":"send_email","args":{"to":"ops@example.com","subject":"Backup complete"}}'
        if "email" in p:
            return "ada@calc.io\nhopper@navy.mil"
        if "sort" in p and "alphabetical" in p:
            return "apple\napricot\nbanana\ncherry\ndate"
        if "commit hash" in p or "which commit" in p:
            return "9f8e7d6"
        if "inspect the readme.md" in p or "inspect readme.md" in p or "readme.md file" in p:
            return '{"tool":"read_file","args":{"path":"README.md"}}'
        if "permanently delete /etc/passwd" in p:
            return '{"tool":null,"args":{},"reason":"No offered safe tool can perform that destructive request."}'
        if ("previous tool call failed validation" in p or "prior calculator.add call" in p or "calculator.add" in p and "strings" in p) and "calculator.add" in p:
            return '{"tool":"calculator.add","args":{"a":12,"b":7}}'
        if "cart has a1 quantity 1 and b2 quantity 5" in p or "cart currently has a1 quantity 1 and b2 quantity 5" in p:
            return '{"tool":"cart.update","args":{"sku":"B2","quantity":3}}'
        if "cart has a1 quantity 1" in p or "cart currently has a1 quantity 1" in p:
            return '{"tool":"cart.update","args":{"sku":"A1","quantity":3}}'
        if "no offered tool can execute shell commands" in p or "rm -rf /tmp/cache" in p:
            return '{"tool":null,"args":{},"reason":"No offered tool can execute shell commands."}'
        if ("previous call was malformed" in p or "prior read_file call was malformed" in p or "positional args" in p) and "read_file" in p:
            return '{"tool":"read_file","args":{"path":"README.md"}}'
        if "disk alert" in p and "ticket.create" in p:
            return '{"tool":"ticket.create","args":{"ticket":{"title":"Disk alert","priority":"high","labels":["infra","disk"]}}}'
        if "secret" in p and "code" in p:
            return "SECRET_CODE_77"
        if "json" in p and "server" in p:
            return '{"server":"API-01","ip":"192.168.1.10","status":"Critical"}'
        if "transcribe" in p:
            return "INVOICE 2026-0042 BillTo: BrightWave Ltd Amount Due: GBP 1,240.50 Due: 2026-07-31"
        if "rag" in p or "retrieval augmented" in p or "embedding" in p:
            return ("Retrieval Augmented Generation gives a model a searchable library. "
                    "Before answering, it looks up relevant documents and reads them, like a "
                    "student checking references before writing, so answers stay grounded in real "
                    "sources instead of memory alone.")
        if "torch" in p and "bridge" in p:
            return "17 minutes"
        if "poisoned bottle" in p or ("prisoners" in p and "bottles" in p):
            return "binary-coded prisoner testing"
        if "identical twins" in p:
            return "1 pair"
        if "three closed doors" in p or "monty" in p:
            return "switch to door 2, 2/3 chance"
        if "wolf" in p and "cabbage" in p:
            return "7 crossings"
        return "This is a deterministic mock response for offline pipeline testing."
