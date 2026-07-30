"""Read-only external llama-server client for RC21 Stage 5.

This module deliberately contains no server lifecycle, model switching, or
mutable-property operations.  It speaks only to an already-running endpoint.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from .backend import (
    BackendCapabilities, BackendCapability, BackendCapabilityError,
    BackendIdentity, CapabilityStatus,
)

MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_ERROR_BYTES = 8192
_DIGEST = re.compile(r"sha256-([0-9a-f]{64})", re.I)


class LlamaCppError(RuntimeError):
    """A bounded, user-safe external llama-server error."""


Transport = Callable[[str, str, str, Optional[Dict[str, Any]], float], Any]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def _default_transport(base: str, method: str, path: str, payload: Optional[Dict[str, Any]], timeout: float) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        base + path, data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method=method,
    )
    try:
        opener = urllib.request.build_opener(_NoRedirect())
        with opener.open(request, timeout=timeout) as response:  # nosec B310
            body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                raise LlamaCppError(f"{path} response exceeds bounded-size limit ({MAX_RESPONSE_BYTES} bytes)")
    except urllib.error.HTTPError as exc:
        detail = exc.read(MAX_ERROR_BYTES + 1).decode("utf-8", "replace")[:MAX_ERROR_BYTES]
        if 300 <= exc.code < 400:
            raise LlamaCppError(f"{path} redirect rejected") from exc
        raise LlamaCppError(f"{path} HTTP {exc.code}: {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise LlamaCppError(f"{path} connection failure: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LlamaCppError(f"{path} timed out") from exc
    except LlamaCppError:
        raise
    except Exception as exc:
        raise LlamaCppError(f"{path} request failed: {exc}") from exc
    try:
        value = json.loads(body.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        if path == "/metrics":
            return {"text": body.decode("utf-8", "replace")}
        raise LlamaCppError(f"{path} did not return JSON") from exc
    return value


def _integer(value: Any) -> Optional[int]:
    """Return a JSON integer, excluding Python's boolean subtype."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _required_integer(value: Any, label: str) -> int:
    integer = _integer(value)
    if integer is None:
        raise LlamaCppError(f"{label} must be an integer")
    return integer


def _validate_model_metadata(value: Dict[str, Any], label: str) -> None:
    for key, description in (
        ("size", "size"),
        ("n_params", "parameter count"),
        ("n_ctx", "active context"),
        ("n_ctx_train", "training context"),
    ):
        if value.get(key) is not None:
            _required_integer(value[key], f"{label} {description}")


def _object(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise LlamaCppError(f"{label} returned an invalid JSON shape")
    return value


def _string_list(value: Any, label: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise LlamaCppError(f"{label} must be a list of strings")
    return list(dict.fromkeys(item for item in value if item))


def _digest(value: Any) -> Optional[str]:
    match = _DIGEST.search(str(value or ""))
    return f"sha256-{match.group(1).lower()}" if match else None


class LlamaCppClient:
    def __init__(self, base: str, seed: int = 42, temperature: float = 0.0,
                 timeout: int = 300, transport: Optional[Transport] = None):
        parsed = urllib.parse.urlsplit(str(base))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("llama.cpp base URL must be an http(s) URL without embedded credentials")
        self.base = str(base).rstrip("/")
        self.seed, self.temperature, self.timeout = seed, temperature, timeout
        self._transport = transport or _default_transport
        self._props_cache: Optional[Dict[str, Any]] = None
        self._models_cache: Optional[List[Dict[str, Any]]] = None

    def _request(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        method = "POST" if payload is not None else "GET"
        return self._transport(self.base, method, path, payload, float(self.timeout))

    def health(self) -> Dict[str, Any]:
        value = self._request("/health")
        value = _object(value, "/health")
        if str(value.get("status") or "").lower() not in {"ok", "healthy"}:
            raise LlamaCppError("/health did not report an available server")
        return value

    def props(self) -> Dict[str, Any]:
        if self._props_cache is None:
            value = self._request("/props")
            value = _object(value, "/props")
            if value.get("build_info") is not None and not isinstance(value.get("build_info"), str):
                raise LlamaCppError("/props build_info must be a string or null")
            if value.get("total_slots") is not None and (_integer(value.get("total_slots")) is None or value["total_slots"] < 0):
                raise LlamaCppError("/props total_slots must be a non-negative integer")
            settings = value.get("default_generation_settings")
            if settings is not None and not isinstance(settings, dict):
                raise LlamaCppError("/props default_generation_settings must be an object")
            if isinstance(settings, dict) and settings.get("n_ctx") is not None:
                _required_integer(settings["n_ctx"], "/props default_generation_settings n_ctx")
            modalities = value.get("modalities")
            if modalities is not None and not isinstance(modalities, dict):
                raise LlamaCppError("/props modalities must be an object")
            template_caps = value.get("chat_template_caps")
            if template_caps is not None and not isinstance(template_caps, dict):
                raise LlamaCppError("/props chat_template_caps must be an object")
            for label, mapping in (("modalities", modalities), ("chat_template_caps", template_caps)):
                if isinstance(mapping, dict) and any(not isinstance(flag, bool) for flag in mapping.values()):
                    raise LlamaCppError(f"/props {label} values must be booleans")
            if value.get("chat_template") is not None and not isinstance(value.get("chat_template"), str):
                raise LlamaCppError("/props chat_template must be a string")
            self._props_cache = value
        return self._props_cache

    def _models(self) -> List[Dict[str, Any]]:
        if self._models_cache is not None:
            return self._models_cache
        payload = self._request("/v1/models")
        payload = _object(payload, "/v1/models")
        data = payload.get("data")
        fallback = payload.get("models")
        if data is not None and not isinstance(data, list):
            raise LlamaCppError("/v1/models data must be a list")
        if fallback is not None and not isinstance(fallback, list):
            raise LlamaCppError("/v1/models models must be a list")
        authoritative = data if data else fallback
        if not authoritative:
            raise LlamaCppError("llama.cpp endpoint reports no served model")
        rows: List[Dict[str, Any]] = []
        for index, raw in enumerate(authoritative):
            raw = _object(raw, f"/v1/models row {index}")
            key = "id" if data else ("name" if raw.get("name") is not None else "model")
            name = raw.get(key)
            if not isinstance(name, str) or not name.strip():
                raise LlamaCppError(f"/v1/models row {index} has no valid model name")
            aliases = _string_list(raw.get("aliases") if data else raw.get("tags"), f"/v1/models row {index} aliases")
            meta = raw.get("meta") if data else raw.get("details")
            if meta is None:
                meta = {}
            if not isinstance(meta, dict):
                raise LlamaCppError(f"/v1/models row {index} metadata must be an object")
            _validate_model_metadata(meta, f"/v1/models row {index}")
            rows.append({"name": name.strip(), "aliases": aliases, "meta": dict(meta), "raw": raw})
        if len({row["name"] for row in rows}) != 1:
            raise LlamaCppError("llama.cpp router or multi-model server mode is outside RC21 Stage 5")
        row = rows[0]
        # Compatibility metadata is only merged when it names this exact model
        # or an explicitly reported authoritative alias.
        if data and fallback:
            for raw in fallback:
                raw = _object(raw, "/v1/models compatibility row")
                name = raw.get("name") if raw.get("name") is not None else raw.get("model")
                if not isinstance(name, str) or not name:
                    raise LlamaCppError("/v1/models compatibility row has no valid model name")
                if name != row["name"] and name not in row["aliases"]:
                    continue
                details = raw.get("details")
                if details is not None and not isinstance(details, dict):
                    raise LlamaCppError("/v1/models compatibility details must be an object")
                row["aliases"] = list(dict.fromkeys(row["aliases"] + _string_list(raw.get("tags"), "/v1/models compatibility tags")))
                for key, value in (details or {}).items():
                    row["meta"].setdefault(key, value)
                _validate_model_metadata(row["meta"], "/v1/models compatibility metadata")
        self._models_cache = [row]
        return self._models_cache

    def _model(self, requested: str) -> Dict[str, Any]:
        row = self._models()[0]
        if requested != row["name"] and requested not in row["aliases"]:
            raise LlamaCppError(f"requested model is not served by this llama.cpp endpoint: {requested}")
        return row

    def version(self) -> Optional[str]:
        return str(self.props().get("build_info") or "") or None

    def tags(self) -> List[Dict[str, Any]]:
        row = self._models()[0]
        meta = row["meta"]
        observed_digest = next((value for value in [_digest(row["name"]), *(_digest(alias) for alias in row["aliases"])] if value), None)
        item: Dict[str, Any] = {"name": row["name"], "model": row["name"]}
        if observed_digest: item["digest"] = observed_digest
        if _integer(meta.get("size")) is not None: item["size"] = _integer(meta.get("size"))
        details = {key: meta[key] for key in ("format", "n_params", "parameter_size", "ftype", "quantization_level") if key in meta}
        if details: item["details"] = details
        if row["aliases"]: item["aliases"] = list(row["aliases"])
        return [item]

    def _contexts(self, row: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        meta = row["meta"]
        props = self.props()
        settings = props.get("default_generation_settings") or {}
        active = _integer(meta.get("n_ctx")) or _integer(settings.get("n_ctx"))
        return active, _integer(meta.get("n_ctx_train"))

    def show(self, model: str) -> Dict[str, Any]:
        row = self._model(model)
        active, training = self._contexts(row)
        caps = self.capabilities(model)
        info = {key: value for key, value in {
            "general.context_length": active,
            "general.training_context_length": training,
            "general.parameter_count": _integer(row["meta"].get("n_params")),
            "general.quantization": row["meta"].get("ftype"),
        }.items() if value is not None}
        result: Dict[str, Any] = {"model_info": info, "capabilities": caps}
        if row["aliases"]: result["aliases"] = list(row["aliases"])
        return result

    def model_info(self, model: str) -> Dict[str, Any]:
        return dict(self.show(model).get("model_info") or {})

    def model_size_bytes(self, model: str) -> Optional[int]:
        return _integer(self._model(model)["meta"].get("size"))

    def context_length(self, model: str) -> Optional[int]:
        return self._contexts(self._model(model))[0]

    def capabilities(self, model: str) -> List[str]:
        self._model(model)
        props = self.props()
        caps = ["completion"]
        template = props.get("chat_template_caps") or {}
        if isinstance(template, dict) and template.get("supports_tools") and template.get("supports_tool_calls"):
            caps.append("tools")
        return caps

    def supports_thinking(self, model: str) -> bool:
        self._model(model)
        return "enable_thinking" in str(self.props().get("chat_template") or "")

    def _messages(self, prompt: str, system: Optional[str], messages: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if messages is not None:
            if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
                raise LlamaCppError("messages must be a list of message objects")
            result = [dict(item) for item in messages]
            roles = {"system", "user", "assistant", "tool"}
            for index, item in enumerate(result):
                role = item.get("role")
                if not isinstance(role, str) or not role or role not in roles:
                    raise LlamaCppError("message role must be a supported non-empty string")
                if role == "system" and index != 0:
                    raise LlamaCppError("system messages are allowed only at index zero")
                content = item.get("content")
                if content is not None and not isinstance(content, str):
                    raise LlamaCppError("message content must be a string or null")
                if content is None and (role != "assistant" or not isinstance(item.get("tool_calls"), list)):
                    raise LlamaCppError("null message content is allowed only for assistant tool calls")
        else:
            result = [{"role": "user", "content": prompt}]
        # A supplied first system message wins; otherwise the explicit system
        # argument is prepended exactly once.
        if system is not None and not isinstance(system, str):
            raise LlamaCppError("system message must be a string")
        if system and not (result and result[0].get("role") == "system"):
            result.insert(0, {"role": "system", "content": system})
        return result

    def _chat_payload(self, model: str, prompt: str, *, system: Optional[str], num_predict: int,
                      num_ctx: Optional[int], think: str, messages: Optional[List[Dict[str, Any]],],
                      tools: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> Dict[str, Any]:
        row = self._model(model)
        active, _ = self._contexts(row)
        if num_ctx is not None:
            requested_ctx = _required_integer(num_ctx, "requested context")
            if active is not None and requested_ctx > active:
                raise LlamaCppError(f"requested context {num_ctx} exceeds served context {active}")
        if kwargs.get("images"):
            raise LlamaCppError("vision is unsupported by this llama.cpp endpoint")
        max_tokens = _required_integer(num_predict, "maximum generated tokens")
        seed = _required_integer(self.seed, "request seed")
        payload: Dict[str, Any] = {"model": model, "messages": self._messages(prompt, system, messages),
                                   "stream": False, "temperature": self.temperature, "seed": seed,
                                   "max_tokens": max_tokens}
        if tools is not None:
            payload["tools"] = tools
        stop = kwargs.get("stop")
        if stop is not None:
            payload["stop"] = stop
        requested_format = kwargs.get("response_format", kwargs.get("format"))
        if requested_format == "json":
            payload["response_format"] = {"type": "json_object"}
        elif isinstance(requested_format, dict):
            if requested_format.get("type") in {"json_object", "json_schema"}:
                response_format = dict(requested_format)
                kind = response_format["type"]
                if kind == "json_object":
                    if set(response_format) - {"type", "schema"} or ("schema" in response_format and not isinstance(response_format["schema"], dict)):
                        raise LlamaCppError("json_object response_format must contain only an optional schema object")
                else:
                    wrapper = response_format.get("json_schema")
                    if not isinstance(wrapper, dict) or not isinstance(wrapper.get("schema"), dict):
                        raise LlamaCppError("json_schema response_format requires an object schema wrapper")
                payload["response_format"] = response_format
            else:
                payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "response", "schema": requested_format}}
        elif requested_format is not None:
            raise LlamaCppError("response_format must be 'json' or an object")
        if think in {"on", "off"}:
            if not self.supports_thinking(model):
                raise LlamaCppError("thinking control is unsupported by this llama.cpp template")
            template_kwargs = kwargs.get("chat_template_kwargs")
            if template_kwargs is not None and not isinstance(template_kwargs, dict):
                raise LlamaCppError("chat_template_kwargs must be an object")
            if template_kwargs and "enable_thinking" in template_kwargs and not isinstance(template_kwargs["enable_thinking"], bool):
                raise LlamaCppError("chat_template_kwargs enable_thinking must be a boolean")
            if template_kwargs and "enable_thinking" in template_kwargs and template_kwargs["enable_thinking"] != (think == "on"):
                raise LlamaCppError("chat_template_kwargs enable_thinking conflicts with explicit think mode")
            payload["chat_template_kwargs"] = {**(template_kwargs or {}), "enable_thinking": think == "on"}
        elif kwargs.get("chat_template_kwargs") is not None:
            if not isinstance(kwargs["chat_template_kwargs"], dict):
                raise LlamaCppError("chat_template_kwargs must be an object")
            if "enable_thinking" in kwargs["chat_template_kwargs"] and not isinstance(kwargs["chat_template_kwargs"]["enable_thinking"], bool):
                raise LlamaCppError("chat_template_kwargs enable_thinking must be a boolean")
            payload["chat_template_kwargs"] = dict(kwargs["chat_template_kwargs"])
        return payload

    @staticmethod
    def _timings(data: Dict[str, Any], usage: Dict[str, Any]) -> Dict[str, Any]:
        timings = data.get("timings") if isinstance(data.get("timings"), dict) else {}
        for label, value in usage.items():
            if label in {"prompt_tokens", "completion_tokens", "total_tokens"} and _integer(value) is None:
                raise LlamaCppError(f"/v1/chat/completions usage {label} must be an integer")
        for label, value in timings.items():
            if label in {"prompt_n", "predicted_n", "prompt_tokens", "completion_tokens", "total_tokens", "n_prompt_tokens_processed", "n_tokens_predicted"} and _integer(value) is None:
                raise LlamaCppError(f"/v1/chat/completions timings {label} must be an integer")
            if label in {"prompt_per_second", "predicted_per_second", "total_ms", "prompt_ms", "predicted_ms"} and not isinstance(value, (int, float)):
                raise LlamaCppError(f"/v1/chat/completions timings {label} must be numeric")
        prompt = _integer(usage.get("prompt_tokens"))
        tokens = _integer(usage.get("completion_tokens")) or 0
        prompt_tps = timings.get("prompt_per_second")
        tps = timings.get("predicted_per_second")
        return {"tokens": tokens, "eval_count": tokens, "prompt_eval_count": prompt,
                "tps": round(float(tps), 2) if isinstance(tps, (int, float)) and tps else None,
                "prompt_tps": round(float(prompt_tps), 2) if isinstance(prompt_tps, (int, float)) and prompt_tps else None,
                "server_total_duration_ms": (float(timings["total_ms"]) if isinstance(timings.get("total_ms"), (int, float)) else None),
                "server_prompt_eval_duration_ms": (float(timings["prompt_ms"]) if isinstance(timings.get("prompt_ms"), (int, float)) else None),
                "server_eval_duration_ms": (float(timings["predicted_ms"]) if isinstance(timings.get("predicted_ms"), (int, float)) else None)}

    def _normalise_chat(self, data: Dict[str, Any], *, num_predict: int, num_ctx: Optional[int], think: str) -> Dict[str, Any]:
        data = _object(data, "/v1/chat/completions")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LlamaCppError("/v1/chat/completions returned no choice")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise LlamaCppError("/v1/chat/completions choice message must be an object")
        raw_content = message.get("content")
        if raw_content is not None and not isinstance(raw_content, str):
            raise LlamaCppError("/v1/chat/completions message content must be a string or null")
        content = raw_content or ""
        raw_reasoning = message.get("reasoning_content")
        if raw_reasoning is not None and not isinstance(raw_reasoning, str):
            raise LlamaCppError("/v1/chat/completions reasoning_content must be a string or null")
        thinking = raw_reasoning or ""
        if not thinking:
            match = re.search(r"<think>(.*?)</think>", content, re.S)
            if match:
                thinking = match.group(1).strip()
        usage = data.get("usage") if data.get("usage") is not None else {}
        if not isinstance(usage, dict):
            raise LlamaCppError("/v1/chat/completions usage must be an object")
        if data.get("timings") is not None and not isinstance(data.get("timings"), dict):
            raise LlamaCppError("/v1/chat/completions timings must be an object")
        result = {"ok": True, "text": content, "thinking": thinking,
                  "done_reason": choice.get("finish_reason"), "num_predict": num_predict,
                  "num_ctx": num_ctx, "thinking_chars": len(thinking), "think_sent": think in {"on", "off"},
                  "think_unsupported": False, "think_requested": think, **self._timings(data, usage)}
        result["truncated"] = choice.get("finish_reason") in {"length", "max_tokens"}
        return result

    def chat(self, model: str, prompt: str, *, images: Optional[List[str]] = None, system: Optional[str] = None,
             num_predict: int = 1024, num_ctx: Optional[int] = None, think: str = "auto",
             messages: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> Dict[str, Any]:
        try:
            payload = self._chat_payload(model, prompt, system=system, num_predict=num_predict, num_ctx=num_ctx,
                                         think=think, messages=messages, images=images, **kwargs)
            return self._normalise_chat(self._request("/v1/chat/completions", payload), num_predict=num_predict, num_ctx=num_ctx, think=think)
        except LlamaCppError as exc:
            return {"ok": False, "error": str(exc), "text": "", "thinking": "", "tokens": 0,
                    "eval_count": 0, "prompt_eval_count": None, "num_predict": num_predict, "num_ctx": num_ctx,
                    "think_requested": think}

    def chat_tools(self, model: str, prompt: str, *, tools: List[Dict[str, Any]], system: Optional[str] = None,
                   num_predict: int = 512, num_ctx: Optional[int] = None, think: str = "auto",
                   messages: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> Dict[str, Any]:
        try:
            if "tools" not in self.capabilities(model):
                return {"ok": False, "error": "llama.cpp tool calls are unsupported by this endpoint template",
                        "text": "", "tool_calls": [], "num_predict": num_predict, "num_ctx": num_ctx}
            if not isinstance(tools, list) or not all(isinstance(item, dict) for item in tools):
                raise LlamaCppError("tools must be a list of objects")
            payload = self._chat_payload(model, prompt, system=system, num_predict=num_predict, num_ctx=num_ctx,
                                         think=think, messages=messages, tools=tools, **kwargs)
            data = self._request("/v1/chat/completions", payload)
            result = self._normalise_chat(data, num_predict=num_predict, num_ctx=num_ctx, think=think)
            message = (data.get("choices") or [{}])[0].get("message")
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise LlamaCppError("/v1/chat/completions returned invalid tool calls")
            normalized = []
            for call in calls:
                if not isinstance(call, dict):
                    raise LlamaCppError("/v1/chat/completions tool call must be an object")
                function = call.get("function")
                if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
                    raise LlamaCppError("/v1/chat/completions tool call function must have a name")
                arguments = function.get("arguments")
                if "id" in call and not isinstance(call["id"], str):
                    raise LlamaCppError("/v1/chat/completions tool call id must be a string")
                if "type" in call and not isinstance(call["type"], str):
                    raise LlamaCppError("/v1/chat/completions tool call type must be a string")
                if isinstance(arguments, str):
                    try:
                        decoded = json.loads(arguments)
                        arguments = decoded if isinstance(decoded, dict) else arguments
                    except json.JSONDecodeError:
                        pass
                elif not isinstance(arguments, dict):
                    raise LlamaCppError("/v1/chat/completions tool arguments must be an object or JSON string")
                entry = {"function": {"name": function["name"], "arguments": arguments}}
                for key in ("id", "type"):
                    if key in call: entry[key] = call[key]
                normalized.append(entry)
            result["tool_calls"] = normalized
            return result
        except LlamaCppError as exc:
            return {"ok": False, "error": str(exc), "text": "", "tool_calls": [], "num_predict": num_predict, "num_ctx": num_ctx}

    def tokenize(self, content: str, *, add_special: bool = False, parse_special: bool = False,
                 with_pieces: bool = False) -> Dict[str, Any]:
        data = self._request("/tokenize", {"content": content, "add_special": add_special,
                                            "parse_special": parse_special, "with_pieces": with_pieces})
        if not isinstance(data, dict):
            raise LlamaCppError("/tokenize returned an invalid JSON shape")
        if not isinstance(data.get("tokens"), list) or not all(_integer(value) is not None for value in data["tokens"]):
            raise LlamaCppError("/tokenize returned an invalid token array")
        return data

    def slots(self) -> List[Dict[str, Any]]:
        data = self._request("/slots")
        if not isinstance(data, list):
            raise LlamaCppError("/slots returned an invalid shape")
        if not all(isinstance(slot, dict) for slot in data):
            raise LlamaCppError("/slots entries must be objects")
        for slot in data:
            for key, kind in (("id", int), ("n_ctx", int), ("is_processing", bool), ("speculative", bool)):
                if key in slot and (not isinstance(slot[key], kind) or (kind is int and isinstance(slot[key], bool))):
                    raise LlamaCppError(f"/slots {key} has an invalid value")
        return data

    def metrics(self) -> Dict[str, Any]:
        value = self._request("/metrics")
        if not isinstance(value, dict) or not isinstance(value.get("text"), str):
            raise LlamaCppError("/metrics returned an invalid response shape")
        return value

    def loaded_model_stats(self, model: str) -> Optional[Dict[str, Any]]:
        row = self._model(model)
        slots = self.slots()
        active, _ = self._contexts(row)
        return {"name": row["name"], "digest": _digest(row["name"]), "size_bytes": self.model_size_bytes(model),
                "context_length": active, "slot_count": len(slots),
                "slot_states": [{"id": slot.get("id"), "is_processing": slot.get("is_processing")} for slot in slots if isinstance(slot, dict)]}

    def offload_fraction(self, model: str, exact: bool = True) -> Optional[float]:
        return None

    def generate_suffix(self, model: str, prompt: str, *, suffix: str, **kwargs: Any) -> Dict[str, Any]:
        return {"ok": False, "error": "llama.cpp suffix generation is unsupported", "text": ""}

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        raise BackendCapabilityError("llama.cpp embeddings are unsupported by this endpoint")

    def unload(self, model: str) -> None:
        raise BackendCapabilityError("llama.cpp model unload is unsupported")

    def flush_all(self) -> None:
        raise BackendCapabilityError("llama.cpp flush-all is unsupported")


_CAPABILITIES = BackendCapabilities({
    BackendCapability.INVENTORY: CapabilityStatus.SUPPORTED, BackendCapability.VERSION: CapabilityStatus.SUPPORTED,
    BackendCapability.MODEL_METADATA: CapabilityStatus.SUPPORTED, BackendCapability.CHAT: CapabilityStatus.SUPPORTED,
    BackendCapability.NATIVE_TOOLS: CapabilityStatus.SUPPORTED, BackendCapability.SUFFIX_GENERATION: CapabilityStatus.UNSUPPORTED,
    BackendCapability.EMBEDDINGS: CapabilityStatus.UNSUPPORTED, BackendCapability.LOADED_MODEL_STATS: CapabilityStatus.SUPPORTED,
    BackendCapability.OFFLOAD_FRACTION: CapabilityStatus.UNAVAILABLE, BackendCapability.MODEL_UNLOAD: CapabilityStatus.UNSUPPORTED,
    BackendCapability.FLUSH_ALL: CapabilityStatus.UNSUPPORTED, BackendCapability.OLLAMA_SERVICE_REPAIR: CapabilityStatus.UNSUPPORTED,
    BackendCapability.OLLAMA_KV_REPAIR: CapabilityStatus.UNSUPPORTED,
})


class LlamaCppBackendAdapter:
    """Protocol adapter for one externally managed llama-server endpoint."""
    def __init__(self, client: LlamaCppClient): self.client = client
    def backend_identity(self) -> BackendIdentity: return BackendIdentity("llama_cpp", "llama.cpp", self.client.base)
    def backend_capabilities(self) -> BackendCapabilities:
        states = dict(_CAPABILITIES.states)
        try:
            states[BackendCapability.NATIVE_TOOLS] = (
                CapabilityStatus.SUPPORTED if "tools" in self.client.capabilities(self.client.tags()[0]["name"])
                else CapabilityStatus.UNSUPPORTED
            )
        except LlamaCppError:
            states[BackendCapability.NATIVE_TOOLS] = CapabilityStatus.UNAVAILABLE
        return BackendCapabilities(states)
    def tags(self) -> List[Dict[str, Any]]: return self.client.tags()
    def version(self) -> Optional[str]: return self.client.version()
    def show(self, model: str) -> Dict[str, Any]: return self.client.show(model)
    def capabilities(self, model: str) -> List[str]: return self.client.capabilities(model)
    def supports_thinking(self, model: str) -> bool: return self.client.supports_thinking(model)
    def model_info(self, model: str) -> Dict[str, Any]: return self.client.model_info(model)
    def model_size_bytes(self, model: str) -> Optional[int]: return self.client.model_size_bytes(model)
    def context_length(self, model: str) -> Optional[int]: return self.client.context_length(model)
    def chat(self, model: str, prompt: str, **kwargs: Any) -> Dict[str, Any]: return self.client.chat(model, prompt, **kwargs)
    def chat_tools(self, model: str, prompt: str, **kwargs: Any) -> Dict[str, Any]: return self.client.chat_tools(model, prompt, **kwargs)
    def generate_suffix(self, model: str, prompt: str, **kwargs: Any) -> Dict[str, Any]: return self.client.generate_suffix(model, prompt, **kwargs)
    def embed(self, model: str, texts: List[str]) -> List[List[float]]: return self.client.embed(model, texts)
    def loaded_model_stats(self, model: str) -> Optional[Dict[str, Any]]: return self.client.loaded_model_stats(model)
    def offload_fraction(self, model: str, exact: bool = True) -> Optional[float]: return self.client.offload_fraction(model, exact=exact)
    def unload(self, model: str) -> None: self.client.unload(model)
    def flush_all(self) -> None: self.client.flush_all()
    def tokenize(self, content: str, **kwargs: Any) -> Dict[str, Any]: return self.client.tokenize(content, **kwargs)
    def slots(self) -> List[Dict[str, Any]]: return self.client.slots()
    def metrics(self) -> Dict[str, Any]: return self.client.metrics()
