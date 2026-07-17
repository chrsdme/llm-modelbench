"""Capability interrogation and evidence-aware task routing.

Metadata is cheap and always collected. Actual ``run`` commands perform small
functional probes by default before routing scored lanes; ``plan`` remains
metadata-only unless ``--auto`` is explicit. Probes never execute a proposed
tool call, and ``--no-auto-probe`` is available for deliberate metadata-only runs.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Dict, Iterable, List, Optional

from . import media
from .classify import (
    CODE_HINTS,
    FAMILY_ORDER,
    families_for,
    families_from_capabilities,
    hinted_families,
    profile_for,
)


def _normalise_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _probe_result(
    ok: bool, *, detail: str = "", elapsed: float = 0.0,
    error: Optional[str] = None, responded: Optional[bool] = None,
) -> Dict[str, Any]:
    return {
        "ok": bool(ok),
        "responded": bool(ok if responded is None else responded),
        "detail": detail[:500],
        "elapsed_seconds": round(float(elapsed), 3),
        "error": error,
    }


def _exception_detail(exc: Exception) -> str:
    """Preserve HTTP status and response body for capability decisions.

    ``repr(HTTPError)`` discards the server's actual reason, which previously
    turned definitive unsupported-build responses into ambiguous probe failures.
    """
    parts = [repr(exc)]
    code = getattr(exc, "code", None)
    reason = getattr(exc, "reason", None)
    if code is not None:
        parts.append(f"http_status={code}")
    if reason:
        parts.append(f"http_reason={reason}")
    try:
        body = exc.read() if hasattr(exc, "read") else b""
        if isinstance(body, bytes):
            body = body.decode(errors="replace")
        if body:
            parts.append(f"response_body={str(body)[:1000]}")
    except Exception:
        pass
    return "; ".join(parts)


def _probe_text(client: Any, model: str) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        res = client.chat(model, "Return exactly AIW_TEXT_OK and nothing else.", num_predict=16, think="off")
        text = str(res.get("text") or "")
        return _probe_result(bool(res.get("ok")) and "aiwtextok" in _normalise_text(text),
                             detail=text, elapsed=time.perf_counter() - started,
                             error=None if res.get("ok") else str(res.get("error") or "chat failed"),
                             responded=bool(res.get("ok")))
    except Exception as exc:
        return _probe_result(False, elapsed=time.perf_counter() - started, error=_exception_detail(exc))


def _probe_vision(client: Any, model: str) -> Dict[str, Any]:
    token = "V7K9Q2"
    image = media.render_text_png(token, noisy=False, seed=42)
    if not image:
        return _probe_result(False, error="Pillow unavailable; vision functional probe not run")
    started = time.perf_counter()
    try:
        # The token deliberately appears only in the image, never in the prompt.
        res = client.chat(model, "Read the short code shown in the image. Return only that code.",
                          images=[image], num_predict=32, think="off")
        text = str(res.get("text") or "")
        return _probe_result(bool(res.get("ok")) and token.lower() in _normalise_text(text),
                             detail=text, elapsed=time.perf_counter() - started,
                             error=None if res.get("ok") else str(res.get("error") or "vision chat failed"),
                             responded=bool(res.get("ok")))
    except Exception as exc:
        return _probe_result(False, elapsed=time.perf_counter() - started, error=_exception_detail(exc))


def _probe_embedding(client: Any, model: str) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        vecs = client.embed(model, ["alpha", "beta"])
        good = (
            isinstance(vecs, list) and len(vecs) == 2 and
            all(isinstance(v, list) and len(v) > 0 for v in vecs) and
            len(vecs[0]) == len(vecs[1]) and
            all(isinstance(x, (int, float)) for v in vecs for x in v[:32])
        )
        detail = f"vectors={len(vecs) if isinstance(vecs, list) else 0} dims={len(vecs[0]) if good else 0}"
        return _probe_result(good, detail=detail, elapsed=time.perf_counter() - started, responded=True)
    except Exception as exc:
        return _probe_result(False, elapsed=time.perf_counter() - started, error=_exception_detail(exc))


def _probe_tools(client: Any, model: str) -> Dict[str, Any]:
    if not hasattr(client, "chat_tools"):
        return _probe_result(False, error="client has no native tool-call method")
    started = time.perf_counter()
    tool = {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Look up weather for one city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city", "units"],
            },
        },
    }
    try:
        res = client.chat_tools(
            model,
            "Use lookup_weather for Paris in celsius. Do not answer from memory.",
            tools=[tool],
            num_predict=128,
            think="off",
        )
        calls = res.get("tool_calls") or []
        good = False
        detail = json.dumps(calls, sort_keys=True, default=str)
        for call in calls:
            fn = call.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if fn.get("name") == "lookup_weather" and str(args.get("city", "")).lower() == "paris" and args.get("units") == "celsius":
                good = True
                break
        return _probe_result(bool(res.get("ok")) and good, detail=detail,
                             elapsed=time.perf_counter() - started,
                             error=None if res.get("ok") else str(res.get("error") or "tool call failed"),
                             responded=bool(res.get("ok")))
    except Exception as exc:
        return _probe_result(False, elapsed=time.perf_counter() - started, error=_exception_detail(exc))


def _probe_insert(client: Any, model: str) -> Dict[str, Any]:
    """Task-equivalent FIM probe using the real held-out suffix contract.

    Ollama metadata can advertise ``insert`` even when the installed model
    template/build cannot satisfy suffix-conditioned generation.  This probe
    therefore uses the same fixture shape and hidden assertion as the scored
    ``fim_suffix_assertion`` task, rather than a loose capability label.
    """
    if not hasattr(client, "generate_suffix"):
        return _probe_result(False, error="client has no suffix/FIM method")
    from .tasks import TASKS
    from . import sandbox, scoring

    task = next((t for t in TASKS if t.id == "fim_suffix_assertion"), None)
    if task is None:
        return _probe_result(False, error="fim_suffix_assertion fixture unavailable")
    started = time.perf_counter()
    try:
        res = client.generate_suffix(
            model, task.prompt, suffix=str(task.meta.get("suffix") or ""),
            num_predict=int(task.num_predict),
        )
        if not res.get("ok"):
            return _probe_result(False, elapsed=time.perf_counter() - started,
                                 error=str(res.get("error") or "suffix generation failed"))
        insertion = str(res.get("text") or "").strip()
        fenced = scoring.extract_blocks(insertion, "python", include_raw=False)
        if fenced:
            insertion = fenced[0].strip()
        score, reason = sandbox.run_python_checks(
            str(task.prompt) + insertion + str(task.meta.get("suffix") or ""),
            [""], timeout=10,
        )
        detail = json.dumps({"output": insertion[:500], "score": score, "reason": reason}, sort_keys=True)
        return _probe_result(bool(score >= 100.0), detail=detail, elapsed=time.perf_counter() - started,
                             error=None if score >= 100.0 else "task-equivalent FIM assertion failed",
                             responded=True)
    except Exception as exc:
        return _probe_result(False, elapsed=time.perf_counter() - started, error=_exception_detail(exc))


def _probe_state(result: Dict[str, Any]) -> str:
    if result.get("ok"):
        return "confirmed_supported"
    text = " ".join(str(result.get(k) or "") for k in ("error", "detail")).lower()
    definitive = any(token in text for token in (
        "unsupported", "does not support", "not supported", "missing projector",
        "mmproj", "capability unavailable", "invalid option",
    ))
    if definitive:
        return "confirmed_unavailable"
    if result.get("responded"):
        # The endpoint/lane responded, but the tiny probe did not satisfy its
        # semantic contract. Route the real scored task so quality failure is
        # measured rather than hidden as a capability absence.
        return "responded_contract_failed"
    transient = any(token in text for token in (
        "timeout", "timed out", "connection reset", "connection refused",
        "temporarily unavailable", "http 429", "http 500", "http 502",
        "http 503", "http 504",
    ))
    return "transient_failure" if transient else "probe_failed"


def interrogate_model(
    client: Any, model: str, *, functional: bool = False,
    probe_families: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Collect metadata and optionally run small capability probes.

    Failed functional probes remove only weak name-hint routes. Explicit
    operator profiles and Ollama-declared capabilities remain routed so that a
    real benchmark task records the failure instead of silently hiding it.
    """
    declared = client.capabilities(model) if hasattr(client, "capabilities") else []
    declared = [str(c).lower() for c in declared or []]
    profile = profile_for(model)
    declared_families = families_from_capabilities(declared)
    name_families = hinted_families(model)
    profile_families = list(profile.get("families") or [])

    sources: Dict[str, List[str]] = {}
    for family in profile_families:
        sources.setdefault(family, []).append("operator_profile")
    for family in declared_families:
        sources.setdefault(family, []).append("ollama_metadata")
    if not profile_families and not declared_families:
        for family in name_families:
            sources.setdefault(family, []).append("name_hint")

    initial = families_for(model, declared)
    probes: Dict[str, Dict[str, Any]] = {}
    positive: List[str] = []
    warnings: List[str] = []
    requested_probes = (
        {str(family) for family in probe_families}
        if probe_families is not None else None
    )

    def wants(family: str) -> bool:
        return requested_probes is None or family in requested_probes

    if functional:
        if "text" in initial and wants("text"):
            probes["text"] = _probe_text(client, model)
            if probes["text"]["ok"]:
                positive.append("text")
        if "embedding" in initial and wants("embedding"):
            probes["embedding"] = _probe_embedding(client, model)
            if probes["embedding"]["ok"]:
                positive.append("embedding")
        # Probe vision when it is declared/configured *or* conservatively
        # suggested by the model name.  Ollama community GGUF conversions can
        # return partial metadata such as ["completion"] and omit ``vision``;
        # requiring vision to already be present in ``initial`` made --auto
        # unable to discover exactly that failure mode.
        if wants("vision") and ("vision" in initial or "vision" in name_families):
            probes["vision"] = _probe_vision(client, model)
            if probes["vision"]["ok"]:
                positive.extend(("vision", "text"))
        # Native tool calling can be useful even when Ollama omitted the flag.
        if "text" in initial and wants("tools"):
            probes["tools"] = _probe_tools(client, model)
            if probes["tools"]["ok"]:
                positive.extend(("tools", "text"))
        # FIM probing is restricted to declared insert models and coding-like names
        # to avoid two unnecessary generations for every general chat model.
        n = model.lower()
        if wants("insert") and ("insert" in initial or any(hint in n for hint in CODE_HINTS)):
            probes["insert"] = _probe_insert(client, model)
            if probes["insert"]["ok"]:
                positive.append("insert")

    supported = families_for(model, declared, positive)
    unavailable: List[str] = []
    unverified: List[str] = []
    probe_states: Dict[str, str] = {}

    if functional:
        # Functional evidence is authoritative for routing. Metadata/name hints
        # remain visible as declarations, but a lane is not sent into the full
        # scored suite until its probe responds successfully.
        probed_families = set(probes)
        supported = [family for family in supported if family not in probed_families]
        for family, result in probes.items():
            state = _probe_state(result)
            probe_states[family] = state
            if state in {"confirmed_supported", "responded_contract_failed"}:
                supported.append(family)
                sources.setdefault(family, []).append(
                    "functional_probe" if state == "confirmed_supported" else "functional_response"
                )
                if state == "responded_contract_failed":
                    warnings.append(f"routed {family}: endpoint responded, but probe contract failed; scored task will measure quality")
            elif state == "confirmed_unavailable":
                unavailable.append(family)
                warnings.append(f"removed {family}: functional probe confirmed the installed build cannot serve this lane")
            else:
                unverified.append(family)
                warnings.append(f"withheld {family}: functional probe failed without definitive capability evidence")

    supported = [f for f in FAMILY_ORDER if f in set(supported)]
    if not supported:
        warnings.append("no capability lane survived interrogation")

    payload = {
        "model": model,
        "declared_capabilities": declared,
        "profile": profile,
        "initial_families": initial,
        "supported_families": supported,
        "sources": sources,
        "functional_probes_enabled": bool(functional),
        "requested_probe_families": sorted(requested_probes) if requested_probes is not None else None,
        "probes": probes,
        "probe_states": probe_states,
        "capability_decisions": {
            family: {
                "sources": list(sources.get(family) or []),
                "probe_state": probe_states.get(family, "not_probed"),
                "route_scored_tasks": family in supported,
            }
            for family in FAMILY_ORDER
            if family in set(initial) | set(probes) | set(supported) | set(unavailable) | set(unverified)
        },
        "confirmed_unavailable_families": [f for f in FAMILY_ORDER if f in set(unavailable)],
        "unverified_families": [f for f in FAMILY_ORDER if f in set(unverified)],
        "routing_policy": "functional_probe_required" if functional else "metadata_only",
        "warnings": warnings,
    }
    payload["evidence_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return payload


def interrogate_models(
    client: Any, models: Iterable[str], *, functional: bool = False,
    probe_families: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    return {
        model: interrogate_model(
            client, model, functional=functional, probe_families=probe_families
        )
        for model in models
    }
