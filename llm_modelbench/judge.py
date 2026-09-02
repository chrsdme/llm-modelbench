"""Local judge helpers for subjective tasks.

The judge is a screening aid, not a final truth source. V9.5.8 makes judge output
structured and diagnosable: no silent fallback score, thinking blocks are stripped,
and invalid judge output returns (None, "judge_error: ...") instead of a fake 50.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional, Tuple

from .backend import InferenceClient
from .scoring import strip_thinking

# NOTE (Anvil Stage 3.5D): ANCHORS and JUDGE_PANEL_PERSONAS are judge-request
# *scoring semantics* — they are embedded verbatim in every judge request and a
# material edit shifts judged scores for the same model output. Any such edit
# MUST be accompanied by a bump of JUDGE_REQUEST_CONTRACT_VERSION (the shared
# algorithm/interpretation version), and is picked up automatically by
# JUDGE_ANCHOR_POLICY_HASH below, which is recorded on every newly written
# judge_results.jsonl entry as `judge_anchor_policy_hash`. Comments around these
# constants are display-only and do not move the hash.
ANCHORS = """
Use these anchors:
- 30 = incomplete, vague, wrong structure, or materially inaccurate.
- 60 = usable but generic, misses important details or constraints.
- 90 = accurate, structured, practical, concise, and follows all constraints.
""".strip()

# Panel-judge persona instructions — request material, same class as ANCHORS.
# Hoisted from judge_panel_result so it is a hashable module constant.
JUDGE_PANEL_PERSONAS = (
    "strict correctness judge: penalize factual errors and missed constraints",
    "pragmatic usefulness judge: reward usable, actionable, well-structured answers",
    "clarity judge: reward concise, readable writing and clear organization",
)

JUDGE_REQUEST_CONTRACT_VERSION = "posthoc-judge-request-v1"


def _judge_anchor_policy_hash() -> str:
    """Deterministic canonical identity of the materially scoring-relevant judge
    request semantics (Anvil Stage 3.5D).

    Mixes the shared algorithm/interpretation version
    (JUDGE_REQUEST_CONTRACT_VERSION) with the concrete scoring anchors and panel
    personas. A material change to any of the three moves the hash; a
    display-only/comment change around them does not (the exact constant values
    are hashed, nothing else). Uses the project's stable-hash idiom
    (sorted-keys, compact separators) — see judge_dumps._compatibility_fingerprint.
    """
    payload = {
        "judge_request_contract_version": JUDGE_REQUEST_CONTRACT_VERSION,
        "anchors": ANCHORS,
        "panel_personas": list(JUDGE_PANEL_PERSONAS),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


JUDGE_ANCHOR_POLICY_HASH = _judge_anchor_policy_hash()


def _extract_json(text: str) -> Optional[dict]:
    text = strip_thinking(text or "")
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def _parse_score(text: str) -> Tuple[Optional[float], str]:
    data = _extract_json(text)
    if not isinstance(data, dict):
        return None, "judge_error: invalid json"
    score = data.get("score")
    try:
        score_f = float(score)
    except Exception:
        return None, "judge_error: missing numeric score"
    if not (0 <= score_f <= 100):
        return None, f"judge_error: score out of range {score_f}"
    confidence = data.get("confidence")
    verdict = str(data.get("verdict") or data.get("justification") or "").strip()
    reason = f"judge_json score={round(score_f, 2)}"
    if confidence is not None:
        reason += f" confidence={confidence}"
    if verdict:
        reason += f" verdict={verdict[:160]}"
    return round(score_f, 2), reason


def _backend_failure_disposition(response: dict) -> str:
    kind = str(response.get("error_kind") or response.get("kind") or "").lower()
    if kind in {"timeout", "timed_out"}:
        return "timeout"
    if kind in {"structural_incompatibility", "request_incompatibility", "invalid_request"}:
        return "structural_incompatibility"
    status = response.get("http_status")
    try:
        status_i = int(status) if status is not None else 0
    except (TypeError, ValueError):
        status_i = 0
    if status_i == 408:
        return "timeout"
    if status_i == 429:
        return "transient_backend_failure"
    if 500 <= status_i <= 599:
        return "transient_backend_failure"
    if status_i in {400, 404, 405, 415, 422}:
        return "structural_incompatibility"
    if kind in {"backend_failure", "transport_failure", "connection_error", "rate_limited", "transient_backend_failure"}:
        return "transient_backend_failure"
    return "backend_failure"


def _failure_result(reason: str, disposition: str, *, backend_response: Optional[dict] = None) -> dict:
    backend_response = backend_response or {}
    return {
        "score": None,
        "reason": reason,
        "failure_disposition": disposition,
        "backend": {
            "ok": bool(backend_response.get("ok")),
            "error": backend_response.get("error"),
            "error_kind": backend_response.get("error_kind") or backend_response.get("kind"),
            "http_status": backend_response.get("http_status"),
            "failure_disposition": disposition,
        },
        "request_contract_version": JUDGE_REQUEST_CONTRACT_VERSION,
    }


def _success_result(score: float, reason: str) -> dict:
    return {
        "score": score,
        "reason": reason,
        "failure_disposition": None,
        "backend": {"ok": True},
        "request_contract_version": JUDGE_REQUEST_CONTRACT_VERSION,
    }


def judge_single(client: InferenceClient, judge_model: str, prompt: str, output: str, rubric: str, *, num_ctx=None, think="auto"):
    result = judge_single_result(client, judge_model, prompt, output, rubric, num_ctx=num_ctx, think=think)
    return result.get("score"), result.get("reason")


def judge_single_result(client: InferenceClient, judge_model: str, prompt: str, output: str, rubric: str, *, num_ctx=None, think="auto") -> dict:
    judge_prompt = f"""You are grading a model answer for a benchmark.

Rubric: {rubric}
{ANCHORS}

Original task:
{prompt}

Model answer:
{output}

Return ONLY valid JSON with this schema:
{{"score": <number 0-100>, "confidence": <number 0-1>, "verdict": "brief reason"}}
Do not include markdown, prose outside JSON, or hidden reasoning.
"""
    try:
        res = client.chat(
            judge_model,
            judge_prompt,
            system="Grade strictly. Return only the requested JSON. Do not include chain of thought.",
            num_predict=1024,
            num_ctx=num_ctx,
            think=think,
        )
    except TimeoutError as exc:
        return _failure_result(f"judge_error: {exc}", "timeout", backend_response={"ok": False, "error": str(exc), "error_kind": "timeout"})
    if not res.get("ok"):
        disposition = _backend_failure_disposition(res)
        return _failure_result(f"judge_error: {res.get('error', 'failed')}", disposition, backend_response=res)
    score, reason = _parse_score(res.get("text") or "")
    if isinstance(score, (int, float)):
        return _success_result(score, reason)
    return _failure_result(reason, "judge_output_failure", backend_response={"ok": True})


def judge_panel(client: InferenceClient, judge_model: str, prompt: str, output: str, rubric: str, *, num_ctx=None, think="auto"):
    result = judge_panel_result(client, judge_model, prompt, output, rubric, num_ctx=num_ctx, think=think)
    return result.get("score"), result.get("reason")


def judge_panel_result(client: InferenceClient, judge_model: str, prompt: str, output: str, rubric: str, *, num_ctx=None, think="auto") -> dict:
    personas = list(JUDGE_PANEL_PERSONAS)
    scores = []
    reasons = []
    failures = []
    for persona in personas:
        try:
            res = client.chat(
                judge_model,
                f"Persona: {persona}\n\nRubric: {rubric}\n{ANCHORS}\n\nTask:\n{prompt}\n\nAnswer:\n{output}\n\nReturn ONLY JSON: {{\"score\": <0-100>, \"confidence\": <0-1>, \"verdict\": \"brief reason\"}}",
                system="Grade strictly. Return only JSON. Do not include chain of thought.",
                num_predict=1024,
                num_ctx=num_ctx,
                think=think,
            )
        except TimeoutError as exc:
            res = {"ok": False, "error": str(exc), "error_kind": "timeout"}
        if not res.get("ok"):
            disposition = _backend_failure_disposition(res)
            failures.append(disposition)
            reasons.append(f"judge_error: {res.get('error', 'failed')}")
            continue
        score, reason = _parse_score(res.get("text") or "")
        if isinstance(score, (int, float)):
            scores.append(float(score))
        else:
            failures.append("judge_output_failure")
        reasons.append(reason)
    if not scores:
        disposition = failures[0] if failures and all(failure == failures[0] for failure in failures) else "backend_failure"
        if failures and all(failure == "judge_output_failure" for failure in failures):
            disposition = "judge_output_failure"
        return _failure_result("judge_error: panel produced no valid scores; " + "; ".join(reasons[:3]), disposition)
    scores.sort()
    mid = scores[len(scores)//2] if len(scores) % 2 else (scores[len(scores)//2 - 1] + scores[len(scores)//2]) / 2
    spread = max(scores) - min(scores) if len(scores) > 1 else 0.0
    return _success_result(round(mid, 2), f"panel_median={round(mid,2)} spread={round(spread,2)} scores={','.join(str(round(s,1)) for s in scores)}")
