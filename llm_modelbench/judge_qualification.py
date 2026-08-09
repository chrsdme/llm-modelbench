"""Universal ModelBench judge qualification protocol.

This module is deliberately model-agnostic.  Selene, Prometheus, or any future
judge candidate must satisfy the same ModelBench-owned synthetic controls
before campaign post-hoc judging uses it.  The protocol is separate from live
campaign judging and is fully testable with fake backends.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

PROTOCOL_VERSION = "judge-qualification-v1"

SCORE_SCHEMA = {
    "score": "number 0-100",
    "confidence": "number 0-1",
    "verdict": "short reason",
    "rubric_adherence": "boolean",
    "reference_used": "boolean",
}
PAIRWISE_SCHEMA = {
    "winner": "A|B|equal",
    "confidence": "number 0-1",
    "verdict": "short reason",
}


@dataclass(frozen=True)
class QualificationControl:
    control_id: str
    mode: str
    task_prompt: str
    candidate_answer: str = ""
    reference_answer: str = ""
    rubric: str = ""
    expected: Dict[str, Any] | None = None
    answer_a: str = ""
    answer_b: str = ""


@dataclass(frozen=True)
class QualificationRequest:
    protocol_version: str
    control_id: str
    mode: str
    task_prompt: str
    rubric: str
    reference_answer: str
    candidate_answer: str = ""
    answer_a: str = ""
    answer_b: str = ""
    response_schema: Dict[str, str] | None = None

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


def qualification_controls() -> List[QualificationControl]:
    """Deterministic synthetic qualification controls owned by ModelBench."""
    return [
        QualificationControl(
            "obviously_correct",
            "score",
            "What is 2 + 2?",
            candidate_answer="2 + 2 = 4.",
            reference_answer="4",
            rubric="Reward exact correctness. Penalize any answer other than 4.",
            expected={"score_min": 80},
        ),
        QualificationControl(
            "obviously_wrong",
            "score",
            "What is 2 + 2?",
            candidate_answer="2 + 2 = 5.",
            reference_answer="4",
            rubric="Reward exact correctness. Penalize any answer other than 4.",
            expected={"score_max": 30},
        ),
        QualificationControl(
            "partial_credit",
            "score",
            "Name two safety checks before deleting generated files.",
            candidate_answer="Check the target path, but I forgot to mention preserving unrelated files.",
            reference_answer="Confirm the exact target and preserve unrelated files.",
            rubric="Award partial credit for one valid safety check; full credit requires both.",
            expected={"score_min": 40, "score_max": 75},
        ),
        QualificationControl(
            "irrelevant_answer",
            "score",
            "Name the capital of France.",
            candidate_answer="A good benchmark should be reproducible.",
            reference_answer="Paris",
            rubric="Score only the answer to the asked question.",
            expected={"score_max": 35},
        ),
        QualificationControl(
            "unsupported_hallucination",
            "score",
            "According to the reference, which library is approved?",
            candidate_answer="The approved library is HyperNova, and it is the fastest option.",
            reference_answer="The reference says no library has been approved.",
            rubric="Penalize unsupported claims and reward faithful use of the reference.",
            expected={"score_max": 35, "reference_used": True},
        ),
        QualificationControl(
            "malformed_candidate_output",
            "score",
            "Return valid JSON with key status.",
            candidate_answer="{status: ok,,,",
            reference_answer='{"status":"ok"}',
            rubric="Penalize malformed candidate output even if intent is guessable.",
            expected={"score_max": 35},
        ),
        QualificationControl(
            "rubric_adherence",
            "score",
            "Answer with exactly one word: yes or no. Is water wet?",
            candidate_answer="Yes. Water is wet because liquid molecules adhere to surfaces.",
            reference_answer="yes",
            rubric="The answer must be exactly one word. Penalize extra explanation despite factual correctness.",
            expected={"score_max": 60, "rubric_adherence": True},
        ),
        QualificationControl(
            "reference_answer_use",
            "score",
            "Use the reference only: what port is configured?",
            candidate_answer="The configured port is 8080.",
            reference_answer="The configured port is 9090.",
            rubric="The reference answer is authoritative. Penalize plausible but reference-conflicting answers.",
            expected={"score_max": 35, "reference_used": True},
        ),
        QualificationControl(
            "pair_equal",
            "pairwise",
            "Pick the better answer.",
            rubric="Choose equal when both answers are equally correct and complete.",
            reference_answer="Paris",
            answer_a="Paris.",
            answer_b="Paris.",
            expected={"winner": "equal"},
        ),
        QualificationControl(
            "pair_a_better",
            "pairwise",
            "Pick the better answer.",
            rubric="Prefer the answer that matches the reference.",
            reference_answer="Paris",
            answer_a="Paris.",
            answer_b="Lyon.",
            expected={"winner": "A"},
        ),
        QualificationControl(
            "pair_b_better",
            "pairwise",
            "Pick the better answer.",
            rubric="Prefer the answer that matches the reference.",
            reference_answer="4",
            answer_a="5",
            answer_b="4",
            expected={"winner": "B"},
        ),
    ]


def _request_for(control: QualificationControl) -> QualificationRequest:
    return QualificationRequest(
        protocol_version=PROTOCOL_VERSION,
        control_id=control.control_id,
        mode=control.mode,
        task_prompt=control.task_prompt,
        rubric=control.rubric,
        reference_answer=control.reference_answer,
        candidate_answer=control.candidate_answer,
        answer_a=control.answer_a,
        answer_b=control.answer_b,
        response_schema=SCORE_SCHEMA if control.mode == "score" else PAIRWISE_SCHEMA,
    )


def _reversed_pair(control: QualificationControl) -> QualificationControl:
    expected = dict(control.expected or {})
    winner = expected.get("winner")
    expected["winner"] = {"A": "B", "B": "A"}.get(winner, winner)
    return QualificationControl(
        f"{control.control_id}_reversed",
        "pairwise",
        control.task_prompt,
        rubric=control.rubric,
        reference_answer=control.reference_answer,
        answer_a=control.answer_b,
        answer_b=control.answer_a,
        expected=expected,
    )


def _parse_strict_json(text: str) -> Optional[dict]:
    raw = str(text or "").strip()
    if not raw or not raw.startswith("{") or not raw.endswith("}"):
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _classify_backend_failure(response: Dict[str, Any]) -> str:
    kind = str(response.get("error_kind") or response.get("kind") or "").lower()
    if kind in {"timeout", "timed_out"}:
        return "timeout"
    if kind in {"unsupported_backend", "unsupported"}:
        return "unsupported_backend"
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
    if kind in {"backend_failure", "transport_failure", "connection_error", "transient_backend_failure", "rate_limited"}:
        return "transient_backend_failure"
    return "transient_backend_failure"


def _render_prompt(request: QualificationRequest) -> str:
    return (
        "MODELbench judge qualification request.\n"
        "Return only JSON matching response_schema.\n"
        + json.dumps(request.to_payload(), sort_keys=True)
    )


def _call_backend(client: Any, model: str, request: QualificationRequest, *, timeout_seconds: float) -> Dict[str, Any]:
    if not hasattr(client, "chat"):
        return {"ok": False, "error": "client has no chat method", "error_kind": "unsupported_backend"}
    try:
        return client.chat(
            model,
            _render_prompt(request),
            system="You are being qualified as a benchmark judge. Return only the requested JSON.",
            num_predict=512,
            think="off",
            timeout=timeout_seconds,
        )
    except TimeoutError as exc:
        return {"ok": False, "error": str(exc), "error_kind": "timeout"}
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "error_kind": "transient_backend_failure"}


def _schema_violation(message: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": False, "disposition": "schema_violation", "error": message, "parsed": parsed}


def _required_number(data: Dict[str, Any], field: str, *, low: float, high: float) -> tuple[Optional[float], Optional[Dict[str, Any]]]:
    if field not in data:
        return None, _schema_violation(f"missing_{field}", data)
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, _schema_violation(f"{field}_must_be_numeric", data)
    number = float(value)
    if not math.isfinite(number):
        return None, _schema_violation(f"{field}_must_be_finite_number", data)
    if not low <= number <= high:
        disposition = "score_out_of_range" if field == "score" else "schema_violation"
        return None, {"ok": False, "disposition": disposition, "error": f"{field}_out_of_range:{number}", "parsed": data}
    return round(number, 2), None


def _required_nonempty_string(data: Dict[str, Any], field: str) -> Optional[Dict[str, Any]]:
    if field not in data:
        return _schema_violation(f"missing_{field}", data)
    if not isinstance(data.get(field), str) or not str(data.get(field) or "").strip():
        return _schema_violation(f"{field}_must_be_nonempty_string", data)
    return None


def _required_bool(data: Dict[str, Any], field: str) -> Optional[Dict[str, Any]]:
    if field not in data:
        return _schema_violation(f"missing_{field}", data)
    if not isinstance(data.get(field), bool):
        return _schema_violation(f"{field}_must_be_boolean", data)
    return None


def _parse_response(request: QualificationRequest, response: Dict[str, Any]) -> Dict[str, Any]:
    if not response.get("ok"):
        disposition = _classify_backend_failure(response)
        return {
            "ok": False,
            "disposition": disposition,
            "error": response.get("error") or disposition,
            "parsed": None,
        }
    text = str(response.get("text") or "")
    data = _parse_strict_json(text)
    if not isinstance(data, dict):
        return {"ok": False, "disposition": "malformed_judge_output", "error": "invalid_json", "parsed": None}
    if request.mode == "score":
        score, violation = _required_number(data, "score", low=0, high=100)
        if violation:
            return violation
        confidence, violation = _required_number(data, "confidence", low=0, high=1)
        if violation:
            return violation
        for field in ("verdict",):
            violation = _required_nonempty_string(data, field)
            if violation:
                return violation
        for field in ("rubric_adherence", "reference_used"):
            violation = _required_bool(data, field)
            if violation:
                return violation
        data["score"] = score
        data["confidence"] = confidence
    else:
        violation = _required_nonempty_string(data, "winner")
        if violation:
            return violation
        confidence, violation = _required_number(data, "confidence", low=0, high=1)
        if violation:
            return violation
        violation = _required_nonempty_string(data, "verdict")
        if violation:
            return violation
        winner = str(data.get("winner") or "").strip().lower()
        normalised = {"a": "A", "b": "B", "equal": "equal"}.get(winner)
        if normalised is None:
            return {"ok": False, "disposition": "schema_violation", "error": "invalid_pairwise_winner", "parsed": data}
        data["winner"] = normalised
        data["confidence"] = confidence
    return {"ok": True, "disposition": "parsed", "error": None, "parsed": data}


def _evaluate_control(control: QualificationControl, parsed: Dict[str, Any]) -> Dict[str, Any]:
    expected = control.expected or {}
    failures: List[str] = []
    if control.mode == "score":
        score = float(parsed["score"])
        if "score_min" in expected and score < float(expected["score_min"]):
            failures.append("score_below_expected")
        if "score_max" in expected and score > float(expected["score_max"]):
            failures.append("score_above_expected")
        if expected.get("rubric_adherence") and parsed.get("rubric_adherence") is not True:
            failures.append("rubric_adherence_not_signaled")
        if expected.get("reference_used") and parsed.get("reference_used") is not True:
            failures.append("reference_use_not_signaled")
    else:
        if parsed.get("winner") != expected.get("winner"):
            failures.append("pairwise_winner_mismatch")
    return {"passed": not failures, "failures": failures}


def _control_result(control: QualificationControl, request: QualificationRequest, response: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "control_id": control.control_id,
        "mode": control.mode,
        "request": request.to_payload(),
        "backend_ok": bool(response.get("ok")),
        "raw_response": str(response.get("text") or response.get("error") or "")[:500],
        "parse": parsed,
        "passed": False,
        "failures": [],
    }
    if parsed.get("ok"):
        evaluation = _evaluate_control(control, parsed["parsed"])
        result["passed"] = evaluation["passed"]
        result["failures"] = evaluation["failures"]
    else:
        result["failures"] = [str(parsed.get("disposition") or "parse_failed")]
    return result


def qualify_candidate(
    client: Any,
    candidate: Dict[str, Any],
    *,
    repeats: int = 3,
    timeout_seconds: float = 30.0,
) -> Dict[str, Any]:
    """Run the complete ModelBench judge qualification protocol."""
    started = datetime.now(timezone.utc)
    model = str(candidate.get("name") or candidate.get("model") or "")
    controls = qualification_controls()
    control_results: List[Dict[str, Any]] = []
    failure_reasons: List[str] = []
    checks: Dict[str, Any] = {
        "backend_request_compatible": True,
        "structured_output": True,
        "parser_schema_compliance": True,
        "score_range": True,
        "good_bad_discrimination": False,
        "partial_credit_sanity": False,
        "rubric_adherence": False,
        "reference_answer_use": False,
        "pairwise_reversal_order_bias": False,
        "repeat_stability": False,
        "malformed_candidate_output_handling": False,
        "timeout": False,
        "structural_incompatibility": False,
        "unsupported_backend": False,
        "transient_backend_failure": False,
    }
    parsed_by_id: Dict[str, Dict[str, Any]] = {}

    def run(control: QualificationControl) -> Optional[Dict[str, Any]]:
        request = _request_for(control)
        response = _call_backend(client, model, request, timeout_seconds=timeout_seconds)
        parsed = _parse_response(request, response)
        result = _control_result(control, request, response, parsed)
        control_results.append(result)
        if not parsed.get("ok"):
            disposition = str(parsed.get("disposition"))
            if disposition == "structural_incompatibility":
                checks["structural_incompatibility"] = True
                checks["backend_request_compatible"] = False
                failure_reasons.append("structural_incompatibility")
                return None
            if disposition == "unsupported_backend":
                checks["unsupported_backend"] = True
                checks["backend_request_compatible"] = False
                failure_reasons.append("unsupported_backend")
                return None
            if disposition == "timeout":
                checks["timeout"] = True
                checks["backend_request_compatible"] = False
                failure_reasons.append("timeout")
                return None
            if disposition == "transient_backend_failure":
                checks["transient_backend_failure"] = True
                checks["backend_request_compatible"] = False
                failure_reasons.append("transient_backend_failure")
                return None
            if disposition == "malformed_judge_output":
                checks["structured_output"] = False
                failure_reasons.append("malformed_judge_output")
            elif disposition == "score_out_of_range":
                checks["score_range"] = False
                failure_reasons.append("score_out_of_range")
            else:
                checks["parser_schema_compliance"] = False
                failure_reasons.append(disposition)
            return None
        parsed_by_id[control.control_id] = parsed["parsed"]
        if not result["passed"]:
            failure_reasons.extend(f"{control.control_id}:{failure}" for failure in result["failures"])
        return parsed["parsed"]

    for control in controls:
        parsed = run(control)
        if parsed is None and _operational_failure_seen(checks):
            return _result(candidate, started, control_results, checks, failure_reasons)
        if control.mode == "pairwise":
            reversed_control = _reversed_pair(control)
            parsed_reversed = run(reversed_control)
            if parsed_reversed is None and _operational_failure_seen(checks):
                return _result(candidate, started, control_results, checks, failure_reasons)

    repeat_scores = []
    repeat_count = max(2, int(repeats))
    repeat_control = next(control for control in controls if control.control_id == "obviously_correct")
    for index in range(repeat_count):
        repeated = QualificationControl(
            f"repeat_stability_{index + 1}",
            "score",
            repeat_control.task_prompt,
            candidate_answer=repeat_control.candidate_answer,
            reference_answer=repeat_control.reference_answer,
            rubric=repeat_control.rubric,
            expected=repeat_control.expected,
        )
        parsed = run(repeated)
        if parsed is None and _operational_failure_seen(checks):
            return _result(candidate, started, control_results, checks, failure_reasons)
        if parsed and isinstance(parsed.get("score"), (int, float)):
            repeat_scores.append(float(parsed["score"]))

    if "obviously_correct" in parsed_by_id and "obviously_wrong" in parsed_by_id:
        checks["good_bad_discrimination"] = (
            float(parsed_by_id["obviously_correct"]["score"]) - float(parsed_by_id["obviously_wrong"]["score"]) >= 50
        )
    if "partial_credit" in parsed_by_id:
        score = float(parsed_by_id["partial_credit"]["score"])
        checks["partial_credit_sanity"] = 40 <= score <= 75
    checks["rubric_adherence"] = bool(parsed_by_id.get("rubric_adherence", {}).get("score", 101) <= 60 and parsed_by_id.get("rubric_adherence", {}).get("rubric_adherence") is True)
    checks["reference_answer_use"] = bool(
        parsed_by_id.get("reference_answer_use", {}).get("score", 101) <= 35
        and parsed_by_id.get("reference_answer_use", {}).get("reference_used") is True
        and parsed_by_id.get("unsupported_hallucination", {}).get("reference_used") is True
    )
    checks["malformed_candidate_output_handling"] = bool(parsed_by_id.get("malformed_candidate_output", {}).get("score", 101) <= 35)
    pair_ok = True
    for base in ("pair_equal", "pair_a_better", "pair_b_better"):
        expected = next(control.expected["winner"] for control in controls if control.control_id == base)
        reversed_expected = {"A": "B", "B": "A"}.get(expected, expected)
        pair_ok = pair_ok and parsed_by_id.get(base, {}).get("winner") == expected
        pair_ok = pair_ok and parsed_by_id.get(f"{base}_reversed", {}).get("winner") == reversed_expected
    checks["pairwise_reversal_order_bias"] = bool(pair_ok)
    checks["repeat_stability"] = bool(repeat_scores and max(repeat_scores) - min(repeat_scores) <= 5)

    for check, passed in checks.items():
        if check in {"timeout", "structural_incompatibility", "unsupported_backend", "transient_backend_failure"}:
            continue
        if not passed and check not in failure_reasons:
            failure_reasons.append(check)

    return _result(candidate, started, control_results, checks, failure_reasons)


def _operational_failure_seen(checks: Dict[str, Any]) -> bool:
    return bool(
        checks.get("structural_incompatibility")
        or checks.get("unsupported_backend")
        or checks.get("timeout")
        or checks.get("transient_backend_failure")
    )


def _result(
    candidate: Dict[str, Any],
    started: datetime,
    control_results: List[Dict[str, Any]],
    checks: Dict[str, Any],
    failure_reasons: List[str],
) -> Dict[str, Any]:
    finished = datetime.now(timezone.utc)
    if checks.get("structural_incompatibility"):
        disposition = "rejected_structural_incompatibility"
    elif checks.get("unsupported_backend"):
        disposition = "rejected_unsupported_backend"
    elif checks.get("timeout"):
        disposition = "rejected_timeout"
    elif checks.get("transient_backend_failure"):
        disposition = "rejected_transient_backend_failure"
    elif failure_reasons:
        if "malformed_judge_output" in failure_reasons:
            disposition = "rejected_malformed_output"
        elif "schema_violation" in failure_reasons:
            disposition = "rejected_schema_violation"
        elif "score_out_of_range" in failure_reasons:
            disposition = "rejected_score_out_of_range"
        elif "repeat_stability" in failure_reasons:
            disposition = "rejected_unstable"
        elif "pairwise_reversal_order_bias" in failure_reasons:
            disposition = "rejected_pairwise_inconsistent"
        else:
            disposition = "rejected_quality_controls"
    else:
        disposition = "qualified"
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model": candidate.get("name") or candidate.get("model"),
        "digest": candidate.get("digest") or candidate.get("model_digest_resolved"),
        "runtime_identity": candidate.get("runtime_identity") or {},
        "capabilities": candidate.get("capabilities") or candidate.get("runtime_capabilities") or [],
        "canonical_families": candidate.get("canonical_families") or candidate.get("supported_families") or [],
        "qualified": disposition == "qualified",
        "aggregate_disposition": disposition,
        "failure_reasons": sorted(set(str(reason) for reason in failure_reasons)),
        "checks": checks,
        "controls": control_results,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "latency_seconds": (finished - started).total_seconds(),
    }
