"""Local judge helpers for subjective tasks.

The judge is a screening aid, not a final truth source. V9.5.8 makes judge output
structured and diagnosable: no silent fallback score, thinking blocks are stripped,
and invalid judge output returns (None, "judge_error: ...") instead of a fake 50.
"""
from __future__ import annotations

import json
from typing import Optional, Tuple

from .scoring import strip_thinking

ANCHORS = """
Use these anchors:
- 30 = incomplete, vague, wrong structure, or materially inaccurate.
- 60 = usable but generic, misses important details or constraints.
- 90 = accurate, structured, practical, concise, and follows all constraints.
""".strip()


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


def judge_single(client, judge_model: str, prompt: str, output: str, rubric: str, *, num_ctx=None, think="auto"):
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
    res = client.chat(
        judge_model,
        judge_prompt,
        system="Grade strictly. Return only the requested JSON. Do not include chain of thought.",
        num_predict=1024,
        num_ctx=num_ctx,
        think=think,
    )
    if not res.get("ok"):
        return None, f"judge_error: {res.get('error', 'failed')}"
    return _parse_score(res.get("text") or "")


def judge_panel(client, judge_model: str, prompt: str, output: str, rubric: str, *, num_ctx=None, think="auto"):
    personas = [
        "strict correctness judge: penalize factual errors and missed constraints",
        "pragmatic usefulness judge: reward usable, actionable, well-structured answers",
        "clarity judge: reward concise, readable writing and clear organization",
    ]
    scores = []
    reasons = []
    for persona in personas:
        res = client.chat(
            judge_model,
            f"Persona: {persona}\n\nRubric: {rubric}\n{ANCHORS}\n\nTask:\n{prompt}\n\nAnswer:\n{output}\n\nReturn ONLY JSON: {{\"score\": <0-100>, \"confidence\": <0-1>, \"verdict\": \"brief reason\"}}",
            system="Grade strictly. Return only JSON. Do not include chain of thought.",
            num_predict=1024,
            num_ctx=num_ctx,
            think=think,
        )
        if not res.get("ok"):
            reasons.append(f"judge_error: {res.get('error', 'failed')}")
            continue
        score, reason = _parse_score(res.get("text") or "")
        if isinstance(score, (int, float)):
            scores.append(float(score))
        reasons.append(reason)
    if not scores:
        return None, "judge_error: panel produced no valid scores; " + "; ".join(reasons[:3])
    scores.sort()
    mid = scores[len(scores)//2] if len(scores) % 2 else (scores[len(scores)//2 - 1] + scores[len(scores)//2]) / 2
    spread = max(scores) - min(scores) if len(scores) > 1 else 0.0
    return round(mid, 2), f"panel_median={round(mid,2)} spread={round(spread,2)} scores={','.join(str(round(s,1)) for s in scores)}"
