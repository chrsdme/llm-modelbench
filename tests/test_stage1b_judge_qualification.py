import json

from llm_modelbench import campaign
from llm_modelbench.judge_qualification import PROTOCOL_VERSION, qualify_candidate


class FakeJudgeBackend:
    def __init__(self, behavior="qualified"):
        self.behavior = behavior
        self.calls = []
        self.repeat_scores = [95, 70, 90]

    def chat(self, model, prompt, **kwargs):
        request = self._request(prompt)
        self.calls.append({"model": model, "request": request, "kwargs": kwargs})
        control_id = request["control_id"]
        if self.behavior == "timeout":
            return {"ok": False, "error": "deadline exceeded", "error_kind": "timeout"}
        if self.behavior == "structural_http":
            return {"ok": False, "error": "unsupported request schema", "http_status": 415}
        if self.behavior == "unsupported":
            return {"ok": False, "error": "chat unavailable", "error_kind": "unsupported_backend"}
        if self.behavior == "malformed_output" and control_id == "obviously_correct":
            return {"ok": True, "text": "not json"}
        if self.behavior == "out_of_range" and control_id == "obviously_correct":
            return {"ok": True, "text": '{"score": 120, "confidence": 1, "verdict": "too high"}'}
        if self.behavior == "rubric_failure" and control_id == "rubric_adherence":
            return {"ok": True, "text": '{"score": 95, "confidence": 1, "verdict": "factually correct", "rubric_adherence": false, "reference_used": true}'}
        if self.behavior == "unstable" and control_id.startswith("repeat_stability_"):
            score = self.repeat_scores.pop(0)
            return {"ok": True, "text": json.dumps({"score": score, "confidence": 1, "verdict": "repeat", "rubric_adherence": True, "reference_used": True})}
        if request["mode"] == "pairwise":
            winner = self._pairwise_winner(control_id)
            if self.behavior == "reversed_pair_inconsistent" and control_id.endswith("_reversed"):
                winner = {"A": "B", "B": "A"}.get(winner, winner)
            return {"ok": True, "text": json.dumps({"winner": winner, "confidence": 1, "verdict": "pairwise"})}
        return {"ok": True, "text": json.dumps(self._score_response(control_id))}

    @staticmethod
    def _request(prompt):
        marker = "Return only JSON matching response_schema.\n"
        return json.loads(prompt.split(marker, 1)[1])

    @staticmethod
    def _pairwise_winner(control_id):
        return {
            "pair_equal": "equal",
            "pair_equal_reversed": "equal",
            "pair_a_better": "A",
            "pair_a_better_reversed": "B",
            "pair_b_better": "B",
            "pair_b_better_reversed": "A",
        }[control_id]

    @staticmethod
    def _score_response(control_id):
        scores = {
            "obviously_correct": 95,
            "obviously_wrong": 5,
            "partial_credit": 60,
            "irrelevant_answer": 5,
            "unsupported_hallucination": 5,
            "malformed_candidate_output": 5,
            "rubric_adherence": 50,
            "reference_answer_use": 5,
        }
        score = scores.get(control_id, 95)
        return {
            "score": score,
            "confidence": 1,
            "verdict": f"{control_id} scored {score}",
            "rubric_adherence": True,
            "reference_used": True,
        }


def _candidate(name="judge"):
    return {
        "name": name,
        "digest": f"digest-{name}",
        "capabilities": ["completion"],
        "canonical_families": ["text"],
        "runtime_identity": {"provider": "fake", "model": name},
    }


def test_fully_qualified_fake_judge_exercises_complete_protocol():
    backend = FakeJudgeBackend()
    result = qualify_candidate(backend, _candidate(), repeats=3)
    assert result["qualified"] is True
    assert result["aggregate_disposition"] == "qualified"
    assert result["protocol_version"] == PROTOCOL_VERSION
    assert result["runtime_identity"] == {"provider": "fake", "model": "judge"}
    assert result["failure_reasons"] == []
    assert all(result["checks"][key] for key in (
        "backend_request_compatible",
        "structured_output",
        "parser_schema_compliance",
        "score_range",
        "good_bad_discrimination",
        "partial_credit_sanity",
        "rubric_adherence",
        "reference_answer_use",
        "pairwise_reversal_order_bias",
        "repeat_stability",
        "malformed_candidate_output_handling",
    ))
    control_ids = {item["control_id"] for item in result["controls"]}
    assert {
        "obviously_correct",
        "obviously_wrong",
        "partial_credit",
        "irrelevant_answer",
        "unsupported_hallucination",
        "malformed_candidate_output",
        "rubric_adherence",
        "reference_answer_use",
        "pair_equal",
        "pair_equal_reversed",
        "pair_a_better",
        "pair_a_better_reversed",
        "pair_b_better",
        "pair_b_better_reversed",
        "repeat_stability_1",
        "repeat_stability_2",
        "repeat_stability_3",
    } <= control_ids
    first_request = backend.calls[0]["request"]
    assert first_request["protocol_version"] == PROTOCOL_VERSION
    assert first_request["response_schema"]["score"] == "number 0-100"


def test_malformed_structured_judge_output_rejects_candidate():
    result = qualify_candidate(FakeJudgeBackend("malformed_output"), _candidate())
    assert result["qualified"] is False
    assert result["aggregate_disposition"] == "rejected_malformed_output"
    assert result["checks"]["structured_output"] is False
    assert "malformed_judge_output" in result["failure_reasons"]


def test_out_of_range_score_rejects_candidate():
    result = qualify_candidate(FakeJudgeBackend("out_of_range"), _candidate())
    assert result["qualified"] is False
    assert result["aggregate_disposition"] == "rejected_score_out_of_range"
    assert result["checks"]["score_range"] is False


def test_rubric_failure_rejects_candidate():
    result = qualify_candidate(FakeJudgeBackend("rubric_failure"), _candidate())
    assert result["qualified"] is False
    assert result["checks"]["rubric_adherence"] is False
    assert any(reason.startswith("rubric_adherence:") or reason == "rubric_adherence" for reason in result["failure_reasons"])


def test_reversed_pair_inconsistency_rejects_candidate():
    result = qualify_candidate(FakeJudgeBackend("reversed_pair_inconsistent"), _candidate())
    assert result["qualified"] is False
    assert result["aggregate_disposition"] == "rejected_pairwise_inconsistent"
    assert result["checks"]["pairwise_reversal_order_bias"] is False


def test_unstable_repeat_scoring_rejects_candidate():
    result = qualify_candidate(FakeJudgeBackend("unstable"), _candidate(), repeats=3)
    assert result["qualified"] is False
    assert result["aggregate_disposition"] == "rejected_unstable"
    assert result["checks"]["repeat_stability"] is False


def test_timeout_is_typed_and_rejects_candidate_immediately():
    backend = FakeJudgeBackend("timeout")
    result = qualify_candidate(backend, _candidate())
    assert result["qualified"] is False
    assert result["aggregate_disposition"] == "rejected_timeout"
    assert result["checks"]["timeout"] is True
    assert len(backend.calls) == 1


def test_typed_http_request_incompatibility_rejects_candidate_immediately():
    backend = FakeJudgeBackend("structural_http")
    result = qualify_candidate(backend, _candidate())
    assert result["qualified"] is False
    assert result["aggregate_disposition"] == "rejected_structural_incompatibility"
    assert result["checks"]["structural_incompatibility"] is True
    assert len(backend.calls) == 1


def test_unsupported_backend_rejects_candidate():
    result = qualify_candidate(object(), _candidate())
    assert result["qualified"] is False
    assert result["aggregate_disposition"] == "rejected_unsupported_backend"
    assert result["checks"]["unsupported_backend"] is True


def test_fallback_ready_rejection_uses_protocol_result_not_descriptive_strings():
    class RoutedBackend:
        def __init__(self):
            self.by_model = {
                "bad": FakeJudgeBackend("structural_http"),
                "good": FakeJudgeBackend("qualified"),
            }

        def chat(self, model, prompt, **kwargs):
            return self.by_model[model].chat(model, prompt, **kwargs)

    selection = campaign.build_judge_selection([
        _candidate("bad"),
        _candidate("good"),
    ], [], campaign.JudgePolicy(configured_fallbacks=("bad", "good")))
    selected, chain = campaign.select_qualified_campaign_judge(RoutedBackend(), selection)
    assert selected["name"] == "good"
    assert [item["model"] for item in chain] == ["bad", "good"]
    assert chain[0]["aggregate_disposition"] == "rejected_structural_incompatibility"
    assert chain[1]["aggregate_disposition"] == "qualified"
