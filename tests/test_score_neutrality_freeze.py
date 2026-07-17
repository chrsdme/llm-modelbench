import hashlib
import json

from llm_modelbench import scoring
from llm_modelbench.tasks import AGENTIC_CONTRACT, TASKS


AGENTIC_CONTRACT_SHA256 = "0cf156353898ffd9c46347daf7e9088886f65d71e77a6cf8fc6ffc3ead897ab0"

AGENTIC_PROMPT_SHA256 = {
    "agent_tool_select": "c942e68c8c7377de9bf28bbb31d85421314a8da9443e2c3a67ad2bf588f32c20",
    "agent_tool_refuse": "2702395987fecafb00676d687e360afec033f6977578c23b2735f77ccb4fea2e",
    "agent_tool_repair": "c79edd604b0d752c8544c44cd9148de656f7a762d21824ba81b40b46470f7827",
    "agent_tool_state": "417d094742532b4c6baaf936daf59fae928e57300024fa11a6afee7a2b1c0e5e",
    "agent_schema_strict": "8fcf702907e970ec378f3a7f8e2be317503a98c31c24e899b7018e6730ee8bd2",
    "agent_unknown_tool_reject": "2e2c18659022d5abf254801cf2828bbb504ca3d01e2cbf6f1415050894537464",
    "agent_schema_collision": "f4059f450def2a3e2e3527df1be7e5e8f8ae990a663c985b8e34abfd922e6967",
    "agent_state_delta": "c104c6d7dd7ced177bcd3fbebea340f8dbc145fdb96fe805a0413f88988c37f0",
    "agent_malformed_repair": "adc0790e87a4fc991e486e4007e8f04d255e1e634cb5dc37ea9a7f5998183984",
    "agent_nested_args": "f8ca74dbfaf608918d98e83d99f64f1b2ec1fc3fb41dce1474264b530dbc3ada",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _agentic_tasks():
    return [t for t in TASKS if t.scorer == "agentic_action"]


def _perfect(meta):
    data = {"tool": meta.get("expected_tool"), "args": meta.get("expected_args") or {}}
    if meta.get("require_reason"):
        data["reason"] = "No available safe tool can perform this request."
    return json.dumps(data, separators=(",", ":"))


# Prompt edits move benchmark scores. A prompt edit is allowed only when this
# freeze test is deliberately updated in the same commit with a comparability note.
def test_agentic_contract_and_prompts_are_frozen():
    assert _sha(AGENTIC_CONTRACT) == AGENTIC_CONTRACT_SHA256
    actual = {t.id: _sha(t.prompt) for t in _agentic_tasks()}
    assert actual == AGENTIC_PROMPT_SHA256


# P1: JSON copied from a prompt must not become a passing answer.
def test_p1_prompt_echo_no_prompt_object_scores_as_passing_answer():
    for task in _agentic_tasks():
        best = 0.0
        for obj in scoring._agentic_json_candidates(task.prompt):
            score, _ = scoring.score_agentic_action(json.dumps(obj[0]), task.meta)
            best = max(best, score)
        assert best < 75.0, f"{task.id}: prompt echo scores {best}"


# P2: every exact string expected by the scorer must be visible in the task.
def test_p2_string_valued_expected_args_are_stated_in_prompt():
    def leaves(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for v in value.values():
                yield from leaves(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                yield from leaves(v)

    for task in _agentic_tasks():
        for want in leaves(task.meta.get("expected_args") or {}):
            assert want in task.prompt, f"{task.id}: expected string {want!r} is not stated"


# P3: the ideal answer remains exactly achievable.
def test_p3_perfect_agentic_answers_score_100():
    for task in _agentic_tasks():
        score, _ = scoring.score_agentic_action(_perfect(task.meta), task.meta)
        assert score == 100.0, task.id


# P4: no component of the agentic envelope is free.
def test_p4_agentic_components_are_not_free():
    for task in _agentic_tasks():
        meta = task.meta
        base = json.loads(_perfect(meta))
        mutations = [
            {k: v for k, v in base.items() if k != "tool"},
            {**base, "args": list((meta.get("expected_args") or {}).values()) or ["x"]},
            {**base, "tool": "not_a_tool"},
            {**base, "comment": "x"},
        ]
        if meta.get("expected_args"):
            bad_args = dict(meta["expected_args"])
            first = next(iter(bad_args))
            bad_args[first] = "__wrong__"
            mutations.append({**base, "args": bad_args})
        else:
            mutations.append({**base, "args": {"x": 1}})
        if meta.get("require_reason"):
            mutations.append({k: v for k, v in base.items() if k != "reason"})
        for mut in mutations:
            score, _ = scoring.score_agentic_action(json.dumps(mut, separators=(",", ":")), meta)
            assert score < 100.0, f"{task.id}: mutation is free: {mut}"


# P5: any accepted correct formatting band must beat every wrong decision cap.
def test_p5_wrong_decision_ceiling_below_accepted_correct_floor():
    for task in _agentic_tasks():
        meta = task.meta
        base = json.loads(_perfect(meta))
        variants = [json.dumps(base, separators=(",", ":")), f"```json\n{json.dumps(base)}\n```", "Sure. " + json.dumps(base)]
        if base["args"]:
            variants.append("Sure. " + json.dumps({**base, "args": [base["args"]]}))
        floor = min(scoring.score_agentic_action(v, meta)[0] for v in variants)

        wrongs = []
        if meta.get("require_reason"):
            wrongs.append('{"tool":null,"args":{}}')
        if meta.get("expected_tool") is not None:
            wrongs.append(json.dumps({"tool": "wrong.tool", "args": meta.get("expected_args") or {}}))
            wrongs.append(json.dumps({"tool": None, "args": {}}))
        if meta.get("expected_args"):
            bad = dict(meta["expected_args"])
            bad[next(iter(bad))] = "WRONG"
            wrongs.append(json.dumps({"tool": meta.get("expected_tool"), "args": bad}))
        wrongs.append(json.dumps({**base, "comment": "x"}))
        ceiling = max(scoring.score_agentic_action(w, meta)[0] for w in wrongs)
        assert ceiling < floor, f"{task.id}: wrong {ceiling} >= accepted correct {floor}"


def test_caps_fired_extracted_from_existing_agentic_reason_without_rescoring():
    reason = "agentic action 20.0/100, missing: tool not allowed, tool=calculator.add, decision_cap=20"
    assert scoring.agentic_caps_fired_from_reason(reason) == ["tool not allowed", "tool=calculator.add"]

    reason = "agentic action 65.0/100, missing: empty args, refusal reason, decision_cap=65"
    assert scoring.agentic_caps_fired_from_reason(reason) == ["empty args", "refusal reason"]

    reason = "agentic action 50.0/100, missing: extra_top_level_key=reason, decision_cap=50"
    assert scoring.agentic_caps_fired_from_reason(reason) == ["extra_top_level_key=reason"]

    assert scoring.agentic_caps_fired_from_reason("agentic action ok") == []
    assert scoring.agentic_caps_fired_from_reason("invalid_json:empty") == ["invalid_json:empty"]
