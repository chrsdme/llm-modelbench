"""Agentic finals specification, hardened in V9.5.15.

The first finals lane is deliberately local and deterministic: models emit a single
JSON action decision, and the harness scores schema validity plus expected tool/args.
No real external tools are called by these tasks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class AgenticFinalSpec:
    id: str
    capability: str
    scorer: str
    expected_tool: str | None
    expected_args: Dict[str, Any]
    negative_fixtures: List[str]
    mutant_fixtures: List[str]


AGENTIC_FINALS_SEED: List[AgenticFinalSpec] = [
    AgenticFinalSpec(
        id="agent_tool_select",
        capability="choose the correct tool from valid alternatives and one distractor",
        scorer="agentic_action",
        expected_tool="read_file",
        expected_args={"path": "README.md"},
        negative_fixtures=["wrong_tool", "missing_args"],
        mutant_fixtures=["markdown_contamination", "path_mutation"],
    ),
    AgenticFinalSpec(
        id="agent_tool_refuse",
        capability="decline tool use when no offered tool is appropriate",
        scorer="agentic_action",
        expected_tool=None,
        expected_args={},
        negative_fixtures=["unnecessary_tool_call", "empty_refusal_reason"],
        mutant_fixtures=["invented_tool", "unsafe_action"],
    ),
    AgenticFinalSpec(
        id="agent_tool_repair",
        capability="repair arguments after a simulated tool validation error",
        scorer="agentic_action",
        expected_tool="calculator.add",
        expected_args={"a": 12, "b": 7},
        negative_fixtures=["repeat_bad_string_args", "wrong_tool"],
        mutant_fixtures=["schema_missing_args", "numeric_string_mutation"],
    ),
    AgenticFinalSpec(
        id="agent_tool_state",
        capability="carry state across a short state update task",
        scorer="agentic_action",
        expected_tool="cart.update",
        expected_args={"sku": "A1", "quantity": 3},
        negative_fixtures=["state_drop", "wrong_quantity"],
        mutant_fixtures=["wrong_sku", "extra_prose"],
    ),
    AgenticFinalSpec(
        id="agent_schema_strict",
        capability="emit strict structured calls without prose contamination",
        scorer="agentic_action",
        expected_tool="send_email",
        expected_args={"to": "ops@example.com", "subject": "Backup complete"},
        negative_fixtures=["prose_contamination", "missing_subject"],
        mutant_fixtures=["fenced_json", "extra_top_level_text"],
    ),

    AgenticFinalSpec(
        id="agent_unknown_tool_reject",
        capability="refuse unavailable or invented tools",
        scorer="agentic_action",
        expected_tool=None,
        expected_args={},
        negative_fixtures=["invented_shell_tool", "unsafe_call"],
        mutant_fixtures=["tool_hallucination", "empty_refusal_reason"],
    ),
    AgenticFinalSpec(
        id="agent_schema_collision",
        capability="resist action/kwargs schema collision and emit tool/args",
        scorer="agentic_action",
        expected_tool="send_email",
        expected_args={"to": "ops@example.com", "subject": "Incident acknowledged"},
        negative_fixtures=["action_kwargs", "function_call_shape"],
        mutant_fixtures=["wrong_schema", "wrong_subject"],
    ),
    AgenticFinalSpec(
        id="agent_state_delta",
        capability="apply state deltas to final state updates",
        scorer="agentic_action",
        expected_tool="cart.update",
        expected_args={"sku": "B2", "quantity": 3},
        negative_fixtures=["delta_quantity", "wrong_sku"],
        mutant_fixtures=["quantity_2", "quantity_5"],
    ),
    AgenticFinalSpec(
        id="agent_malformed_repair",
        capability="repair malformed prior calls into strict tool/args JSON",
        scorer="agentic_action",
        expected_tool="read_file",
        expected_args={"path": "README.md"},
        negative_fixtures=["positional_args", "unquoted_tool"],
        mutant_fixtures=["array_args", "missing_path"],
    ),
    AgenticFinalSpec(
        id="agent_nested_args",
        capability="emit nested argument objects without flattening",
        scorer="agentic_action",
        expected_tool="ticket.create",
        expected_args={"ticket": {"title": "Disk alert", "priority": "high", "labels": ["infra", "disk"]}},
        negative_fixtures=["flattened_ticket", "missing_labels"],
        mutant_fixtures=["wrong_priority", "label_order_change"],
    ),
]


def manifest() -> Dict[str, object]:
    return {
        "status": "active_hardened",
        "use_case_priority": 1,
        "use_case": "agentic tool execution and repair loops",
        "registered_in_tasks": True,
        "task_count": len(AGENTIC_FINALS_SEED),
        "tasks": [x.__dict__ for x in AGENTIC_FINALS_SEED],
    }


def spec_by_id() -> Dict[str, AgenticFinalSpec]:
    return {s.id: s for s in AGENTIC_FINALS_SEED}
