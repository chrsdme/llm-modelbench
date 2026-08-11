# Recommended routing

Level: `short` | sample_mode: `smart` | judge: `single`
Tasks: 31 | task_ids: `agent_malformed_repair, agent_native_tool_call, agent_nested_args, agent_plan, agent_schema_collision, agent_schema_strict, agent_state_delta, agent_tool_refuse, agent_tool_repair, agent_tool_select, agent_tool_state, agent_unknown_tool_reject, file_ext, fim_suffix_assertion, git_commit, git_conflict, js_debounce, json_extract, kb_taxonomy, py_anagram, py_csv, py_dedupe, reasoning_birthday_twins, reasoning_bridge_crossing, reasoning_monty_hall, reasoning_poisoned_wine, reasoning_wolf_goat_cabbage, txt_emails, txt_sort, web_nav, wr_rag`
num_ctx_used: `server-default` | num_predict: `task-default` | think: `auto` | needle_max_ctx: `none`
Runtime variants: llama3.1:8b=1; qwen2.5-coder:14b=1

Routing refuses single winners when category coverage is too small or the top decision score is tied.
Agentic routing ranks decision quality first, then orders tied ceiling bands by VRAM and throughput.

- **agentic_tool** -> ceiling band tied at 94.32; no exact rank inside band. Route by VRAM, then throughput:
  - `llama3.1:8b` — 4.9 GB, 42.0 tok/s recommended
  - `qwen2.5-coder:14b` — 9.0 GB, 42.0 tok/s
- **coding_js** -> no recommendation, insufficient coverage: 1 task(s), minimum 2
- **coding_python** -> no single winner, tied at 100.0: `qwen2.5-coder:14b`, `llama3.1:8b`
- **coding_web** -> no recommendation, insufficient coverage: 1 task(s), minimum 2
- **file_ops** -> no recommendation, insufficient coverage: 1 task(s), minimum 2
- **knowledge_base** -> no single winner, tied at 88.0: `qwen2.5-coder:14b`, `llama3.1:8b`
- **reasoning** -> no single winner, tied at 100.0: `qwen2.5-coder:14b`, `llama3.1:8b`
- **tech_writing** -> no recommendation, insufficient coverage: 1 task(s), minimum 2

Note: thinking_only rows at fixed num_predict are model failures for scoring, but budget-limited rows should be read as lower bounds on reasoning-model agentic ability.