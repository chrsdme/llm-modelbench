# Clone detection

Level: `short` | sample_mode: `smart` | judge: `single`
Tasks: 31 | task_ids: `agent_malformed_repair, agent_native_tool_call, agent_nested_args, agent_plan, agent_schema_collision, agent_schema_strict, agent_state_delta, agent_tool_refuse, agent_tool_repair, agent_tool_select, agent_tool_state, agent_unknown_tool_reject, file_ext, fim_suffix_assertion, git_commit, git_conflict, js_debounce, json_extract, kb_taxonomy, py_anagram, py_csv, py_dedupe, reasoning_birthday_twins, reasoning_bridge_crossing, reasoning_monty_hall, reasoning_poisoned_wine, reasoning_wolf_goat_cabbage, txt_emails, txt_sort, web_nav, wr_rag`
num_ctx_used: `server-default` | num_predict: `task-default` | think: `auto` | needle_max_ctx: `none`
Runtime variants: llama3.1:8b=1; qwen2.5-coder:14b=1

Certain clone evidence (same digest) is separated from advisory probe evidence.

## Certain clones, same Ollama digest / ID

- none detected

## Advisory probe matches

Probe matches are not used for prune recommendations. Empty or near-empty probe outputs are ignored, and models with too few valid probe answers are marked non-actionable.

- `qwen2.5-coder:14b` ~ `llama3.1:8b` (valid-probe similarity 1.0)