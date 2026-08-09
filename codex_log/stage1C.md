# Stage 1C log — Model roles, independent judge resolution, no self-judging

## Session context
- Branch: `rc21-post1-topology-budget`
- Baseline commit: `a0a6eca09d52377eb8dac1ed9d934a1c1721b4bb`
- Approved stage: Stage 1C only
- Stage 1B approval: explicit human approval in the Stage 1C prompt
- Real-host/model work: prohibited; only mocks, fixtures, synthetic controls, static checks, and safe tests are in scope.

## Instructions read
- `AGENTS.md`
- `CODEX_START.md`
- `PR_RC21POST1_ACCEPTANCE_CONTROLS.md`
- `codex_prompts/stage1C.md`

## Analysis
- Stage 1A already owns deterministic judge candidate ordering and capability/family policy.
- Stage 1B already owns model-agnostic judge qualification.
- Stage 1C requires roles and row-level independence on top of those layers.
- The existing implementation still excluded judge candidates if they were in the tested cohort, which conflicts with Stage 1C because a model may be both `judge` and `benchmark_candidate`.
- The existing post-hoc judging path accepted one judge model for a whole run, so it could not switch to a fallback judge for rows authored by the primary judge.

## Implementation notes
- Added `MODEL_ROLE_POLICY_VERSION = "rc21.post1.roles"`.
- Added normalized role handling in `campaign._model_roles(...)`; roles are distinct from runtime capabilities.
- Preserved legacy role-less fixtures/evidence as role-unspecified, not as implicit capabilities.
- Explicit role metadata is enforced for judge selection: when roles are supplied, a judge candidate must include `judge`; `benchmark_candidate` alone is rejected as `missing_judge_role`.
- Models with both `judge` and `benchmark_candidate` roles remain eligible as judges and remain benchmark candidates.
- Added stable model identity helpers:
  - `stable_model_identity(...)`;
  - `same_stable_model_identity(...)`.
- Stable identity prefers digest/resolved digest when both sides provide it.
- When digest metadata is incomplete, matching names are treated conservatively as the same identity to avoid accidental self-judging.
- Replaced cohort-level judge rejection with row-level independent-judge resolution.
- Added `resolve_independent_judge_for_row(...)`, which consumes the already-qualified deterministic judge order and selects the first non-self judge per source row.
- Added `select_qualified_campaign_judges(...)` to collect all Stage 1B-qualified judges in Stage 1A order for row-level fallback.
- Updated campaign judging integration to persist `qualified_judges`, role policy version, and qualification chain evidence.
- Updated `judge_dumps.judge_run(...)` to resolve the judge per row from `qualified_judges`.
- If no independent qualified judge exists, `judge_dumps` writes an explicit `awaiting_independent_judge` sidecar entry and does not call the judge backend for that row.
- Added `awaiting_independent_judge` as a terminal disposition so pending independent-judge rows remain explicit rather than being silently dropped.
- Updated adoption validation to reject actual self-judged sidecar rows, not merely a qualified judge digest also appearing in the campaign cohort.
- Did not implement Stage 1D judge provenance/evidence expansion beyond the Stage 1C fields needed for role/no-self decisions.

## Validation
- Focused Stage 1C tests:
  - `.venv/bin/python -m pytest tests/test_stage1c_model_roles.py -q`
  - Result: `8 passed in 0.09s`.
- Directly affected Stage 1A/1B/judge/campaign/config/backend tests:
  - `.venv/bin/python -m pytest tests/test_stage1c_model_roles.py tests/test_stage1a_judge_policy.py tests/test_stage1b_judge_qualification.py tests/test_judge_dumps.py tests/test_campaign.py tests/test_campaign_final_acceptance.py tests/test_campaign_adoption_transaction.py tests/test_cli_subcommands.py tests/test_config_validation.py tests/test_capability_workflow.py tests/test_backend.py -q`
  - Initial result: one expected failure in the old adoption test that asserted cohort-level judge digest conflict.
  - Fix: updated the test to assert the Stage 1C contract, namely rejection of actual self-judged sidecar evidence.
  - Final result: `160 passed in 1.84s`.
- Broader offline campaign/Ollama-adjacent compatibility tests:
  - `.venv/bin/python -m pytest tests/test_campaign_runtime_identity_resume.py tests/test_campaign_adoption_transaction.py tests/test_campaign_cleanup_migration_hygiene.py tests/test_campaign_package_integrity.py tests/test_campaign_recovery_matrix.py tests/test_ollama_service_conflict_warning.py tests/test_ollama_service_active_unit.py tests/test_ollama_kv_broker_boundary.py -q`
  - Result: `137 passed in 1.08s`.
- Static/check results:
  - `.venv/bin/python -m ruff check llm_modelbench/campaign.py llm_modelbench/judge_dumps.py llm_modelbench/cli.py tests/test_stage1c_model_roles.py tests/test_campaign_adoption_transaction.py` — passed.
  - `.venv/bin/python -m compileall -q llm_modelbench tests/test_stage1c_model_roles.py tests/test_campaign_adoption_transaction.py` — passed.
  - `git diff --check` — passed.

## Manual inspection
- Stage 1A selection remains the deterministic source of judge order.
- Stage 1B qualification remains the source of qualified judge status.
- Stage 1C consumes the ordered qualified chain only after those layers complete.
- Roles do not alter capability classification.
- A benchmark candidate with a judge role can still qualify as a judge.
- A qualified judge role does not remove benchmark-candidate role state.
- Stable digest identity detects alias/self cases.
- Per-row post-hoc judging skips a self-identical primary judge and deterministically uses the next independent qualified judge.
- When no independent qualified judge exists, the row is recorded as `awaiting_independent_judge` and no judge backend call is made.
- No real Selene/Ollama/llama.cpp generation, real judge, real benchmark, or real campaign evidence was used.
- Stage 1D was not started.
