# Session 5 - Stage 4B

## Session identity
- Session: 5
- Stage: Stage 4B - `campaign execute --config`, immutable plan, resume/idempotency
- Branch: `rc21-post1-topology-budget`
- Starting HEAD: `aeeb6e0efd710555b8b1a1a2dcc4899df20f6fae`

## Files/specs read
- `AGENTS.md`
- `CODEX_START.md`
- `PR_RC21POST1_ACCEPTANCE_CONTROLS.md`
- `codex_prompts/stage4B.md`
- `codex_logs/Session 5/stage4A.md`

## Requirements
- Parse only validated Stage 4A configs.
- Create one immutable campaign config plan.
- Never silently replan an existing campaign with changed settings.
- Interrupted execution resolves to explicit resume semantics.
- Resume/idempotency must not duplicate primary evidence.
- Same config/state gives deterministic next action.
- Invalid state transitions fail clearly.
- Preserve legacy explicit commands unless intentionally documented.
- Mock/fake backend only.

## Current Stage 4B audit
- `campaign execute --config` validated Stage 4A config but then directly delegated to `campaign run`.
- No immutable config signature/plan was persisted for `execute --config`.
- Re-running the same config against a completed campaign was not explicitly idempotent.
- Running a changed config against an existing campaign could fall through to existing command behavior rather than a config-specific refusal.
- Interrupted campaigns did not get a config-execute-specific next-action message.

## Design decisions
- Added deterministic `campaign_config_signature(...)` and `campaign_config_plan_record(...)`.
- `campaign execute --config` writes `plan/campaign_config.json` after the delegated mocked/normal campaign command succeeds.
- Existing campaigns must have `plan/campaign_config.json` to be managed by `execute --config`; legacy explicit campaign commands remain available but are not silently adopted into config execution.
- Existing campaigns with a different config signature fail closed.
- Existing packaged campaigns with the same config return a deterministic no-op and do not call the run path again.
- Existing interrupted campaigns with the same config fail with an explicit `campaign resume <id>` next action.
- Other existing states fail with a clear status/next-action message rather than silently replanning.

## Commands/tests
- `sed -n '1,260p' codex_prompts/stage4B.md`
- Source reads around `campaign execute`, `campaign run`, state transitions, and manifest handling.
- `pytest -q tests/test_stage4b_campaign_execute.py tests/test_stage4a_campaign_config.py tests/test_campaign.py` -> 79 passed.
- `pytest -q tests/test_stage4b_campaign_execute.py tests/test_stage4a_campaign_config.py tests/test_stage3b_supersession.py tests/test_stage3a_supersession.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign.py tests/test_campaign_package_integrity.py tests/test_campaign_final_acceptance.py tests/test_campaign_recovery_matrix.py tests/test_stage2a_recovery_reconciliation.py tests/test_stage2b_recovery_post_execution.py tests/test_repair.py tests/test_rc9_context_repair.py tests/test_rc10_repair_truth.py tests/test_stage1a_judge_policy.py tests/test_stage1b_judge_qualification.py tests/test_stage1c_model_roles.py tests/test_stage1d_judge_integration.py tests/test_judge_dumps.py` -> 385 passed, 11 skipped.
- `./.venv/bin/ruff check llm_modelbench tests/test_stage4a_campaign_config.py tests/test_stage4b_campaign_execute.py tests/test_stage3a_supersession.py tests/test_stage3b_supersession.py` -> passed.
- `python -m compileall -q llm_modelbench tests/test_stage4a_campaign_config.py tests/test_stage4b_campaign_execute.py tests/test_stage3a_supersession.py tests/test_stage3b_supersession.py` -> passed.
- `git diff --check` -> passed.

## Failures and corrections
- No focused or regression test failures after implementation.

## Manual inspection
- Confirmed config signatures are canonical over validated Stage 4A config.
- Confirmed first mocked execute records an immutable config plan.
- Confirmed repeated identical packaged execute is no-op and does not duplicate primary evidence.
- Confirmed changed config is rejected.
- Confirmed interrupted state reports explicit resume next action.
- Confirmed explicit legacy `campaign run`, `campaign plan`, and `campaign resume` command paths were not redesigned.
- Confirmed no real model execution, real Ollama request, adoption, push, or evidence mutation beyond synthetic tests.

## Safety confirmation
- No real inference, real recovery, real judging, real catch-up, real acceptance campaign, Ollama/llama.cpp/GPU/service mutation, adoption, evidence rewrite, model pull/delete, or push performed.
