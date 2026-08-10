# Session 5 - Stage 4A

## Session identity
- Session: 5
- Stage: Stage 4A - strict versioned campaign config schema and `campaign init`
- Branch: `rc21-post1-topology-budget`
- Starting HEAD: `dabbe6f6265d522897b7706aeae9d8dae54b2418`

## Files/specs read
- `AGENTS.md`
- `CODEX_START.md`
- `PR_RC21POST1_ACCEPTANCE_CONTROLS.md`
- `codex_prompts/stage4A.md`
- `codex_logs/Session 4/session4_20260810T002148Z.md`

## Requirements
- Versioned campaign config schema.
- `campaign init` emits a valid usable template.
- Advertised fields must be implemented or rejected/removed; no silently ignored policy fields.
- Validate unknown keys, types, required values, campaign IDs, selectors, incompatible settings, judge policy, and context values.
- Safe documented defaults; no automatic adoption.
- No real execution and no Stage 4B execution-plan/resume/idempotency work.

## Current Stage 4A audit
- Existing `campaign init` emitted unversioned config and advertised ignored/deferred policies: topology, KV fallback, recovery, telemetry, reporting package, deferred models, and enabled judge policy.
- Existing `campaign execute --config` performed only loose campaign ID/model checks and would silently ignore unknown or unsupported fields.
- No strict versioned config validation helper existed.

## Design decisions
- Added `CAMPAIGN_CONFIG_SCHEMA_VERSION = 1`.
- Added `campaign_config_template()` as the single source for `campaign init` output.
- Added `validate_campaign_config(...)` for Stage 4A schema validation without executing real work.
- Template includes only fields that are safe and not silently ignored by current behavior: schema version, campaign ID, models, level, samples, runtime auto flag, optional needle max context, disabled judge policy, executable scorer host-code flag, and mandatory `stop_before_adoption=true`.
- Deferred/ignored policy fields are rejected as unknown instead of advertised.
- `judge_policy.enabled=true` is rejected until Stage 4B wires execution semantics; disabled judge policy is valid and explicit.
- `campaign execute --config` now validates with the strict schema before invoking the existing mock/normal campaign path, but Stage 4A does not add immutable execution-plan/resume/idempotency behavior.
- Contradictory `campaign_id` and legacy `id` aliases fail closed.

## Commands/tests
- `sed -n '1,260p' codex_prompts/stage4A.md`
- `sed -n '1,220p' codex_logs/Session 4/session4_20260810T002148Z.md`
- `git rev-parse HEAD && git status -sb`
- Source/test reads around `campaign init`, `campaign execute`, level choices, and existing campaign tests.
- `pytest -q tests/test_stage4a_campaign_config.py tests/test_campaign.py` -> 74 passed before the ID conflict fix; 75 passed after adding ID conflict coverage.
- `pytest -q tests/test_stage4a_campaign_config.py tests/test_stage3b_supersession.py tests/test_stage3a_supersession.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign.py tests/test_campaign_package_integrity.py tests/test_campaign_final_acceptance.py tests/test_campaign_recovery_matrix.py tests/test_stage2a_recovery_reconciliation.py tests/test_stage2b_recovery_post_execution.py tests/test_repair.py tests/test_rc9_context_repair.py tests/test_rc10_repair_truth.py tests/test_stage1a_judge_policy.py tests/test_stage1b_judge_qualification.py tests/test_stage1c_model_roles.py tests/test_stage1d_judge_integration.py tests/test_judge_dumps.py` -> 380 passed, 11 skipped before the ID conflict fix; 381 passed, 11 skipped after.
- `./.venv/bin/ruff check llm_modelbench tests/test_stage4a_campaign_config.py tests/test_stage3a_supersession.py tests/test_stage3b_supersession.py` -> passed.
- `python -m compileall -q llm_modelbench tests/test_stage4a_campaign_config.py tests/test_stage3a_supersession.py tests/test_stage3b_supersession.py` -> passed.
- `git diff --check` -> passed.

## Failures and corrections
- Manual inspection found contradictory `campaign_id`/`id` aliases were not rejected. Added `campaign_config_id_conflict` validation and test coverage.

## Manual inspection
- Confirmed template round-trips through strict schema validation.
- Confirmed ignored/deferred fields from the previous template are no longer advertised and are rejected if supplied.
- Confirmed unknown top-level and nested keys fail closed.
- Confirmed invalid campaign IDs, models, level, samples, runtime policy, context value, judge policy, executable scorer policy, and stop-before-adoption values fail closed.
- Confirmed `campaign init` refuses to overwrite existing files.
- Confirmed Stage 4B execution-plan/resume/idempotency behavior was not implemented.
- Confirmed no real execution, real model work, adoption, push, or evidence mutation.

## Safety confirmation
- No real inference, real recovery, real judging, real catch-up, real campaign execution beyond synthetic/mock tests, Ollama/llama.cpp/GPU/service mutation, adoption, evidence rewrite, model pull/delete, or push performed.
