# Stage 2A log - Recovery eligibility and exact pre-execution reconciliation

## Start
- Stage 2A was explicitly approved by Cristian in the Session 3 prompt.
- Stage 2B is not approved and must not be started.
- Required files read completely: `AGENTS.md`, `CODEX_START.md`, `PR_RC21POST1_ACCEPTANCE_CONTROLS.md`, `codex_prompts/stage2A.md`, `codex_logs/Session 2/session2_20260809T181554Z.md`.
- Approved Stage 1D log reviewed for compatibility context: `codex_logs/Session 2/stage1D.md`.
- Expected functional baseline from closed Judge subsystem: `92cb80df1fa9f488b2f770f62ebee7eb485774c9`.
- Actual starting commit: `92cb80df1fa9f488b2f770f62ebee7eb485774c9`.
- Worktree had pre-existing unrelated/untracked items from the handoff, including deleted `codex_log/stage1*.md` files and untracked spec/log assets. These are preserved.
- Scope: Stage 2A only. No post-execution Stage 2B semantics.
- Safety: synthetic/mocked tests only; no real inference, campaigns, recovery execution, judge work, GPU tests, model/service mutation, adoption, or push.

## Requirements tracked
- Complete task-agnostic recovery eligibility for every primary evidence row.
- Preserve no-score-fishing: visible/scorable answers at scores 100, 50, and 0 are not retry eligible.
- Explicit row classification/accounting for eligible, planned, terminal/non-retry, and explicit exclusions.
- Immutable source-row attribution for every native recovery action.
- Remove unattributed native-plan execution bypass.
- Exact set reconciliation before execution using identities, not counts.
- Failed reconciliation must prevent `apply_plan_fn` / recovery execution callback.
- Explicit exclusions must be attributable and have non-empty reasons.
- Deterministic reconciliation evidence and primary evidence immutability.
- Preserve closed Judge subsystem behavior.

## Initial findings
- `campaign.classify_recovery_row(...)` is already mostly task-agnostic and preserves score-zero no-score-fishing.
- `campaign.assert_recovery_reconciled(...)` currently has a compatibility bypass for plans with actions but no source hashes; native execution can appear balanced without attribution.
- `repair.build_plan(...)` skips rows when `task is None` or `task.difficulty <= 0` before checking recovery policy. The six historical omissions (`git_conflict`, `txt_emails`, `agent_plan`, `txt_sort`, `json_extract`, `git_commit`) are all `difficulty=0.0`, so eligible unresolved rows can be silently omitted.

## Implementation
- Updated `llm_modelbench/campaign.py`:
  - kept `classify_recovery_row(...)` as the canonical recovery eligibility path;
  - added judge-lane protection so judge pending/failure states and judge-lane timeout/transient failures do not become generation retries;
  - added explicit generation transient eligibility for `timeout` and `transient_backend_failure`;
  - added row/source identity evidence with source-row hash, task, model and digest;
  - added `recovery_reconciliation_evidence(...)` to compute exact pre-execution accounting without mutating inputs;
  - made `assert_recovery_reconciled(...)` fail closed on omitted, unexpected, duplicated, overlapping, excluded, malformed or unattributed native plan sources;
  - made `execute_recovery_phase(...)` persist candidate-plan reconciliation evidence before any `apply_plan_fn` invocation, and stop before execution when reconciliation fails.
- Updated `llm_modelbench/repair.py`:
  - `build_plan(...)` now consults `classify_recovery_row(...)` before task difficulty/family routing can omit an eligible primary row;
  - difficulty-zero rows with eligible recovery states are now planned with immutable source-row attribution;
  - current-route family metadata no longer silently excludes recovery-eligible primary rows.
- Updated existing recovery execution tests so native mocked plans include `source_row_hashes`.
- Added `tests/test_stage2a_recovery_reconciliation.py` with focused synthetic coverage for the Stage 2A matrix.

## Validation
- `.venv/bin/python -m pytest tests/test_stage2a_recovery_reconciliation.py -q` - passed, 34 tests.
- `.venv/bin/python -m pytest tests/test_campaign_recovery_matrix.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign.py -q` - passed, 80 tests.
- `.venv/bin/python -m pytest tests/test_campaign_final_acceptance.py tests/test_campaign_runtime_identity_resume.py tests/test_campaign_package_integrity.py tests/test_cli_subcommands.py tests/test_config_validation.py -q` - passed, 43 tests.
- `.venv/bin/python -m pytest tests/test_stage1a_judge_policy.py tests/test_stage1b_judge_qualification.py tests/test_stage1c_model_roles.py tests/test_stage1d_judge_integration.py -q` - passed, 76 tests.
- `.venv/bin/python -m pytest tests/test_repair.py tests/test_rc9_context_repair.py tests/test_rc10_repair_truth.py -q` - passed, 46 tests, 11 skipped.
- Combined requested/affected validation:
  - `.venv/bin/python -m pytest tests/test_campaign_recovery_matrix.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign.py tests/test_campaign_final_acceptance.py tests/test_campaign_runtime_identity_resume.py tests/test_campaign_package_integrity.py tests/test_cli_subcommands.py tests/test_config_validation.py tests/test_stage1a_judge_policy.py tests/test_stage1b_judge_qualification.py tests/test_stage1c_model_roles.py tests/test_stage1d_judge_integration.py tests/test_repair.py tests/test_rc9_context_repair.py tests/test_rc10_repair_truth.py -q` - passed, 245 tests, 11 skipped.
- `.venv/bin/python -m ruff check llm_modelbench/campaign.py llm_modelbench/repair.py tests/test_stage2a_recovery_reconciliation.py tests/test_campaign.py tests/test_campaign_recovery_matrix.py` - passed.
- `.venv/bin/python -m compileall -q llm_modelbench tests/test_stage2a_recovery_reconciliation.py` - passed.
- `git diff --check` - passed.

## Manual inspection
- Confirmed every applicable primary row receives a `row_classifications` entry with `recovery_eligible` or `terminal_non_retry`.
- Confirmed exact reconciliation uses source-row hash sets and duplicate detection, not count-only balancing.
- Confirmed native actions with missing or empty `source_row_hashes` fail with `recovery_plan_incomplete`.
- Confirmed planned/excluded overlap, unexpected planned sources, unexpected exclusions, missing eligible rows and empty exclusion reasons all fail.
- Confirmed failed reconciliation writes structured evidence into `evidence/recovery/recovery_plan.json` and does not call `apply_plan_fn`.
- Confirmed successful reconciliation permits only the mocked execution boundary.
- Confirmed visible score 100, 50 and 0 rows remain non-retryable and score quality is not converted into recovery eligibility.
- Confirmed judge-side pending/failure states remain non-generation-recovery states.
- Confirmed primary raw evidence remains byte-identical across classification, successful reconciliation and failed reconciliation tests.
- Confirmed no post-execution recovery-result semantics, terminal attempt ledger redesign, or Stage 2B completion rules were implemented.
- Confirmed closed Judge subsystem tests remain compatible.

## Defects found and fixed
- Found native execution tests relying on the old unattributed-plan compatibility path. Fixed by giving mocked native plans explicit source-row hashes.
- Found failed reconciliation evidence was not persisted before raising. Fixed by separating non-raising evidence calculation from the fail-closed assertion and persisting the evidence before the execution boundary.
- Found the planner still had a family-route omission path for recovery-eligible rows. Fixed by allowing canonical policy eligibility to override the legacy family/difficulty omission guards for existing primary rows.

## Deferred to Stage 2B
- Post-execution reconciliation.
- Child-run outcome interpretation and terminal recovery completion rules.
- Final executable-lane recovery-result semantics.
- Recovery attempt terminal ledger redesign.

## Safety confirmation
- No real Ollama, llama.cpp, Selene, judge, benchmark, campaign recovery, GPU, model pull/delete, service/KV mutation, adoption, canonical ranking mutation or push occurred.
- Stage 2B was not started.

## Corrective pass after independent review
- Corrective baseline: `38f1b8acd301597214b7bf5bc4a25becda115d50`.
- Stage 2B remains not approved and was not started.
- Required files re-read before corrective work: `AGENTS.md`, `CODEX_START.md`, `PR_RC21POST1_ACCEPTANCE_CONTROLS.md`, `codex_prompts/stage2A.md`, `codex_logs/Session 3/stage2A.md`.
- Residual defects addressed:
  - native source-hash set reconciliation did not validate task/model semantics for each source attribution;
  - legacy transient text fallback was too broad because arbitrary prose containing a digit `5` could become transient recovery eligibility.
- Implementation:
  - added narrow `LEGACY_TRANSIENT_TEXT_RE` for legacy generation transport failures, preserving typed `timeout` / `transient_backend_failure`;
  - removed the broad `" 5"` / loose HTTP text classifier;
  - added native action declared-task extraction and invalid attribution evidence;
  - validated every native `source_row_hashes[task] = hash` against the exact primary row resolved by hash;
  - rejected source hash not found, ambiguous hash, task/source mismatch, action task mapping mismatch, missing task source mapping, model mismatch where digest is absent, and digest mismatch where both digests are present;
  - kept grouped multi-task action support by validating each task/hash pair independently;
  - kept failed reconciliation persistence before the execution boundary.
- Tests added/extended:
  - wrong task mapping fails;
  - source mapping key absent from `action.tasks` fails;
  - declared action task missing a source mapping fails;
  - contradictory model digest fails;
  - matching task/model/digest succeeds;
  - grouped multi-task attribution succeeds;
  - grouped attribution with one wrong pair fails;
  - arbitrary digit-5 prose is not transient;
  - narrow legacy timeout/HTTP 5xx text remains generation-transient;
  - same HTTP 500 text in the judge lane is not generation recovery;
  - typed transient generation and judge-lane behavior remain unchanged;
  - semantic attribution failure prevents `apply_plan_fn`, persists invalid attribution evidence, and preserves primary raw bytes.
- Corrective validation:
  - `.venv/bin/python -m pytest tests/test_stage2a_recovery_reconciliation.py -q` - passed, 54 tests.
  - `.venv/bin/python -m pytest tests/test_repair.py tests/test_rc9_context_repair.py tests/test_rc10_repair_truth.py -q` - passed, 46 tests, 11 skipped.
  - `.venv/bin/python -m pytest tests/test_campaign_recovery_matrix.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign.py tests/test_campaign_final_acceptance.py tests/test_campaign_runtime_identity_resume.py tests/test_campaign_package_integrity.py tests/test_cli_subcommands.py tests/test_config_validation.py tests/test_stage1a_judge_policy.py tests/test_stage1b_judge_qualification.py tests/test_stage1c_model_roles.py tests/test_stage1d_judge_integration.py tests/test_judge_dumps.py -q` - passed, 203 tests.
  - Combined corrective validation over Stage 2A, repair, campaign, CLI/config and Judge files - passed, 303 tests, 11 skipped.
  - `.venv/bin/python -m ruff check llm_modelbench/campaign.py tests/test_stage2a_recovery_reconciliation.py tests/test_campaign.py tests/test_campaign_recovery_matrix.py` - passed.
  - `.venv/bin/python -m compileall -q llm_modelbench tests/test_stage2a_recovery_reconciliation.py` - passed.
  - `git diff --check` - passed.
- Corrective manual inspection:
  - source hashes cannot be attached to the wrong task;
  - action task declarations and `source_row_hashes` keys must agree for native plans;
  - grouped actions are checked per task/hash pair;
  - contradictory positive digest evidence fails;
  - matching digest remains authoritative and permits aliases where both digests match;
  - invalid semantic attribution blocks `apply_plan_fn` and persists `invalid_planned_attributions`;
  - legacy transient recognition no longer matches arbitrary digit `5` prose;
  - judge-lane text cannot become generation recovery;
  - no Stage 2B post-execution semantics were implemented.
