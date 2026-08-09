# Stage 2B log - Recovery post-execution reconciliation and terminal states

## Start
- Stage 2A was independently reviewed and CLOSED / APPROVED by Cristian.
- Stage 2B was explicitly approved by Cristian.
- Stage 3A and later stages are not approved and must not be started.
- Required files read before editing:
  - `PR_RC21POST1_ACCEPTANCE_CONTROLS.md` Stage 2B line;
  - `codex_prompts/stage2B.md`;
  - `codex_logs/Session 3/stage2A.md`.
- Current approved Stage 2A HEAD: `420b84cf06aa2dd75ddbaf20752b9b39be2d063d`.
- Actual Stage 2B starting HEAD: `420b84cf06aa2dd75ddbaf20752b9b39be2d063d`.
- Worktree has pre-existing unrelated/untracked items including deleted `codex_log/stage1*.md`, untracked prompt/spec/log assets, and the Stage 2A independent audit log. These are preserved.
- Scope: Stage 2B only. No Stage 3A, Stage 4, Stage 5, watcher, telemetry, KV/context, real evidence recovery, real judging, or real model work.
- Safety: synthetic/mocked tests only; no real Ollama, llama.cpp, GPU, model/service mutation, benchmark, recovery execution against real evidence, adoption, or push.

## Requirements tracked
- Authoritative post-execution reconciliation for every Stage 2A planned source row.
- Per-row outcome accounting for grouped actions.
- Terminal recovery dispositions for scored/recovered, thinking-only, empty-output, transient exhausted, measured capability failure, unresolved/harness failure and exclusions where applicable.
- Narrow post-execution legacy transient classification matching Stage 2A.
- No score-fishing after recovery: visible score 100, partial and 0 are terminal evidence.
- Executable-lane recovery result semantics remain task-family independent.
- Child/result provenance must validate source hash, task, model/digest, action, child, attempt and policy evidence.
- Persist deterministic Stage 2B reconciliation evidence.
- Integrate only enough effective/readiness behavior to avoid misleading pending retry after terminal recovery.
- Reconciliation is idempotent and does not mutate primary raw evidence.

## Implementation
- Added `reconcile_recovery_post_execution(...)` in `llm_modelbench/campaign.py` as the single Stage 2B post-execution accounting path.
- Planned recovery source identities come from the Stage 2A native plan `source_row_hashes`.
- Recovery outcome identities come from copied child `raw_results.jsonl` rows when present, or from unambiguous mocked/result action rows mapped through the Stage 2A plan.
- Persisted structured `post_execution_reconciliation` in `evidence/recovery/recovery_result.json`.
- Fail closed with `recovery_post_execution_incomplete` after persisting evidence when post-execution reconciliation is not exact.
- Added per-row terminal disposition normalization for:
  - visible/scorable recovered result, including score 100, partial and zero;
  - terminal thinking-only;
  - terminal empty-output;
  - terminal transient timeout/backend failure;
  - measured capability failure;
  - harness/unresolved recovery failure.
- Reused Stage 2A narrow legacy transport matching for post-execution transient classification and removed broad digit-5 post-recovery transient logic.
- Added child/result attribution validation for source hash, task, action mapping, model/digest, duplicates, unexpected sources, missing sources and ambiguous source hashes.
- Grouped actions remain supported by validating each task/source pair independently.
- Updated readiness child-row loading so invalid persisted post-execution reconciliation does not become effective recovered evidence.
- Updated one Stage 2A compatibility test to assert that successful pre-execution reconciliation still calls `apply_plan_fn`, while Stage 2B now fails afterward if no post-execution outcome exists.

## Tests added/changed
- Added `tests/test_stage2b_recovery_post_execution.py`.
- Covered:
  - one planned row to visible/scored recovery;
  - visible recovered scores 100, 50 and 0 as terminal scored evidence;
  - terminal thinking-only, empty-output and typed transient outcomes;
  - narrow legacy HTTP 500/503 terminal transient;
  - arbitrary digit-5 prose not terminal transient;
  - missing planned outcome;
  - unexpected child source;
  - wrong child task;
  - contradictory child digest;
  - duplicate child attribution;
  - grouped action all rows resolved;
  - grouped action one missing member;
  - grouped action one invalid member;
  - executable empty-output and thinking-only reconciliation;
  - visible executable zero-score as terminal evidence;
  - primary evidence immutability and idempotency;
  - persisted post-reconciliation evidence after execution;
  - invalid post-reconciliation persistence before fail-closed raise;
  - invalid post-reconciliation not used as effective recovered evidence.

## Validation
- `.venv/bin/python -m pytest tests/test_stage2b_recovery_post_execution.py -q` - passed, 28 tests.
- `.venv/bin/python -m pytest tests/test_repair.py tests/test_rc9_context_repair.py tests/test_rc10_repair_truth.py tests/test_campaign_recovery_matrix.py tests/test_stage2a_recovery_reconciliation.py -q` - passed, 119 tests, 11 skipped.
- `.venv/bin/python -m pytest tests/test_campaign_recovery_matrix.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign.py tests/test_campaign_final_acceptance.py tests/test_campaign_runtime_identity_resume.py tests/test_campaign_package_integrity.py tests/test_cli_subcommands.py tests/test_config_validation.py tests/test_stage1a_judge_policy.py tests/test_stage1b_judge_qualification.py tests/test_stage1c_model_roles.py tests/test_stage1d_judge_integration.py tests/test_judge_dumps.py -q` - passed, 203 tests.
- Combined Stage 2B, recovery, campaign, CLI/config and closed Judge validation:
  - `.venv/bin/python -m pytest tests/test_stage2b_recovery_post_execution.py tests/test_repair.py tests/test_rc9_context_repair.py tests/test_rc10_repair_truth.py tests/test_campaign_recovery_matrix.py tests/test_stage2a_recovery_reconciliation.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign.py tests/test_campaign_final_acceptance.py tests/test_campaign_runtime_identity_resume.py tests/test_campaign_package_integrity.py tests/test_cli_subcommands.py tests/test_config_validation.py tests/test_stage1a_judge_policy.py tests/test_stage1b_judge_qualification.py tests/test_stage1c_model_roles.py tests/test_stage1d_judge_integration.py tests/test_judge_dumps.py -q` - passed, 331 tests, 11 skipped.
- `.venv/bin/python -m ruff check llm_modelbench/campaign.py tests/test_stage2b_recovery_post_execution.py tests/test_stage2a_recovery_reconciliation.py` - passed.
- `.venv/bin/python -m compileall -q llm_modelbench tests/test_stage2b_recovery_post_execution.py tests/test_stage2a_recovery_reconciliation.py` - passed.
- `git diff --check` - passed.

## Manual inspection
- Confirmed every Stage 2A planned source is represented exactly once by recovered or terminal post-execution evidence for exact success.
- Confirmed grouped action resolution is per row and one missing/invalid member prevents exact reconciliation.
- Confirmed action-level success cannot hide a missing planned row.
- Confirmed recovered visible score 0 is preserved as `scored`, not retried.
- Confirmed no score-fishing path was introduced.
- Confirmed broad digit-5 terminal transient logic is gone from `_terminal_after_recovery`.
- Confirmed malformed or contradictory child attribution cannot become effective evidence.
- Confirmed primary raw evidence remains byte-identical in tests.
- Confirmed Judge subsystem tests remain green.
- Confirmed no Stage 3A or later work was started.

## Deferred
- Stage 3A supersession redesign.
- Stage 3B supersession CLI/effective ledger work.
- Stage 4 campaign config/init/execute redesign.
- Stage 5 full acceptance/readiness integration.
- Real q8/q4 fallback validation, real model inference, real recovery, real judging and Selene qualification.

## Safety confirmation
- No real Ollama, llama.cpp, Selene, judge, benchmark, campaign recovery against production evidence, GPU, model pull/delete, service/KV mutation, adoption, canonical ranking mutation or push occurred.

## Corrective pass after independent review
- Corrective baseline: `065262c5d54fbae4b33ab4ee8786fed8367bd5f7`.
- Stage 3A is not approved and was not started.
- Required files re-read before corrective work: `AGENTS.md`, `CODEX_START.md`, `PR_RC21POST1_ACCEPTANCE_CONTROLS.md`, `codex_prompts/stage2B.md`, `codex_logs/Session 3/stage2B.md`.
- Residual defects addressed:
  - post-execution provenance recorded action, child, attempt and policy fields but did not validate all of them;
  - sequential bounded attempts for one source were treated as duplicate final outcomes;
  - execution resolved evidence with global `child_rows OR action_result_rows`, which could suppress unrelated action-result evidence.
- Implementation:
  - validates child/action `action_id` against the Stage 2A owning action;
  - rejects native child evidence with missing action ID, missing child run ID, invalid attempt number, or missing/mismatched policy version;
  - validates policy version against the repair plan policy version when present;
  - preserves deterministic `attempt_history`;
  - derives exactly one `final_per_row_outcomes` entry per planned source from the highest valid bounded attempt;
  - permits legitimate multi-attempt bounded histories without duplicate failure;
  - rejects duplicate/conflicting evidence for the same source/action/attempt;
  - distinguishes `evidence_source = child_raw` from `evidence_source = action_result`;
  - suppresses aggregate action-result attempt metadata when valid child raw evidence exists for that exact source/task/attempt, but still allows explicit row-outcome action metadata to reveal contradictions;
  - resolves child raw and action-result evidence per source instead of globally.
- Tests added/extended:
  - wrong action ID fails;
  - missing native child action ID fails;
  - invalid native attempt numbers fail;
  - contradictory native policy version fails;
  - complete child action/attempt/policy provenance succeeds;
  - legitimate two-attempt chain does not fail as duplicate;
  - three-attempt chain ending in visible score 0 has final `scored` outcome;
  - repeated bounded transient attempts end in terminal transient;
  - same action/same attempt incompatible rows fail;
  - incompatible child and explicit action-result row outcomes fail;
  - deterministic attempt history and one final outcome per source;
  - mixed child/action-result evidence resolves H1 and H2 independently;
  - child evidence for one source does not suppress action-result evidence for another source.
- Corrective validation:
  - `.venv/bin/python -m pytest tests/test_stage2b_recovery_post_execution.py -q` - passed, 44 tests.
  - `.venv/bin/python -m pytest tests/test_stage2a_recovery_reconciliation.py tests/test_campaign_recovery_matrix.py tests/test_repair.py tests/test_rc9_context_repair.py tests/test_rc10_repair_truth.py -q` - passed, 119 tests, 11 skipped.
  - `.venv/bin/python -m pytest tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign.py tests/test_campaign_final_acceptance.py tests/test_campaign_runtime_identity_resume.py tests/test_campaign_package_integrity.py tests/test_cli_subcommands.py tests/test_config_validation.py tests/test_stage1a_judge_policy.py tests/test_stage1b_judge_qualification.py tests/test_stage1c_model_roles.py tests/test_stage1d_judge_integration.py tests/test_judge_dumps.py -q` - passed after policy-version correction.
  - Final combined corrective validation:
    `.venv/bin/python -m pytest tests/test_stage2b_recovery_post_execution.py tests/test_stage2a_recovery_reconciliation.py tests/test_campaign_recovery_matrix.py tests/test_repair.py tests/test_rc9_context_repair.py tests/test_rc10_repair_truth.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign.py tests/test_campaign_final_acceptance.py tests/test_campaign_runtime_identity_resume.py tests/test_campaign_package_integrity.py tests/test_cli_subcommands.py tests/test_config_validation.py tests/test_stage1a_judge_policy.py tests/test_stage1b_judge_qualification.py tests/test_stage1c_model_roles.py tests/test_stage1d_judge_integration.py tests/test_judge_dumps.py -q` - passed, 347 tests, 11 skipped.
  - `.venv/bin/python -m ruff check llm_modelbench/campaign.py tests/test_stage2b_recovery_post_execution.py tests/test_stage2a_recovery_reconciliation.py` - passed.
  - `.venv/bin/python -m compileall -q llm_modelbench tests/test_stage2b_recovery_post_execution.py tests/test_stage2a_recovery_reconciliation.py` - passed.
  - `git diff --check` - passed.
- Corrective manual inspection:
  - action identity cannot contradict the plan;
  - native child/attempt/policy provenance is checked truthfully;
  - bounded attempt history is retained and no longer mistaken for duplicate final evidence;
  - duplicate or conflicting same-attempt/final evidence still fails;
  - exactly one final outcome is produced per planned source on exact success;
  - mixed child/action-result evidence resolves per source;
  - child evidence for one source cannot suppress action-result evidence for another;
  - visible final score 0 remains `scored`;
  - no Stage 3A or later work leaked in;
  - no prohibited real-host/model work occurred.
