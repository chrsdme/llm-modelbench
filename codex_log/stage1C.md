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

## Corrective pass — integration review findings
- Baseline for corrective pass: `a13f349dfe612f3c3e6a323f89ac460ae0ff754b`.
- Approved scope: Stage 1C corrective pass only; Stage 1D not approved.
- Re-read `AGENTS.md`, `CODEX_START.md`, `PR_RC21POST1_ACCEPTANCE_CONTROLS.md`, and `codex_prompts/stage1C.md`.

### Corrective implementation
- Made `judge_dumps.scan_run(...)` resolve the actual per-row judge before idempotency checks.
- Rerun idempotency now keys judged rows against the resolved per-row judge identity, not the global preferred judge name.
- Pending `awaiting_independent_judge` sidecars carry a deterministic judge-pool signature; unchanged reruns skip rather than appending duplicate pending rows.
- If the qualified judge pool changes and adds an independent judge, a previously pending row becomes eligible and can be judged.
- Removed fabricated `qualified=True` / `roles=["judge"]` evidence from legacy/manual `judge_dumps` calls.
- Legacy/manual judge-dumps now records a truthful `manual_unqualified_designation` pool and fails closed when independence cannot be proven.
- Added `stable_model_identity_relation(...)` with `same`, `independent`, and `indeterminate` states.
- Digest-backed independence is required when digest evidence is present; aliases with unresolved digest are not assumed independent merely because names differ.
- Added `latest_judge_sidecars(...)` and updated `apply_judgements(...)` so `awaiting_independent_judge` sidecars propagate into row state without claiming `posthoc_judged`.
- Updated readiness state calculation so `awaiting_independent_judge` blocks adoption as `not_ready_external_judge`.
- Added `apply_campaign_roles_to_judge_candidates(...)` so new campaign judge-candidate records receive explicit role evidence from judge policy/cohort, separate from capabilities.
- Normal campaign flow now persists explicit roles and role sources in judge selection evidence.
- Added bounded qualification helper `select_qualified_campaign_judges_for_rows(...)`; it qualifies candidates in Stage 1A order only until the qualified pool provides independent coverage for every subjective source row, then records skipped candidates as `not_considered_coverage_satisfied`.
- Adoption validation now requires exact per-row `source_model_digest` and `judge_model_digest` for judged sidecars.
- Adoption rejects actual self digest and incomplete/mixed identity; it no longer borrows top-level preferred judge digest for a sidecar row.

### Corrective validation
- Focused Stage 1C tests:
  - `.venv/bin/python -m pytest tests/test_stage1c_model_roles.py -q`
  - Result: `15 passed in 0.07s`.
- Directly affected Stage 1A/1B/judge/campaign/config/backend tests:
  - `.venv/bin/python -m pytest tests/test_stage1c_model_roles.py tests/test_stage1a_judge_policy.py tests/test_stage1b_judge_qualification.py tests/test_judge_dumps.py tests/test_campaign.py tests/test_campaign_final_acceptance.py tests/test_campaign_adoption_transaction.py tests/test_cli_subcommands.py tests/test_config_validation.py tests/test_capability_workflow.py tests/test_backend.py -q`
  - Result: `169 passed in 1.84s`.
- Broader offline campaign/Ollama-adjacent compatibility tests:
  - `.venv/bin/python -m pytest tests/test_campaign_runtime_identity_resume.py tests/test_campaign_adoption_transaction.py tests/test_campaign_cleanup_migration_hygiene.py tests/test_campaign_package_integrity.py tests/test_campaign_recovery_matrix.py tests/test_ollama_service_conflict_warning.py tests/test_ollama_service_active_unit.py tests/test_ollama_kv_broker_boundary.py -q`
  - Result: `139 passed in 1.13s`.
- Static/check results:
  - `.venv/bin/python -m ruff check llm_modelbench/campaign.py llm_modelbench/judge_dumps.py llm_modelbench/cli.py tests/test_stage1c_model_roles.py tests/test_judge_dumps.py tests/test_campaign_adoption_transaction.py` — passed.
  - `.venv/bin/python -m compileall -q llm_modelbench tests/test_stage1c_model_roles.py tests/test_judge_dumps.py tests/test_campaign_adoption_transaction.py` — passed.
  - `git diff --check` — passed.

### Corrective manual inspection
- Rerun/no-op behavior: scan resolves the same actual fallback judge and skips an already-judged row without appending.
- Pending rerun behavior: unchanged pending rows are skipped by matching source hash, mode, and judge-pool signature.
- Pending-to-judged transition: when a new independent judge appears, the pool signature changes and the row is eligible for judging.
- No fabricated qualification: manual judge-dumps does not mark the judge qualified and does not fabricate roles/digest.
- Alias identity safety: `indeterminate` identity is pending, not judged.
- Normal role propagation: new campaign candidate records get role evidence from judge policy/cohort, not from text capability.
- Bounded pool: qualification stops once independent coverage is satisfied and records unconsidered tail candidates.
- Adoption: judged sidecars must carry exact per-row source and judge digests; incomplete identity fails closed.
- No real Selene/Ollama/llama.cpp generation, real judge, real benchmark, or real campaign evidence was used.
- Stage 1D was not started.
