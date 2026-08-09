# Stage 1D log — Judge provenance/evidence and mocked integration

## Start
- Stage 1C approval was explicitly stated by Cristian in the Stage 1D prompt.
- Required files read completely: `AGENTS.md`, `CODEX_START.md`, `PR_RC21POST1_ACCEPTANCE_CONTROLS.md`, `codex_prompts/stage1D.md`.
- Baseline commit: `f29704ccbc9cfec04c5ecf6b28c1564a511a0925`.
- Scope: Stage 1D only. No Stage 2A work.
- Safety: mocks, fixtures, synthetic evidence and offline/static checks only; no real inference, judge calls, benchmarks, GPU tests, model/service mutation, adoption, push, or real evidence mutation.

## Work log
- Inspecting existing Stage 1A/1B/1C judge selection, qualification, no-self-judging, judge dump, campaign lifecycle, effective-row and readiness code before editing.
- Existing Stage 1C implementation already covered per-row independent fallback, pending `awaiting_independent_judge`, fallback-row idempotency, pending idempotency, manual judge-dumps fail-closed behavior, bounded qualification, and readiness propagation for independent-judge pending rows.
- Implemented Stage 1D evidence expansion in `llm_modelbench/judge_dumps.py`:
  - persisted qualified judge pool evidence with stable identities and qualification protocol/disposition;
  - persisted source identity, actual per-row judge identity/digest, identity relation, judge mode configuration, policy versions, pool signature, output hashes, and judgement attempt chains;
  - recorded structural judging failures as bounded rejected candidate attempts;
  - added fallback to the next qualified independent judge after structural judging incompatibility;
  - added `judge_exhausted_unavailable` sidecar state when all independent qualified judges fail structurally;
  - added idempotency for unchanged exhausted sidecars so structurally incompatible judges are not re-hammered on rerun.
- Implemented Stage 1D readiness/effective-row integration in `llm_modelbench/campaign.py`:
  - overlays `judge_exhausted_unavailable`;
  - carries actual per-row judge digest/identity/relation/failure disposition into effective-row provenance;
  - blocks readiness as `not_ready_external_judge` for exhausted judge rows.
- Added `tests/test_stage1d_judge_integration.py` covering Stage 1D mocked flow, evidence, structural qualification rejection, structural judging fallback/exhaustion, rerun idempotency, manual designation truthfulness, malformed/incomplete identity fail-closed behavior, and campaign readiness propagation.

## Validation
- `.venv/bin/python -m pytest tests/test_stage1d_judge_integration.py -q` — passed, 7 tests.
- `.venv/bin/python -m pytest tests/test_stage1d_judge_integration.py tests/test_stage1c_model_roles.py tests/test_stage1b_judge_qualification.py tests/test_stage1a_judge_policy.py tests/test_judge_dumps.py tests/test_campaign.py tests/test_campaign_final_acceptance.py tests/test_campaign_adoption_transaction.py tests/test_cli_subcommands.py tests/test_config_validation.py tests/test_capability_workflow.py tests/test_backend.py -q` — passed, 176 tests.
- `.venv/bin/python -m ruff check llm_modelbench/judge_dumps.py llm_modelbench/campaign.py tests/test_stage1d_judge_integration.py` — passed.
- `.venv/bin/python -m compileall -q llm_modelbench tests/test_stage1d_judge_integration.py` — passed.
- `git diff --check` — passed.

## Manual inspection
- Compared implementation against Stage 1D requirements. The requested/configured judge policy and deterministic candidate order remain in `judge_selection.json`; qualification protocol/results and bounded coverage remain in selection evidence; per-row sidecars now identify the actual row judge and failure/fallback attempt chain; effective rows and readiness consume the sidecar state.
- Defect found during inspection: the first structural-fallback edit would have allowed unchanged `judge_exhausted_unavailable` rows to retry the same incompatible judge on a rerun. Fixed by adding exhausted-sidecar idempotency and a regression test.
- No Stage 2A work was started.
- No prohibited real-host/model work occurred.

## Corrective pass
- Moved the Stage 1D operational log from `codex_logs/stage1D.md` to `codex_logs/Session 2/stage1D.md` as requested. Session 1 history was not modified.
- Replaced text-derived post-hoc judge failure classification with typed backend-result preservation:
  - added backward-compatible structured internal judge helpers in `llm_modelbench/judge.py`;
  - preserved public `judge_single(...)` / `judge_panel(...)` tuple behavior;
  - carried `error_kind`, `http_status`, backend failure disposition, and request contract version through `judge_dumps` sidecar samples.
- Preserved non-structural judgement failures as explicit `judge_error` sidecars with typed `failure_disposition` values including `timeout`, `transient_backend_failure`, `backend_failure`, and `judge_output_failure`.
- Updated sidecar overlay and readiness/effective rows so non-structural judge failures do not disappear, keep `score=None`, do not become model-quality failures, and block readiness as external judge work.
- Added deterministic compatibility/execution fingerprints:
  - per-judge compatibility fingerprint includes stable judge identity, runtime identity, qualification protocol/disposition, judge mode configuration, and post-hoc request contract version;
  - row execution fingerprint covers the ordered qualified pool compatibility fingerprints and judge config;
  - unchanged exhausted rows no-op;
  - expanded pools reuse known-bad structural evidence and skip calling the known-bad judge;
  - changed judge config/fingerprint reconsiders prior structural failures.
- Added a mocked campaign artifact regression that links `evidence/judge/judge_selection.json`, `evidence/judge/judge_results.jsonl`, `evidence/effective_rows.jsonl`, and readiness evidence while distinguishing requested judge `j1` from actual per-row fallback judge `j2`.

## Corrective validation
- `.venv/bin/python -m pytest tests/test_stage1d_judge_integration.py -q` — passed, 12 tests.
- `.venv/bin/python -m pytest tests/test_judge_dumps.py tests/test_stage1c_model_roles.py tests/test_stage1d_judge_integration.py -q` — passed, 31 tests.
- `.venv/bin/python -m pytest tests/test_stage1d_judge_integration.py tests/test_stage1c_model_roles.py tests/test_stage1b_judge_qualification.py tests/test_stage1a_judge_policy.py tests/test_judge_dumps.py tests/test_campaign_final_acceptance.py tests/test_campaign_adoption_transaction.py tests/test_backend.py -q` — passed, 95 tests.
- `.venv/bin/python -m pytest tests/test_stage1d_judge_integration.py tests/test_stage1c_model_roles.py tests/test_stage1b_judge_qualification.py tests/test_stage1a_judge_policy.py tests/test_judge_dumps.py tests/test_campaign.py tests/test_campaign_final_acceptance.py tests/test_campaign_adoption_transaction.py tests/test_cli_subcommands.py tests/test_config_validation.py tests/test_capability_workflow.py tests/test_backend.py -q` — passed, 181 tests.
- `.venv/bin/python -m ruff check llm_modelbench/judge.py llm_modelbench/judge_dumps.py llm_modelbench/campaign.py tests/test_stage1d_judge_integration.py` — passed.
- `.venv/bin/python -m compileall -q llm_modelbench tests/test_stage1d_judge_integration.py` — passed.
- `git diff --check` — passed.

## Corrective manual inspection
- Confirmed `llm_modelbench/judge_dumps.py` no longer performs authoritative HTTP/status text matching; post-hoc sidecars consume structured judge result fields.
- Confirmed typed backend failures preserve `error_kind`/`http_status` where supplied and map statuses by typed fields: 400/404/405/415/422 structural, 408 timeout, 429 transient, 5xx transient, direct `TimeoutError` timeout.
- Confirmed `judge_error` sidecars are included by `latest_judge_sidecars(...)`, overlay into rows, and feed effective/readiness evidence.
- Confirmed known-bad structural fingerprint reuse skips a previously failed judge after pool expansion, while a changed `think` config changes the fingerprint and allows reconsideration.
- Confirmed campaign artifact regression proves requested/preferred judge `j1` and actual per-row judge `j2` remain distinct in persisted evidence.
- Corrective defect found and fixed: `scan_run(..., force=True)` initially referenced `previous` only assigned in the non-force path after adding prior-sidecar provenance. Fixed by assigning previous sidecars before the force branch and reran affected tests.
