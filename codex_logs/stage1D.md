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
