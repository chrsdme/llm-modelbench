# Stage 1B log — Universal ModelBench judge qualification framework

## Session context
- Branch: `rc21-post1-topology-budget`
- Baseline commit: `d2d38c5237dba1ecb563cadf2b6161d0e42feadc`
- Approved stage: Stage 1B only
- Stage 1A approval: explicit human approval in the Stage 1B prompt
- Real-host/model work: prohibited; only mocks, fixtures, synthetic controls, static checks, and safe tests are in scope.

## Instructions read
- `AGENTS.md`
- `CODEX_START.md`
- `PR_RC21POST1_ACCEPTANCE_CONTROLS.md`
- `codex_prompts/stage1B.md`

## Initial analysis
- Existing Stage 1A candidate selection is model-agnostic and produces a deterministic eligible order.
- Existing `campaign.qualify_judge(...)` is still a small campaign-local probe using one arithmetic prompt and textual error matching.
- Stage 1B requires a reusable ModelBench-owned protocol with real control-flow checks, typed outcomes, and complete evidence.

## Implementation notes
- Added `llm_modelbench/judge_qualification.py` as the ModelBench-owned, model-agnostic judge qualification protocol module.
- Defined protocol version `judge-qualification-v1`.
- Defined structured request schema via `QualificationRequest` and deterministic synthetic controls via `QualificationControl`.
- Implemented score-response schema checks for numeric score range, confidence range, verdict payload, rubric adherence signal, and reference-use signal.
- Implemented pairwise-response schema checks for `winner: A|B|equal`.
- Added deterministic controls for:
  - obviously correct;
  - obviously wrong;
  - partial credit;
  - irrelevant answer;
  - unsupported hallucination;
  - malformed candidate output;
  - rubric adherence;
  - reference-answer use;
  - equal pair;
  - A-better;
  - B-better;
  - reversed-order pairwise controls generated from each pairwise case;
  - repeated stability controls.
- Implemented actual control-flow evaluation for backend/request compatibility, structured output parsing, parser/schema compliance, score range, good/bad discrimination, partial-credit sanity, rubric adherence, reference-answer use, pairwise reversal/order bias, repeat stability, malformed candidate output handling, timeout, structural incompatibility, and unsupported backend.
- Added typed backend failure classification using `error_kind`/`kind` and typed HTTP status fields; no textual `http 400` matching is used.
- Structural incompatibility, timeout, and unsupported backend reject immediately after the first typed failure.
- Qualification result includes protocol version, candidate model/digest/runtime identity/capability fields, per-control request/parse/evaluation evidence, aggregate disposition, checks, failure reasons, and timing.
- Updated `campaign.qualify_judge(...)` to delegate to the universal protocol after Stage 1A capability eligibility.
- Updated mocked campaign path to persist the new qualification evidence in `judge_selection.json`.
- Added deterministic offline qualification responses to `MockClient` only, so mock campaign lifecycle tests remain fully offline.
- Did not implement Stage 1C model-role/no-self-judging behavior.

## Validation log
- Focused Stage 1B tests initially exposed that `MockClient` did not understand the new qualification prompt, so the existing mocked campaign lifecycle produced no judged row. Fixed by adding mock-only qualification responses to `MockClient`.
- Manual inspection found the qualification helper was initially inserted before the tail of `MockClient._answer`, making later mock answers unreachable. Fixed by restoring the `_answer` tail before `_qualification_answer`.
- Ruff found one unused import in `judge_qualification.py`; removed it.
- Final directly affected tests:
  - `.venv/bin/python -m pytest tests/test_stage1b_judge_qualification.py tests/test_stage1a_judge_policy.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign.py tests/test_campaign_final_acceptance.py tests/test_judge_dumps.py tests/test_cli_subcommands.py tests/test_config_validation.py -q`
  - Result: `113 passed in 1.54s`.
- Static/check results:
  - `.venv/bin/python -m ruff check llm_modelbench/judge_qualification.py llm_modelbench/campaign.py llm_modelbench/ollama.py tests/test_stage1b_judge_qualification.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign_final_acceptance.py` — passed.
  - `.venv/bin/python -m compileall -q llm_modelbench tests/test_stage1b_judge_qualification.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign_final_acceptance.py` — passed.
  - `git diff --check` — passed.
- Manual inspection:
  - Protocol is ModelBench-owned and model-agnostic; no Selene/Prometheus-specific selection or qualification behavior.
  - Qualification is separate from live campaign judging and is fully mockable through fake backends.
  - Tests exercise real protocol calls, parser/schema validation, scoring checks, pairwise reversal checks, repeat stability, typed timeout/structural/unsupported dispositions, and fallback flow.
  - Campaign integration consumes the universal qualification result without adding role/no-self-judging logic.
  - No real Selene/Ollama/llama.cpp generation, real judge, real benchmark, or real campaign evidence was used.

## Corrective pass — review findings
- Baseline for corrective pass: `f850d10cf24702310402e32a4b9daeb6ed5dcab1`.
- Approved scope: Stage 1B corrective pass only; Stage 1C not approved.
- Re-read `AGENTS.md`, `CODEX_START.md`, `PR_RC21POST1_ACCEPTANCE_CONTROLS.md`, and `codex_prompts/stage1B.md`.

### Corrective implementation
- Added a real per-request timeout contract to `OllamaClient.chat(...)` and `MockClient.chat(...)`.
- `OllamaClient._post_stream(...)` now accepts an optional timeout and passes it to `urllib.request.urlopen(...)`; qualification timeout therefore controls that individual streaming request.
- Removed the unsafe generic `TypeError` retry-without-timeout path from qualification backend calls.
- Added explicit `transient_backend_failure` handling for operational transport/backend failures.
- Classified backend failures as:
  - `structural_incompatibility` for structural 4xx request incompatibility;
  - `unsupported_backend`;
  - `timeout` for typed timeout and HTTP 408;
  - `transient_backend_failure` for HTTP 429, 5xx, and operational transport/backend errors;
  - `malformed_judge_output`;
  - `schema_violation`;
  - quality-control failures.
- Operational backend failures now terminate the current candidate qualification attempt after the first failed control and allow higher-level fallback to continue.
- Replaced permissive first-object JSON extraction with strict raw JSON parsing: the response may contain only a JSON object plus surrounding whitespace.
- Enforced all declared score response fields:
  - numeric `score` in 0..100;
  - numeric `confidence` in 0..1;
  - non-empty string `verdict`;
  - boolean `rubric_adherence`;
  - boolean `reference_used`.
- Enforced all declared pairwise response fields:
  - `winner` in `A|B|equal`;
  - numeric `confidence` in 0..1;
  - non-empty string `verdict`.
- Added aggregate disposition `rejected_schema_violation`.

### Corrective validation
- Focused Stage 1B corrective tests:
  - `.venv/bin/python -m pytest tests/test_stage1b_judge_qualification.py -q`
  - Result: `17 passed in 0.08s`.
- Final directly affected tests:
  - `.venv/bin/python -m pytest tests/test_stage1b_judge_qualification.py tests/test_stage1a_judge_policy.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign.py tests/test_campaign_final_acceptance.py tests/test_judge_dumps.py tests/test_cli_subcommands.py tests/test_config_validation.py tests/test_capability_workflow.py tests/test_backend.py -q`
  - Result: `141 passed in 1.65s`.
- Static/check results:
  - `.venv/bin/python -m ruff check llm_modelbench/judge_qualification.py llm_modelbench/ollama.py llm_modelbench/campaign.py tests/test_stage1b_judge_qualification.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign_final_acceptance.py` — passed.
  - `.venv/bin/python -m compileall -q llm_modelbench tests/test_stage1b_judge_qualification.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign_final_acceptance.py` — passed.
  - `git diff --check` — passed.
- Manual inspection:
  - Production timeout propagation: qualification passes `timeout_seconds` to `client.chat`; `OllamaClient.chat` accepts `timeout`; `_post_stream` passes it to `urlopen`; think-parameter retry also preserves the same timeout.
  - Failure classification: HTTP 408 maps to timeout, 429/5xx and transport failures map to transient backend failure, structural 4xx maps to structural incompatibility.
  - Bounded call behavior: timeout, structural incompatibility, unsupported backend, and transient backend failure terminate after one control.
  - Schema validation: all declared score and pairwise fields are required and type/range checked.
  - Strict structured output: raw JSON and whitespace-wrapped JSON pass; prose before/after JSON fails as malformed output.
  - Stage isolation maintained; no Stage 1C role or no-self-judging logic was implemented.
  - No real Selene/Ollama/llama.cpp generation, real judge, real benchmark, or real campaign evidence was used.

## Final corrective pass — residual contract defects
- Baseline for final corrective pass: `3b4a550397f12074c9d87d923b932d06fcc69d18`.
- Approved scope: final Stage 1B corrective pass only; Stage 1C not approved.
- Re-read `AGENTS.md`, `CODEX_START.md`, `PR_RC21POST1_ACCEPTANCE_CONTROLS.md`, and `codex_prompts/stage1B.md`.

### Final corrective implementation
- Preserved real Ollama transport timeouts as typed timeout payloads in `OllamaClient` error normalization.
- `_exception_payload(...)` now marks both direct `TimeoutError`/`socket.timeout` and transport wrappers whose typed `reason` is `TimeoutError`/`socket.timeout` with `error_kind="timeout"`.
- Did not infer timeout from error-message strings.
- Tightened qualification numeric schema validation so `score` and `confidence` accept only JSON number values represented as Python `int`/`float`.
- Explicitly reject `bool`, strings including numeric strings, `null`, arrays, objects, and non-finite numeric values.
- Preserved existing score/confidence range rules.

### Final corrective validation
- Focused Stage 1B tests:
  - `.venv/bin/python -m pytest tests/test_stage1b_judge_qualification.py -q`
  - Result: `21 passed in 0.11s`.
- Directly affected judge/config/campaign/backend tests:
  - `.venv/bin/python -m pytest tests/test_stage1b_judge_qualification.py tests/test_stage1a_judge_policy.py tests/test_campaign.py tests/test_cli_subcommands.py tests/test_config_validation.py tests/test_capability_workflow.py tests/test_backend.py tests/test_judge_dumps.py -q`
  - Result: `139 passed in 0.87s`.
- Additional campaign/Ollama-adjacent offline compatibility tests:
  - `.venv/bin/python -m pytest tests/test_campaign_runtime_identity_resume.py tests/test_campaign_final_acceptance.py tests/test_campaign_adoption_transaction.py tests/test_campaign_cleanup_migration_hygiene.py tests/test_campaign_package_integrity.py tests/test_campaign_recovery_matrix.py tests/test_ollama_service_conflict_warning.py tests/test_ollama_service_active_unit.py tests/test_ollama_kv_broker_boundary.py -q`
  - Result: `138 passed in 1.89s`.
- Static/check results:
  - `.venv/bin/python -m ruff check llm_modelbench/ollama.py llm_modelbench/judge_qualification.py tests/test_stage1b_judge_qualification.py` — passed.
  - `.venv/bin/python -m compileall -q llm_modelbench tests/test_stage1b_judge_qualification.py` — passed.
  - `git diff --check` — passed.
- Manual inspection:
  - Production timeout preservation: `OllamaClient.chat(...)` catches transport exceptions and normalizes direct or wrapped typed timeout exceptions to `ok=False, error_kind="timeout"`; qualification maps that to `rejected_timeout`.
  - No timeout classification depends on substring matching.
  - Qualification `_required_number(...)` no longer coerces values with `float(value)` unless the parsed JSON value is already an `int`/`float`; booleans and numeric strings fail as schema violations.
  - The final changes are limited to Stage 1B contracts and tests.
  - No Stage 1C model-role or no-self-judging logic was implemented.
  - No real Selene/Ollama/llama.cpp generation, real judge, real benchmark, or real campaign evidence was used.
