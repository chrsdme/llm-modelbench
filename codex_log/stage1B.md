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
