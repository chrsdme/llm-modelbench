# Stage 1A log — Judge policy, canonical capability source, candidate selection

## Session context
- Branch: `rc21-post1-topology-budget`
- Baseline commit: `555028e925e7f35c89b7ec7f15432bcd80aff811`
- Approved stage: Stage 1A only
- Real-host/model work: prohibited; only synthetic/mock tests and static checks are in scope.

## Instructions read
- `AGENTS.md`
- `CODEX_START.md`
- `PR_RC21POST1_ACCEPTANCE_CONTROLS.md`
- `codex_prompts/stage1A.md`

## Initial analysis
- Existing judge selection lives in `llm_modelbench/campaign.py`.
- Existing selection uses `_judge_candidate_is_generative`, a judge-local capability interpretation.
- Canonical runtime/planning capability families are resolved by `llm_modelbench.classify.families_for(...)` and capability interrogation payloads use `supported_families`.
- `Config.judge_model` exists but campaign post-hoc automatic selection currently only passes `Config.judge_candidates`, so the configured primary is not first in selection order.
- Existing selection returns only the chosen candidate/qualification chain; Stage 1A requires structured selection information with requested primary, fallbacks, automatic candidates, exclusions, rejection reasons, and final eligible order.

## Implementation notes
- Added `JudgePolicy` and `JudgeSelectionResult` in `llm_modelbench.campaign`.
- Replaced the judge-local raw-capability allowlist with judge eligibility based on canonical `families_for(...)` / `families_from_capabilities(...)` routing representation.
- Judge eligibility now requires canonical text-generation support and fails closed for unknown/non-generative capabilities.
- Embedding-only, reranker, and vision-only/non-text candidates are rejected before fallback.
- Configured primary is represented separately from ordered configured fallbacks and is evaluated first.
- Added explicit `Config.judge_allow_excluded_primary`; default is `False`.
- Removed hard-coded default judge models from `Config`; `judge_model` defaults to empty and `judge_candidates` defaults to an empty list. Operators can still configure a primary/fallbacks through config/env/CLI.
- Qwen-family exclusion remains generic token/family based and applies to automatic candidates and configured fallbacks by default.
- Excluded configured primary is eligible only when `judge_allow_excluded_primary` is true.
- Candidate ordering is duplicate-free and deterministic: configured primary, configured fallbacks, then deterministic automatic inventory order.
- Selection result records requested primary, configured fallbacks, automatic candidates, exclusions, rejection reasons, final eligible order, selected candidate, and policy metadata.
- Campaign post-hoc judging now persists structured `judge_policy_selection` in `evidence/judge/judge_selection.json`.
- Did not implement Stage 1B qualification framework, Stage 1C roles/no-self-judging, or Stage 1D judge provenance integration.

## Validation log
- First focused test run found unknown raw capabilities were still eligible through generic name-hint text fallback; fixed the judge gate so supplied unknown capabilities fail closed unless canonical capability mapping resolves text.
- Focused test after fix: `17 passed in 0.06s`.
- Manual requirement review found hard-coded default named judges in `Config`; removed those defaults and updated the mocked campaign persistence assertion to expect inventory-driven automatic selection.
- Final focused/adjacent tests:
  - `.venv/bin/python -m pytest tests/test_campaign.py tests/test_campaign_final_acceptance.py tests/test_rc21_post1_acceptance_repairs.py tests/test_stage1a_judge_policy.py tests/test_config_validation.py tests/test_cli_subcommands.py -q`
  - Result: `89 passed in 1.49s`.
- Static/check results:
  - `.venv/bin/python -m ruff check llm_modelbench/campaign.py llm_modelbench/cli.py llm_modelbench/config.py tests/test_stage1a_judge_policy.py tests/test_campaign_final_acceptance.py` — passed.
  - `.venv/bin/python -m compileall -q llm_modelbench tests/test_stage1a_judge_policy.py tests/test_campaign_final_acceptance.py` — passed.
  - `git diff --check` — passed.
- Manual inspection:
  - Capability source: judge code delegates to canonical classification helpers and persisted `supported_families`.
  - Ordering: primary/fallback/automatic order is deterministic and duplicate-free.
  - Rejection evidence: missing primary, exclusions, cohort conflicts, non-generative capabilities, unknown capabilities, and context floor preserve structured reasons before fallback.
  - Config semantics: `Config.judge_model` is primary, `judge_candidates` are fallbacks, `judge_allow_excluded_primary` is the explicit override.
  - Stage isolation: no Stage 1B qualification framework, Stage 1C role semantics, or Stage 1D full provenance work started.
  - Real-host/model prohibition: no real inference, judge qualification against live backends, benchmarks, GPU tests, model pulls/deletes, recovery against production evidence, or pushes were performed.

## Corrective pass — review findings
- Baseline for corrective pass: `f9836f47b56ef9f0d01cc42bce7779fe8a6c6d28`.
- Approved scope: Stage 1A corrective pass only; Stage 1B not approved.
- Re-read `AGENTS.md`, `CODEX_START.md`, `PR_RC21POST1_ACCEPTANCE_CONTROLS.md`, and `codex_prompts/stage1A.md`.

### Corrective implementation
- Replaced nondeterministic tied cohort majority resolution with `Counter` plus deterministic family-name tie-break.
- Removed duplicate qualification-time selection rebuild. Campaign CLI now uses:
  - `Config`
  - `JudgePolicy.from_config(...)`
  - `build_judge_selection(...)`
  - `select_qualified_campaign_judge(client, judge_selection)`
- `select_qualified_campaign_judge(...)` now consumes `JudgeSelectionResult.final_eligible_order` directly.
- Fixed empty exclusion semantics:
  - Defaults are applied when creating `JudgePolicy` from config.
  - Explicit empty exclusion tuples/lists remain empty after policy creation and through the legacy wrapper.
- Added deterministic inventory identity resolution:
  - same name + same digest deduplicates safely;
  - same name + different digest is rejected fail-closed with `conflicting_candidate_identity` independent of inventory order.
- Clarified capability evidence contract:
  - raw runtime capabilities are authoritative when present and are resolved via canonical `families_for(...)`;
  - `supported_families` is accepted only when raw capabilities are absent, because it is canonical when produced by the capability interrogation/planning pipeline;
  - contradictory raw embedding capability plus stale `supported_families=["text"]` is rejected.
- Replaced arbitrary substring family exclusion with canonical family identity matching where metadata exists, and a narrow controlled name fallback for Qwen-family names.
- Added a CLI guard so `judge-dumps` fails clearly when `Config.judge_model` is empty instead of constructing/invoking an empty judge model.

### Corrective validation
- Focused corrective tests:
  - `.venv/bin/python -m pytest tests/test_stage1a_judge_policy.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign.py::test_judge_selection_excludes_cohort_and_prefers_calibrated_other_family tests/test_campaign_final_acceptance.py::test_cli_forced_mock_campaign_runs_full_terminal_lifecycle tests/test_cli_subcommands.py::test_judge_dumps_requires_configured_or_cli_judge_model_before_client -q`
  - Result after fixing a test fixture bug: `31 passed in 0.87s`.
- Final directly affected tests:
  - `.venv/bin/python -m pytest tests/test_stage1a_judge_policy.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign.py tests/test_campaign_final_acceptance.py tests/test_cli_subcommands.py tests/test_config_validation.py tests/test_judge_dumps.py -q`
  - Result: `103 passed in 1.59s`.
- Static/check results:
  - `.venv/bin/python -m ruff check llm_modelbench/campaign.py llm_modelbench/cli.py llm_modelbench/config.py tests/test_stage1a_judge_policy.py tests/test_rc21_post1_acceptance_repairs.py tests/test_cli_subcommands.py tests/test_campaign_final_acceptance.py` — passed.
  - `.venv/bin/python -m compileall -q llm_modelbench tests/test_stage1a_judge_policy.py tests/test_rc21_post1_acceptance_repairs.py tests/test_cli_subcommands.py tests/test_campaign_final_acceptance.py` — passed.
  - `git diff --check` — passed.
- Manual inspection:
  - No remaining `max(set(cohort_families), key=cohort_families.count)`.
  - Campaign CLI no longer reconstructs judge policy for qualification after persistence selection.
  - No remaining `excluded_families or ["qwen"]` in the corrected qualification path.
  - `judge-dumps` cannot proceed with an empty judge model.
  - Stage isolation maintained; no Stage 1B qualification framework, Stage 1C roles, or Stage 1D provenance integration was implemented.
  - No real-host/model work was performed.
