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
