# Session 6 - Stage 6 documentation/operator UX

## Session identity
- Session number: 6
- Stage: 6
- Baseline HEAD: 464ed3f68ffbb070fb7102509a97b4a6cffcb20c
- Branch: rc21-post1-topology-budget

## Files/specs read
- AGENTS.md
- CODEX_START.md
- PR_RC21POST1_ACCEPTANCE_CONTROLS.md
- codex_prompts/stage6.md
- codex_logs/Session 5/session5_20260810T002907Z.md
- README.md
- docs/CAMPAIGNS.md
- docs/USAGE.md
- docs/README.md
- docs/REPAIR.md
- docs/JUDGE_DUMPS.md
- llm_modelbench/cli.py
- llm_modelbench/campaign.py

## Requirements
- Document verified behavior from Stages 1A-5 only.
- Cover judge eligibility/precedence/Qwen/default override boundaries, qualification, roles/no-self-judging, provenance, recovery terminal states/no-score-fishing, immutable supersession, config/init/execute, resume/idempotency, readiness semantics, and evidence immutability.
- Do not document deferred watcher, telemetry, KV/context, real Selene qualification, benchmark expansion, real recovery/catch-up, production acceptance, or adoption as completed.
- Verify docs/examples and local help.

## Design decisions
- Added one focused public guide at docs/ACCEPTANCE_CONTROLS.md rather than scattering all acceptance-control detail across older docs.
- Updated README, docs/README.md, docs/USAGE.md, and docs/CAMPAIGNS.md with concise links and examples.
- Updated campaign argparse descriptions/help text only; no CLI execution semantics were changed.
- Added docs/UX tests that assert required wording and campaign help surfaces exist.

## Commands/tests
- `./.venv/bin/pytest -q tests/test_stage6_docs_ux.py tests/test_docs_hygiene.py` -> initial 7 passed, 3 failed; final 10 passed.
- `./.venv/bin/python -m llm_modelbench campaign --help` -> inspected.
- `./.venv/bin/python -m llm_modelbench campaign execute --help` -> inspected.
- `./.venv/bin/python -m llm_modelbench campaign supersede --help` -> inspected.
- `./.venv/bin/pytest -q tests/test_stage6_docs_ux.py tests/test_docs_hygiene.py tests/test_stage5_readiness_integration.py tests/test_stage4b_campaign_execute.py tests/test_stage4a_campaign_config.py tests/test_stage3b_supersession.py tests/test_stage3a_supersession.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign.py tests/test_campaign_package_integrity.py tests/test_campaign_final_acceptance.py tests/test_campaign_recovery_matrix.py tests/test_stage2a_recovery_reconciliation.py tests/test_stage2b_recovery_post_execution.py tests/test_repair.py tests/test_rc9_context_repair.py tests/test_rc10_repair_truth.py tests/test_stage1a_judge_policy.py tests/test_stage1b_judge_qualification.py tests/test_stage1c_model_roles.py tests/test_stage1d_judge_integration.py tests/test_judge_dumps.py` -> 400 passed, 11 skipped.
- `./.venv/bin/ruff check llm_modelbench tests/test_stage6_docs_ux.py tests/test_docs_hygiene.py tests/test_stage5_readiness_integration.py tests/test_stage4a_campaign_config.py tests/test_stage4b_campaign_execute.py tests/test_stage3a_supersession.py tests/test_stage3b_supersession.py` -> passed.
- `python -m compileall -q llm_modelbench tests/test_stage6_docs_ux.py tests/test_docs_hygiene.py` -> passed.
- `git diff --check` -> passed.

## Failures and corrections
- `tests/test_stage6_docs_ux.py` initially expected "immutable plan signature" in `campaign execute --help`; the help text said "config signature". Corrected the help description to say "immutable plan signature".
- `tests/test_docs_hygiene.py` initially failed on pre-existing local Codex audit logs containing absolute local paths and unrelated deleted tracked `codex_log/` files. Updated the hygiene test to treat `codex_log/`, `codex_logs/`, and `codex_prompts/` as local generated material and skip unrelated missing tracked paths during privacy scans. No local logs were cleaned or restored.

## Manual inspection
- Verified docs describe only implemented Stage 1A-5 controls and do not claim real Selene qualification, real recovery/catch-up, production acceptance, or adoption.
- Verified `campaign init`, `campaign execute --config`, and `campaign supersede` examples match current CLI surfaces.
- Verified CLI help edits are description/help-only and do not change execution behavior.
- Verified acceptance-control doc covers judge independence, recovery no-score-fishing, immutable supersession, config plan signatures, readiness blockers, and evidence immutability.
- Verified Stage 7 was not started during Stage 6.

## Safety confirmation
- Stage 6 only.
- No Stage 7 implementation started during Stage 6.
- No real inference, real recovery, real judging, real catch-up, real acceptance campaign, evidence adoption, push, model pull/delete, or service/GPU/KV mutation.
