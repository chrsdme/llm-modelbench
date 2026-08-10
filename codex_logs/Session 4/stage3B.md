# Session 4 - Stage 3B

## Session identity
- Session: 4
- Stage: Stage 3B - supersession CLI, effective ledger, synthetic catch-up integration
- Branch: `rc21-post1-topology-budget`
- Starting HEAD: `66e5265652d01e927fa6d846ab689c520b48fd3b`

## Files/specs read
- `AGENTS.md`
- `CODEX_START.md`
- `PR_RC21POST1_ACCEPTANCE_CONTROLS.md`
- `codex_prompts/stage3B.md`
- `codex_logs/Session 4/stage3A.md`

## Startup verification
- `git branch --show-current` -> `rc21-post1-topology-budget`
- `git rev-parse HEAD` -> `66e5265652d01e927fa6d846ab689c520b48fd3b`
- Stage 3A corrective closure audit: `codex_logs/Session 4/stage3A_codex_closure_audit_20260810T001717Z.log`

## Requirements
- Harden `campaign supersede` so operator input is anchored to immutable campaign evidence, not arbitrary row JSON.
- Validate campaign/run, row identity/hash, graph conflicts, chain safety, and useful preview/dry-run behavior.
- Resolve effective ledger supersession transitively using the validated Stage 3A graph.
- Preserve provenance in effective rows/readiness metadata.
- Add synthetic historical 65536 -> 66560 needle correction fixture/test.
- Prove old raw evidence remains unchanged and effective row is corrected.
- Add CLI tests.
- Do not run real model work, real catch-up benchmarks, real recovery/judging, acceptance evidence supersession, adoption, or push.

## Current Stage 3B audit
- Current CLI `campaign supersede` accepted arbitrary source row JSON and passed it directly into `record_supersession(...)`.
- Effective ledger/readiness previously applied only direct one-hop `supersession_map(...)` records.
- Stage 3A graph helpers existed and were suitable for deterministic transitive resolution, but were not yet used by effective rows.
- No CLI dry-run/preview path existed.
- No synthetic 65536 -> 66560 supersession/catch-up fixture existed.

## Design decisions
- Added `campaign.primary_row_by_hash(...)` to resolve a source row from immutable campaign primary evidence by exact row hash, failing closed on missing or duplicate source hashes.
- Hardened `campaign supersede` so the source row must resolve from campaign primary evidence through `--source-row-hash`, or through `--source-row` only as a way to derive the immutable hash. The source JSON body is not trusted as evidence.
- Added `--replacement-row-hash`, `--source-run-id`, `--replacement-campaign-id`, and `--dry-run` CLI options.
- CLI dry-run builds and validates the candidate native record and graph but does not append.
- Effective row generation now loads the validated supersession graph and resolves each primary source transitively with `resolve_supersession_chain(...)`.
- Superseded effective rows record terminal replacement hash, chain length, full chain metadata, and provenance fields while preserving primary raw evidence unchanged.
- Replacement rows remain synthetic/operator-supplied for Stage 3B tests, with optional exact hash validation. No real catch-up execution is performed.

## Commands/tests
- `sed -n '1,260p' codex_prompts/stage3B.md`
- Source reads around `campaign supersede`, parser options, `write_readiness`, and existing CLI campaign tests.
- `pytest -q tests/test_stage3b_supersession.py tests/test_stage3a_supersession.py tests/test_rc21_post1_acceptance_repairs.py::test_supersession_is_immutable_traceable_and_unambiguous` -> 18 passed.
- `pytest -q tests/test_stage3b_supersession.py tests/test_stage3a_supersession.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign.py tests/test_campaign_package_integrity.py tests/test_campaign_final_acceptance.py tests/test_campaign_recovery_matrix.py tests/test_stage2a_recovery_reconciliation.py tests/test_stage2b_recovery_post_execution.py tests/test_repair.py tests/test_rc9_context_repair.py tests/test_rc10_repair_truth.py tests/test_stage1a_judge_policy.py tests/test_stage1b_judge_qualification.py tests/test_stage1c_model_roles.py tests/test_stage1d_judge_integration.py tests/test_judge_dumps.py` -> 362 passed, 11 skipped.
- `./.venv/bin/ruff check llm_modelbench tests/test_stage3a_supersession.py tests/test_stage3b_supersession.py` -> passed.
- `python -m compileall -q llm_modelbench tests/test_stage3a_supersession.py tests/test_stage3b_supersession.py` -> passed.
- `git diff --check` -> passed.

## Failures and corrections
- No focused or regression test failures after Stage 3B implementation.

## Manual inspection
- Confirmed CLI no longer trusts arbitrary source row JSON as evidence; source must resolve to campaign primary evidence by row hash.
- Confirmed replacement row hash mismatch fails closed.
- Confirmed dry-run does not create `supersessions.jsonl`.
- Confirmed effective ledger uses validated graph resolution transitively and records chain provenance.
- Confirmed synthetic 65536 -> 66560 needle correction updates the effective row while preserving `evidence/primary/raw_results.jsonl` bytes.
- Confirmed graph conflict/cycle errors still fail closed through Stage 3A helpers.
- Confirmed no real catch-up benchmark, real model work, adoption, push, or acceptance evidence mutation.
- Confirmed no Stage 4 config/execute behavior was implemented.

## Deferred work
- Real acceptance evidence supersession, adoption, and any real catch-up execution remain prohibited/deferred.

## Safety confirmation
- No real inference, real catch-up benchmark, real recovery, real judging, Ollama/llama.cpp/GPU/service mutation, Selene qualification, model pull/delete, adoption, acceptance evidence rewrite, or push performed.
