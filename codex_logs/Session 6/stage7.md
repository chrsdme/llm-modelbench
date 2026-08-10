# Session 6 - Stage 7 final offline validation

## Session identity
- Session number: 6
- Stage: 7
- Baseline HEAD: 4bdf2e0252555376f6444692f56e2e4f84649467
- Branch: rc21-post1-topology-budget

## Files/specs read
- AGENTS.md
- PR_RC21POST1_ACCEPTANCE_CONTROLS.md
- codex_prompts/stage7.md
- codex_logs/Session 6/stage6.md

## Requirements
- Implement no new feature except fixes required by validation.
- Run full offline/non-inference validation: full pytest, Ruff, Bandit, compileall, release check, safe selftest, `git diff --check`, package/wheel build, installed-wheel non-inference smoke/selftest.
- Audit PR plan and Stage 1A-6 requirements with `Requirement | Implementation | Tests | Status | Notes`.
- Confirm no real inference/judging/recovery, no model pull/delete, no service/KV mutation, no evidence rewrite, no adoption, no push.

## Commands/tests
- `./.venv/bin/pytest -q` -> initial 1 failed, 1042 passed, 11 skipped; final 1043 passed, 11 skipped.
- `./.venv/bin/ruff check llm_modelbench tests` -> passed.
- `./.venv/bin/ruff check llm_modelbench tests tools/release_check.py` -> passed after Stage 7 correction.
- `./.venv/bin/python -m bandit -q -r llm_modelbench tools -x tests -ll` -> passed.
- `python -m compileall -q llm_modelbench tests tools` -> passed.
- `./.venv/bin/python tools/release_check.py` -> initial failed on local Codex logs; final `RELEASE CHECK: PASS`.
- `./.venv/bin/python -m llm_modelbench selftest` -> `SELFTEST: ALL GOOD`.
- `git diff --check` -> passed.
- `./.venv/bin/python -m build` -> built sdist and wheel.
- Temporary installed-wheel smoke with `--force-reinstall --no-deps`: `python -m llm_modelbench --help` and `python -m llm_modelbench selftest` -> passed.

## Requirement audit
| Requirement | Implementation | Tests | Status | Notes |
| --- | --- | --- | --- | --- |
| Judge 1A canonical capability source, judge policy, deterministic candidate selection | Campaign judge policy and candidate selection implemented in campaign/judge paths | Full pytest; `tests/test_stage1a_judge_policy.py`; regression bundle | Complete | Offline/mock only |
| Judge 1B qualification framework/protocol | ModelBench-owned judge qualification and coverage evidence | Full pytest; `tests/test_stage1b_judge_qualification.py` | Complete | No real judge qualification claimed |
| Judge 1C roles/no-self-judging | Role policy and cohort name/digest exclusion | Full pytest; `tests/test_stage1c_model_roles.py` | Complete | Machine judging remains provisional |
| Judge 1D provenance/evidence integration | Selection, sidecars, readiness blockers, mocked integration | Full pytest; `tests/test_stage1d_judge_integration.py`; `tests/test_judge_dumps.py` | Complete | No real judging |
| Recovery 2A eligibility and pre-execution reconciliation | Exact source identity and eligible-only recovery planning | Full pytest; `tests/test_stage2a_recovery_reconciliation.py`; recovery regressions | Complete | No score-fishing |
| Recovery 2B post-execution reconciliation and terminal states | Final outcome reconciliation and effective recovery rows | Full pytest; `tests/test_stage2b_recovery_post_execution.py`; recovery regressions | Complete | Primary evidence immutable |
| Stage 3A supersession graph safety | Versioned schema, semantic IDs, hash validation, chains, cycles, forks | Full pytest; `tests/test_stage3a_supersession.py` | Complete | Append-only graph |
| Stage 3B supersession CLI/effective resolution | `campaign supersede`, transitive effective-row resolution, synthetic catch-up tests | Full pytest; `tests/test_stage3b_supersession.py`; campaign readiness tests | Complete | Synthetic only |
| Stage 4A strict config/init | Versioned strict config schema and `campaign init` template | Full pytest; `tests/test_stage4a_campaign_config.py` | Complete | Unknown keys fail |
| Stage 4B execute/resume/idempotency | `campaign execute --config`, immutable plan signature, resume guidance | Full pytest; `tests/test_stage4b_campaign_execute.py` | Complete | Mocked lifecycle tests |
| Stage 5 readiness integration | Cross-subsystem readiness blockers and effective-ledger semantics | Full pytest; `tests/test_stage5_readiness_integration.py` | Complete | Synthetic evidence |
| Stage 6 docs/operator UX | Public acceptance-controls docs and help text | Full pytest; `tests/test_stage6_docs_ux.py`; `tests/test_docs_hygiene.py` | Complete | Truthful offline scope |

## Manual inspection
- Confirmed Judge 1A-1D remain covered by focused tests and full pytest.
- Confirmed Recovery 2A-2B remain covered by focused tests and full pytest.
- Confirmed Supersession 3A-3B remain covered by focused tests and full pytest.
- Confirmed Campaign config/lifecycle 4A-4B remain covered by focused tests and full pytest.
- Confirmed Stage 5 readiness integration remains covered by focused tests and full pytest.
- Confirmed Stage 6 docs match implemented offline behavior and do not claim real-host work.
- Confirmed no score-fishing, no self-judging regression, no recovery evidence mutation, no supersession graph ambiguity/cycle acceptance, no unknown-schema trust upgrade, no mutable native supersession deactivation, no campaign-config silent unknown keys, no resume-plan mutation, no evidence adoption, no real-host work, and no push.

## Failures and corrections
- Full pytest initially failed `tests/test_needle_output_fields.py::test_think_false_not_sent_to_non_thinking_model` because the synthetic `_NoThinkClient._post_stream` test double did not accept the production `timeout` keyword. Corrected the test double signature; production behavior already avoided sending `think` to non-thinking models.
- `tools/release_check.py` initially failed on local untracked `codex_logs/` containing absolute home paths. Corrected release-check file discovery to exclude local generated materials (`codex_log/`, `codex_logs/`, `codex_prompts/`, and `local_only`) in Git-checkout mode as well as non-Git mode. No local files were cleaned or restored.

## Safety confirmation
- No real inference, real judging, real recovery, real catch-up, real acceptance campaign, model pull/delete, service/GPU/KV mutation, evidence adoption, evidence rewrite, or push.
