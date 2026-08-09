# Session 4 - Stage 3A

## Session identity
- Session: 4
- Stage: Stage 3A - immutable supersession schema, provenance, hashing, chains, cycles
- Branch: `rc21-post1-topology-budget`
- Starting HEAD: `35fdac0b437f74b2da83d564fbf50bfa14c5ea0e`

## Files/specs read
- `AGENTS.md`
- `CODEX_START.md`
- `PR_RC21POST1_ACCEPTANCE_CONTROLS.md`
- `codex_prompts/stage3A.md`
- `codex_logs/Session 3/session3_20260809T233725Z.md`
- `llm_modelbench/campaign.py`
- `llm_modelbench/cli.py`
- `tests/test_rc21_post1_acceptance_repairs.py`
- relevant campaign/readiness/recovery test listings

## Startup verification
- `git branch --show-current` -> `rc21-post1-topology-budget`
- `git rev-parse HEAD` -> `35fdac0b437f74b2da83d564fbf50bfa14c5ea0e`
- Stage 2B handoff confirms Judge 1A-1D and Recovery 2A-2B were closed/approved.

## Current supersession audit
- `record_supersession(...)` currently computes source/replacement hashes from supplied mutable rows, checks only active one-hop conflicts by source hash, embeds `replacement_row`, and appends directly with `open("a")`.
- Native provenance is incomplete: no schema version, no source run/origin, no replacement campaign, no structured row identity, no semantic supersession ID.
- Current identity/hash contract is weak: no canonical supersession semantic hash, no proof that timestamp/key order is excluded from identity, and no declared source/replacement hash validation path for externally supplied records.
- Append behavior is not atomic at record level; a partial JSONL write can later corrupt reads.
- `supersession_map(...)` reads via generic JSONL loading, filters `active`, and returns one-hop source -> replacement mappings only.
- Effective-row/readiness consumption calls `supersession_map(...)` and applies only the direct replacement row. This remains Stage 3B territory for transitive effective-ledger behavior.
- There is no chain resolution helper, no deterministic graph contract, no cycle detection, and fork handling is only one-hop active replacement comparison.
- Ambiguity/conflict validation is insufficient for duplicate semantic records, malformed rows, legacy evidence, and order-independent graph behavior.
- `CampaignPaths.supersessions` is already `evidence/supersessions.jsonl`.
- Package/readiness references include `effective_rows.jsonl`; supersession currently only affects readiness through direct effective row rewriting.

## Requirements
- Implement only Stage 3A internal supersession contract.
- Add versioned native schema/provenance.
- Add canonical semantic hashing excluding volatile audit timestamp.
- Validate source/replacement row hashes and provenance.
- Preserve append-only ledger semantics and idempotent duplicate behavior.
- Add atomic/corruption-safe ledger write/read behavior.
- Add deterministic graph construction, chain resolution, cycle detection, and fork detection.
- Keep legacy compatibility explicit and conservative.
- Add synthetic tests for Stage 3A matrix.
- Do not implement Stage 3B CLI redesign, effective-ledger transitive adoption, catch-up workflow, real recovery, judging, adoption, or model work.

## Design decisions
- Added `SUPERSESSION_SCHEMA_VERSION = 2` while preserving `SUPERSESSION_POLICY_VERSION = rc21.post1`.
- Native records now contain structured `source` and `replacement` identities: campaign ID, run ID, exact row hash, task, model, and model digest where available.
- Native records retain compatibility aliases (`source_row_hash`, `replacement_row_hash`, `replacement_run_id`, etc.) so existing direct readiness consumption stays green.
- Semantic supersession identity is canonical JSON over stable relationship material only: schema version, policy version, source identity, replacement identity, reason, operator, and tool. `recorded_at` and JSON key order are excluded.
- `record_supersession(...)` still returns a record and writes `evidence/supersessions.jsonl`, but now validates source/replacement rows before write, deduplicates exact semantic IDs, builds the graph with the candidate edge, and fails closed on invalid graph state.
- Atomic append is implemented as an atomic full-file rewrite that preserves existing ledger bytes and adds one complete JSONL row; failed writes do not report success.
- `load_supersession_ledger(...)`, `build_supersession_graph(...)`, and `resolve_supersession_chain(...)` provide the Stage 3A internal API for validation, graph construction, fork/cycle detection, chain resolution, and full chain evidence.
- `supersession_map(...)` now loads through the graph and fails closed on malformed/ambiguous/cyclic evidence, but still returns direct source-hash mappings only. Transitive effective-ledger behavior is deferred to Stage 3B.
- Legacy records are accepted only through an explicit compatibility path when they contain source hash, replacement hash, and an embedded replacement row that hashes to the declared replacement hash. Insufficient legacy records fail closed.

## Commands/tests
- `pwd && ls`
- `sed -n '1,240p' AGENTS.md`
- `sed -n '1,240p' CODEX_START.md`
- `git branch --show-current`
- `git rev-parse HEAD`
- `sed -n '1,260p' PR_RC21POST1_ACCEPTANCE_CONTROLS.md`
- `sed -n '1,280p' codex_prompts/stage3A.md`
- `sed -n '1,260p' codex_logs/Session 3/session3_20260809T233725Z.md`
- `find codex_prompts -maxdepth 1 -type f -printf '%f\n' | sort`
- `rg -n "supersess|row_hash|effective" llm_modelbench tests codex_prompts PR_RC21POST1_ACCEPTANCE_CONTROLS.md`
- `rg -n "class CampaignPaths|supersession|superseded|effective" llm_modelbench -g '*.py'`
- `find tests -type f | sort | rg 'supersess|campaign|readiness|recover|repair|judge'`
- `git status -sb`
- Targeted source/test reads around `record_supersession`, `supersession_map`, row hashes, readiness, CLI, and tests.
- `pytest -q tests/test_stage3a_supersession.py tests/test_rc21_post1_acceptance_repairs.py::test_supersession_is_immutable_traceable_and_unambiguous` -> 9 passed.
- `pytest -q tests/test_stage3a_supersession.py tests/test_rc21_post1_acceptance_repairs.py tests/test_campaign.py tests/test_campaign_package_integrity.py tests/test_campaign_final_acceptance.py tests/test_campaign_recovery_matrix.py tests/test_stage2a_recovery_reconciliation.py tests/test_stage2b_recovery_post_execution.py tests/test_stage1a_judge_policy.py tests/test_stage1b_judge_qualification.py tests/test_stage1c_model_roles.py tests/test_stage1d_judge_integration.py` -> 303 passed.
- `ruff check llm_modelbench tests/test_stage3a_supersession.py` -> failed because `ruff` was not on PATH.
- `python -m compileall llm_modelbench tests/test_stage3a_supersession.py` -> passed.
- `git diff --check` -> passed.
- `python -m ruff check llm_modelbench tests/test_stage3a_supersession.py` -> failed because the system Python has no `ruff` module.
- `find . -maxdepth 3 -type f -name ruff -o -type f -name ruff.exe` -> found `./.venv/bin/ruff`.
- `./.venv/bin/ruff check llm_modelbench tests/test_stage3a_supersession.py` -> passed.

## Failures and corrections
- `ruff` was unavailable on PATH and as a system Python module. Corrected by using the repository-local `./.venv/bin/ruff`.
- No focused test failures after implementation.

## Manual inspection
- Confirmed schema is versioned and native records carry source/replacement structured identity plus compatibility aliases.
- Confirmed semantic identity excludes `recorded_at` and uses canonical JSON serialization, not Python dict order or filesystem order.
- Confirmed row hash validation checks supplied source/replacement rows and embedded replacement row contents.
- Confirmed malformed supersession JSONL raises `malformed_supersession_ledger` instead of being silently accepted.
- Confirmed duplicate semantic edge is idempotent and does not create a second physical JSONL row through `record_supersession(...)`.
- Confirmed fork `A -> B` and `A -> C` is represented as `ambiguous_supersession_fork` and graph validity false.
- Confirmed self, two-node, and three-node cycles are rejected/detected as `cyclic_supersession`.
- Confirmed `A -> B -> C` resolves to terminal `C` through the internal helper while preserving both edge records.
- Confirmed graph edge ordering and repeated loads are deterministic.
- Confirmed legacy compatibility is explicit and insufficient legacy evidence fails closed.
- Confirmed readiness still uses direct `supersession_map(...)` only; no Stage 3B transitive effective-ledger adoption was implemented.
- Confirmed tests use synthetic rows and temporary campaign directories only.
- Confirmed no primary, recovery, judge, acceptance, model, or service evidence was mutated.

## Deferred Stage 3B work
- CLI redesign and operator UX.
- Effective-ledger transitive supersession adoption.
- Synthetic catch-up workflow integration.
- Acceptance evidence supersession and adoption.

## Safety confirmation
- No real inference, Ollama/llama.cpp/GPU/service mutation, real recovery, real judging, Selene qualification, model pull/delete, adoption, acceptance evidence rewrite, or push performed.
