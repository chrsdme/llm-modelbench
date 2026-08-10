# H2 - Anchor Applied Replacement Evidence

## Scope
- Branch: `rc21-post1-topology-budget`
- Baseline commit: `9ccdac3f0a8f0699b1148693beced44a45265179`
- Stage: H2 only

## Requirements Implemented
- Non-dry `campaign supersede` now treats replacement JSON as a preview only.
- Applied replacement evidence is resolved from stored campaign evidence by:
  - replacement campaign id;
  - replacement run id;
  - exact replacement row hash;
  - unique stored row match.
- Apply fails closed for absent replacement campaign, absent replacement evidence location, missing row hash, duplicate hash, source/replacement model-task-digest contradictions, and preview/stored evidence contradictions.
- Dry-run remains synthetic/non-mutating and can validate a preview without appending authoritative native evidence.
- Source and replacement evidence files remain unchanged during apply.

## Design Decisions
- Used `evidence/primary/raw_results.jsonl` for `replacement_run_id=primary`.
- Used `evidence/<replacement_run_id>/raw_results.jsonl` for synthetic stored replacement runs. This is intentionally narrow and avoids implementing real catch-up execution.
- Kept existing direct unit helper behavior for synthetic graph construction; the CLI apply path is the authoritative operator path hardened here.

## Tests
- Valid stored replacement -> applied edge succeeds.
- Arbitrary JSON without stored replacement -> apply fails.
- Correct JSON/hash but no stored replacement campaign/evidence source -> apply fails.
- Wrong replacement hash -> fails.
- Duplicate replacement hash -> fails.
- Wrong model/task/digest provenance -> fails.
- Preview JSON contradicting stored replacement -> fails.
- Dry-run synthetic preview remains non-mutating.
- Source and replacement stored evidence are unchanged.

## Focused Validation
- `./.venv/bin/pytest -q tests/test_stage3b_supersession.py tests/test_stage3a_supersession.py tests/test_rc21_post1_acceptance_repairs.py::test_supersession_is_immutable_traceable_and_unambiguous` -> 26 passed.
- `./.venv/bin/pytest -q tests/test_stage3b_supersession.py tests/test_stage3a_supersession.py tests/test_campaign.py tests/test_campaign_package_integrity.py tests/test_campaign_final_acceptance.py` -> 102 passed.
- `./.venv/bin/ruff check llm_modelbench/campaign.py llm_modelbench/cli.py tests/test_stage3b_supersession.py tests/test_stage3a_supersession.py` -> passed.
- `python -m compileall -q llm_modelbench/campaign.py llm_modelbench/cli.py tests/test_stage3b_supersession.py tests/test_stage3a_supersession.py` -> passed.
- `git diff --check` -> passed.

## Manual Inspection
- Confirmed non-dry CLI apply loads the stored row before `record_supersession(...)`.
- Confirmed supplied preview JSON cannot replace stored row content.
- Confirmed provenance checks cover source/replacement task, model, and model digest.
- Confirmed no real catch-up execution was introduced.
- Confirmed no later hardening stage was started in this commit.
- Confirmed no real model, judge, recovery, catch-up, adoption, evidence rewrite, or push work occurred.

