# H3 - Package Integrity For New Evidence

## Scope
- Branch: `rc21-post1-topology-budget`
- Baseline commit: `eaa250133a5d1895154c9143277c56eda6390300`
- Stage: H3 only

## Requirements Implemented
- Package verification now validates superseded effective rows against packaged `evidence/supersessions.jsonl`.
- Verification follows the graph from the effective row source hash through the packaged supersession chain to the terminal replacement hash.
- Verification fails for missing/malformed supersession ledger, tampered edge, missing intermediate edge, changed terminal replacement, fork, cycle, and effective-row provenance inconsistency.
- Config-managed campaigns now carry an explicit manifest note with config signature and path.
- Package verification now requires config-managed packages to include `plan/campaign_config.json`, validates the canonical config signature, and checks manifest binding.
- Legacy non-config packages remain compatible and are not treated as config-managed merely because `plan/campaign_config.json` exists.

## Design Decisions
- The explicit config-managed indicator is stored in `manifest.notes.campaign_config`.
- `verify_package_details(...)` now reports `supersession_references_valid` and `config_references_valid`.
- Package tests use a helper that rebuilds internally consistent tampered archives so semantic verifier failures are tested independently from generic checksum failures.

## Tests
- Valid A->B and A->B->C supersession packages verify.
- Missing/malformed/tampered supersession ledgers fail.
- Missing intermediate edge, changed terminal replacement, fork, cycle, and inconsistent effective provenance fail.
- Valid config-managed package verifies.
- Missing/malformed/modified config, signature mismatch, and manifest binding mismatch fail.
- Legacy package with stray config file verifies without config-management inference.

## Focused Validation
- `./.venv/bin/pytest -q tests/test_campaign_package_integrity.py tests/test_stage4b_campaign_execute.py tests/test_stage3b_supersession.py tests/test_stage3a_supersession.py` -> 63 passed.
- `./.venv/bin/pytest -q tests/test_campaign.py tests/test_campaign_package_integrity.py tests/test_campaign_final_acceptance.py tests/test_stage4a_campaign_config.py tests/test_stage4b_campaign_execute.py tests/test_stage3b_supersession.py tests/test_stage3a_supersession.py` -> 139 passed.
- `./.venv/bin/ruff check llm_modelbench/campaign.py llm_modelbench/cli.py tests/test_campaign_package_integrity.py tests/test_stage4b_campaign_execute.py` -> passed.
- `python -m compileall -q llm_modelbench/campaign.py llm_modelbench/cli.py tests/test_campaign_package_integrity.py tests/test_stage4b_campaign_execute.py` -> passed.
- `git diff --check` -> passed.

## Manual Inspection
- Confirmed superseded effective rows cannot verify without packaged graph support.
- Confirmed config-managed state is explicit in manifest notes, not inferred from file presence.
- Confirmed legacy compatibility remains.
- Confirmed no real model, judge, recovery, catch-up, adoption, evidence rewrite, or push work occurred.
- Confirmed no later hardening stage was started in this commit.
