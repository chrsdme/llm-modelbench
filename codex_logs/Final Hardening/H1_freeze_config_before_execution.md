# H1 - Freeze Config Before Execution

## Scope
- Branch: `rc21-post1-topology-budget`
- Baseline commit: `1169ac2c07f486f78914e7ecf7530d5470de7f74`
- Stage: H1 only

## Requirements Implemented
- `campaign execute --config` now validates config, creates the campaign manifest, transitions to `planned`, atomically persists `plan/campaign_config.json`, verifies the exact frozen record, and only then delegates to `campaign run`.
- First execution leaves the frozen config unchanged after success.
- Delegate exception/interruption leaves the frozen config present and byte-stable.
- Same-config retry follows existing deterministic lifecycle behavior for interrupted campaigns.
- Changed-config retry refuses before delegation.
- Existing legacy campaigns without immutable config plans remain refused and are not silently made config-managed.
- The frozen config is not rewritten on resume or retry.

## Tests
- `tests/test_stage4b_campaign_execute.py` now asserts from inside the fake delegated execution function that `campaign_config.json` already exists before execution.
- Added a fake delegated execution that raises after checking the frozen config and marks the synthetic campaign interrupted.
- Proved frozen config signature and bytes remain unchanged, changed config refuses before the fake delegate is called, and no primary evidence appears unexpectedly.

## Focused Validation
- `./.venv/bin/pytest -q tests/test_stage4b_campaign_execute.py tests/test_stage4a_campaign_config.py tests/test_campaign.py` -> 80 passed.
- `./.venv/bin/ruff check llm_modelbench/cli.py tests/test_stage4b_campaign_execute.py tests/test_stage4a_campaign_config.py` -> passed.
- `python -m compileall -q llm_modelbench/cli.py tests/test_stage4b_campaign_execute.py tests/test_stage4a_campaign_config.py` -> passed.
- `git diff --check` -> passed.

## Manual Inspection
- Confirmed `plan/campaign_config.json` is written before `main(invocation)`.
- Confirmed the old post-execution config write was removed.
- Confirmed existing config-managed campaign checks remain before any delegation.
- Confirmed legacy campaigns without a config plan still fail closed.
- Confirmed no later hardening stage was started in this commit.
- Confirmed no real model, judge, recovery, catch-up, adoption, evidence rewrite, or push work occurred.
