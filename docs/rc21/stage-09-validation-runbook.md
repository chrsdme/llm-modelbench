# RC21 Stage 9 validation runbook

Status: `stage9_testing_complete_rc21_release_closeout_pending`.

This is a release-readiness record, not an instruction to rerun accepted live
GPU lanes. See [Stage 9 evidence index](stage-09-evidence-index.md).

## Completed validation

- Focused RC21 validation: 172 passed.
- Clean full suite: 763 passed, 0 failed.
- `llm-modelbench doctor --json`: passed.
- `llm-modelbench selftest`: passed.
- Release-blocking warnings: 0; unresolved warnings: 0.
- Repository and clean-attempt production runtime-profile state: unchanged.
- Services: unchanged; endpoints healthy; owned processes remaining: 0.

The earlier result collected 763 tests, passed 762, and failed one. It is
superseded as a `validation_environment_conflict`, not a source regression:
global `TMPDIR`, `TEMP`, and `TMP` overrides contradicted the intentional
fixed-`/tmp` behaviour in `tests/test_repair.py`. The clean attempt retained
normal application temporary semantics and used a persistent pytest
`--basetemp`.

## Final readiness procedure

Phase 5 must validate this documentation and the proposed release allowlist.
It must not rerun accepted Stage 9 GPU lanes merely to repeat acceptance.

1. Verify Phase 1A, Phase 2, Phase 3, and Phase 4 summaries and checksum
   records beneath `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-closeout-20260806T000245Z`.
2. Review the proposed allowlist and documented exclusions.
3. Only then prepare the separate version, commit, and local-tag plan. This
   runbook does not authorize a push.
