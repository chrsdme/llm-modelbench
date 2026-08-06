# RC21 Stage 9 release closeout

Status: `stage9_testing_complete_rc21_release_closeout_pending`.

Stage 9 implementation, acceptance, evidence reconciliation, repository validation,
and documentation closeout are complete. RC21 has not been versioned, committed,
tagged, pushed, or released. The pre-documentation repository identity is
`rc21-runtime-multigpu` at `903b7470f05500fe859a0ddcae0e8bf0cb66a470`.

This canonical closeout replaces the superseded Stage 9A planning document. See
[the evidence index](stage-09-evidence-index.md),
[the validation runbook](stage-09-validation-runbook.md), and
[the handover](../handovers/RC21_STAGE9_RELEASE_HANDOVER.md).

## Accepted evidence

Phase 1A records 13 accepted areas: 11 by native passing summary and two by
explicit adjudication. No accepted package was modified during reconciliation.

| Area | Accepted root |
| --- | --- |
| Stage 9A preflight | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T132823Z-00` |
| Stage 9A dry run | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T133253Z-00` |
| Ollama GPU0 | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T141313Z-00` |
| Ollama GPU1 | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T155928Z-00` |
| llama.cpp GPU0 | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T160944Z-00` |
| llama.cpp GPU1 | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T161212Z-00` |
| llama.cpp dual GPU | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T161911Z-00` |
| ThinkingCap exact 64K | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T212009Z-00` |
| Runtime contract | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T212737Z-00` |
| Runtime selection | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T213004Z-00` |
| Resume-gate tests | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T213100Z-00` |
| Standard resume mismatch | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-recovery-20260804T220033Z/live-resume-mismatch` |
| Campaign resume mismatch | `/media/storage/tmp/modelbench/rc21-stage9/live-campaign-resume-mismatch-v9-20260805T235859Z` |

## Adjudications

Ollama GPU0 is accepted by recorded adjudication: its substantive request,
telemetry, cleanup, idle-endpoint, repository, and checksum evidence passed,
but its native summary checked finalization artifacts before creating them. It
is not presented as a clean native-summary pass.

llama.cpp GPU1 is accepted by adjudication. Excluded GPU0 memory delta exceeded
the nominal threshold while its utilization remained zero; this is a retained
threshold-calibration note.

ThinkingCap records `tokens_evaluated: 64000`, `tokens_predicted: 16`, and
`truncated: false`. The outer wrapper did not recognize the server response
field names; this wrapper-schema mismatch was not a model-lane failure and the
lane was not rerun for it.

The final campaign manifest referenced transient `.sampler-ready` and
`.sampler-stop` markers absent during reconciliation. All extant manifest
members matched, actual content hash mismatches were zero, and Phase 1A
recorded the exception without repairing the accepted package. Phase 1A
transparently superseded the strict original Phase 1 result. Earlier campaign
v4, v8, and related attempts are diagnostic or superseded, not accepted.

## Campaign refusal and final validation

The accepted campaign root is
`/media/storage/tmp/modelbench/rc21-stage9/live-campaign-resume-mismatch-v9-20260805T235859Z`.
Its interruption exited `130`, mismatched resume exited `1`, and the execution
script exited `0` because refusal was the expected result. The refusal reported
`device_order_changed`, `endpoint_changed`, `physical_gpu_uuids_changed`, and
`profile_changed` before campaign mutation or GPU1 model execution. Its
acceptance summary and checksum verification passed.

Phase 2 reused 172 focused passing tests and completed a clean 763-test full
suite. Doctor and self-test passed; release-blocking and unresolved warnings
were zero; repository and clean-attempt production state were unchanged;
services were unchanged; endpoints remained healthy; and no owned process
remained. The earlier run collected 763 tests, passed 762, and failed one. It
is a superseded `validation_environment_conflict`: forcing global `TMPDIR`,
`TEMP`, and `TMP` contradicted the intentional fixed-`/tmp` behaviour tested by
`tests/test_repair.py`, not RC21 source behavior.

Phase 3 found eight changed paths (five tracked, three untracked), no staged or
unresolved path, and seven source/test release candidates. Its historical
allowlist is
`/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-closeout-20260806T000245Z/phase3-repository-audit/rc21-release-file-allowlist.txt`.

## Scope and next action

Completed scope covers dynamic NVIDIA inventory, stable GPU UUID/PCI identity,
runtime-profile discovery and selection, Ollama and llama.cpp selection, frozen
runtime identities, standard/campaign mismatch refusal, and cleanup/migration
hygiene. Version bump, release commit, local tag, and push remain pending.
Existing RC22 deferred work remains in [RC22 deferred scope](RC22_DEFERRED_SCOPE.md).

**Next action:** Validate the Phase 4 documentation and final release allowlist
in Phase 5 before preparing the version, commit and tag plan.
