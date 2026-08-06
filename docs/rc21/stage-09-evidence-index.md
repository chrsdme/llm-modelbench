# RC21 Stage 9 evidence index

Status: `stage9_testing_complete_rc21_release_closeout_pending`.

The governing reconciliation is
`/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-closeout-20260806T000245Z/phase1a-reconciliation/phase1a-summary.json`.

| Acceptance area | Root | Basis |
| --- | --- | --- |
| Stage 9A preflight | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T132823Z-00` | Native passing summary |
| Stage 9A dry run | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T133253Z-00` | Native passing summary |
| Ollama GPU0 | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T141313Z-00` | Finalization-ordering adjudication |
| Ollama GPU1 | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T155928Z-00` | Native passing summary |
| llama.cpp GPU0 | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T160944Z-00` | Native passing summary |
| llama.cpp GPU1 | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T161212Z-00` | Threshold-calibration adjudication |
| llama.cpp dual GPU | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T161911Z-00` | Native passing summary |
| ThinkingCap exact 64K | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T212009Z-00` | Native lane evidence; wrapper-schema adjudication |
| Runtime contract | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T212737Z-00` | Native passing summary |
| Runtime selection | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T213004Z-00` | Native passing summary |
| Resume-gate tests | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-replay-20260805T213100Z-00` | Native passing summary |
| Standard resume mismatch | `/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-recovery-20260804T220033Z/live-resume-mismatch` | Accepted live refusal |
| Campaign resume mismatch | `/media/storage/tmp/modelbench/rc21-stage9/live-campaign-resume-mismatch-v9-20260805T235859Z` | Accepted live refusal |

The final campaign root records interruption exit `130`, resume exit `1`, and
the refusal codes `device_order_changed`, `endpoint_changed`,
`physical_gpu_uuids_changed`, and `profile_changed`. Earlier v4, v8, and
related campaign roots are diagnostic or superseded and are not acceptance.

See [Stage 9 release closeout](stage-09-release-closeout.md) for adjudications
and reconciliation context.
