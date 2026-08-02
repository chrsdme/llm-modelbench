# RC21 Stage 8 live acceptance and closeout

## Scope and baseline

Stage 8 freezes backend, model, physical GPU, and execution-contract evidence in campaigns and result rows. It adds fail-closed compatibility comparison, run-level identity artifacts, runtime-variant report provenance, and additive reporting without changing task scoring, score validity, canonical sample weighting, ranking eligibility, or ranking-store update policy.

The reviewed committed baseline was `3c3864acb83f2e86c563ec0d204b09531cf25b2c` (`feat: add conservative runtime fit profiling`). The accepted fixture-only, non-inference acceptance evidence is `/tmp/llmb-rc21-stage8-acceptance-20260802T150345Z`. Its harness is `/tmp/llmb-rc21-stage8-acceptance.sh`, SHA-256 `b8bf932dc00e9461f774227ea047636511b3d5213096116a7dadb47877cf47a3`.

The fixture identities use the real-host physical NVIDIA UUID forms `GPU-78077308-f4b2-3330-6d4e-19581d7b1511` and `GPU-5b99bce2-35ab-f6db-857b-72162069fa72`. They are identity fixtures only: acceptance made no inference request, runtime lifecycle action, model/configuration/service mutation, or ranking-store mutation.

## Accepted evidence

`acceptance-summary.json` records 28/28 unique Stage 8 checks passing. `report-acceptance.json` records 28/28 production-report checks passing. The compatibility matrix validates deterministic, field-specific refusal codes for backend, endpoint, profile, server version, model artifact, physical UUIDs, device order, strategy, allocation weights, context, batch, micro-batch, KV type, parallel sequences, spill policy, and offload policy. Timestamp-only changes remain compatible; missing legacy identity fails closed.

The refusal fixtures show backend, digest, allocation, and legacy mismatches rejected before client construction or an inference callback. The atomic identity artifact has the full authoritative hash and its row reference; missing, corrupt, and future artifacts remain bounded evidence. Exact runtime-variant duplicates and resumed copies collapse, while distinct variants remain separately reportable and do not create additional canonical sample weight. The canonical fixture score remains unchanged.

The isolated ranking store contains two records, is 231 bytes, and has SHA-256 `3ea07938a0c2b8e4202d83a159fac2b99ec4db9505cd4c63e57c2f75c7852576` before and after reporting. The snapshots have identical parsed content, byte count, record count, and hash. Repository and bounded runtime-state snapshots are identical. The recorded application version is `1.0.0rc20.post1`, not a fixture placeholder. No `harness-failure.json` exists; `harness-traceback.txt` is empty.

## Report provenance and compatibility

Structured per-model summaries retain schema/hash/variant, backend/profile, model digest, UUID/order, declared strategy/allocation/context/batch/KV/parallel/spill/offload evidence where supplied. `summary_meta.json` adds run-level `runtime_provenance`: variant totals and per-model counts, backend/profile sets, legacy and identity-bearing row counts, run-level identity-artifact state, telemetry state, and advisory runtime-fit state. This global state is clearly separate from model provenance.

`scorecard.md` presents concise identity information. `scorecard.csv` has deterministic additive columns `runtime_identity_state`, `runtime_variant_count`, `runtime_backends`, `runtime_profiles`, and `runtime_identity_hashes`; existing columns and values retain their meanings. A legacy aggregate stays `legacy_unknown` and cannot inherit a neighbouring model's run-level identity artifact. Runtime telemetry and Stage 7 fit evidence are advisory only and do not claim layer, tensor, KV-cache, or offload placement beyond supplied declared evidence.

## Validation and status

Final focused validation passed: `6 passed` (`tests/test_runtime_identity.py` and `tests/test_report_offline.py`). The full suite passed: `748 passed`. `python3 -m compileall -q llm_modelbench tests`, `./llmb selftest`, `python3 tools/release_check.py`, and `git diff --check` passed. The import-side-effect guard blocked subprocess, procfs, and network entry points during imports and passed. The added-lines secret/environment-value review found no matches.

Score-preservation review confirms the runtime summary is attached after aggregate scoring and the accepted fixture retains quality `50.0`; no scorer, weighting, eligibility, or ranking formula changed. The one-variant report path remains compatible because its added fields contain one deterministic variant. Mixed current/legacy validation confirms a legacy row remains readable, retains its score, and has `legacy_unknown` without identity injection. Stage 8 is complete and ready for Stage 9 review; Stage 9 has not started.

Earlier fixture roots are not authoritative: `20260802T141634Z` was provisional and omitted required gates; `20260802T142557Z` stopped on a harness `Path + str` error; `20260802T143004Z` falsely passed without enforcing required evidence; `20260802T143606Z` asserted report evidence with an empty ranking store; `20260802T144116Z` lacked production identity metadata; and `20260802T144647Z` still had incomplete report checks, an empty-store comparison, fixture application version, generic summary details, and legacy artifact injection. The accepted `20260802T150345Z` evidence replaces those attempts.

Final status: `stage8_complete_ready_for_stage9_review`.
