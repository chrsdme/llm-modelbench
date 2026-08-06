# RC21 Stage 9 release handover

Status: `stage9_testing_complete_rc21_release_closeout_pending`.

Repository: `<repo>`  
Branch: `rc21-runtime-multigpu`  
Pre-documentation HEAD: `903b7470f05500fe859a0ddcae0e8bf0cb66a470`  
Current version: `1.0.0rc20.post1`

## Completed scope

RC21 completed dynamic NVIDIA GPU inventory, canonical UUID/PCI identity,
runtime-profile discovery and selection, Ollama and external llama.cpp runtime
selection, frozen runtime identities, standard and campaign resume mismatch
refusal, and Stage 9 cleanup/migration hygiene. The thirteen accepted areas and
their roots are in [the Stage 9 evidence index](../rc21/stage-09-evidence-index.md).

The accepted campaign mismatch evidence is
`/media/storage/tmp/modelbench/rc21-stage9/live-campaign-resume-mismatch-v9-20260805T235859Z`.
It proves pre-mutation refusal when selected runtime profile, endpoint, GPU UUIDs,
or device order differ.

## Validation and limitations

The final focused validation passed 172 tests; the clean full suite passed 763;
doctor and self-test passed. The Phase 1A adjudications retain the Ollama GPU0
finalization-ordering issue, llama.cpp GPU1 threshold-calibration note, and
ThinkingCap wrapper-schema mismatch transparently. See
[Stage 9 release closeout](../rc21/stage-09-release-closeout.md).

No version bump, release commit, local tag, or push has occurred. The seven
historical source/test candidates are in
`/media/storage/tmp/modelbench/rc21-stage9/rc21-stage9-closeout-20260806T000245Z/phase3-repository-audit/rc21-release-file-allowlist.txt`.
Existing RC22 deferred work remains in [RC22 deferred scope](../rc21/RC22_DEFERRED_SCOPE.md).

**Next action:** Validate the Phase 4 documentation and final release allowlist
in Phase 5 before preparing the version, commit and tag plan.
