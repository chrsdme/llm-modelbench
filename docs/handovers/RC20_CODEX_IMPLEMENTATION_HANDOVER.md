# RC20 Codex implementation handover

## Status

Partial. RC19.1 workspace foundations are implemented and validated; RC19.2,
RC19.3, and RC20 automation/adoption acceptance remain unimplemented. Canonical
rankings were not changed and no real Ollama campaign was run.

## Implemented architecture

- `campaigns/<id>` owns manifests, plan, primary/recovery/judge evidence,
  candidate rankings, reports, logs, packages, checksums, readiness, and adoption
  record locations.
- `llmb campaign plan`, `run`, and `status` create/use isolated primary evidence
  and candidate rankings. Existing legacy commands remain compatible.
- Manifest schema v1 validates structural fields while ignoring unknown optional
  future fields. Campaign locks only reclaim a conclusively dead same-host PID.
- Review packages have a SHA-256 inventory. Cleanup is deliberately conservative
  and only removes primary raw dump files after a verified package. Legacy runs
  are copied, never moved.

## Commits and tag

- `d506f4b fix: harden campaign lifecycle state handling`
- `090037c feat: add campaign evidence resolver and locking`
- `dd77e5a feat: route benchmark subsystems through campaign workspaces`
- `4db83cc feat: add campaign packaging and retention`
- `f3802a5 chore: prepare RC19.1 campaign workspace release`
- `840ef09 fix: restore release changelog version heading`
- Local tag: `v1.0.0rc19.post1`

The tag was created before `840ef09`; it has deliberately not been rewritten.

## Validation

Final completed validation: `python3 -m pytest -q` — 438 passed; `./llmb
selftest` — ALL GOOD. A disposable mock campaign succeeded under its campaign
root and was removed. Full session evidence is in
`/tmp/llm-modelbench-rc20-codex-20260718_034148`.

## Known limitations and review

Recovery/capability/judge orchestration, readiness, transactional canonical
adoption, RC19.2/RC19.3/RC20 release gates, and real-Ollama acceptance must be
implemented before claiming RC20. Do not perform adoption: there is no adopted
campaign. When that work is complete, the intended human preview command is:

`./llmb rankings adopt --campaign <campaign-id> --dry-run`

No canonical adoption was performed. Review commits, the local tag position,
the log directory, and campaign tests before continuing. Rollback of the
implemented changes is by normal review/revert commits; primary legacy evidence
is never modified by migration.
