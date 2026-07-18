# RC20 Codex implementation handover

## Final architecture

RC20 campaign mode is an isolated evidence pipeline rooted at
`campaigns/<campaign-id>/`. The production CLI persists the campaign manifest
and state history, plan, inventory, capability profiles, primary evidence,
recovery evidence, judge evidence, effective terminal rows, readiness, candidate
rankings, checksums, cleanup previews, and one verified review package.

Legacy `runs/` evidence remains readable. `campaign migrate-legacy` copies
legacy evidence into a new campaign and records explicit migration provenance;
it does not move or mutate the source run.

Canonical rankings are isolated from campaigns until a human performs
campaign-only adoption. Candidate rankings are generated under the campaign.
Canonical adoption verifies readiness and package signatures, previews the
semantic diff, requires the exact typed confirmation `ADOPT <campaign-id>`, and
rebuilds transactionally with rollback.

## Supported CLI commands

- `./llmb campaign plan --campaign-id <id> --level <smoke|short|full> --models '<model;...>' [--samples N] [--judge off]`
- `./llmb campaign run --campaign-id <id> --unattended-safe --yes --level <smoke|short|full> --models '<model;...>' [--live-ui off]`
- `./llmb campaign resume <id>`
- `./llmb campaign status <id>`
- `./llmb campaign package <id>`
- `./llmb campaign clean <id>` for dry-run cleanup
- `./llmb campaign clean <id> --apply` for eligible cleanup apply
- `./llmb campaign clean --all [--apply]`
- `./llmb campaign migrate-legacy --run-id <run-id> --campaign-id <id> [--runs-dir runs] [--dry-run|--apply]`
- `./llmb rankings --adopt <campaign-id> --dry-run [--out <temporary-rankings>]`
- `./llmb rankings --adopt <campaign-id> [--out rankings]` followed by the exact interactive confirmation

There is no `./llmb rankings adopt` subcommand; adoption is exposed as
`./llmb rankings --adopt <campaign-id>`.

## Release/version mapping

- Runtime/package version: `1.0.0rc20`
- README release identity: `1.0.0rc20`
- CHANGELOG first heading: `1.0.0rc20`
- Local tags audited before finalization:
  - `v1.0.0rc19.post1` -> `840ef098bca47557fe47402ea2ef4a57bb491178`
  - `v1.0.0rc19.post2` -> `bbedc82fdc2c71564e9bfe63d7fb5a8f56381769`
  - `v1.0.0rc19.post3` -> `fc90b9b51e8323afd4e40a7f4f69922afa861801`
  - old `v1.0.0rc20` -> `beffcff4e9531547a17030fb7926afa23d988a57`
- The final release documentation commit is the commit containing this handover;
  the corrected local `v1.0.0rc20` tag is recorded in
  `/tmp/llm-modelbench-rc20-remediation-20260718_042149/final-tag-audit.md`
  and `FINAL_SUMMARY.md`.

## Remediation commit list from RC18 through RC20

- `840ef098` fix: restore release changelog version heading
- `c14bbf6` docs: document RC20 campaign methodology and handover
- `dcdbcd2` feat: add unattended bounded terminal recovery
- `b8e4044` feat: add deterministic capability reprobe policy
- `39e5401` feat: add automatic campaign judge selection
- `4a594c4` feat: orchestrate unattended full campaign lifecycle
- `bbedc82` chore: prepare RC19.2 unattended campaign release
- `7cb2156` feat: add transactional campaign rankings adoption
- `fc90b9b` chore: prepare RC19.3 canonical adoption release
- `f356ce8` fix: persist complete campaign plan and inventory
- `4e856fe` fix: integrate post-hoc campaign judging and recovery evidence
- `8978d2a` feat: materialize terminal campaign evidence and readiness
- `487a559` fix: make campaign packages internally verifiable
- `a5ae7d0` fix: classify post-hoc subjective evidence correctly
- `f8fcf5a` fix: exclude generated campaign evidence from docs hygiene
- `beffcff` docs: record remediated real acceptance evidence
- `cf82288` feat: execute bounded campaign recovery attempts
- `f9d6481` test: prove campaign recovery execution boundary
- `200eb33` fix: expand campaign adoption preview semantics
- `e945f2f` test: complete bounded campaign recovery matrix
- `3c86d31` test: enforce complete self-verifying campaign packages
- `ba22fdc` fix: propagate package verification failures to readiness
- `8bb22a4` test: verify complete recovery and judge package references
- `1f83ae8` fix: harden transactional campaign adoption
- `112c676` fix: harden campaign cleanup and legacy migration
- `cf10134` fix: complete final campaign lifecycle integration

## Matrix evidence

Recovery matrix: bounded recovery is limited to thinking-only, empty, and
transient failures. Visible primary answers are not retried; score-zero visible
answers are terminal; first visible recovery result stops retries; exhausted
recovery persists terminal dispositions and child provenance.

Package matrix: packages contain the complete plan, primary evidence, recovery
evidence, judge evidence, effective rows, readiness, candidate rankings, one
authoritative report tree, inventory, and internal checksums. Strict
verification rejects stale, incomplete, duplicate, traversal, absolute-path, and
symlink-escape packages, and readiness reflects package verification status.

Adoption matrix: adoption accepts campaigns only, validates readiness and
package signatures, produces a detailed dry-run diff, preserves same-signature
no-ops, replaces changed signatures through a transaction, converts candidate
scope to canonical scope, requires typed confirmation, and rolls back on
failure. Real canonical adoption was not performed.

Cleanup/migration matrix: cleanup is dry-run by default, applies only to
eligible terminal campaigns, retains forensic evidence and final packages, and
is idempotent and path-contained. Migration is copy-only, refuses ambiguous or
unsafe sources, preserves source checksums, records unavailable historical
fields explicitly, keeps candidate rankings isolated, and avoids root leakage.

## Acceptance campaigns

Forced mock campaign: `rc20_forced_mock_final_20260718_123936`.

- Tested model: `qwen2.5-coder:14b`
- Tested model digest: `mock-qwen25coder14b`
- Judge model: `llama3.1:8b`
- Judge digest: `mock-llama318b`
- State history: `created -> planned -> generating -> recovering -> judging -> packaged`
- Rows: 13 terminal rows covering primary correct, primary visible wrong,
  primary partial, recovered correct, recovered visible wrong, exhausted
  recovery, terminal thinking-only, terminal transient, confirmed capability
  unavailable, environment limited, subjective judged, and operator excluded.
- Readiness: `ready_for_adoption`
- Package verification: passed
- Adoption: strict dry-run only
- Cleanup: dry-run only

Real acceptance campaign: `rc20_real_acceptance_v3_20260718_112539`.

- Tested model: `qwen2.5-coder:3b`
- Tested model digest:
  `f72c60cabf6237b07f6e632b2c48d533cef25eda2efbd34bed21c5e9c01e6225`
- Tasks: `json_extract`, `kb_taxonomy`, `txt_emails`
- Task hashes: `0f29cd37bba5bb28`, `355dbfa8f1f4d8fd`,
  `7e2af9dd07e131f5`
- Judge model: `HammerAI/tiger-gemma-v3:latest`
- Judge digest: `8fcf495de194215ea244cb6af6e652407c771dfae4b4d87eb91abfeb3018abfb`
- State history: `created -> planned -> generating -> judging -> packaged`
- Readiness: `ready_for_adoption`
- Package verification: passed
- Adoption: strict dry-run only
- Cleanup: dry-run only
- Primary evidence hash was unchanged during revalidation.

## Canonical-ranking immutability proof

Production canonical rankings default to `rankings/`. The canonical master
source file is `rankings/master_raw.jsonl`; generated summary/report files are
`rankings/master_summary.json`, `rankings/master_report_data.json`,
`rankings/master_report.html`, `rankings/master_report_v3_data.json`,
`rankings/master_report_v3.html`, `rankings/master_report_v3_1_data.json`, and
`rankings/master_report_v3_1.html`.

At final audit, `rankings/` contained no tracked or untracked files. The earlier
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` value was
the SHA-256 of an empty command stream and is not meaningful proof. It is
corrected by the deterministic structured manifest at
`/tmp/llm-modelbench-rc20-remediation-20260718_042149/canonical-rankings-final-manifest.json`.
That manifest hashes repository-relative paths, mode, byte size, and SHA-256 of
regular files in stable lexical order, plus explicit root/tracked/status
metadata, so an absent or empty canonical root is recorded as data rather than
as an empty input stream. The earlier adoption unit’s non-empty before/after
digest was
`abcfa6a9d4df344d1781bc2560b5e4cdcae08b39ed303063535e7e1e926a304a`.

## Validation

Final validation commands:

- `python3 -m compileall -q llm_modelbench tests`
- focused campaign/recovery/judge/readiness/package/adoption/cleanup/migration/root-hygiene/acceptance/release tests
- `python3 -m pytest -q`
- `./llmb selftest`
- `python tools/release_check.py`
- `git diff --check`
- `git status --short`

The final counts and command results are recorded in
`/tmp/llm-modelbench-rc20-remediation-20260718_042149/FINAL_SUMMARY.md`.

## Known limitations

- Real canonical adoption has not been performed.
- Machine judging remains provisional and must be reviewed before publication.
- Historical legacy migrations cannot fabricate missing legacy model digests,
  task hashes, or judge provenance; unavailable values are explicit.
- Campaign cleanup removes only proven redundant artifacts and intentionally
  leaves the campaign independently auditable.

## Human review sequence

1. Review this handover, the changelog, and the README campaign workflow.
2. Review the forced mock and real acceptance JSON reports under the remediation
   log root.
3. Inspect both campaign review packages and package checksum manifests.
4. Review the corrected canonical-ranking manifest.
5. Run the exact adoption preview:

   ```bash
   ./llmb rankings --adopt rc20_real_acceptance_v3_20260718_112539 --dry-run --out /tmp/llm-modelbench-rc20-review-rankings
   ```

6. If and only if the preview is acceptable, perform canonical adoption in an
   interactive terminal and type `ADOPT rc20_real_acceptance_v3_20260718_112539`
   when prompted.

## Rollback and revert guidance

Do not delete or rewrite campaign primary evidence. If code or documentation
must be changed after review, use normal follow-up commits or `git revert` for
published commits. If a local unpushed tag target is wrong, move only the local
tag before pushing. If canonical adoption is later performed and found
incorrect, revert through a new audited adoption transaction or restore the
previous canonical rankings directory from the transaction backup recorded by
the adoption record.

No remote push, real canonical adoption, model pull/delete, service mutation,
sudo action, or KV mutation was performed by the RC20 remediation work.
