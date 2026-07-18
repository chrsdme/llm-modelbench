# RC20.post1 lifecycle hotfix handover

## Release identity

- Version: `1.0.0rc20.post1`
- Tag to create: `v1.0.0rc20.post1`
- Base hotfix commit: `8add25d fix: refuse unsafe campaign replanning`
- Existing `v1.0.0rc20` tag remains unchanged and must not be moved for this
  post-release hotfix.

## Reproduction

Real stale campaign: `rc20_postrelease_smoke_20260718_183710`.

Observed bug: after the campaign reached
`created -> planned -> generating -> judging -> packaged`, a later
`./llmb campaign plan ...` command was accepted and rewrote
`plan/plan.json`.

Evidence:

- current plan SHA-256:
  `73716aeb27d30e23264f458d4eb815be802787fb17fe9c4a56a27f2d80598d23`
- plan stored in package SHA-256:
  `050a9e7f632ceafaa332b711d23a8b4f66470984e7ee28a1c0ecddaccbb75681`
- preserved evidence directory, relative to this checkout's parent:
  `../llm-modelbench-evidence/rc20_postrelease_smoke_20260718_183710_packaged_replan_bug`

The strict verifier and adoption path correctly refused the stale package with
`CampaignError: campaign package checksums do not verify`. That fail-closed
behavior is retained.

## Fix summary

`campaign plan` now:

- may create a new campaign;
- may operate on state `created`;
- returns an explicit deterministic no-op for an identical request in state
  `planned`;
- refuses changed settings in state `planned` because there is no audited safe
  replacement policy;
- refuses `generating`, `recovering`, `judging`, `packaged`, `accepted`,
  `rejected`, `archived_diagnostic`, and `interrupted` before writing any file;
- directs interrupted campaigns to `campaign resume`;
- tells operators to use a new campaign ID when settings must change.

The refusal happens before filesystem mutation. Existing `plan.json`,
`inventory.json`, `capabilities.json`, `manifest.json`, readiness, package, and
primary evidence remain byte-for-byte unchanged after refusal.

The package verifier and adoption validation were not weakened. Stale packages
remain blocked from adoption.

Post-hoc judge cohort identities are deduplicated by model name and digest
before judge selection.

## Post-fix validation

Successful post-fix campaign: `rc20_post1_smoke_20260718_190739`.

- packaged campaign replan refusal: PASS
- zero mutation: PASS
- package verifier fail-closed: PASS
- fresh campaign lifecycle: PASS
- judge cohort entries: 2
- unique judge cohort identities: 2
- judged rows: 2
- judge errors: 0
- readiness: `ready_for_adoption`
- package verification: PASS
- adoption dry-run: PASS
- real canonical adoption: not performed

## Non-blocking follow-ups

The two ETA values and duplicate refusal display observed during smoke review
are cosmetic follow-ups for a later UI pass. They are not release blockers for
`1.0.0rc20.post1` because the lifecycle mutation bug is fixed, package/adoption
validation remains fail-closed, and the post-fix campaign reached
`ready_for_adoption`.

## Human release sequence

1. Review `8add25d` and the release commit for `1.0.0rc20.post1`.
2. Confirm `v1.0.0rc20` still targets the original RC20 release commit.
3. Review the preserved stale-campaign evidence directory.
4. Review post-fix campaign `rc20_post1_smoke_20260718_190739`.
5. Push the release commit and `v1.0.0rc20.post1` tag only after review.

No remote push or real canonical adoption was performed while preparing this
handover.
