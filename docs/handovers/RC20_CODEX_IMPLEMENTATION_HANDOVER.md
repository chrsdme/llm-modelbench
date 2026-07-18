# RC20 Codex implementation handover

## Final architecture

Campaign work is isolated below `campaigns/<id>/`: manifest/state history,
plan, primary/recovery/judge evidence, candidate rankings, reports, logs,
checksums, readiness, adoption record, and a review ZIP. Legacy `runs/` remain
readable; migration copies rather than moves source evidence.

The campaign policy treats every visible scored answer—including score zero—as
immutable. Recovery policy primitives only permit thinking-only, empty-output,
and transient retries, with progressive budgets. Capability resolution does not
misclassify transport failures as unsupported. Judge selection excludes cohort
names/digests and prefers calibrated compatible text judges. Candidate adoption
is campaign-only, verifies readiness/package inventory, previews first, and
requires an interactive `ADOPT <campaign-id>` confirmation.

## Commands

`./llmb campaign plan --campaign-id <id> ...` creates a plan.

`./llmb campaign run --campaign-id <id> --unattended-safe --yes ...` runs an
isolated primary campaign, readiness summary, and package.

`./llmb campaign status <id>`, `package <id>`, and `clean <id> [--apply]`
inspect/package/retain evidence. `migrate-legacy` is copy-only.

`./llmb rankings --adopt <id> --dry-run` is the exact human adoption preview.
Do not run non-dry adoption without a human review; canonical adoption was not
performed by this campaign work.

## Releases and validation

Local tags: `v1.0.0rc19.post1`, `v1.0.0rc19.post2`,
`v1.0.0rc19.post3`, and `v1.0.0rc20`.

The RC19.1 tag intentionally remains on its original release-metadata commit;
the subsequent changelog-heading correction is a separate commit and the tag
was not rewritten.

Remediation acceptance campaign: `campaigns/rc20_real_acceptance_v3_20260718_112539`.
It used `qwen2.5-coder:3b`, one sample, `json_extract`, `kb_taxonomy`, and
`txt_emails`; primary generation used judge-off and the post-hoc judge evidence
is persisted under `evidence/judge/`. The package internally verifies and the
campaign is `ready_for_adoption`; adoption was dry-run only. No model
pull/delete, service/KV/sudo operation, or canonical ranking mutation occurred.

## Review and rollback

Review local commits/tags, `/tmp/llm-modelbench-rc20-codex-20260718_034148`,
the real package/checksums, and the adoption preview. Roll back code only with
normal follow-up revert commits. Do not delete or overwrite primary evidence.
