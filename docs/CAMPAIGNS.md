# Campaign methodology

Campaign mode is the release-quality workflow for benchmark evidence that may
later be adopted into canonical rankings. All campaign artifacts are rooted
under `campaigns/<campaign-id>/`; campaign commands must not write root-level
packages, logs, recovery children, judge sidecars, or candidate ranking stores.

## Operator sequence

The normal sequence is:

1. optional smoke;
2. `./llmb campaign plan --campaign-id <id> ...`;
3. primary generation with generation-time judge mode off;
4. bounded automatic recovery for non-scorable failures only;
5. deterministic capability and environment terminal classification;
6. automatic post-hoc judge selection and execution;
7. terminal effective-row ledger generation;
8. readiness calculation;
9. candidate ranking generation;
10. complete verified package creation;
11. human adoption preview;
12. exact typed canonical adoption decision.

`./llmb campaign run --campaign-id <id> --unattended-safe --yes ...` uses the
same orchestration path for planning, generation, recovery, judging, readiness,
candidate rankings, package creation, package verification, adoption dry-run
eligibility, and cleanup dry-run eligibility.

`./llmb campaign plan` may create a new campaign, complete a campaign still in
`created`, or return an explicit no-op for an identical already-planned
campaign. It refuses in-place changes after `planned`; use a new campaign ID
when models, tasks, samples, context, thinking, or output limits must change.
Interrupted campaigns must be resumed with `campaign resume`, not replanned.
`campaign plan` does not accept `--no-fingerprint` because fingerprint probes
are not executed by planning; fingerprint controls belong to `campaign run`.
If any retained evidence is mutated after packaging, package verification fails
closed and adoption remains blocked.

## Recovery fairness

Recovery is only for rows that are not scorable as primary evidence:
thinking-only output, empty output, or transient transport/runtime failures. A
visible primary answer is never retried. A visible answer with score zero is
terminal and stops recovery. Recovery therefore cannot fish for a better score.

Recovery uses bounded progressive budgets and circuit breakers. Primary
`raw_results.jsonl` remains immutable. Recovery planning, result summaries,
attempt rows, and child run evidence are retained so reviewers can reconstruct
which model/task cell was retried and why.

## Judging

Campaign generation runs with judge mode off. Subjective rows are judged
post-hoc only, after recovery and terminal classification. Judge selection
excludes tested cohort names, exact digests, and digest-equivalent aliases. The
tested cohort is deduplicated by model name and digest before selection. The
selection record persists architecture, calibration, and selection reason.
`manifest.json` field `judge_model`, when present, refers to generation-time
judging; post-hoc judge model and digest are recorded in
`evidence/judge/judge_selection.json` and `evidence/judge/judge_summary.json`.

Machine-judged subjective evidence is labelled provisional. If no qualified
judge is available, readiness records the unresolved external-judge blocker
instead of accepting the campaign.

## Readiness and effective rows

The effective terminal ledger is the authoritative adoption source. Each row
records model identity, task hash, primary row hash, recovery/judge references
when applicable, result origin, score, reason, terminal disposition, and
capability/environment/harness status.

Readiness requires every applicable cell to be terminal, package verification
to pass, and no unresolved harness, manual, or external-judge blocker. Recovered
does not imply correct: recovered visible wrong rows remain wrong.

## Package

Each campaign has one final review package. The package includes the manifest,
complete plan, primary evidence, recovery evidence, judge evidence, effective
rows, readiness JSON/Markdown, candidate ranking evidence, one authoritative
report tree, package inventory, and internal SHA-256 checksums. Verification
rejects missing required files, duplicate members, duplicate authoritative
report trees, traversal, absolute paths, symlink escapes, stale checksums, and
invalid recovery/judge references.

## Adoption

Adoption accepts campaigns only, not arbitrary directories. The dry-run preview
validates readiness and package signatures, compares incoming and existing
source signatures, reports additions, replacements, unchanged rows, score and
reason changes, disposition changes, coverage changes, aggregate movement,
ranking movement, task hashes, judge provenance, and candidate-to-canonical
scope conversion.

Apply requires an interactive exact confirmation: `ADOPT <campaign-id>`. There
is no generic `--yes` bypass. Canonical rebuild is transactional and rolls back
if validation or filesystem operations fail. Same-signature rows are no-ops;
changed signatures replace prior rows only through the audited transaction.

## Cleanup and migration

`./llmb campaign clean <id>` is a dry-run by default. Apply is allowed only for
terminal campaigns whose final package verifies and whose retained evidence is
complete. Cleanup may remove only explicitly classified redundant files; it
does not remove primary raw evidence, recovery attempt provenance, judge
evidence, effective rows, readiness, final packages, or checksums.

`./llmb campaign migrate-legacy --run-id <run-id> --campaign-id <id>` is
copy-only. It preserves source bytes, refuses ambiguous or unsafe layouts,
generates migration provenance and source checksums, marks unavailable
historical fields explicitly, keeps candidate rankings isolated, and never
updates canonical rankings.
