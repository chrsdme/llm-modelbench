# Acceptance controls

This page describes the verified offline acceptance-control contracts. It does
not describe real-host qualification, real recovery, real catch-up, or adoption
as already performed.

## Judge controls

Campaign primary generation runs with generation-time judging off. Subjective
answers are judged post-hoc only after primary evidence, recovery
classification, and terminal effective-row generation are available.

Judge selection uses ModelBench-owned eligibility and qualification evidence.
The policy records candidate source, configured precedence, role filtering,
qualification outcome, and the selected judge. Models in the tested cohort are
excluded by model name and exact digest, including digest-equivalent aliases, so
a model cannot independently judge its own subjective output. Qwen-family judge
exclusion is the default policy unless explicitly overridden by configuration.

If no qualified independent judge is available, readiness records an external
judge blocker such as `awaiting_independent_judge` or
`judge_exhausted_unavailable`. It does not silently accept the campaign.
Machine-judged subjective evidence remains provisional.

## Recovery controls

Recovery is limited to rows that are not already scorable primary evidence:
thinking-only output, empty output, transient backend/runtime failures, or
other explicitly eligible non-scorable failures. A visible answer, including a
visible answer scored zero, is terminal and is not retried for a better score.

Recovery planning and post-execution reconciliation preserve immutable primary
evidence. Any effective recovered row records the source row hash, recovery
policy version, action ID, child run ID when applicable, attempt number, final
outcome, and whether the final evidence came from child raw output or an action
result. Terminal failures remain terminal evidence rather than being hidden.

## Supersession controls

Supersession is an append-only evidence graph stored in
`evidence/supersessions.jsonl`. Native records are schema-versioned, include
source and replacement campaign/run provenance, bind exact source and
replacement row hashes, and carry a stable semantic `supersession_id` that
excludes display-only timestamps.

Use `campaign supersede` to append a corrected-evidence relationship:

```bash
llm-modelbench campaign supersede \
  --campaign-id CAMPAIGN_ID \
  --source-row-hash SOURCE_ROW_HASH \
  --replacement-run-id synthetic-catchup \
  --replacement-row corrected-row.json \
  --replacement-row-hash REPLACEMENT_ROW_HASH \
  --reason corrected_synthetic_evidence \
  --operator operator \
  --dry-run
```

The source row hash must identify a row in campaign primary evidence. The
replacement row hash, task, model, and digest are validated when available.
Malformed ledgers, unsupported schema versions, contradictory aliases, mutable
native deactivation flags, forks such as `A -> B` and `A -> C`, and cycles all
fail closed. Multiple semantic records for the same effective successor
`A -> B` are retained deterministically as supporting evidence.

Effective rows resolve valid chains transitively, for example `A -> B -> C`,
while preserving the full chain evidence. The graph is not flattened or
rewritten.

## Campaign config and execution

`campaign init` writes the strict versioned JSON schema accepted by
`campaign execute --config`:

```bash
llm-modelbench campaign init campaign.json
llm-modelbench campaign execute --config campaign.json --mock
```

The current schema requires `schema_version`, `campaign_id`, non-empty
`models`, `level`, positive `samples`, `runtime_policy.auto`,
`context_needle_policy.needle_max_ctx`, `judge_policy.enabled`,
`executable_scorer_policy.allow_host_code_execution`, and
`stop_before_adoption`. Unknown keys and unsupported schema versions fail
before a campaign is created.

The generated template intentionally does not advertise telemetry, recovery,
KV fallback, service control, or adoption fields. `judge_policy.enabled` must
remain false in the current config workflow. `stop_before_adoption` must remain
true.

`campaign execute --config` records an immutable config plan signature under
the campaign plan directory. Re-running the same config after a completed
packaged campaign is a no-op. Re-running with changed config is refused. An
interrupted campaign must be resumed with `campaign resume <campaign-id>` so
the recorded lifecycle phase and runtime identities remain authoritative.

## Readiness controls

Readiness is computed from terminal effective rows, recovery/judge sidecars,
supersession graph validation, and package verification. It blocks campaigns
with unresolved manual, harness, external-judge, malformed-supersession, or
package-verification issues.

`ready_for_adoption` means the offline evidence package is internally complete
and candidate adoption preview may be considered. It does not mean canonical
rankings have been changed. Adoption remains a separate interactive operation
requiring exact typed confirmation.

## Evidence immutability

Primary evidence remains immutable. Recovery, judging, supersession,
configuration plans, readiness, packages, and candidate rankings add
traceable sidecar or derived evidence. They do not rewrite source raw rows, do
not mutate canonical rankings automatically, and do not perform real model
work unless the operator invokes a real benchmark/judge/recovery command.
