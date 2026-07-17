# Benchmark Levels and Interpretation

## Levels

| Level | Includes | Use |
|---|---|---|
| `smoke` | Fast baseline tasks applicable to each family. | Screen a broad local roster for broken, off-task, or unsuitable models. |
| `short` | Smoke plus harder coding, agentic, OCR/PDF, retrieval, and other applicable tasks. | Compare viable models in a lane. |
| `full` | Short plus long-context work. | Focused validation of a small shortlist and context variants. |

Levels are cumulative. A family only receives tasks it is eligible to run.

## Selecting a run

Use `inventory` and `plan` first. Use a mock run to validate a command or
artifact workflow. Run a real smoke pass only with operator approval, then
apply short/full runs to a deliberate shortlist rather than treating every
installed tag as a final candidate.

## Samples and repeats

`--samples N` controls task samples; the configured sampling mode decides which
tasks receive the additional samples. Samples improve evidence only when they
are comparable. After a candidate comparison, use:

```bash
llm-modelbench repeat-report runs/run_a runs/run_b
```

A single observed cell is `insufficient-repeats`, not stable. Missing rows are
missing, not zero. Moving rows or reason changes are evidence to investigate,
not an automatic model ranking outcome.

## Deltas and sensitivity

Use an empirical repeat range as a conservative comparison band:

```bash
llm-modelbench diff --a runs/baseline --b runs/candidate --noise-band 2.0
```

Scores inside the supplied band are labelled `tied/noise-band`; the command
does not alter quality or ranking values. `sensitivity-plan` and
`sensitivity-report` examine configuration fragility, which is distinct from
same-config repeatability.

## Comparability cautions

Do not compare scores blindly across benchmark versions, task-set changes,
different filters, context settings, judge modes, or partial coverage. Reports
and diffs surface metadata differences, but the operator remains responsible
for deciding whether two runs answer the same question.
