# Scoring and Interpretation

## Score contract

Public benchmark scores are bounded to `0.0` through `100.0`. Quality measures
task correctness only. Throughput, VRAM, offload, and hardware information are
reported separately and do not reduce a correct score.

## Deterministic scorers

- **Python and file operations:** execute candidate code in restricted temporary
  environments against assertions or expected layouts.
- **Web and JavaScript:** check required structural/behavioral contracts.
- **JSON, exact, regex, contains, and line-set:** validate the stated output
  contract; they do not grant credit merely because a token appears in prose.
- **OCR/PDF:** compare transcriptions or narrow code targets with the task's
  deterministic reference contract.
- **Needle:** records declared-depth coverage. Partial coverage is diagnostic
  and is not silently promoted to a fully verified context result.
- **Retrieval:** evaluates labelled ranks with recall@1 and MRR.

Empty or stripped-only output receives a bounded explicit result rather than a
negative sentinel or an exception.

## Category and composite boundaries

Category quality is a difficulty-weighted mean over scored task rows. The
standard report renormalizes only across categories that actually ran; missing
coverage is not zero. Report-time weight overrides create a separate report
copy and do not rewrite raw rows.

The dossier is a separate, read-only cross-run composite. It includes only
covered, non-stale categories and renormalizes their weights. Stale and pending
coverage remain visible instead of becoming zero-quality evidence.

## Retrieval diagnostics

For future instrumented retrieval runs, every case preserves its target rank,
top three IDs, target and nearest-distractor similarities, margin, pass-at-one
status, and `embed_model`. These values explain an already-computed score; they
do not alter the recall@1 or MRR formula. Diagnostics avoid full query and
document text.

## Agentic boundaries

Agentic tool tasks score the requested tool decision, named arguments, refusal
rules, and JSON envelope contract. Output-format deviations can reduce a
correct decision but cannot make an incorrect decision competitive. No external
tool is executed.

## Reliability and calibration

`repeat-report` interprets already-recorded matching cells as `stable`,
`moving`, `reason-moving`, `insufficient-repeats`, or `missing`. A one-run cell
is insufficient evidence, and a missing cell remains missing rather than zero.

When comparable repeated cells exist, the empirical noise band is the maximum
observed repeat score range. `diff --noise-band N` labels a quality delta whose
absolute value is at most `N` as `tied/noise-band`; it leaves scores, sorting,
and aggregate formulas unchanged. Configuration sensitivity is separate from
repeatability noise and belongs in sensitivity reports.
