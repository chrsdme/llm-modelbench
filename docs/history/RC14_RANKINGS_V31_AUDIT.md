# LLM ModelBench 1.0.0rc14 Rankings V3.1 audit

## Scope

RC14 adds a calm, split-page, human-facing Rankings V3.1 layout and ranking-scope controls. It keeps the existing V3 operational report and master evidence intact.

## User-visible changes

- `rankings/v3_1/index.html` decision dashboard.
- Dedicated model pages under `rankings/v3_1/models/`.
- `compare.html` page for two-model comparison.
- `methodology.html` explaining evidence tiers and capability/recovery semantics.
- Larger text, lighter slate panels, wider rows and no tiny model subcards.
- Current selected task evidence is readable in full-width tables.
- Prompts, rubrics, expectations and raw details are hidden behind expandable evidence blocks rather than clipped panels.
- Missing, capability-limited, measured-failure and recovery-limited evidence are grouped in plain language.
- Internal task IDs remain visible, but human-readable task names are primary.

## Ranking update controls

Evidence-producing commands refresh rankings by default:

```text
run
repair
context-profile
judge-dumps
```

New flags:

```text
--no-ranking-update
--separate-ranking
```

`--no-ranking-update` writes evidence normally but skips the immediate rankings refresh. Future canonical rescans may still include the run.

`--separate-ranking` writes an isolated `rankings-separate/<run-id>/` report and marks the run as separate/diagnostic so canonical rankings do not import it by default.

Explicit `--rankings-out` remains the manual redirect and wins when supplied.

## Non-destructive exclusion controls

`llmb rankings` now supports:

```text
--exclude-model / --include-model
--exclude-run / --include-run
--archive-run / --unarchive-run
--list-excluded
--reason
--include-separate
```

The controls update:

```text
rankings/exclusions.json
rankings/audit_log.jsonl
```

They do not delete or rewrite source evidence. Raw rows remain preserved and can be restored into the ranked view by reversing the exclusion and rescanning.

Public-release privacy note: exclusion entries deliberately avoid personal operator names or user identifiers. They store only the target, reason, timestamps and event type.

## Generated artifacts

```text
rankings/master_report_v3_1_data.json
rankings/master_report_v3_1.html
rankings/v3_1/index.html
rankings/v3_1/compare.html
rankings/v3_1/methodology.html
rankings/v3_1/assets/report.css
rankings/v3_1/assets/report.js
rankings/v3_1/data/site_manifest.json
rankings/v3_1/models/*.html
```

## Scoring impact

None. RC14 does not change task scores or V3 score formulas. It adds presentation, automatic update routing, separate ranking scope and non-destructive view exclusions.

## Freeze impact

Freeze includes V3.1 data, landing page and split-site files. Canonical freezes should also preserve `exclusions.json` and `audit_log.jsonl` when present.

## Validation

Performed in the build environment:

```text
compileall: passed
pytest: 379 passed
selftest: SELFTEST: ALL GOOD
version: llm-modelbench 1.0.0rc14
patch dry-run on RC13: passed
clean patch apply smoke: passed
focused RC14 tests: passed
```

No Ollama call, GPU inference, sudo command, service restart, model deletion, hard evidence deletion or live ranking mutation was performed while creating this patch.
