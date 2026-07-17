# Persistent rankings and model cards

```bash
llmb rankings --runs-dir runs --out rankings
llmb rankings --runs-dir runs --out rankings --rescan
llmb rankings --runs-dir runs --out rankings --watch --interval 5
```

`--rescan` rereads every run currently present. Deleted-run evidence already imported into `rankings/master_raw.jsonl` remains preserved. `--watch` rescans for run and judge-sidecar changes.

## Current evidence selection

All historical rows remain in each model card. One current row per model digest and task is selected by:

1. current task hash;
2. canonical, non-diagnostic configuration;
3. highest cumulative level;
4. newest run.

## Scoring

Within a category:

```text
sum(task score * task difficulty) / sum(task difficulty)
```

Difficulty-zero tasks are gates and add no positive quality. Overall quality is the configured category-weighted mean renormalised over eligible measured categories. Speed, VRAM and model size never alter quality.

Scores within 0.5 points share a tie band. Display order inside a band uses greater scope coverage, fewer errors, fewer failed gates, higher measured speed and smaller size. These are operational differentiators, not extra quality points.

## Status

- `complete`: a cumulative full-level run exists and every currently applicable positive-difficulty task has a current, numeric, non-error result.
- `provisional`: a numeric score exists, but some current applicable scope is absent, stale, unjudged or incomplete.
- `ineligible`: no eligible positive-difficulty quality result exists.

The model card lists exact missing/stale tasks and preserves every historical attempt, including run ID, benchmark version, task configuration, score, reason, timing, TPS and judge provenance.
