# Judging existing subjective dumps

Subjective answers can be judged after a run without regenerating the tested model's answer.

```bash
llmb judge-dumps --run-id RUN_ID --judge single --judge-model qwen2.5:14b
llmb judge-dumps --run-id RUN_ID --judge panel --judge-model gpt-oss:20b
llmb judge-dumps --everything --runs-dir runs --judge single --judge-model qwen2.5:14b
```

Preview eligibility without calling the judge:

```bash
llmb judge-dumps --everything --runs-dir runs --judge-model qwen2.5:14b --dry-run
```

`--everything` scans every directory directly under `runs/` containing `raw_results.jsonl`, then processes eligible subjective rows run by run. It does not judge deterministic tasks because they already have deterministic scorers.

## Evidence and resume behavior

- The tested/source model is never called.
- `raw_results.jsonl` is never modified.
- Results append to `judge_results.jsonl` and are overlaid by reports and rankings.
- Rows already judged by the same judge model and mode are skipped, making the batch resumable.
- `--force` deliberately rejudges them.
- Missing/empty dumps, stale task hashes and source-model errors are skipped with reasons.
- `single` makes one judge request per stored sample. `panel` uses three judge personas through the configured judge model.

A real judge batch consumes judge-model inference time. In a non-interactive shell, `--yes` is required after reviewing the printed count. `--dry-run` never calls a model.
