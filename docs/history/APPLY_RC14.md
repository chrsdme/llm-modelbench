# Apply LLM ModelBench 1.0.0rc14

## Purpose

RC14 adds Rankings V3.1 and ranking-scope controls:

- calm split rankings site under `rankings/v3_1/`;
- one readable page per model;
- portable V3 files are still generated and frozen;
- automatic ranking refresh remains the default after evidence-producing commands;
- `--no-ranking-update` skips automatic refresh while preserving evidence;
- `--separate-ranking` writes an isolated `rankings-separate/<run-id>/` report instead of touching canonical rankings;
- non-destructive model/run exclusion controls are available through `llmb rankings`;
- exclusion metadata is public-safe and does not store personal operator names.

## Apply

```bash
cd ~/llm-modelbench

git diff > ../llm-modelbench-pre-rc14-local.patch

patch --dry-run -p1 \
  < llm-modelbench-1.0.0rc14-rankings-v31-scope.patch

patch -p1 \
  < llm-modelbench-1.0.0rc14-rankings-v31-scope.patch

python3 -m compileall -q llm_modelbench tests
python3 -m pytest -q
python3 -m llm_modelbench selftest
./llmb --version
```

Expected:

```text
379 passed
SELFTEST: ALL GOOD
llm-modelbench 1.0.0rc14
```

## Generate canonical V3.1

```bash
./llmb rankings --runs-dir runs --out rankings --rescan
```

Open:

```text
rankings/v3_1/index.html
```

Also generated:

```text
rankings/master_report_v3_1.html
rankings/master_report_v3_1_data.json
```

## Ranking-scope controls

Default auto-update:

```bash
./llmb run ...
./llmb repair ...
./llmb context-profile ...
./llmb judge-dumps ...
```

Skip automatic ranking refresh:

```bash
./llmb run ... --no-ranking-update
```

Create isolated diagnostic rankings without touching canonical `rankings/`:

```bash
./llmb run ... --separate-ranking
```

The isolated output is:

```text
rankings-separate/<run-id>/
```

Explicit `--rankings-out <path>` still wins when supplied.

## Non-destructive exclusions

Hide a model from canonical rankings without touching raw evidence:

```bash
./llmb rankings --exclude-model 'model-name-or-digest' --reason 'diagnostic run' --rescan
```

Reverse it:

```bash
./llmb rankings --include-model 'model-name-or-digest' --rescan
```

Hide or restore a run:

```bash
./llmb rankings --exclude-run run_id --reason 'diagnostic run' --rescan
./llmb rankings --include-run run_id --rescan
```

Archive or unarchive a run:

```bash
./llmb rankings --archive-run run_id --reason 'old baseline' --rescan
./llmb rankings --unarchive-run run_id --rescan
```

Inspect controls:

```bash
./llmb rankings --list-excluded
```

Files:

```text
rankings/exclusions.json
rankings/audit_log.jsonl
```

These controls do not delete `runs/*/raw_results.jsonl`. They only change the ranked/summary view.

## Freeze

```bash
./llmb freeze \
  --repo-root . \
  --runs-dir runs \
  --rankings-dir rankings \
  --out snapshots/rc14-rankings-v31 \
  --label rc14-rankings-v31

./llmb freeze --out snapshots/rc14-rankings-v31 --verify
sha256sum -c snapshots/rc14-rankings-v31/SHA256SUMS.txt
snapshots/rc14-rankings-v31/VERIFY.sh
```
