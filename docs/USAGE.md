# Usage

All commands are subcommands of `llm-modelbench`. From a checkout, `./llmb` is
the local-venv wrapper and is equivalent to `.venv/bin/python -m llm_modelbench`.

```bash
.venv/bin/python -m llm_modelbench --help
```

## Executable coding tasks

Python, file-operation, JavaScript, and FIM scorers execute candidate code. `run` refuses these tasks unless `--allow-host-code-execution` is supplied. Use that flag only inside a disposable container or VM. See [SAFETY.md](SAFETY.md).

## Discovery and inventory

```bash
llm-modelbench inventory [--json] [--mock] [--auto]
llm-modelbench doctor [--json]
llm-modelbench plan [--mock] [--level smoke|short|full] [--models 'a;b'|--all|--select] [--auto]
```

`inventory` and a non-mock `plan` read the local Ollama inventory. They do not
execute benchmark prompts. `plan --mock` is fully offline.

## Benchmark execution

```bash
llm-modelbench run --level smoke --allow-host-code-execution --run-id smoke1
llm-modelbench run --level short --samples 3 --allow-host-code-execution --include-regex 'qwen|llama' --run-id short1
llm-modelbench run --level full --tasks needle --run-id context1
llm-modelbench run --level short --categories coding_python,coding_js --allow-host-code-execution --run-id coders_only
llm-modelbench run --level short --allow-host-code-execution --exclude-regex 'gguf|abliterat' --run-id stable_only
llm-modelbench run --level full --context-only --run-id context_sweep
```

Model selectors are mutually exclusive: `--models` uses exact semicolon-delimited names, `--all` explicitly selects the entire installed inventory (also the default), and `--select` opens a model-only TUI. There is no `-all` alias. `--auto` runs small functional capability probes before task routing. For interactive model plus test editing use `llm-modelbench wizard`.

Useful filters include `--categories`, `--tasks`, `--task-regex`,
`--include-regex`, `--exclude-regex`, `--family-base-only`,
`--context-aliases-only`, and `--context-only`. Real runs execute models and
require operator approval. A TTY prompts after showing the exact plan. A non-TTY invocation stops before benchmark task requests unless `--yes` is supplied; `--yes` approves exactly that printed plan. When `--auto` was explicitly requested, small capability-probe calls occur while the plan is built.

## Mock and offline usage

```bash
llm-modelbench selftest
llm-modelbench plan --mock
llm-modelbench run --mock --level short --allow-host-code-execution --run-id demo --yes
```

Mock mode is deterministic and useful for validating the harness, reports, and
operator workflows without Ollama or a GPU.

## Reporting

```bash
llm-modelbench report --run-id demo
llm-modelbench report --out runs/demo
llm-modelbench report --run-id demo --weights coding_python=0.4,agentic_tool=0.3
```

`--weights` is a report-time override. It writes a separate report copy by
default, preserving raw run artifacts and original aggregate evidence.

## Judge existing dumps

```bash
llm-modelbench judge-dumps --run-id demo --judge single --judge-model qwen2.5:14b
llm-modelbench judge-dumps --everything --runs-dir runs --judge single --judge-model qwen2.5:14b --dry-run
llm-modelbench judge-dumps --everything --runs-dir runs --judge single --judge-model qwen2.5:14b --yes
```

`--everything` scans each prior run sequentially. It judges only eligible subjective dumps, never reruns tested models, and appends immutable overlays to `judge_results.jsonl`. See [JUDGE_DUMPS.md](JUDGE_DUMPS.md).

## Persistent rankings

```bash
llm-modelbench rankings --runs-dir runs --out rankings
llm-modelbench rankings --runs-dir runs --out rankings --rescan
llm-modelbench rankings --runs-dir runs --out rankings --watch --interval 5
```

The HTML contains overall, category, class and multimodal rankings plus per-model cards with full historical attempts. See [RANKINGS.md](RANKINGS.md).

## Review and export

```bash
llm-modelbench export-review runs/demo --out demo_review.zip
llm-modelbench pack-subjective --run-id demo
llm-modelbench grade --run-id demo --export-blind
```

`export-review` accepts one or more positional run directories and includes
retrieval diagnostics when present.

## Coverage and gaps

```bash
llm-modelbench coverage update --ledger runs/coverage_ledger.json --run-dir runs/demo
llm-modelbench coverage show --ledger runs/coverage_ledger.json
llm-modelbench gaps --ledger runs/coverage_ledger.json --mock --json
```

Coverage is keyed by model digest when identity artifacts are available. Gaps
is advisory only: it never schedules or starts a benchmark. `--mock` checks
against the offline stub roster instead of Ollama.

## Dossier

```bash
llm-modelbench dossier --ledger runs/coverage_ledger.json --runs-dir runs --json
llm-modelbench dossier --ledger runs/coverage_ledger.json --runs-dir runs --out runs/dossier.json
```

The dossier combines already-computed non-stale category evidence. Missing or
stale categories are explicit and are not scored as zero.

## Reliability and comparisons

```bash
llm-modelbench repeat-report runs/repeat_a runs/repeat_b
llm-modelbench diff --a runs/baseline --b runs/candidate --noise-band 2.0
```

Repeat reports classify matching cells and derive an empirical noise band when
repeat evidence exists. `diff --noise-band` adds interpretation only; it does
not alter reported quality.

## Sensitivity

```bash
llm-modelbench sensitivity-plan --tasks needle --level full
llm-modelbench sensitivity-report runs/config_a runs/config_b --out runs/sensitivity.md
```

The planner prints a diagnostic script; it does not execute it. Sensitivity
describes configuration fragility and is separate from same-config repeats.

## Simulation and read-only serving

```bash
llm-modelbench simulate --run-dir runs/demo --simulate-vram 24 --json
llm-modelbench serve --runs-dir runs/demo --port 8080
```

Simulation replays stored environment-class VRAM skips only. The server reads
existing `summary.json` artifacts and never refreshes or benchmarks models.

## Watch and wrappers

Run and watch in the same terminal, one after the other:

```bash
./llmb-run --mock --level short --allow-host-code-execution --run-id demo --yes && \
./llmb-watch --run-id demo --layout compact --once
```

Run in the background, watch it live in the same window:

```bash
./llmb-run --level short --allow-host-code-execution --run-id demo &
./llmb-watch --run-id demo --layout compact
```

Run and watch in two separate terminals (the common case for a long real
run): start the run in one terminal, then in a second terminal:

```bash
./llmb-watch --run-id demo --layout interactive
```

If you don't pass `--run-id`, `watch` looks in `--runs-dir` (default `runs/`)
and picks for you: the only run if there's just one, the only in-progress run
if several exist but only one is still running, or an interactive prompt to
choose if it's genuinely ambiguous. Without a terminal attached (for example,
piped output), an ambiguous case refuses and lists the candidates instead of
guessing:

```bash
./llmb-watch --layout compact
```

Interactive watch requires a TTY. The local wrappers are `llmb`, `llmb-run`,
and `llmb-watch`; bare `llm` is deliberately not installed.

## Environment variables

`LLM_MODELBENCH_*` configuration variables override the equivalent `Config`
field: `LLM_MODELBENCH_OLLAMA_URL`, `LLM_MODELBENCH_JUDGE_MODEL`,
`LLM_MODELBENCH_EMBED_MODEL`, `LLM_MODELBENCH_VRAM_BUDGET_GB`,
`LLM_MODELBENCH_SEED`, `LLM_MODELBENCH_CTX` / `LLM_MODELBENCH_NUM_CTX`,
`LLM_MODELBENCH_NUM_PREDICT`, `LLM_MODELBENCH_THINK`, and
`LLM_MODELBENCH_NEEDLE_MAX_CTX`.

## Help

Use the CLI as the source of truth for supported options:

```bash
.venv/bin/python -m llm_modelbench report --help
.venv/bin/python -m llm_modelbench export-review --help
.venv/bin/python -m llm_modelbench coverage --help
.venv/bin/python -m llm_modelbench gaps --help
.venv/bin/python -m llm_modelbench dossier --help
.venv/bin/python -m llm_modelbench repeat-report --help
.venv/bin/python -m llm_modelbench sensitivity-plan --help
.venv/bin/python -m llm_modelbench sensitivity-report --help
.venv/bin/python -m llm_modelbench diff --help
.venv/bin/python -m llm_modelbench simulate --help
.venv/bin/python -m llm_modelbench serve --help
.venv/bin/python -m llm_modelbench judge-dumps --help
.venv/bin/python -m llm_modelbench rankings --help
.venv/bin/python -m llm_modelbench wizard --help
```
