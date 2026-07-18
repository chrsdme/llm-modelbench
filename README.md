# LLM ModelBench

LLM ModelBench is a local benchmark for installed Ollama models. It measures
task correctness by lane, preserves raw evidence, and provides read-only tools
for reviewing coverage, reliability, routing, and pruning decisions.

## Naming

The installed package, import, and CLI are `llm-modelbench` / `llm_modelbench`;
the repository is [`chrsdme/llm-modelbench`](https://github.com/chrsdme/llm-modelbench).
The local wrappers are `llmb`, `llmb-run`, and `llmb-watch`; bare `llm` is
deliberately not used to avoid colliding with an unrelated, widely installed
tool of that name.

## Versioning

This is a release candidate: `1.0.0rc19.post3`. Public semantic versioning starts at
`1.0.0`: after the release candidate, `1.0.1` denotes a patch release, `1.1.0`
and `1.2.0` denote minor releases, and `2.0.0` denotes a major release.

## Security warning

Coding and file-operation scorers may execute model-generated code. The built-in guards, temporary directories, timeouts, environment scrubbing, and Linux resource limits reduce accidental damage but are not a complete security boundary. Run untrusted models inside a container, VM, or disposable host. See [docs/SAFETY.md](docs/SAFETY.md).

## Install and verify

From the public repository:

```bash
git clone https://github.com/chrsdme/llm-modelbench.git llm-modelbench
cd llm-modelbench
./install.sh
./llmb selftest
```

The offline selftest requires no Ollama server or GPU. The project uses the
local `.venv`; the direct equivalent is:

```bash
.venv/bin/python -m llm_modelbench selftest
```

Removing the tool later:

```bash
./uninstall.sh
```

`uninstall.sh` removes the virtualenv and build artefacts only. It never
touches `runs/`, so past benchmark results are always preserved.

## Safe workflow

Start by inspecting what would run. `inventory` and `plan` read the local
Ollama inventory; `--mock` exercises the pipeline without Ollama.

```bash
./llmb inventory
./llmb plan --mock
./llmb plan --models 'qwen2.5-coder:14b;qwen2.5-vl:7b' --auto --level short
./llmb-run --mock --level short --allow-host-code-execution --run-id demo --yes
./llmb report --run-id demo
./llmb export-review runs/demo --out demo_review.zip
```

Only run real models after explicit operator approval. A normal staged flow is
inventory, plan, a smoke pass, a repeatable short pass for survivors, then
focused full/context checks where needed. See [benchmark levels](docs/BENCHMARK_LEVELS.md).


## Validation and diagnostics

The current release provides deterministic repair-watcher replay, a dedicated repair/context-profile
dashboard, controlled long-context telemetry and behavior profiles, standalone
operating model cards, and a verifiable pre-Rankings-V3 regression freeze. These tools are offline unless `context-profile` is invoked.

Replay the repair watcher without Ollama or GPU work:

```bash
./llmb simulate repair-watch --scenario capability-repair --speed 1.0
./llmb simulate repair-watch --scenario needle-current --speed 1.0
```

To validate a standalone watcher in a second terminal, start the watcher first:

```bash
./llmb-watch --runs-dir runs --follow-queue --layout repair --refresh 0.5 --idle-grace 30
```

Then emit deterministic campaign files from another terminal:

```bash
./llmb simulate repair-watch --scenario kv-cascade --speed 1.0 \
  --write-only --run-id rc12_watch_fixture
```

Run one explicitly approved 64k operating profile using current/default KV:

```bash
./llmb context-profile \
  --model 'deepseek-coder-v2:16b' \
  --run-id rc12_deepseek_coder_v2_16b_64k_profile \
  --runs-dir runs --rankings-out rankings --cards-out model_cards \
  --target-ctx 64000 --gpu-vram-gb 15.93 \
  --emergency-headroom-gb 0.25 --max-spill-gb 2.5 \
  --min-tps 10 --critical-tps 3 --live-ui compact --yes
```

Generate cards and freeze the validated pre-V3 state:

```bash
./llmb model-cards --rankings-dir rankings --runs-dir runs --out model_cards
./llmb freeze --repo-root . --runs-dir runs --rankings-dir rankings \
  --out snapshots/rc12-pre-rv3 --label rc12-pre-rv3
./llmb freeze --out snapshots/rc12-pre-rv3 --verify
```

Context-profile output is diagnostic evidence. It uses advisory memory preflight plus
a live available-RAM safety floor, refuses reuse of a non-empty run ID, and can
enrich long-context operating profiles without replacing canonical quality rows.
Its 64k behavior probe does not certify agentic readiness.

## Benchmark lanes

- **Text and coding:** Python, web, JavaScript, text operations, JSON, git,
  file operations, and technical writing.
- **Agentic:** deterministic JSON tool-action decisions plus native structured
  Ollama tool-call validation; proposed tools are never executed.
- **FIM/insert:** suffix-conditioned completion through Ollama's generate API.
- **Vision, OCR, and PDF:** capability-routed VLM tasks, including small local
  image fixtures where a task declares `image_path`.
- **Retrieval and embeddings:** recall@1 and MRR over labelled synthetic
  documents, with per-case rank diagnostics for instrumented runs.
- **Long context:** needle probes with explicit coverage and safety gates.

Task and fixture rules are in [docs/TASKS.md](docs/TASKS.md); scorer contracts
are in [docs/SCORING.md](docs/SCORING.md).

## Run artifacts

Each run lives under `runs/<run-id>/`. Important artifacts include:

- `raw_results.jsonl`: append-only task rows and model evidence.
- `summary.json` and `summary_meta.json`: aggregate results and run metadata.
- `model_identities.json`: digest-backed model identities when available.
- `scorecard.md` / `.csv`, `report.html`, `routing.md`, `prune.md`, and
  `clones.md`: generated reports.
- `retrieval_diagnostics.json` / `.md`: case-level retrieval ranks when the
  run persisted them; old rows state explicitly when details are unavailable.
- `export-review` packs the relevant reports, raw rows, identities, and
  retrieval diagnostics for review.

Artifacts can contain model output. Treat review packs and raw rows as data
that must be inspected before sharing.

## Selection, capability routing, and approval

`--models 'a;b'` selects exact installed names, `--all` explicitly selects the
whole Ollama inventory, and `--select` opens a model-only selector. Test scope
still comes from `--level`, `--categories`, and `--tasks`. `llmb wizard` edits
both models and test scope. `--auto` adds small functional capability probes
before routing, including a vision probe for conservative VLM-name candidates even when Ollama reports incomplete `completion`-only metadata. Narrow explicit profiles cover known fleet conversions with this defect. There is no redundant `-all` alias.

Every run prints its exact plan. In a real terminal it asks for confirmation.
In a non-interactive shell it exits before any benchmark task call unless `--yes` is
present; `--yes` approves only the printed plan and does not broaden it. An
explicit `--auto` may already have made the small capability-probe calls needed
to construct that plan.

See [capability routing](docs/CAPABILITY_ROUTING.md).

## Post-hoc judging and persistent rankings

```bash
./llmb judge-dumps --run-id demo --judge single --judge-model qwen2.5:14b
./llmb judge-dumps --everything --runs-dir runs --judge single --judge-model qwen2.5:14b
./llmb rankings --runs-dir runs --out rankings --rescan
./llmb rankings --runs-dir runs --out rankings --watch --interval 5
```

Post-hoc judging calls only the judge model and keeps source raw rows immutable.
Rankings preserve all historical attempts in each model card while selecting
one current comparable row per task. See [judge dumps](docs/JUDGE_DUMPS.md) and
[rankings](docs/RANKINGS.md).

## Operator tools

All of the following operate on artifacts unless stated otherwise:

```bash
./llmb coverage update --ledger runs/coverage_ledger.json --run-dir runs/demo
./llmb gaps --ledger runs/coverage_ledger.json --mock --json
./llmb dossier --ledger runs/coverage_ledger.json --runs-dir runs --json
./llmb repeat-report runs/a runs/b
./llmb diff --a runs/a --b runs/b --noise-band 2.0
./llmb sensitivity-plan
./llmb sensitivity-report runs/a runs/b
./llmb simulate --run-dir runs/a --simulate-vram 24
./llmb serve --runs-dir runs/a --port 8080
```

`simulate`, `serve`, coverage, gaps, dossier, report rebuilding, diff,
repeat-report, sensitivity-report, and export-review do not run models. The
full command reference is in [docs/USAGE.md](docs/USAGE.md).

## Reliability interpretation

Scores are not precision claims. `repeat-report` labels cells as stable,
moving, reason-moving, insufficient-repeats, or missing. Missing data is never
zero; a single observed row is insufficient repeat evidence, not stability.

When repeated comparable rows exist, the empirical noise band is the maximum
observed repeat score range. `diff --noise-band N` labels deltas at or below
`N` as `tied/noise-band`; it does not change scores or rankings. Configuration
sensitivity is a separate diagnostic and is reported by the sensitivity tools.

## Safety and privacy

Model execution may evaluate generated code. Review [docs/SAFETY.md](docs/SAFETY.md)
before a real run. Public fixtures must be synthetic; see
[docs/PRIVACY_FIXTURES.md](docs/PRIVACY_FIXTURES.md).

## Repair incomplete evidence

Plan bounded, targeted recovery without changing source evidence:

```bash
llmb repair --run-id RUN_ID
llmb repair --run-prefix overnight_v2 --dry-run
llmb repair --everything --dry-run
llmb repair --run-prefix overnight_v2 --apply --yes
llmb repair --run-prefix overnight_v2 --apply --yes --gpu-vram-gb 15.93 --kv-cascade --restart-ollama
```

See [docs/REPAIR.md](docs/REPAIR.md) for thinking-only recovery, capability gates, post-hoc judging, needle/KV handling, provenance, and q8-to-q4 operational steps.

## Documentation

- [docs/USAGE.md](docs/USAGE.md) — full command reference
- [docs/CAPABILITY_ROUTING.md](docs/CAPABILITY_ROUTING.md) — selectors, probes, and routing
- [docs/JUDGE_DUMPS.md](docs/JUDGE_DUMPS.md) — retroactive automated judging
- [docs/RANKINGS.md](docs/RANKINGS.md) — formulas, status, ties, and model history
- [docs/README.md](docs/README.md) — current documentation index
- [docs/history/](docs/history/) — archived release audits and application notes
- [docs/TASKS.md](docs/TASKS.md) — task and fixture rules
- [docs/SCORING.md](docs/SCORING.md) — scorer contracts
- [docs/FEATURES.md](docs/FEATURES.md) — feature summary
- [docs/BENCHMARK_LEVELS.md](docs/BENCHMARK_LEVELS.md) — levels and interpretation
- [docs/SAFETY.md](docs/SAFETY.md) — safety boundaries
- [docs/PRIVACY_FIXTURES.md](docs/PRIVACY_FIXTURES.md) — fixture content policy
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — contribution rules
- [docs/PUBLISHING.md](docs/PUBLISHING.md) — release process
- [CHANGELOG.md](CHANGELOG.md) — release notes

## License

MIT. See [LICENSE](LICENSE).
