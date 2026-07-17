# LLM ModelBench 1.0.0rc5 repair correction audit

Date: 2026-07-15

## Purpose

This release corrects the gaps found during independent review of the RC4 repair package. The audit was performed against the packaged RC4 source, the supplied overnight run archive, and the actual command-line interface. No real Ollama model inference was used while building or validating RC5.

## RC4 findings reproduced

The following review findings were confirmed:

1. The packaged RC4 test suite produced **274 passed**, not the previously claimed 280.
2. `llmb repair --dry-run` was documented but not accepted by the RC4 CLI.
3. `--gpu-vram-gb` was documented but absent from the RC4 CLI.
4. Without detected GPU capacity, RC4 needle planning fell through to a generic non-repairable observation.
5. RC4 prose over-interpreted measured VRAM slope as evidence of an effective KV precision. The same slope can include activations, compute workspaces, allocator effects and offload.
6. The ranking-selection fix, embedding-only routing fix and HTTP error-body capture were present and retained.

## RC5 corrections

### CLI and planning

The following are real, parser-tested options:

```text
--dry-run
--gpu-vram-gb
--confirm-kv-server
--force
```

`--apply` and `--dry-run` are mutually exclusive. Planning remains the default when neither is supplied.

Inputs that could weaken safety are validated:

- GPU VRAM override must be greater than zero;
- emergency headroom cannot be negative;
- permitted spill cannot be negative;
- recovery output budget must be greater than zero.

### Needle classification

Needle gaps now surface explicit evidence classes:

- `MODEL_OR_OPERATOR_LIMIT`
- `GPU_CAPACITY_UNKNOWN`
- `ESTIMATE_UNAVAILABLE`
- `MARGINAL_SOFT_LIMIT`
- `GUARDED_RETRY_AVAILABLE`
- `HARD_VRAM_OR_SPILL_LIMIT`
- `KV_SERVER_MISMATCH`

The rendered text plan includes each actionable depth, estimate, old budget, soft-budget overage, physical GPU value, emergency headroom, spill allowance and guarded total.

### KV evidence

RC5 does not infer effective KV precision from VRAM slope. It reports estimator divergence and the explicit caveat that total-memory slope can include non-KV components.

The planner performs best-effort inspection of:

- `/proc/<ollama-pid>/environ` for a running `ollama serve` process;
- the configured systemd unit environment;
- the repair shell environment.

Only `OLLAMA_KV_CACHE_TYPE` is retained. Complete process or service environments are never persisted, preventing unrelated credentials from entering repair plans.

Explicit q8/q4 application requires:

1. matching `OLLAMA_KV_CACHE_TYPE` in the repair shell; and
2. a matching verified live process environment, or explicit operator confirmation after service restart.

The command does not edit or restart Ollama. q8 and q4 are separate operational passes because Ollama KV quantisation is server-global.

### Repair evidence

- Source `raw_results.jsonl` remains immutable.
- Generation retries create child runs.
- Repair actions are appended to `repair_results.jsonl`.
- Judge repairs now record and count every planned task action, even when one judge batch handles several tasks in a source run.
- Capability failures remain recorded in `capability_repair.json`.

## Real overnight dry-run verification

Command:

```bash
python -m llm_modelbench repair \
  --run-prefix overnight_v2_20260715 \
  --runs-dir <extracted-runs> \
  --dry-run \
  --gpu-vram-gb 15.93 \
  --emergency-headroom-gb 0.25 \
  --max-spill-gb 2 \
  --kv-type q8_0
```

Observed plan:

```text
70 runs selected
67 automatic actions
51 observations/manual items
54 bounded generation retries
4 capability gates
9 guarded needle retries
49 obsolete historical misrouting observations
2 model/operator-limited needle observations
```

The `gpt-oss:20b` 65k depth is rendered as:

```text
MARGINAL_SOFT_LIMIT
estimated total: 14.487 GB
old soft budget: 14.400 GB
overage: 0.087 GB
guarded total: 17.680 GB
```

This is no longer labelled a hard hardware ceiling.

The dry-run does not prove q8 will succeed. It proves that the gap is eligible for a guarded attempt once the live Ollama service setting is verified.

## Validation

Validated from the RC5 working tree:

```text
compileall: passed
pytest: 284 passed
version: llm-modelbench 1.0.0rc5
selftest: SELFTEST: ALL GOOD
```

A full mock CLI application was also executed:

- one thinking-only task planned;
- one repair child run created;
- repaired task scored successfully;
- source raw file remained byte-for-byte unchanged;
- provenance and rankings refresh were written;
- final repair outcome was `COMPLETE`.

## Unverified on this audit host

The audit host has no live Ollama/GPU service. Therefore RC5 does **not** claim that:

- the user's running Ollama service currently uses q8 or q4;
- any specific missing needle depth will fit;
- CPU spill has a fixed performance penalty;
- the three community VLM builds can serve image requests after a reload.

Those questions require the real host and are intentionally left to guarded repair execution with recorded evidence.
