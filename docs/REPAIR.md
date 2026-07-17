# Post-run repair

`llmb repair` scans existing run evidence, classifies unresolved current results, and builds a bounded recovery plan. Planning is the default and makes no model calls. Source `raw_results.jsonl` files are never overwritten.

## Plan only

```bash
llmb repair --run-id RUN_ID --runs-dir runs --dry-run
llmb repair --run-prefix overnight_v2_20260715 --runs-dir runs --dry-run
llmb repair --everything --runs-dir runs --dry-run
```

`--dry-run` is explicit but optional; omitting both `--dry-run` and `--apply` also plans only.

For reproducible offline needle classification, provide the physical GPU capacity:

```bash
llmb repair \
  --run-prefix overnight_v2_20260715 \
  --runs-dir runs \
  --dry-run \
  --gpu-vram-gb 15.93 \
  --emergency-headroom-gb 0.25 \
  --max-spill-gb 2 \
  --kv-type q8_0
```

The rendered plan shows the physical GPU value, reserved headroom, permitted spill, guarded total, requested KV type, observed server setting when inspectable, and each needle-depth classification.

## Apply a reviewed plan

```bash
llmb repair \
  --run-prefix overnight_v2_20260715 \
  --runs-dir runs \
  --apply \
  --yes
```

Add post-hoc judging without rerunning the source model:

```bash
llmb repair \
  --run-prefix overnight_v2_20260715 \
  --runs-dir runs \
  --judge single \
  --judge-model qwen2.5:14b \
  --apply \
  --yes
```

Use `--force` only to repeat an action already recorded in `repair_results.jsonl`.

## Recovery policy

- `thinking_only` or empty text output: retry with `think=off` at the original output budget, then once at the bounded recovery budget (default 4096).
- HTTP 5xx or timeout: unload and retry once.
- Repeated vision/tool/FIM failures: run one functional capability gate before repeating that lane.
- Subjective output awaiting grading: write judge-sidecar evidence; do not rerun the source model.
- Missing or stale current task: rerun only that task.
- Historical tasks no longer applicable after corrected capability routing: record as obsolete misrouting; do not retry.
- Needle gaps: classify model/operator limits, unknown GPU capacity, marginal soft-budget skips, guarded retries, and hard GPU-plus-spill limits.

## Needle memory policy

The original run may have used a conservative soft VRAM budget. Repair planning instead separates:

- physical GPU capacity;
- emergency GPU headroom;
- permitted system-RAM spill.

Example for a 15.93 GB card:

```text
physical GPU             15.93 GB
emergency headroom        0.25 GB
permitted spill           2.00 GB
guarded total allowance  17.68 GB
```

`--max-spill-gb` only permits the harness to attempt the missing depth. Ollama decides actual layer/KV placement. The repair child run records real wall time, throughput, telemetry and offload evidence. The tool does not claim a fixed slowdown multiplier.

A small overage such as 14.487 GB versus an old 14.400 GB soft budget is labelled `MARGINAL_SOFT_LIMIT`, not a hard hardware ceiling.

## KV-cache policy

Ollama KV-cache quantisation is global to the running Ollama server. There are two supported workflows.

### Human-supervised automatic cascade

The recommended workflow lets `llmb repair` manage a **temporary, dedicated systemd drop-in** and perform the q8-to-q4 sequence itself:

```bash
llmb repair \
  --run-prefix overnight_v2_20260715 \
  --runs-dir runs \
  --apply \
  --yes \
  --gpu-vram-gb 15.93 \
  --emergency-headroom-gb 0.25 \
  --max-spill-gb 2 \
  --kv-cascade \
  --restart-ollama
```

The command performs this sequence:

1. require the operator to type `DISCOVER`, authenticate through normal `sudo`, identify the PID listening on the configured Ollama port, and match that PID to a systemd unit `MainPID`;
2. reject a manually supplied `--ollama-service` when it does not own the live endpoint;
3. validate any UUID-based `CUDA_VISIBLE_DEVICES` binding and block a restart when the UUID no longer exists;
4. run non-needle repairs under the original Ollama configuration;
5. print the exact privileged q8 phase and require the operator to type `RESTART`;
6. install `/etc/systemd/system/<active-unit>.d/zzzz-llmb-repair-kv.conf` with `q8_0`;
7. reload systemd and verify the **merged effective environment before restart**;
8. restart the active unit, verify it still owns the configured port, and verify the live process environment;
9. run only guarded needle repairs;
10. when some remain unresolved, request a second `RESTART` confirmation and repeat under `q4_0` for those actions only;
11. request a final `RESTART` confirmation, restore the original managed drop-in state, and restart Ollama;
12. refresh rankings and write repair/service audit artifacts.

`--yes` approves the benchmark repair plan. It does **not** bypass privileged service confirmations. Each systemd phase still requires typing `RESTART`. By default the controller runs `sudo -k` followed by `sudo -v`, so sudo genuinely requests the password rather than silently using an unrelated cached timestamp. Use `--reuse-sudo-credentials` only when that behaviour is intentionally unwanted.

The password is never read, piped, logged, cached, or serialised by LLM ModelBench. It is handled only by the normal `sudo` executable.

The cascade requires a real interactive terminal. It refuses to run through a non-TTY pipe, cron job, or unattended automation. It also refuses `--mock` together with service restarts. A pseudo-terminal recorder such as `script` is supported; a plain `| tee` pipeline is not a TTY.

The managed drop-in is narrow and temporary. It uses a late-sorting filename, but systemd's merged `Environment` output remains authoritative: if another unit file or drop-in still wins, RC7 aborts before restarting anything and identifies competing drop-ins when possible. If the managed path already exists without the LLM ModelBench marker, the command refuses to overwrite it. On an exception after a service change, it attempts to restore the original state before propagating the error.

Useful options:

```text
--ollama-service NAME       explicit systemd unit; default auto discovers the live port owner
--keep-final-kv             deliberately leave the final q8/q4 setting active
--reuse-sudo-credentials    do not invalidate sudo's cached timestamp per phase
```

Leaving the final setting active is not the default. Normal behaviour is to restore the pre-existing service state.

Service operations are written incrementally to:

```text
runs/repair_service_<plan-id>.jsonl
```

This audit survives even when a later model call or service phase fails.

### Manual q8 or q4 pass

Manual operation remains available for systems that do not use systemd or where the operator does not want the benchmark to touch the service. Configure and restart Ollama yourself, export the same value in the repair shell, and use `--kv-type` plus `--confirm-kv-server`.

```bash
export OLLAMA_KV_CACHE_TYPE=q8_0
llmb repair \
  --run-prefix overnight_v2_20260715 \
  --runs-dir runs \
  --apply \
  --gpu-vram-gb 15.93 \
  --emergency-headroom-gb 0.25 \
  --max-spill-gb 2 \
  --kv-type q8_0 \
  --confirm-kv-server \
  --yes
```

The shell value alone is insufficient. Manual application is blocked unless the live process environment matches, or the operator explicitly confirms the already-completed restart.

### Interpretation boundary

Measured VRAM slope is total-memory behaviour. It may include KV cache, activations, compute workspaces, allocator effects and offload. A slope larger than the simple KV formula is evidence that the estimator is incomplete; it is not proof that Ollama ignored q8/q4. The supervised workflow verifies the actual live process environment instead of inferring cache precision from memory arithmetic.

## Inspecting the live setting

The repair planner performs best-effort inspection of `/proc/<ollama-pid>/environ` and the systemd unit. It stores only `OLLAMA_KV_CACHE_TYPE`, never the complete service environment.

Manual checks on the host:

```bash
systemctl show ollama.service --property=Environment --value | tr ' ' '\n' | grep '^OLLAMA_KV_CACHE_TYPE='
pid=$(systemctl show ollama.service --property=MainPID --value)
sudo sh -c 'tr "\000" "\n" < "/proc/'"$pid"'/environ" | grep "^OLLAMA_KV_CACHE_TYPE="'
```

## Evidence and outcomes

- Generation repairs create new `repair_*` child runs.
- Source runs receive append-only `repair_results.jsonl` records.
- Repaired rows include parent run, source-row hash, policy version, action ID, attempt number and overrides.
- Capability-gate failures are recorded in `capability_repair.json`.
- Judgements remain in `judge_results.jsonl` sidecars.
- Final repair outcome is `COMPLETE`, `PARTIAL`, `FAILED` or `TIMEOUT`.

## RC9 current-first KV policy

`--kv-cascade --restart-ollama` no longer means that q8 is tried first. Guarded
needle actions run under the current/default service configuration before any
sudo preflight or systemd discovery. Only unresolved actions enter the bounded
fallback:

```text
current/default KV
  -> unresolved-only q8_0
  -> unresolved-only q4_0, unless q8 proved a shared Flash-Attention prerequisite unavailable
  -> restore original service state if a mutation occurred
```

Build/runtime-scoped compatibility is stored in the source run's
`kv_compatibility.json`. It includes model digest and relevant runtime identity,
so a later Ollama/driver/model change can invalidate old incompatibility evidence.

`--auto-confirm` remains opt-in. It suppresses typed confirmation and uses only
`sudo -n`, but the sudo preflight is now lazy and is not run when current KV
recovers all guarded work.

## Long-context telemetry

Needle probe rows include per-depth wall/server timing, prompt/decode throughput,
GPU telemetry, system RAM/swap, aggregate Ollama process RSS/PSS/swap, and
`/api/ps` model residency. `model_host_bytes` is the model-weight portion outside
VRAM reported by Ollama; it is not an exact KV-offload byte count. Process PSS is
the preferred host-memory signal when available.

The default 64k operating profile treats decode below 10 tok/s as slow and below
3 tok/s as generally impractical. These are configurable diagnostics and do not
alter the needle quality score.
