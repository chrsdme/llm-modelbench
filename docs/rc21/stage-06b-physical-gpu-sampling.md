# RC21 Stage 6B: Pure Physical GPU Sampling

## Scope

Stage 6B adds `llm_modelbench.telemetry` as a pure, read-only physical NVIDIA GPU sampling module. It does not redirect or modify `detect_gpu`, `detect_gpus`, `Telemetry`, `ProbeTelemetry`, `nvidia_live`, `live_snapshot`, runner, reports, status, doctor, watcher, context profiling, ranking, campaigns, or repair. Existing scalar collection behavior is unchanged. No scoring, ranking, report-schema, campaign-schema, or lifecycle behavior changed.

## Models and field states

The module provides frozen `TelemetryFieldState`, `PhysicalGPUIdentity`, `GPUSample`, `TelemetryCollectionError`, `GPUCollectionResult`, and `GPUInventoryJoinResult` structures.

`TelemetryFieldState` has `field`, `state`, and bounded `detail`. States are `observed`, `unsupported`, `unavailable`, `failed`, `malformed`, and `not_queried`. Every GPU metric has exactly one state: numeric zero is observed, every non-`None` value is observed, and every `None` value has a non-observed state. Duplicate field records, invalid states, non-finite numbers, negative physical metrics, and out-of-range percentages are rejected. Field states and serialized samples have deterministic ordering.

`PhysicalGPUIdentity` requires an NVIDIA UUID. PCI bus ID is corroborating evidence only; observed index is a non-boolean, non-negative sample-time locator and is not a CUDA ordinal. CUDA runtime-visible mapping is outside Stage 6B.

## Fixed query tiers and bounds

The extended tier is attempted first:

```text
index,uuid,pci.bus_id,name,memory.total,memory.used,memory.free,
utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,
clocks.current.graphics,clocks.current.memory,pstate
```

The one permitted fallback is the fixed baseline tier:

```text
index,uuid,pci.bus_id,name,memory.total,memory.used,memory.free,
utilization.gpu,temperature.gpu,power.draw
```

Commands are argument vectors using `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`; no shell is used. Extended non-zero exit or an extended fixed-schema failure permits one baseline attempt. Missing executable, timeout, stdout/stderr overflow, runner failure, and invalid CSV syntax do not retry. The collector does not guess a failed extended field from stderr. Baseline-only fields not queried after fallback remain explicitly `not_queried`.

Named bounds are `NVIDIA_SMI_TIMEOUT_SECONDS = 5.0`, `MAX_NVIDIA_SMI_STDOUT_BYTES = 1 MiB`, `MAX_NVIDIA_SMI_STDERR_BYTES = 64 KiB`, `MAX_TELEMETRY_ERRORS = 64`, and `MAX_TELEMETRY_DETAIL_CHARS = 512`. The production runner drains both pipes into bounded buffers; injected runners make all collection paths deterministic in tests.

## Parsing and identity policy

The parser uses Python CSV parsing and supports quoted names containing commas. It accepts arbitrary-N rows, sorts accepted samples by UUID, and preserves each row's observed index only as evidence. Every non-empty row must have exactly the selected fixed-tier column count: missing or extra columns invalidate that parser result and produce no samples. CSV syntax or fixed-schema failure means `successful_tier=None`; otherwise `successful_tier` means that the command (when collected) and the selected fixed CSV schema succeeded. A valid zero-row response remains a successful tier with structured no-GPU/unavailable evidence. Malformed optional metric values remain row-level evidence and do not by themselves invalidate a schema-valid tier.

Live UUIDs are pre-scanned before samples are constructed. Every row in a duplicate UUID group is excluded (never first-row-wins), with one deterministic bounded `csv_uuid` error per duplicate UUID; valid unrelated UUID rows still survive. Missing and marker UUIDs are invalid. `N/A`/empty values map to `unavailable`; `[Not Supported]` and `Not Supported` map to `unsupported`; `[Insufficient Permissions]` and `Insufficient Permissions` map to `unavailable`; unparseable numeric values map to `malformed`. No malformed value is coerced to zero. For optional identity text such as PCI bus ID and name, `N/A`, `NA`, `[N/A]`, `unknown`, `[Not Supported]`, `Not Supported`, `[Insufficient Permissions]`, and `Insufficient Permissions` normalize to `None`; they never become physical identity.

`join_inventory_samples()` is pure and accepts `GPUDevice` inventory and parsed samples. It joins only exact UUIDs, copies driver version and compute capability from inventory, and compares PCI only for corroboration. It never joins by row position, observed index, name, or PCI alone. Duplicate inventory UUID groups are identified before the join map is created and are excluded completely: a matching live sample stays un-enriched, regardless of inventory order or duplicate metadata. Inventory-only UUIDs, sample-only UUIDs, PCI disagreement, and duplicate inventory UUIDs become bounded deterministic warnings.

`GPUCollectionResult` validates its internal evidence contract: attempted tiers are valid and unique; a successful tier was attempted; samples require a successful tier; and fallback is exactly `("extended", "baseline")`. This permits direct parser results for either tier and schema-valid zero-GPU success while rejecting contradictory manually constructed results.

## Validation, acceptance, and compatibility

Offline fixture validation is complete. Focused validation passed with `57 passed`; the full suite passed with `664 passed`. `python3 -m compileall -q llm_modelbench tests`, `./llmb selftest`, `python3 tools/release_check.py`, and `git diff --check` all passed. The fixtures cover immutable-model state rules, zero values, missing-state rejection, boolean indices, NaN/infinity, deterministic serialization, zero/one/many CSV rows, quoted names, exact schema rejection, duplicate live UUID group exclusion, identity markers, fixed fallback policy, bounded runner failures, command vectors, result invariants, and UUID-only inventory joins including ambiguous inventory groups.

Real-host acceptance passed at `/tmp/llmb-rc21-stage6b-live-20260731T162617`. The extended query tier succeeded without fallback (`attempted_tiers=("extended",)`, `fallback_used=false`) and returned exactly two physical GPU samples. Their UUID set and UUID-to-PCI mappings matched the expected RTX 5060 Ti and RTX 3060 hardware; deterministic UUID ordering passed. UUID-only inventory enrichment supplied driver version and compute capability, and both collection and inventory-join errors were empty. The collection left repository state unchanged.

Rollback is removal of `telemetry.py`, its focused tests, and this document. No existing collector or consumer needs reversal because none was redirected.

Stage 6B is complete and ready for commit. Stage 6 overall is not complete. Stage 6C subsequently implemented isolated bounded process discovery and process-to-GPU attribution; it did not redirect this collector or any consumer.
