# RC21 Stage 6A: Backend-Neutral Telemetry Audit and Plan

## Status and boundary

This is the Stage 6A audit and implementation plan. Its conclusions remain the boundary for later work. Stages 6B and 6C subsequently implemented only isolated physical-GPU and process-attribution evidence layers; neither integrated telemetry into existing collectors or consumers. Stage 7 has not started, and this document does not claim fresh hardware evidence. Collection in later Stage 6 work must be read-only, bounded, timeout-controlled, and diagnostic unless an existing explicit preflight already declares a required measurement.

The target AI-PC has two physical NVIDIA devices: RTX 5060 Ti (`GPU-78077308-f4b2-3330-6d4e-19581d7b1511`, `00000000:01:00.0`) and RTX 3060 (`GPU-5b99bce2-35ab-f6db-857b-72162069fa72`, `00000000:05:00.0`). These are acceptance expectations, not constants to encode. Ollama and the external llama-server remain independent, operator-managed local runtimes.

## Current-state map

### Producers

| Producer | Collection and cadence | Output / identity | Failure and mutation boundary | Consumers |
| --- | --- | --- | --- | --- |
| `hardware.detect_gpus()` / `_parse_nvidia_gpu_inventory()` | `nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version,compute_cap`; one inventory query per caller, legacy retry without `compute_cap` | Ordered `GPUDevice` list: physical index, UUID, PCI, name, total MiB, driver, capability | Missing executable/failed command yields `[]`; malformed rows omitted; read-only | `doctor.collect`, runtime recommendation in `runtime_profiles.discover_runtimes`, tests |
| `hardware.detect_gpu()` | Compatibility projection of `detect_gpus()[0]`; AMD/Apple fallbacks | Scalar `GPUInfo` with no UUID/PCI | No-device scalar fallback; read-only | Config budget, `runner.run`, `context_profile`, `repair`, doctor, status |
| `hardware.Telemetry` | Background thread at 0.1 seconds while each normal task runs; `nvidia-smi --query-gpu=memory.used,power.draw,temperature.gpu` | First CSV row only: peak VRAM, mean power, peak temperature; no device identity | Sensor failures are ignored and output `None`; read-only | Normal runner row fields and temperature pause |
| `hardware.nvidia_live()` / `live_snapshot()` | On demand; `nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw`; first row only | Scalar GPU name/temp/util/VRAM/power plus RAM/swap/CPU snapshots | `{}` on missing/parse failure; read-only | `ProbeTelemetry`, runner VRAM fallback, doctor, watch, inline UI, interactive navigation |
| `hardware.ProbeTelemetry` | Background thread at 0.25 seconds during needle/context probes, baseline plus stop sample | Scalar GPU sample, system RAM/swap, CPU, and aggregate local runtime RSS/PSS/swap | Individual sampling errors are suppressed; unavailable fields remain `None`; read-only | Needle rows, context-profile validation, model cards, rankings display/selection tie-breaker, watch |
| `hardware._read_ollama_process_memory()` | `/proc/<pid>/{comm,cmdline,status,smaps_rollup}` scan per probe sample | Aggregate process count/RSS/PSS/swap for commands matching `ollama`, `llama_server`, or `llama-server`; legacy keys are all `ollama_*` | Permission/missing `/proc` degrades silently; command line is inspected but not emitted; read-only | `ProbeTelemetry` and downstream long-context evidence |
| `hardware._read_proc_meminfo`, `_read_proc_stat`, `_cpu_temp_c` | `/proc` and sysfs per live/probe sample | RAM/swap, CPU utilisation, CPU temperature | Best-effort `{}`/`None`; read-only | `ProbeTelemetry`, watcher, doctor |
| `OllamaClient.loaded_model_stats()` / `offload_fraction()` | `/api/tags` then `/api/ps`, currently during model/probe paths | Exact loaded model size, `size_vram`, host bytes, context, expiry, derived offload fraction | Exceptions return `None`; read-only HTTP | Runner context probes and rows, model-card/ranking operating evidence |
| `LlamaCppClient.loaded_model_stats()`, `slots()`, `metrics()` | Read-only `/v1/models`, `/props`, `/slots`, optional `/metrics` | Served model ID/size/context and slot state; metrics text bounded but not used per request | Shape/endpoint errors are explicit; no lifecycle HTTP | Runner loaded-model evidence where capability is supported; no per-request metrics attribution |
| `OllamaClient.chat()` and `LlamaCppClient.chat()` | Per benchmark request | Normalized token counts, throughput/durations, finish state; backend-specific server timing if reported | Existing task error handling; generation is outside Stage 6A | Runner rows, reports, scoring diagnostics |
| `doctor.collect()` | One interactive command invocation | Scalar `live_snapshot`, physical `gpus` inventory, Ollama `/api/version`, `/api/tags`, `/api/ps` | Endpoint failures represented as error strings; read-only | Doctor rendering only |
| `runtime_profiles._process_profiles()` | Read-only `/proc/*/cmdline` during bounded discovery | Candidate backend/localhost endpoint based on command and `--port`/`-p` | Missing `/proc` gives no candidates; command line not persisted | Runtime discovery and selection only |

No `pynvml` dependency exists. The repository currently uses only standard-library subprocess and `/proc` reads. Stage 6 should retain that policy unless a later implementation proves `nvidia-smi` cannot provide a required, testable evidence field.

### Consumers and semantic classification

| Consumer | Existing fields / source | Classification | Stage 6 rule |
| --- | --- | --- | --- |
| `runner.run` raw rows (`raw_results.jsonl`) | `vram_peak_mb`, `power_mean_w`, `temp_peak_c`, needle probe scalar telemetry, normalized request timings, loaded-model/offload data | Diagnostic; long-context preflight/validity has explicit existing required scalar fields | Add new optional evidence only; do not change row score, task routing, or existing validity requirements in 6A-6D |
| `progress.StatusWriter` (`status.json`) | Scalar `hardware_config`, current offload and probe telemetry | Display/live UI; status schema-affecting | Preserve `hardware_config` unchanged; later add optional telemetry snapshot reference/summary only |
| `watch.py`, `inline_ui.py`, `interactive_nav.py` | `live_snapshot()` overlays and scalar status fields | Display only | Render list evidence additively in 6F; retain scalar display fallback |
| `doctor.py` | Inventory list plus scalar live telemetry and Ollama endpoint facts | Display/diagnostic | Add read-only per-GPU presentation only after collectors exist; never make doctor mutate or select a runtime |
| `context_profile.py` | Scalar peak VRAM, host memory and offload values in required/optional telemetry checks | Benchmark validity for its existing explicit long-context lane; diagnostic otherwise | Do not reinterpret criteria in Stage 6; additive per-GPU evidence must not make an old probe invalid |
| `model_cards.py` | Context-probe VRAM/host/offload evidence | Display/diagnostic | Preserve legacy scalar card fields; later render per-GPU evidence as supplementary |
| `report.py` | Status `hardware_config`, scalar GPU metadata, request timing rows | Display/report schema | Old reports remain readable; no duplicate-key or aggregate change in Stage 6 |
| `rankings.py`, `aggregate.py`, `rankings_v3.py`, `rankings_v31.py` | Offload, VRAM and long-context telemetry copied to operating profiles; `telemetry_completeness` is a tie-breaker among otherwise selected needle evidence | Ranking-adjacent display/evidence selection, not score formula | Do not feed new telemetry into ranking selection or change current completeness semantics before Stage 8 review |
| `fingerprint.py`, model identities, campaign package | Model digest/metadata only | Identity/campaign-schema-affecting when persisted | Runtime/GPU identity freeze is Stage 8; Stage 6 must not alter model fingerprints or package requirements |
| `planner.py`, config and `repair.py` | Scalar VRAM budget and `detect_gpu`; Ollama KV/service inspection | Planning/lifecycle-affecting | No replacement of scalar planning assumptions in Stage 6; repair stays Ollama-specific |
| `campaign.py` manifests, summaries, effective rows | Existing raw/status/report packaging | Campaign-schema-affecting | Do not add required fields or migrations in Stage 6; defer campaign freeze to Stage 8 |

## Identity policy

The Stage 6 canonical physical GPU key is the NVIDIA UUID when the driver reports one. PCI bus ID is retained as corroborating physical evidence. `physical_index` is a sample-time locator and ordering aid only. GPU name, driver and compute capability are descriptive, never identity keys.

CUDA ordinal is runtime-local. It must be represented separately as an optional observed/configured runtime-visible ordinal with its source, and must never be equated with a `GPUDevice.physical_index`. `RuntimeProfile.physical_gpu_uuids` remains declarative profile metadata: it records an operator-selected intent, not proof of use. A missing UUID makes the physical sample non-joinable; it may be retained in a bounded unkeyed warning/sample list, but it must not be guessed from name or index. Duplicate UUID inventory responses are collector errors and must not merge samples.

Current first-device assumptions are in `Telemetry._query`, `nvidia_live`, `live_snapshot`, `_current_vram_used_mb`, `runner.run` (`detect_gpu` and per-task sampler), `ProbeTelemetry`, scalar `StatusWriter.hardware_config`, watcher/doctor scalar render paths, `Config.vram_budget_gb`, planner/repair/context-profile VRAM gates, report metadata, and model-card/ranking long-context evidence. No current source assumes exactly two devices; the pervasive issue is first-row scalar collapse.

## Process attribution policy

Stage 6 must identify candidate local runtime processes without retaining arbitrary command lines or environments:

1. Discover candidate PIDs from bounded `/proc/<pid>/comm`, a bounded `cmdline` read, executable symlink where readable, and a local listening-port ownership resolver. Use the selected profile's normalized localhost endpoint as the port target; do not scan ports or LAN hosts.
2. Match parent/child processes by PPID from `/proc/<pid>/stat` and retain only a bounded process tree rooted at a selected server candidate. Ollama may have a server and runner/model children; llama-server generally has one server PID.
3. Query `nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory --format=csv,noheader,nounits` once per snapshot where supported. Join PID to the selected process tree and GPU UUID to the physical inventory. One PID may produce records on multiple UUIDs; one UUID may contain multiple PIDs.
4. Capture process start-time ticks from `/proc/<pid>/stat` with PID so later samples can detect stale PID/reuse. Do not persist executable arguments, environment contents, or raw `/proc` text.

Attribution confidence is: **confirmed** when selected endpoint port ownership, PID/start time, and `nvidia-smi` process record join; **probable** when the selected runtime process identity and GPU process record join but port ownership is unavailable; **profile-declared only** for profile UUIDs with no observed matching process allocation; and **unavailable** when any necessary source is absent or ambiguous. A llama-server command such as `-dev CUDA0,CUDA1 -sm layer -ts 80,20` is configuration evidence only. Its CUDA ordinals and split values cannot establish physical UUID mapping without a separately verified mapping, and do not prove actual allocation, layer placement, KV placement, or offload fraction.

## NVIDIA query compatibility

Use CSV with `noheader,nounits`, explicit timeouts, and fixed query tiers. The implementation must not infer which individual field failed by scraping arbitrary stderr text.

The extended tier is attempted first:

```text
index,uuid,pci.bus_id,name,memory.total,memory.used,memory.free,
utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,
clocks.current.graphics,clocks.current.memory,pstate
```

If that invocation fails for any reason, the collector may make exactly one fixed baseline fallback attempt:

```text
index,uuid,pci.bus_id,name,memory.total,memory.used,memory.free,
utilization.gpu,temperature.gpu,power.draw
```

The collector therefore makes at most one successful collection and one bounded fallback attempt per snapshot. The baseline fields are the broadly available live sample contract. Memory-controller utilisation, power limit, clocks, and pstate are extended-only: after a successful baseline fallback they are `None` with an `unsupported`, `unavailable`, or `failed` field state as applicable. It must not parse a partial/truncated response as complete.

Stage 6B uses named, testable internal defaults: `NVIDIA_SMI_TIMEOUT_SECONDS = 5.0`, `MAX_NVIDIA_SMI_STDOUT_BYTES = 1024 * 1024`, `MAX_NVIDIA_SMI_STDERR_BYTES = 64 * 1024`, `MAX_TELEMETRY_ERRORS = 64`, and `MAX_TELEMETRY_DETAIL_CHARS = 512`. They are internally configurable, not public runtime configuration. Empty output, missing executable, timeout, non-zero exit, or bounded/truncated output produces collector warnings and non-observed field states. CSV parsing maps `N/A` to `unavailable`, `[Not Supported]` to `unsupported`, and `[Insufficient Permissions]` to `unavailable`; malformed numeric values become `malformed`. Every non-empty selected-tier row requires the exact field count: missing or extra columns invalidate the tier. Duplicate live UUID groups are excluded entirely, missing UUID rows remain unjoinable, and ambiguous duplicate inventory UUID groups are never used for enrichment. Reordered rows are joined only by UUID. No row position or transient index is a join fallback.

The process query above is also version/permission dependent. If `gpu_uuid` is unavailable, do not fall back to GPU index as a durable identity; retain process evidence with `gpu_uuid=None` and attribution unavailable. `nvidia-smi pmon` is not proposed as a primary source because it is less stable for PID/memory attribution; it may remain an explicitly documented last-resort diagnostic only if a later fixture-backed need emerges.

## Proposed Stage 6 data model

The implementation should use frozen dataclasses plus deterministic `to_dict()` methods in a new telemetry module. Numeric zero is an observed value, never missing. Every `None` metric must have exactly one corresponding non-`observed` field state; duplicate states for one field are invalid. Field-state records sort by field, and details are bounded. Snapshot warnings remain collector-level events and never substitute for per-field state. No structure stores raw command output.

```text
PhysicalGPUIdentity
  uuid: str | None                 # canonical key only when present
  pci_bus_id: str | None
  observed_index: int | None       # sample-time locator, never CUDA ordinal
  name: str | None
  driver_version: str | None
  compute_capability: str | None

TelemetryFieldState
  field: str
  state: str                       # observed | unsupported | unavailable | failed | malformed | not_queried
  detail: str | None                # bounded; one record per field

GPUSample
  identity: PhysicalGPUIdentity     # joined from existing GPUDevice inventory by UUID
  timestamp_utc: str
  source: str                      # e.g. nvidia-smi.query-gpu.v1
  total_vram_mb, used_vram_mb, free_vram_mb: float | None
  gpu_util_pct, memory_util_pct, temperature_c: float | None
  power_draw_w, power_limit_w: float | None
  graphics_clock_mhz, memory_clock_mhz: float | None
  pstate: str | None
  field_states: tuple[TelemetryFieldState, ...]

GPUProcessSample
  pid: int
  process_start_ticks: int | None
  process_name: str | None
  gpu_uuid: str | None
  used_gpu_memory_mb: float | None
  source: str
  field_states: tuple[TelemetryFieldState, ...]

RuntimeProcessIdentity
  backend: str
  endpoint: str
  pid: int | None
  process_start_ticks: int | None
  executable_basename: str | None
  role: str                         # server | runner | child | unknown
  profile_name: str | None

RuntimeGPUAttribution
  runtime_pid: int | None
  gpu_uuid: str | None
  process_sample: GPUProcessSample | None
  confidence: str                   # confirmed | probable | profile-declared only | unavailable
  evidence_sources: tuple[str, ...]
  detail: str | None                # bounded

RuntimeTelemetrySample
  runtime: RuntimeProcessIdentity
  declared_physical_gpu_uuids: tuple[str, ...]
  gpu_attributions: tuple[RuntimeGPUAttribution, ...]
  backend_metrics: dict[str, scalar] | None
  field_states: tuple[TelemetryFieldState, ...]
  warnings: tuple[TelemetryCollectionError, ...]

TelemetrySnapshot
  schema: str                       # new additive, versioned schema
  captured_at_utc: str
  collector_version: str
  gpus: tuple[GPUSample, ...]       # deterministic UUID/index ordering
  runtimes: tuple[RuntimeTelemetrySample, ...]
  host: dict[str, scalar | None]    # existing RAM/swap/CPU data, bounded
  host_field_states: tuple[TelemetryFieldState, ...]
  warnings: tuple[TelemetryCollectionError, ...]

TelemetryCollectionError
  source: str
  state: str                        # unsupported | unavailable | failed | malformed
  detail: str                       # bounded concise detail
```

`backend_metrics` is read-only backend evidence, distinct from request-normalized timing already returned by `chat`. It must not use global `/metrics` deltas as request attribution. Field states apply to every optional metric in `GPUSample`, `GPUProcessSample`, `RuntimeTelemetrySample`, and `TelemetrySnapshot.host`; identity absence is represented separately and is not converted into a numeric metric state. A runtime can have a distinct confidence record for every GPU. Profile-declared-only UUIDs have no process sample; observed-but-undeclared UUIDs are retained as observations; `RuntimeProcessIdentity` describes the process and never supplies per-GPU confidence. Serialization sorts runtime records by normalized backend/endpoint, GPU samples by UUID then observed index, and attribution records by runtime PID, GPU UUID, and process PID. Warnings are deduplicated and capped.

`PhysicalGPUIdentity.driver_version` and `compute_capability` come only from the existing `detect_gpus()`/`GPUDevice` inventory. Stage 6B collects live sample fields separately and joins inventory/sample rows by UUID. PCI bus ID corroborates a UUID match but is never an identity fallback; differing inventory and live UUID sets produce one bounded mismatch warning. Stage 6B must not duplicate or replace the existing arbitrary-N inventory collector.

## Collector boundaries and sampling lifecycle

Proposed modules and responsibilities:

- Stage 6B is restricted to a new `telemetry.py`, immutable sample/state structures, pure CSV parsing, bounded injected subprocess collection, deterministic serialization, fixture tests, and an optional additive `GPUDevice` inventory-to-sample UUID join helper. `hardware.py` should remain unchanged; if later reviewed as necessary, it may expose only an additive wrapper with no existing caller redirected to it.
- Process inspection, PID/GPU joins, backend/server evidence, runner integration, and rendering are not Stage 6B work. Later collectors own `/proc` process inspection, PID/GPU joins, and backend snapshot evidence. All OS/network seams are injectable for fixture tests.
- A backend-neutral adapter helper receives `BackendIdentity`, selected runtime-profile metadata, and only explicitly supported read-only backend methods (`metrics`, `loaded_model_stats`, `slots`). It never constructs another client and never calls lifecycle/model-management endpoints.
- Runner integration is delayed until 6E and receives a `TelemetrySnapshot`/warning result rather than collecting ad hoc fields. Report/status/doctor rendering is delayed until 6F.

Desired later sample points: one pre-run baseline; one pre-model snapshot; post-task only for normal task boundaries; pre/post each long-context probe because that lane already records memory evidence; post-model; post-run; and optional live UI polling at no faster than two seconds. The existing 0.1-second normal-task sampler is too frequent for an arbitrary-N process query. A 1.0-second bounded aggregate GPU sampler only during a task/probe is sufficient for peaks, while PID-to-GPU attribution should occur at baseline, model start, model end, and post-run unless a task is long enough to justify one 5-second refresh. No permanent watcher, daemon, or service is proposed.

Telemetry collection failures remain diagnostic and non-fatal. Existing explicit context-profile validity fields keep their current behavior until a separately reviewed integration decides otherwise. Temperature safety remains the existing scalar behavior until a future Stage 6 substage defines an explicit multi-GPU safety policy; it must not silently choose a device.

## Compatibility and schema plan

Stage 6 implementation should add optional `telemetry`/`telemetry_snapshot` evidence sections to new run artifacts or rows only after 6E, while retaining all scalar keys as compatibility projections. Old rows/reports continue to parse without the section. New telemetry must not alter `report._duplicate_key`, task scores, validity calculations, model fingerprint values, ranking formulas, candidate selection, campaign manifests, package checksums, or resume equivalence.

Likely implementation touchpoints are `hardware.py`, a new `telemetry.py`, `runner.py`, `context_profile.py`, `progress.py`, `watch.py`, `doctor.py`, `report.py`, `model_cards.py`, `inline_ui.py`, `interactive_nav.py`, `rankings.py`, and focused fixtures/tests. `backend.py` may gain only an additive read-only telemetry capability if an actual consumer requires it. `campaign.py`, `fingerprint.py`, campaign manifest schemas, canonical rankings, and report identity freeze are Stage 8 work, not Stage 6 work.

## Test matrix

All tests must use captured CSV/dictionary fixtures and injected subprocess, `/proc`, clock, and backend seams. They must have no live GPU, process tree, profile store, Ollama, llama-server, or network dependency.

| Area | Required deterministic cases |
| --- | --- |
| GPU parser and field state | zero, one, arbitrary-N, reordered indices with stable UUIDs, duplicate/missing UUID, malformed CSV, empty output, every unavailable marker, missing/extra columns, absent `nvidia-smi`, timeout, non-zero exit, truncated stdout, non-NVIDIA host; fixed extended-to-baseline fallback with no stderr field-name guessing |
| Field-state serialization | every state (`observed`, `unsupported`, `unavailable`, `failed`, `malformed`, `not_queried`), numeric zero observed, `None` requiring a non-observed state, duplicate field state rejection, deterministic bounded details |
| Inventory/sample join | reordered inventory/sample indices, matching UUID join, inventory-only UUID, sample-only UUID, bounded UUID-set mismatch warning, PCI corroboration only |
| Process parser/join | one PID on one GPU, one PID on multiple GPUs, multiple PIDs on one GPU, stale PID, reused PID with changed start ticks, no GPU memory, unreadable `/proc`, bounded process tree |
| Attribution | Ollama server/runner tree, llama-server server, one runtime with mixed confirmed/probable/profile-declared-only GPU records, declared UUID not observed, observed-but-undeclared GPU, unavailable state, deterministic attribution ordering |
| Backend evidence | supported bounded metrics, unavailable metrics, slots/loaded-model facts without fabricated placement, no global-metric request attribution |
| Serialization | deterministic order, `None` distinct from zero, bounded deduplicated warnings, no command line/environment/raw output leakage |
| Compatibility | legacy scalar sampler/readers, old status/row/report fixtures, no score/ranking/duplicate-key/campaign mutation |
| Containment | allowed read-only command/API set, no mutation calls, no lifecycle calls, no configuration/profile writes |

## Real-host acceptance plan

Later operator acceptance should use a temporary in-memory or isolated configuration context and read-only commands only. It should first inventory both physical GPUs and compare UUID/PCI evidence to the expected host values; take a current per-GPU sample; identify the selected llama-server PID and, where `nvidia-smi` exposes it, attribute GPU memory per UUID; and observe Ollama's server with no loaded-model process. It must display profile-declared UUIDs separately from observed allocation, including a mismatch case where applicable.

No generation is needed for the initial acceptance. Only after fixtures and the full suite pass may the operator approve one tiny request to observe post-request snapshots. The acceptance record must verify that no service, model, profile, score, ranking, or runtime mutation occurred and must stay outside the repository as raw evidence; committed documentation records only durable conclusions.

## Proposed Stage 6 sequence

| Substage | Scope, tests, acceptance, rollback, non-goals |
| --- | --- |
| **6A audit and plan** | Documentation only: this file and progress ledger. Validate hygiene/release checks. Acceptance is reviewed plan approval. Rollback removes the two documents. No telemetry code. |
| **6B physical GPU sample model and parser** | Scope is only new `telemetry.py`, immutable sample/field-state structures, pure CSV parser, bounded injected subprocess collection, deterministic serialization, fixtures, and optional additive UUID join with existing `GPUDevice` inventory. Accept arbitrary-N UUID joins, explicit per-field states, and fixed extended-to-baseline query fallback. Roll back the new module/tests. Do not alter `detect_gpu`, `detect_gpus`, `Telemetry`, `ProbeTelemetry`, `nvidia_live`, `live_snapshot`, runner, context profile, doctor, watch, reports, rankings, or campaigns; no process attribution. |
| **6C process discovery and GPU attribution** | Read-only `/proc`/port/PID/start-time and `query-compute-apps` collectors with fixtures. Accept confidence-labelled joins and no command-line persistence. Roll back attribution module. No backend HTTP or benchmark integration. |
| **6D backend/runtime snapshot** | Read-only selected-runtime process and capability-gated backend evidence. Accept Ollama/llama.cpp neutral snapshot fixtures and zero lifecycle calls. Roll back backend snapshot adapter. No runner/report schema change. |
| **6E runner evidence integration** | Add optional snapshot references/sections around existing sample points, preserving all score, validity, raw-row and scalar behavior. Accept old/new artifact compatibility and no ranking input change. Roll back optional writes. No campaign migrations. |
| **6F status/report/doctor additive exposure** | Add display-only per-GPU/runtime evidence to status, watch, doctor, report and model cards. Accept old fixtures and bounded rendering. Roll back displays while evidence remains optional. No ranking/campaign changes. |
| **6G real-host acceptance and closeout** | Operator read-only validation on zero/one/many supported environments, including the target two-GPU host. Accept identity/process confidence evidence and no mutations. Roll back by disabling collection/display use, not deleting historical evidence. No Stage 7 profiler. |

## Risks and explicit non-goals

Risks are driver field variability, NVIDIA process-query permissions, PID reuse, unrelated processes sharing a GPU, command-line ambiguity, benchmark perturbation from frequent sampling, and legacy `ollama_*` terminology embedded in historical context evidence. The mitigation is evidence confidence, bounded warnings, low cadence, strict physical UUID joins, additive compatibility fields, and no inference from configuration text.

Stage 6 does not implement model/server launching, stopping, restart, split configuration, CUDA device selection, model switching, cache erasure, systemd/KV repair changes, permanent watcher/session logging, telemetry-based scoring/ranking changes, campaign/report runtime identity freezing, embeddings, vision/OCR, or runtime-fit profiling.

## Exact first implementation step

Begin Stage 6B by adding a new pure `telemetry.py` and fixture tests only: immutable sample and explicit field-state structures, fixed extended/baseline `nvidia-smi` query tiers through an injected bounded subprocess seam, deterministic serialization, and UUID joins with existing `GPUDevice` inventory. Do not integrate with legacy collectors, implement process attribution, or redirect any existing caller.
