# RC21 Progress

## Stage status

| Stage | Status | Durable output |
| --- | --- | --- |
| 0. Working-tree preparation outside Codex | Complete | Clean reviewed working tree |
| 1. Architecture audit and durable plan | Complete | This RC21 documentation set |
| 2. Arbitrary-N real GPU inventory | Complete | Compatibility inventory API, tests, and real-host acceptance |
| 3. Backend protocol and Ollama preservation | Complete | Protocol, adapters, compatibility tests, and real-host Ollama acceptance |
| 4. Runtime profiles and discovery/selection | Complete | Profiles, bounded discovery, fail-closed selection, and real-host acceptance |
| 5. External llama-server backend | Complete | One-served-model external adapter with completed AI-PC direct and runner acceptance |
| 6. Per-GPU/backend-neutral telemetry | Complete; ready for Stage 7 planning/review | UUID-keyed physical/process evidence, snapshot, run artifact, and conservative exposure |
| 7. Runtime-fit profiler | Complete; ready for Stage 8 planning/review | Conservative UUID-keyed capacity and requirement evidence with completed real-host acceptance |
| 8. Campaign/report/ranking/resume integration | Complete; ready for Stage 9 review | Frozen runtime-identity contracts, variant row provenance, compatibility readers, and accepted fixture-only harness |
| 9. Real acceptance and RC21 release | Complete; release closeout pending | Accepted evidence, reconciliation, validation, scope audit, and release documentation |

## Stage 1 record

- Baseline version: `1.0.0rc20.post1` from `llm_modelbench/__init__.py`, `README.md`, and `pyproject.toml` dynamic version metadata.
- Baseline validation: `python3 -m compileall -q llm_modelbench tests`, `python3 -m pytest -q` (`532 passed`), `./llmb selftest`, `python3 tools/release_check.py`, and `git diff --check` all passed.
- No production Python, tests, package version, changelog heading, score, ranking, service, model, commit, tag, or remote changed in Stage 1.

## Human gate

Review `RC21_MASTER_PLAN.md` and `RC21_SOURCE_AUDIT.md`. Do not begin Stage 2 until the operator has prepared a clean reviewed working tree outside Codex and approved the Stage 2 objective below.

**Proposed Stage 2 objective:** replace the single-card hardware discovery internals with arbitrary-N physical GPU inventory while retaining `detect_gpu()` and current scalar fields as compatibility wrappers; no backend/runtime behavior changes in that stage.

## Stage 2 record

- Added `detect_gpus()` and `GPUDevice` in `llm_modelbench/hardware.py`; `detect_gpu()` remains the first-device `GPUInfo` compatibility wrapper.
- Added an inventory list to doctor output without changing the existing scalar `GPU:` line.
- Focused suite passed: 59 tests covering multi-row ordering, identities, optional values, wrapper/no-GPU behavior, and existing scalar consumers.
- Codex could not access the NVIDIA driver, so final acceptance was run on the real AI-PC host. The new API detected the RTX 5060 Ti and RTX 3060 as separate ordered devices with distinct UUIDs and PCI bus IDs, correct VRAM totals, and compute capabilities 12.0 and 8.6 respectively.
- See `docs/rc21/stage-02-multigpu-inventory.md` for the compatibility boundary, known limitations, and rollback point.

**Proposed Stage 3 objective:** define a capability-oriented backend protocol while preserving Ollama and Mock client behavior; do not implement a new backend or runtime selection yet.

## Stage 3 record

- Added the `InferenceClient` protocol, backend identity, four-state capability model, and thin Ollama/Mock adapters in `llm_modelbench/backend.py`.
- `cli._client()` now constructs adapters; the existing Ollama and Mock implementations retain their HTTP and deterministic behavior unchanged.
- KV/systemd repair is capability-guarded as Ollama-only before a cascade can proceed.
- Focused suite passed: 102 tests across adapters, CLI construction, capability routing, runner-adjacent behavior, campaigns, judging, and existing Ollama/Mock regression coverage.
- Codex could not reach the host Ollama endpoint. Final read-only acceptance on the real AI-PC confirmed Ollama `0.30.7`, 57 installed models, zero loaded models, and normal `./llmb doctor` behaviour.
- See `docs/rc21/stage-03-backend-abstraction.md` for the interface, compatibility boundary, limitations, and rollback point.

**Proposed Stage 4 objective:** add runtime-profile schema plus read-only local discovery/selection while preserving the existing Ollama URL configuration contract.

## Stage 4 record

- Added atomic JSON-backed external runtime profiles, bounded local discovery, backend-specific read-only health states, and selection precedence in `llm_modelbench/runtime_profiles.py`.
- Added `llmb runtime discover|list|show|select|save|delete` and `--runtime-profile` wiring for inference commands; selecting llama.cpp before Stage 5 fails clearly.
- Focused suite passed: 90 tests covering profile persistence, selection, recommendation, fail-closed cases, deduplication, and containment.
- Final real-host acceptance detected healthy Ollama and llama-server endpoints, recommended llama.cpp on the dual-GPU host, displayed both interactive choices, and failed closed during unattended ambiguity.
- Isolated Ollama and llama.cpp profiles were saved, shown, deleted, and verified absent. Ollama remained unloaded and its canonical 57-model inventory digest remained unchanged.
- See `docs/rc21/stage-04-runtime-profiles-discovery.md` for storage, discovery bounds, health/selection policy, limitations, and rollback.

### Stage 4 real-host acceptance fix iteration

- Corrected the bounded Ollama `/api/tags` health read: a 4096-byte truncation falsely classified the real 57-model inventory as invalid JSON. Responses now have a deliberate 4 MiB limit and fail explicitly when exceeded.
- Hardened tags-shape validation, runtime subcommand error boundaries, unknown-profile deletion, and `_client()` so only the documented no-healthy-candidate legacy case can fall back to implicit Ollama. Ambiguous or unhealthy explicit/default selection remains fail-closed.
- Regression coverage was added for these cases. Operator acceptance on the AI-PC completed Stage 4: Ollama `0.30.7` was healthy with 57 installed and no loaded models; llama-server was healthy at `127.0.0.1:8081`; two GPUs recommended llama.cpp; interactive selection displayed both runtimes; unattended ambiguity exited 1 requiring `--runtime-profile`; and explicit llama.cpp selection stopped at the Stage 5 boundary. The canonical inventory digest remained `5f553c1450d7f2b1c3e52010bcd0db205a63a9ec42bf037072d795b7972a933c`.

**Proposed Stage 5 objective:** implement the external llama-server inference adapter while preserving Ollama and failing closed for unsupported switching and service operations.

## Stage 5 record

- Added a bounded, read-only external llama-server client and protocol adapter. It accepts exactly one served model ID or reported alias and rejects router/multi-model inventory.
- Chat, structured JSON, reasoning preservation, template-gated tool calls, tokenization, slots, and observed usage/timing normalization are implemented without server lifecycle or property mutation.
- Ollama and Mock behavior remain unchanged. Embeddings, suffix generation, unload, flush-all, service/KV repair, and unreported offload placement remain unsupported or unavailable.
- Human review corrective iteration: made non-empty `/v1/models.data` authoritative, removed invented GGUF/vision metadata, hardened endpoint and nested response validation, rejected redirects, preserved tool/message/structured-output compatibility, and added clean selected-llama.cpp CLI error containment with no Ollama fallback.
- Follow-up review correction: explicit thinking now maps only through `chat_template_kwargs.enable_thinking`; unsupported operations are cold-cache transport-free; and messages, props, slots, tools, and OpenAI response formats receive strict shape validation before HTTP. The redirect/error-size and no-Ollama-fallback containment remains covered.
- Final closeout correction rejects Python boolean values for every interpreted integer request and endpoint field, including context/token controls, model metadata, slots, usage, and timing counters.
- Direct AI-PC adapter acceptance at `/tmp/llmb-rc21-stage5-live-20260730T221916` passed: exactly one served model, build `b10086-66e4bf7e5`, active/training contexts `65536`/`262144`, one slot, tokenization, `think=off` with no separate or visible reasoning, structured JSON `{"status":"ok","value":5}`, a normalized but unexecuted `stage5_probe` tool call, and direct timing evidence. llama-server was healthy and idle afterward.
- The first normal runner attempt failed before generation because `runner.py` called `flush_all()` unconditionally. Runner and reachable repair lifecycle calls are capability-gated before invocation. Unsupported llama.cpp cache/model lifecycle skips are backend routing, not harness or model-quality failures; supported Ollama, Mock, and legacy direct-client lifecycle behavior remains preserved.
- Normal AI-PC runner acceptance at `/tmp/llmb-rc21-stage5-runner-20260730T234849` completed with run ID `rc21_stage5_runner_20260730T234849`: one served model and `json_extract`, exit `0`, score `100.0`, valid run, approximately `23.1 tok/s`, prompt-token metric `380 -> 455`, and predicted-token metric `59 -> 88`. The llama-server was healthy and idle afterward; Ollama remained at 57 installed models and zero loaded models; the canonical inventory digest remained `5f553c1450d7f2b1c3e52010bcd0db205a63a9ec42bf037072d795b7972a933c`; isolated profile stores ended empty. Final closeout validation passed: 136 focused tests and 613 full-suite tests. Stage 6 has not started. See `docs/rc21/stage-05-external-llama-server-backend.md`.

**Proposed Stage 6 objective:** collect per-GPU and backend-neutral server-process telemetry without changing benchmark scoring or runtime lifecycle behavior.

## Stage 6A record

- Completed the read-only architecture audit and staged implementation plan in `docs/rc21/stage-06-telemetry-audit-and-plan.md`.
- The audit confirms that `detect_gpus()` already provides arbitrary-N physical inventory, but current live telemetry and scalar compatibility consumers still sample the first NVIDIA row. Stage 6B will add a UUID-keyed physical sample model without changing those existing fields.
- The planned canonical physical identity is NVIDIA UUID, with PCI bus ID as supporting evidence and physical index only as a sample-time locator. CUDA ordinals and profile-declared GPU UUIDs are not proof of actual process placement.
- The correction adds deterministic per-field telemetry states, moves confidence to per-runtime/GPU attribution records, and fixes Stage 6B to join live rows only to existing `GPUDevice` inventory by UUID. Its fixed extended/baseline `nvidia-smi` query tiers use bounded injected subprocess collection; Stage 6B does not integrate or replace any legacy collector.
- Later process attribution will use bounded read-only local process/port evidence plus `nvidia-smi` compute-process UUID/memory records, and will report confirmed, probable, profile-declared-only, or unavailable confidence per GPU rather than guessing.
- Stage 6A made no telemetry implementation, benchmark, service, model, runtime-profile, score, ranking, campaign-schema, version, or release changes. Stage 7 has not started.

**Exact next action:** begin Stage 6B only after review with a new pure telemetry module, explicit field states, fixed query tiers, deterministic UUID join to existing inventory, and fixtures; do not integrate a legacy collector or add process attribution.

## Stage 6B record

- Added the standalone `llm_modelbench.telemetry` module with immutable physical sample/state structures, deterministic serialization, fixed extended/baseline NVIDIA query tiers, bounded injected subprocess collection, strict CSV parsing, and pure UUID-only inventory joins.
- Review correction: duplicate live UUID groups are excluded rather than first-row-wins; every fixed-tier CSV row requires the exact column count; and `successful_tier` now means command completion plus CSV syntax and fixed schema success.
- Review correction: ambiguous duplicate inventory UUID groups are excluded from enrichment, optional identity markers normalize to `None`, and `GPUCollectionResult` invariants reject contradictory evidence construction.
- Offline fixture validation is complete: focused validation `57 passed`, full validation `664 passed`, compileall passed, selftest passed, release check passed, and `git diff --check` passed.
- Real-host acceptance passed at `/tmp/llmb-rc21-stage6b-live-20260731T162617`: extended succeeded without fallback, exactly two samples were returned, expected UUID and PCI mappings and deterministic UUID ordering passed, inventory enrichment supplied driver and compute capability, and collection/join errors were empty. Collection left repository state unchanged.
- Existing `detect_gpu`, `detect_gpus`, `Telemetry`, `ProbeTelemetry`, `nvidia_live`, `live_snapshot`, runner, reports, status, doctor, watcher, context profiling, ranking, campaigns, and repair were not changed or redirected. No scoring, ranking, report-schema, campaign-schema, or lifecycle behavior changed.
- Stage 6B is complete and ready for commit. Stage 6 overall is not complete. Stage 6C subsequently implemented isolated bounded process discovery and process-to-GPU attribution without redirecting this collector or any consumer.

## Stage 6C record

- Added the standalone `llm_modelbench.process_telemetry` module with bounded injected procfs discovery, stable PID/start-time identity, socket-inode ownership evidence, fixed NVIDIA compute-process CSV parsing, and deterministic per-runtime/per-GPU attribution.
- Declared profile UUIDs and observed process UUIDs remain distinct. Attribution is `confirmed`, `probable`, `profile-declared only`, or `unavailable`; CUDA ordinals and runtime command-line placement options are never mapped to physical UUIDs.
- Review correction: runtime observed UUIDs derive only from unambiguous confirmed/probable runtime matches; stable PID/start-time revalidation and unique compatible endpoint ownership are required for confirmed attribution. Procfs truncation, malformed socket rows, and ambiguous PID/owner groups remain bounded diagnostic evidence.
- Live-evidence correction: empty procfs command lines are valid absent command identity rather than malformed evidence; non-empty malformed or truncated command lines remain bounded diagnostics. The authoritative manual before/query/after rerun passed at `/tmp/llmb-rc21-stage6c-manual-live-20260731T220628Z`: NVIDIA returned a successful empty idle result; Ollama was listening with two processes but no compute allocation; llama-server was unavailable and not started; reconciliation/attribution remained empty; bounded `proc_fd` limitations correctly kept socket evidence incomplete; repository state was unchanged.
- No existing collector or consumer was redirected. `detect_gpu`, `detect_gpus`, `Telemetry`, `ProbeTelemetry`, `nvidia_live`, `live_snapshot`, runner, reports, status, doctor, watcher, context profiling, rankings, campaigns, and repair remain unchanged. No scoring, ranking, report-schema, campaign-schema, or lifecycle behavior changed.
- Fixture validation is complete: the required Stage 6C/Stage 6B/inventory focused set has `104 passed`; the full suite has `711 passed`; compileall, selftest, release check, import side-effect check, and `git diff --check` passed. Stage 6C is complete; Stage 6 overall remains incomplete.

## Stage 6D-6F record

- Stage 6D composes existing Stage 6B/6C evidence into schema-versioned, deterministic `RuntimeTelemetrySnapshot` records. Declared and observed physical UUIDs remain separate; partial, idle, permission-limited, and failed collection states remain bounded evidence.
- Stage 6E writes one optional pre-run `runtime_telemetry.json` artifact for real backend adapters and a small row reference. It is failure-isolated and does not alter benchmark execution, lifecycle behavior, scoring, ranking, or legacy-artifact reading.
- Stage 6F presents a bounded summary in reports and doctor. It does not claim physical layer, KV-cache, tensor-split, or offload placement, and it adds no watcher, repair, or runtime management behavior.
- Focused checkpoints passed: Stage 6D model/assembly `107 passed`; Stage 6E runner/backend/profile integration `35 passed`; Stage 6F report/doctor rendering `14 passed`. Final full validation is recorded with the batch review.
- Consolidated hardening added canonical UUID/inventory invariants, atomic optional-artifact writes, and conservative missing/corrupt/future-schema report handling. Stage 6G host acceptance is prepared at `/tmp/llmb-rc21-stage6g-acceptance.sh` but remains pending execution.
- Consolidated review validation passed: telemetry/runner/report/doctor/profile/backend focus `48 passed`; full suite `725 passed`; compileall, selftest, release check, diff check, and import side-effect guard passed.
- First Stage 6G evidence at `/tmp/llmb-rc21-stage6g-acceptance-20260731T224056Z` found a script-only noninteractive runtime-selection failure, endpoint-port/incomplete-evidence conflation, missing conservative Ollama-worker lineage, and duplicate aggregate errors. These are corrected with a replacement `/tmp/llmb-rc21-stage6g-acceptance.sh`; Stage 6G remains pending rerun and Stage 6 remains incomplete.
- The next Stage 6G evidence at `/tmp/llmb-rc21-stage6g-acceptance-20260731T231022Z` passed pure snapshots, fixed NVIDIA query, doctor, compatibility, and non-mutation checks. Its runner correctly refused protected `py_anagram` before inference; the script now uses safe smoke-level `json_extract`. Timestamp-only process observations now deduplicate by substantive identity. Stage 6G remains pending.
- Stage 6G passed at `/tmp/llmb-rc21-stage6g-acceptance-20260801T001221Z`: fixed query, UUID validation, deterministic snapshots, runner integration, report, doctor, compatibility, and repository non-mutation passed. Pre-run process allocation was idle, while later benchmark metrics remained separate evidence. Stage 6A through 6G and Stage 6 are complete; Stage 7 has not started.

## Stage 7 record

- Added `llm_modelbench.runtime_fit`: a pure, UUID-keyed conservative evaluator, read-only collection wrapper, and `./llmb runtime-fit` command. It preserves lower-bound model-size provenance, separate installed/live-free capacity, per-device reserve, KV evidence, deterministic finite serialization, and explicit reasons without integrating benchmarks, campaigns, reports, rankings, routing, repair, or lifecycle behavior.
- Fit states are `confirmed_fit`, `candidate_fit`, `conditional_fit`, `confirmed_no_fit`, and `unknown`. Unknown runtime overhead and KV inputs never become zero; confirmed fit requires complete modeled evidence; confirmed no-fit requires a reliable exceeding lower bound. The default reserve is 512 MiB per device.
- NVIDIA UUID is canonical identity. PCI is corroborating evidence only; index and CUDA ordinal are never identity. Arbitrary-N inventory is supported, malformed/duplicate UUIDs and positional or incomplete allocation weights are rejected, aggregate VRAM never authorizes multi-GPU fit, explicit layer split evaluates each UUID-keyed allocation, unsupported tensor placement remains unknown, and CPU/RAM spill remains conditional.
- Real-host acceptance passed at `/tmp/llmb-rc21-stage7-acceptance-20260802T135500Z`: the repository and bounded runtime state were unchanged; no inference, runtime lifecycle action, model/configuration mutation, or host-code execution occurred; the RTX 5060 Ti (`GPU-78077308-f4b2-3330-6d4e-19581d7b1511`) and RTX 3060 (`GPU-5b99bce2-35ab-f6db-857b-72162069fa72`) had canonical UUID and separate installed/live-free evidence.
- The host had Ollama 0.32.5 with zero installed models and empty `/api/tags`, empty llama.cpp `/v1/models`, and no configured UUID-bound llama.cpp profile. The optional real-model and configured-split cases were correctly recorded as `environmentally_unavailable`; no model was installed to manufacture evidence. The isolated arbitrary-N pure fixture verified 3:3:1 layer allocation, no aggregate fit without strategy, per-device no-fit, unknown overhead, and explicit spill semantics.
- Final closeout validation passed: focused Stage 7 evaluator/CLI tests `9 passed`; full suite `743 passed`; compileall, selftest, release check, diff check, import-side-effect guard, and added-lines secret/environment-value review passed.
- Stage 7 implementation and acceptance are complete. See `docs/rc21/stage-07-live-acceptance-and-closeout.md`. Stage 8 is ready for planning/review and has not started.

## Stage 8 record

- Added schema-versioned, deterministic `RuntimeIdentity` evidence with canonical NVIDIA UUIDs, credential-free normalized endpoints, backend/profile/server/model evidence, explicit execution settings, SHA-256 compatibility hash, and fail-closed legacy/current compatibility results. Observation timestamps, PCI corroboration, and display-only evidence do not affect the compatibility hash. A review correction replaced generic execution mismatch output with deterministic field-specific codes.
- Campaign plans can freeze per-model runtime identities and separate judge state; runner rows can reference a run-level `runtime_identity.json` by full authoritative hash and runtime variant ID. Report duplicate keys preserve distinct runtime variants while legacy rows remain readable.
- Existing task scoring, canonical ranking selection, ranking-store mutation controls, telemetry semantics, fit advisory semantics, lifecycle behavior, model state, and historical artifacts remain unchanged. The initial fixture evidence `/tmp/llmb-rc21-stage8-acceptance-20260802T141634Z` is provisional because it omitted required gates; the hardened replacement at `/tmp/llmb-rc21-stage8-acceptance.sh` is prepared but not executed.
- The first hardened harness root `/tmp/llmb-rc21-stage8-acceptance-20260802T142557Z` is also non-authoritative: it stopped on an acceptance-harness `Path + str` construction error before substantive acceptance results. The replacement harness records phase and bounded traceback evidence on unexpected errors; Stage 8 remains pending.
- Offline validation passed before the harness hardening: focused identity/Stage 7 tests `13 passed`; full suite `746 passed`; compileall, selftest, release check, diff check, import-side-effect guard, and added-lines secret/environment review passed. Stage 8 implementation remains pending hardened acceptance. Stage 9 has not started.
- Subsequent non-authoritative harness roots through `/tmp/llmb-rc21-stage8-acceptance-20260802T144647Z` found acceptance-only evidence gaps and a report-provenance presentation defect: legacy aggregates could show an unrelated run-level identity artifact. The correction only attaches a run-level artifact to a model aggregate when a matching row hash exists, adds run-level `runtime_provenance` in `summary_meta.json`, and adds deterministic CSV provenance columns. Scores, canonical weighting, ranking eligibility, and legacy score values remain unchanged.
- Fixture-only Stage 8 acceptance passed at `/tmp/llmb-rc21-stage8-acceptance-20260802T150345Z`: all 28 Stage 8 and 28 report checks passed, the populated two-record ranking fixture and bounded runtime/repository snapshots were unchanged, and no inference, lifecycle action, or persistent mutation occurred. See `docs/rc21/stage-08-live-acceptance-and-closeout.md`. Stage 8 is complete and ready for Stage 9 review; Stage 9 has not started.

## Stage 9 record

- Stage 9 acceptance, evidence reconciliation, repository validation, and
  documentation closeout are complete. The accepted evidence covers preflight,
  dry run, Ollama and llama.cpp GPU lanes, ThinkingCap exact 64K, runtime
  contract/selection, resume-gate tests, and live standard/campaign runtime
  identity mismatch refusals.
- Phase 1A reconciled 13 accepted areas: 11 native passing summaries and two
  explicit adjudications. Phase 2 passed 172 focused tests and a clean 763-test
  full suite; doctor and self-test passed. Phase 3 produced a safe seven-path
  source/test allowlist with no unresolved path or unapproved evidence package.
- See `docs/rc21/stage-09-release-closeout.md`,
  `docs/rc21/stage-09-evidence-index.md`, and
  `docs/handovers/RC21_STAGE9_RELEASE_HANDOVER.md`. RC21 has not been
  versioned, committed, tagged, pushed, or released.

**Next action:** validate the Phase 4 documentation and final release allowlist
in Phase 5 before preparing the version, commit and tag plan.
