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
| 6. Per-GPU/backend-neutral telemetry | Not started | Per-device/process evidence |
| 7. Runtime-fit profiler | Not started | Diagnostic fit evidence lane |
| 8. Campaign/report/ranking/resume integration | Not started | Frozen runtime identity and migrations |
| 9. Real acceptance and RC21 release | Not started | Acceptance record and release documentation |

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
