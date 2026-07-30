# RC21 Progress

## Stage status

| Stage | Status | Durable output |
| --- | --- | --- |
| 0. Working-tree preparation outside Codex | Complete | Clean reviewed working tree |
| 1. Architecture audit and durable plan | Complete | This RC21 documentation set |
| 2. Arbitrary-N real GPU inventory | Complete | Compatibility inventory API, tests, and real-host acceptance |
| 3. Backend protocol and Ollama preservation | Complete | Protocol, adapters, compatibility tests, and real-host Ollama acceptance |
| 4. Runtime profiles and discovery/selection | Complete | Profiles, bounded discovery, fail-closed selection, and real-host acceptance |
| 5. External llama-server backend | Not started | Read-only external backend adapter |
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
- Fixture regression coverage was added for these cases. The operator must rerun real-host acceptance outside Codex; Stage 4 remains pending that confirmation.

**Proposed Stage 5 objective:** implement the external llama-server inference adapter while preserving Ollama and failing closed for unsupported switching and service operations.
