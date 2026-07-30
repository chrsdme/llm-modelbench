# RC21 Progress

## Stage status

| Stage | Status | Durable output |
| --- | --- | --- |
| 0. Working-tree preparation outside Codex | Complete | Clean reviewed working tree |
| 1. Architecture audit and durable plan | Complete | This RC21 documentation set |
| 2. Arbitrary-N real GPU inventory | Complete | Compatibility inventory API, tests, and real-host acceptance |
| 3. Backend protocol and Ollama preservation | Implemented; live endpoint smoke unavailable | Protocol, adapter, compatibility tests |
| 4. Runtime profiles and discovery/selection | Not started | Profile schema and selection workflow |
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
- Read-only local Ollama smoke returned `version: None` and no models; no service or model action occurred, so live acceptance remains unavailable in this environment.
- See `docs/rc21/stage-03-backend-abstraction.md` for the interface, compatibility boundary, limitations, and rollback point.

**Proposed Stage 4 objective:** add runtime-profile schema plus read-only local discovery/selection while preserving the existing Ollama URL configuration contract.
