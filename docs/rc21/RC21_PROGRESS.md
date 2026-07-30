# RC21 Progress

## Stage status

| Stage | Status | Durable output |
| --- | --- | --- |
| 0. Working-tree preparation outside Codex | Pending human action | Clean reviewed working tree |
| 1. Architecture audit and durable plan | Complete, awaiting human review | This RC21 documentation set |
| 2. Arbitrary-N real GPU inventory | Not started | Compatibility inventory API and tests |
| 3. Backend protocol and Ollama preservation | Not started | Protocol, adapter, compatibility tests |
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
