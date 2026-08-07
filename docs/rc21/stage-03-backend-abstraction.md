# RC21 Stage 3: Backend Abstraction

## Outcome

Stage 3 adds `llm_modelbench.backend.InferenceClient`, a runtime-checkable protocol that exactly covers the client operations ModelBench currently uses: backend identity/capabilities, inventory/version, model metadata, chat, native tools, suffix generation, embeddings, loaded-model/offload inspection, unload, and flush.

`BackendIdentity` is additive internal diagnostics with `backend`, `implementation`, and optional endpoint. It is not a runtime profile and is not persisted into campaign/ranking schemas in this stage. `BackendCapabilities` maps each operation to one of `supported`, `unsupported`, `unavailable`, or `unknown`. Model-level Ollama `capabilities(model)` remains unchanged and separate from backend-level capability reporting.

`OllamaBackendAdapter` delegates every protocol method directly to the existing `OllamaClient`; it does not rewrite HTTP paths, payloads, response shapes, exception handling, or model metadata behavior. `MockBackendAdapter` delegates to `MockClient`, retains deterministic outputs, identifies itself as `mock`, and explicitly marks Ollama systemd/KV repair unsupported.

`cli._client()` now returns an `InferenceClient` adapter in both normal and `--mock` modes. Existing direct `OllamaClient`/`MockClient` construction remains a temporary RC21 compatibility boundary for tests and internal callers that have not yet moved to the factory.

## Ollama-only operations

The protocol names loaded-model statistics and offload fraction as explicit backend capabilities, preserving their Ollama `/api/ps` basis. Runner reads them only when an adapter supports them, while legacy direct clients retain their historical behavior during migration. Systemd service repair and KV repair are distinct Ollama-only capabilities. `cmd_repair` checks both before entering a KV cascade; an adapter declaring either unsupported, unavailable, or unknown fails closed before a controller can be created. The adapter does not claim that arbitrary backends support service control.

## Files changed

- `llm_modelbench/backend.py`
- `llm_modelbench/cli.py`
- `llm_modelbench/planner.py`
- `llm_modelbench/runner.py`
- `llm_modelbench/context_profile.py`
- `llm_modelbench/judge.py`
- `tests/test_backend.py`
- `docs/rc21/RC21_PROGRESS.md`
- `docs/rc21/stage-03-backend-abstraction.md`

## Tests

Focused validation passed:

```text
python3 -m pytest -q tests/test_backend.py tests/test_cli_subcommands.py tests/test_capability_workflow.py tests/test_planner_doctor_grade.py tests/test_pre_batch_integrity.py tests/test_campaign.py tests/test_judge_dumps.py tests/test_needle_output_fields.py
102 passed
```

The focused tests cover structural protocol conformance, Ollama delegation, Mock behavior, all capability state categories, unsupported Ollama-only guards, CLI construction, and unchanged exception propagation.

Full validation passed:

```text
python3 -m compileall -q llm_modelbench tests
python3 -m pytest -q
543 passed
./llmb selftest
SELFTEST: ALL GOOD
python3 tools/release_check.py
RELEASE CHECK: PASS
git diff --check
passed
```

## Live Ollama smoke

The Codex execution environment could not reach the host Ollama service, so final read-only acceptance was performed separately on the real AI-PC host.

The host endpoint `http://127.0.0.1:11434` returned Ollama version `0.30.7` and an inventory of 57 installed models. `./llmb doctor` reported Ollama healthy, zero models loaded, and preserved the existing Ollama configuration and output behaviour.

The same doctor run also retained the Stage 2 inventory output for both physical NVIDIA GPUs. No model inference, service lifecycle action, configuration mutation, or fallback runtime selection occurred during this smoke test.

## Known limitations and rollback

- No llama-server adapter, runtime discovery/profile, selection, GPU mapping, split configuration, campaign-schema migration, or per-GPU telemetry is implemented.
- Backend capabilities describe adapter operation support, not whether every model supports a feature; existing model-level capability probes remain authoritative for routing.
- Existing direct client construction remains supported temporarily; Stage 4 will introduce runtime-profile selection without removing it prematurely.
- `supports_capability()` treats legacy clients without capability metadata as capable to preserve RC20 behaviour. New RC21 runtime paths must use adapters, and this fallback must be reviewed before RC21 release.
- Ollama repair capability currently represents implementation support, not proof that systemd or privileged service control is available in the active environment. Runtime availability belongs to Stage 4 discovery.
- Rollback removes the adapter/factory layer and restores `cli._client()` direct construction. Ollama HTTP logic, Mock behavior, scoring, reports, campaigns, and rankings were not altered.

## Proposed Stage 4 objective

Add reusable runtime-profile schema and read-only local discovery/selection. Preserve `ollama_url` and `LLM_MODELBENCH_OLLAMA_URL` as compatible Ollama defaults; automatically choose exactly one viable profile and fail before model work when unattended selection is ambiguous.
