# RC21 Source Audit

Audit date: 2026-07-30. Baseline is `1.0.0rc20.post1`; no source behavior changed by this audit.

## Runtime/client topology

`cli._client` (`llm_modelbench/cli.py:43`) is the only direct production construction site: it creates `MockClient(cfg.ollama_url, ...)` for `--mock`, otherwise `OllamaClient(cfg.ollama_url, ...)`. Test construction is intentionally direct in `tests/test_agentic_finals_seed.py`, `test_agentic_finals_hardening.py`, `test_capability_workflow.py`, `test_code_debug_correction.py`, `test_judge_dumps.py`, `test_needle_output_fields.py`, `test_planner_doctor_grade.py`, `test_pre_batch_integrity.py`, `test_reasoning_tasks.py`, `test_rc10_repair_truth.py`, `test_rc9_context_repair.py`, `test_repair.py`, and `test_validity_hotfix.py`.

The current `OllamaClient` method surface (`ollama.py:69-451`) is: `tags`, `version`, `show`, `capabilities`, `supports_thinking`, `model_info`, `model_size_bytes`, `context_length`, `chat`, `chat_tools`, `generate_suffix`, `embed`, `loaded_model_stats`, `offload_fraction`, `unload`, and `flush_all`. `MockClient` subclasses it (`ollama.py:459`) and overrides inventory, metadata, generation, tools, suffix, embedding, and loaded-model behavior.

Exact production callers are:

- `planner.build_plan` and CLI inventory/model selection use `tags`; `capabilities.interrogate_model` uses metadata-derived methods plus `chat`/`embed` probes (`planner.py:51`, `cli.py:58`, `capabilities.py:59-115`).
- `runner` uses `chat`, `chat_tools`, `generate_suffix`, `embed`, `version`, and optional loading/offload behavior (`runner.py:55,116,863,1285,1442,1444,1629`); this is the largest protocol client.
- `judge.py:90,112` and `judge_dumps` route judging through `chat`.
- `context_profile.py:128-141` uses `chat`; `repair.py:1028` uses `version` and its repair execution shares runner/client operations.
- `fingerprint.model_identity` (`fingerprint.py:122`) derives identity from Ollama tag rows/digest; campaign judge selection consumes its inventory identity (`campaign.py:1465`).

RC21 should model basic generation, inventory, metadata/capabilities, embedding, native tools, suffix/FIM, loaded-server inspection, and lifecycle/service control as separately advertised capabilities. This preserves unsupported results rather than falsely mapping llama.cpp to Ollama behavior.

## Ollama configuration and assumptions

`Config.ollama_url` and `_validate_ollama_url` are in `config.py:64-152`; `LLM_MODELBENCH_OLLAMA_URL` maps directly to it at line 129. `cli.py:46-48`, repair port/service discovery (`cli.py:760,806,835,857`), `doctor.py:38-51`, report metadata (`report.py:265`), and watch’s direct environment fallback (`watch.py:780,824`) all depend on it. Examples define `ollama_url`; `docs/USAGE.md:189` documents the environment variable; `tests/test_config_validation.py:31-32` and `tests/test_cli_subcommands.py:23` assert the current contract.

Ollama inventory is assumed to be `tags()` rows with `name`, size, and digest/model metadata in `cli.py:58-114`, `planner.py:51-76`, `runner.py:1376-1379`, and `fingerprint.py:122`. RC21 must retain that path while allowing llama-server inventory to mean served endpoint model(s), never every GGUF on disk.

`ollama_service.py` contains human-supervised, privileged systemd/KV behavior: `discover_active_service` (`:145`), `OllamaServiceController` (`:228`), CUDA UUID binding checks, temporary `OLLAMA_KV_CACHE_TYPE` drop-ins, restart, verification, and restoration. `cli.cmd_repair` invokes it only for `--kv-cascade --restart-ollama` (`cli.py:734-880`); `repair.py` records KV runtime identity and cascade results. These operations are permanently Ollama-specific in RC21. llama.cpp must return an explicit unsupported outcome, not an emulated service action.

## GPU and telemetry audit

`hardware.GPUInfo` (`hardware.py:22`) is scalar. `detect_gpu` (`:41`) asks `nvidia-smi` for all GPUs but selects `splitlines()[0]`; AMD collapses to maximum VRAM; Apple is a unified-memory scalar. It has no UUID, PCI bus ID, physical index, or runtime-visible index. `suggested_vram_budget_gb` (`:68`) accepts one GPU. Direct callers are `Config.load` (`config.py:152`), `runner.run` (`runner.py:1284`), `context_profile.run_context_profile` (`context_profile.py:341`), `repair.build_plan`/runtime checks (`repair.py:669,1026`), and `doctor.collect` (`doctor.py:36`). Tests monkeypatch it in `test_config_validation.py`, `test_repair.py`, `test_rc9_context_repair.py`, and `test_rc10_repair_truth.py`.

Scalar fit assumptions occur at `cfg.vram_budget_gb` in `config.py`, `planner.py:76`, `cli.py:110-114`, `runner.py:267,1309`, `context_profile.py:342-345`, and `repair.py:669-831`. Status exposes one `gpu_vendor`, `gpu_name`, driver, and budget in `progress.StatusWriter._write` (`progress.py:280-305`); doctor prints one GPU (`doctor.py:98`); watcher renders one hardware row (`watch.py:359-423`); report metadata reads those scalar fields (`report.py:257-292`); model-card long-context evidence uses scalar VRAM/offload fields (`model_cards.py:59-126`); ranking aggregates preserve model/task evidence without runtime identity.

`Telemetry` (`hardware.py:75`) samples the first `nvidia-smi` row and is created by `runner.py:1460`. `ProbeTelemetry` (`hardware.py:135`) samples `nvidia_live` and is created by `runner.py:1005` and `context_profile.py:128`; tests replace it in `test_rc12_pre_rv3_truth.py`. `nvidia_live` (`hardware.py:407`) parses only the first row and is called by `ProbeTelemetry._query_raw` (`:161`) and runner’s dynamic estimate (`runner.py:395`). `live_snapshot` (`hardware.py:431`) calls it and is consumed by `doctor.py:37`, `inline_ui.py:138`, `interactive_nav.py:76`, and `watch.py:265,270,895,900`; the watcher integration is tested in `test_rc12_pre_rv3_truth.py`.

`ProbeTelemetry._read_ollama_process_memory` already matches local Ollama/llama-server process names (`hardware.py` around `:304`), but emitted keys remain `ollama_process_count`, `ollama_rss_*`, `ollama_pss_*`, and `ollama_swap_*` (`:237-275`). RC21 needs neutral selected-server process evidence and additive legacy aliases.

## Rows, reports, cards, rankings, and campaigns

Runner persistence is centered on `runner.run` (`runner.py:1274-1663`): it resumes from `raw_results.jsonl`, writes `model_identities.json`, emits per-row `model_digest`, config/filters, run validity, and scalar hardware status. Raw rows do not record backend, runtime profile, endpoint/server identity, GPU inventory, or selected mapping. `report._duplicate_key` (`report.py:22`) keys on model/task/task hash/sample and its metadata includes `ollama_url` and scalar GPU fields (`:257-292`), so provenance additions must not alter quality deduplication semantics.

`model_cards.py:59-126` reads legacy long-context `runtime_identity`/KV fields but cards do not carry a complete selected backend/GPU profile identity. `watch.py:349,359,419,707` uses one current model and scalar hardware/status. `progress.py:280-305` is the source status schema (`llm-modelbench.status.v1.1`). `rankings.py`, `rankings_v3.py`, and `rankings_v31.py` consume report/candidate data with no runtime identity; RC21 must add provenance without touching canonical formulas/ordering.

Campaign persistence is `CampaignManifest` schema v1 (`campaign.py:327-367`), `plan/plan.json`, `plan/inventory.json`, `plan/capabilities.json` (`:400-424`), primary raw rows/model identities, effective rows, readiness, candidates, and package checks (`:747-783`). Plan equivalence only ignores `created_at` (`:426`); resume tracks lifecycle state, not runtime identity (`:450-493`, `:562-609`). Stage 8 must add a versioned runtime contract to manifest/plan/inventory/capabilities, primary and recovery child configs/rows, judge evidence, effective rows, readiness, candidate ranking rows, report metadata/cards, package required-file/checksum rules, and equivalence/resume checks. Legacy migration (`:989-1084`) must label unavailable identity rather than manufacture it.

Campaign automatic post-hoc judging presently selects an outside-cohort identity from the same inventory (`campaign.py:1465-1486`). A one-model external endpoint normally has no distinct judge: RC21 must require another eligible runtime profile/endpoint or retain the existing `external_judge` readiness blocker. This is not a reason to switch the served model automatically.

## Migration test inventory

Likely compatibility test updates/additions belong alongside `test_config_validation.py`, `test_cli_subcommands.py`, `test_planner_doctor_grade.py`, `test_capability_workflow.py`, `test_repair.py`, `test_ollama_service_active_unit.py`, `test_ollama_service_conflict_warning.py`, `test_watch_run_selection.py`, `test_report_provenance.py`, `test_report_offline.py`, `test_campaign.py`, `test_campaign_package_integrity.py`, `test_campaign_recovery_matrix.py`, `test_campaign_final_acceptance.py`, `test_campaign_cleanup_migration_hygiene.py`, `test_campaign_adoption_transaction.py`, `test_rankings.py`, `test_rc11_pre_rankings_v3.py`, and `test_rc12_pre_rv3_truth.py`.

The required approach is additive schema versioning, old-reader compatibility, explicit unavailable values for historical evidence, and invariant tests proving that score values, ranking scope, canonical adoption, raw evidence immutability, and package verification semantics do not change.
