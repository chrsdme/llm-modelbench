# RC21 Stage 1 Codex Prompt

Work in the repository root on the intended RC21 branch.

Perform RC21 Stage 1 only: a read-only architecture audit and durable project plan. Do not change production Python, test behavior, scoring, canonical rankings, package version, changelog release heading, services, systemd, models, commits, tags, or remotes. Do not start Stage 2.

Read the complete repository, including README, changelog, packaging/hygiene files, all runtime/configuration/backend/hardware/runner/context/doctor/progress/watch/report/fingerprint/capabilities/planner/campaign/repair/model-card modules, relevant tests, and historical audits/handovers. Confirm the current version and baseline validation from the checkout.

Plan for preserving Ollama and adding external, already-running `llama-server`; arbitrary-N real GPUs with UUID/PCI identity and separate physical/runtime-visible indices; healthy local runtime discovery; automatic only-viable selection, interactive multi-viable selection, unattended ambiguity failure; reusable profile persistence; single-GPU Ollama and multi-GPU healthy llama.cpp recommendation rules; per-GPU and neutral server-process telemetry; a runtime-fit lane; and frozen runtime identity in campaigns, rows, reports, rankings, and resume. Preserve score/ranking/evidence semantics.

State the external llama-server boundary: never launch/restart/stop/reconfigure it; one served model per endpoint/profile by default; multiple endpoints are profiles; inventory is served models only; no automatic GGUF switching; no-switch multi-model campaigns fail closed; post-hoc judging needs a separate judge runtime profile or external-judge blocker. Keep Ollama systemd/KV repair Ollama-only; llama.cpp reports unsupported.

Document RC22 deferrals: managed llama-server lifecycle, validated-profile Ollama Modelfile generation, permanent user services, permanent watcher/session logger, non-destructive fleet classification with explicit-approval removal, vision/OCR expansion, and embedding changes. No automatic deletion.

Create `AGENTS.md`, `docs/rc21/RC21_MASTER_PLAN.md`, `docs/rc21/RC21_SOURCE_AUDIT.md`, `docs/rc21/RC21_PROGRESS.md`, `docs/rc21/RC22_DEFERRED_SCOPE.md`, and this cleaned prompt. The audit must map construction/config/GPU/telemetry/client surfaces, model identity/inventory, single-GPU evidence, campaign fields, service boundaries, schemas/tests, llama-server one-model boundary, and judge issue with file/symbol references. The plan must use stages 0-9 and specify objective, files, non-goals, compatibility, tests, hardware acceptance, rollback, docs, and exact next action for each future stage.

Use repository-relative paths and `<repo>` placeholders. Keep generated/local evidence outside the repository. Run compileall, pytest, selftest, release check, diff check, and status. Stop for human review after Stage 1.
