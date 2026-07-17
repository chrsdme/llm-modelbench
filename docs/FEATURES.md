# Features

## Benchmark lanes

- Deterministic text, coding, JSON, git, file-operation, and web checks.
- Deterministic agentic JSON tool-action lane with no external tool execution.
- Capability-first vision routing for OCR and PDF tasks.
- Embedding retrieval lane with recall@1, MRR, and provenance for the embedding
  model under test.
- Long-context needle probes with explicit coverage and safety gates.

## Model and execution support

- Local Ollama inventory, classification, family routing, and digest identities.
- One-model-at-a-time runs with resume-safe JSONL artifacts.
- Hardware/offload telemetry where available, without mixing it into quality.
- Local wrappers: `llmb`, `llmb-run`, and `llmb-watch`.
- Deterministic mock mode for offline pipeline validation.

## Reports and artifacts

- HTML, Markdown, CSV, JSON, routing, prune, clone, and regression reports.
- Raw result rows, summary metadata, filters, identities, and review-pack export.
- Retrieval diagnostics artifacts with case-level ranks for instrumented rows.
- Interactive and plain watch layouts, including a TTY-safe interactive layout.

## Operator tools

- Digest-keyed coverage ledger and advisory coverage gaps.
- Read-only dossier over covered, non-stale categories.
- Report-time category-weight overrides without mutating raw artifacts.
- Repeatability report, empirical noise-band guidance, and `diff --noise-band`.
- Sensitivity planning/reporting, VRAM skip simulation, and read-only routing
  server over existing summaries.

## Safety boundaries

- No model deletion or automatic run scheduling.
- Read-only tools do not benchmark or refresh model data.
- Fixture privacy restrictions and archival rules keep public documentation and
  benchmark assets reviewable.
