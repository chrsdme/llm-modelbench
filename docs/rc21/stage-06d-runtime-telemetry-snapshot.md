# RC21 Stage 6D: Runtime telemetry snapshot

Stage 6D adds `RuntimeTelemetrySnapshot` in `llm_modelbench.runtime_telemetry` (schema version 1). It composes, rather than reparses, Stage 6B physical inventory/live samples and Stage 6C process, compute-process, reconciliation, and attribution results.

Collection is explicit and read-only: process discovery before, physical inventory/live sampling, one fixed compute-process query, process discovery after, reconciliation, then attribution. Importing the module performs no process, procfs, GPU, or network access. Every seam is injectable for fixtures.

Snapshots retain partial evidence: unavailable endpoints, idle compute rows, failed NVIDIA queries, incomplete socket evidence, and missing profile declarations remain structured evidence rather than synthetic placement. UUID is always physical identity. Declared UUIDs, observed runtime UUIDs, declared-only UUIDs, and observed-but-undeclared UUIDs remain separate; CUDA ordinals, indices, command-line order, tensor splits, and placement flags are never converted to UUIDs.

The snapshot serializes deterministically with bounded errors and no environment, credential, or raw procfs content. Stage 6D is implemented and fixture-tested; real-host batch acceptance remains Stage 6G work.

Consolidated hardening rejects non-canonical UUID-like values (including CUDA ordinals), duplicate physical-inventory UUIDs, unknown completeness indicators, and observed UUIDs that contradict a non-empty inventory. An unavailable inventory remains valid partial evidence, so it never fabricates placement. Stage 6G host execution is prepared at `/tmp/llmb-rc21-stage6g-acceptance.sh` and remains pending.

The first Stage 6G host attempt at `/tmp/llmb-rc21-stage6g-acceptance-20260731T224056Z` exposed duplicate aggregate `proc_fd` evidence, endpoint-port ambiguity conflated with incomplete socket evidence, and an Ollama compute worker that needed bounded lineage proof. Snapshot-level errors now deduplicate exact normalized records while nested discovery evidence remains intact. Stage 6G remains pending a replacement-script rerun.

The later evidence root `/tmp/llmb-rc21-stage6g-acceptance-20260731T231022Z` passed pure snapshots, the fixed NVIDIA query, doctor, old-artifact compatibility, and repository non-mutation, but exposed timestamp-only process ambiguity. Observation timestamps are now excluded from substantive stable-identity comparison and the later timestamp is retained deterministically. Stage 6 remains incomplete and Stage 6G remains pending.

Stage 6G passed at `/tmp/llmb-rc21-stage6g-acceptance-20260801T001221Z`. Its pre-run compute-process query was empty, producing valid idle-at-capture observed/attribution sets; later run metrics are not process attribution. Stage 6 is complete; Stage 7 has not started.
