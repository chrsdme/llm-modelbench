# RC21 Stage 6: Live acceptance and closeout

Stage 6A audited the legacy boundary; 6B added UUID-keyed physical GPU samples; 6C added bounded process discovery and attribution; 6D assembled `RuntimeTelemetrySnapshot` schema version 1; 6E persisted one optional pre-run run artifact; and 6F exposed bounded report and doctor summaries. UUID is canonical physical identity, PCI is corroboration, and index/CUDA ordinal are never physical identity. Field states and bounded errors retain unavailable, unsupported, malformed, failed, and idle evidence without inventing placement.

Process evidence uses PID plus start ticks, exact endpoint-port ownership, and before/query/after reconciliation. Observation timestamps are not stable identity; substantive metadata conflicts and PID reuse remain rejected. Ollama workers require bounded, loop-safe lineage to the unique Ollama endpoint owner. Incomplete `proc_fd`/socket evidence remains partial and prevents confirmed attribution. A standalone llama-server is not treated as an Ollama worker.

The runner collects one bounded pre-run snapshot into optional `runtime_telemetry.json`; rows contain only its relative reference and schema version. Reports and doctor are observational and backward compatible with absent, corrupt, or future optional artifacts. Neither surface proves layer, tensor, KV-cache, or offload placement. Stage 6 has no continuous monitoring or in-flight placement proof.

## Acceptance

Authoritative acceptance passed at `/tmp/llmb-rc21-stage6g-acceptance-20260801T001221Z`. The fixed compute-process query, known UUID validation, deterministic Ollama/llama.cpp snapshots, no-CUDA-ordinal check, repository non-mutation, report, doctor, and old-artifact compatibility all passed. The accepted command was:

```text
./llmb run --runtime-profile legacy-ollama --out /tmp/llmb-rc21-stage6g-acceptance-20260801T001221Z/runs --run-id stage6g --models hermes3:8b --tasks json_extract --level smoke --no-auto-probe --no-fingerprint --no-ranking-update --live-ui off --yes
```

It exited 0, wrote `runs/stage6g/raw_results.jsonl` and readable `runtime_telemetry.json`, and the row referenced `artifact=runtime_telemetry.json`, `schema_version=1`, `status=collected`. `json_extract` scored 100.0 and the benchmark result remained valid. The pre-run compute-process query returned zero rows, so observed UUIDs and runtime attributions were empty: valid idle-at-capture evidence, not telemetry failure. Later benchmark VRAM/power metrics and doctor’s one loaded model/GPU use are not retroactively converted into process attribution. Doctor conservatively reported compute query supported/idle, socket evidence partial, and schema readable.

Two earlier harness attempts were corrected, not production failures: `/tmp/llmb-rc21-stage6g-acceptance-20260731T224056Z` omitted explicit runtime selection and correctly failed noninteractive ambiguity; `/tmp/llmb-rc21-stage6g-acceptance-20260731T231022Z` selected protected `py_anagram` and correctly hit the host-code gate. No override was added; the passing command uses safe `json_extract`.

The batch also corrected exact endpoint-port selection, timestamp-independent stable identity, bounded worker lineage, canonical UUID/inventory validation, atomic artifact writes, optional artifact compatibility, deduplicated snapshot summaries, and report/doctor state distinctions. Final validation: focused `75 passed`, full `734 passed`, compileall, selftest, release check, diff check, import side-effect guard, and secret review passed.

Stage 6A–6G is complete and ready for Stage 7 planning/review. Stage 7 has not started.
