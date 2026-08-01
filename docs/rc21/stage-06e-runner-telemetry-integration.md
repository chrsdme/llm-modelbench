# RC21 Stage 6E: Runner telemetry evidence

The CLI run path requests one bounded Stage 6D snapshot before benchmark execution for real Ollama and llama.cpp adapters. The artifact is written as optional `runtime_telemetry.json`; result rows retain only a namespaced reference and schema version under `runtime_telemetry`.

Telemetry is optional evidence. Collection failure writes bounded unavailable evidence where possible and never changes prompts, generation parameters, inference count, timeout policy, score calculation, ranking, lifecycle actions, or benchmark result preservation. Older artifacts without this optional file or row field remain readable.

The artifact is serialized with `allow_nan=False` through an atomic same-directory replacement. A failed write produces no row reference, avoiding a partial artifact reference. Report readers treat missing, corrupt, oversized, and future-schema artifacts as bounded unavailable evidence rather than failing an unrelated report.

The first Stage 6G runner attempt failed before benchmark work because its script omitted explicit runtime selection in a noninteractive two-runtime environment; no telemetry artifact was therefore expected. The replacement script passes `--runtime-profile legacy-ollama`, records its exact argv and bounded diagnostics, and discovers only the requested `--out`/`--run-id` directory.

The subsequent attempt at `/tmp/llmb-rc21-stage6g-acceptance-20260731T231022Z` correctly refused `py_anagram` before model execution because it is a protected host-code task. The next script uses smoke-level `json_extract`, which is JSON-schema scoring and does not require `--allow-host-code-execution`.

The passing Stage 6G run at `/tmp/llmb-rc21-stage6g-acceptance-20260801T001221Z` wrote a readable schema-1 artifact and row reference, preserved a valid 100.0 `json_extract` result, and did not change repository state. Stage 6 is complete; Stage 7 has not started.

No model is loaded, unloaded, started, stopped, repaired, or reconfigured for telemetry. Stage 6E is complete; Stage 7 has not started.
