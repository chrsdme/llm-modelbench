# LLM ModelBench 1.0.0rc11 Pre-Rankings-V3 Audit

## Purpose

RC11 closes the observation and evidence-contract blockers before Rankings V3. It does not redesign rankings presentation.

## Implemented

### Deterministic watcher replay

`llmb simulate repair-watch` emits deterministic repair campaigns using the same on-disk parent, child, and `repair_link.json` contracts as live repair. Scenarios cover capability repair, current-KV needle repair, q8/q4 cascade, interrupted child, and failed child.

### Repair dashboard

The repair renderer presents campaign/action progress, phase, model, task, context tier, child state, elapsed time, prompt/decode throughput, VRAM, RAM/process residency, swap, offload, service/KV evidence, and final outcome. It avoids falling back to generic model-count and unknown-context fields when campaign evidence exists.

### Controlled context profile

`context-profile` runs one needle-only model profile up to the requested target context, validates required telemetry, refreshes rankings, and regenerates operating cards. The row is marked diagnostic so it can enrich operating evidence without replacing canonical quality evidence.

### Model cards

`model-cards` generates standalone JSON and Markdown cards containing quality status, capability/recovery limits, long-context depth results, speed, GPU and host residency, offload, KV compatibility, and behavior warnings.

### Semantic cleanup

`recovery_limited` is now reserved for exhausted recovery or ineffective-thinking evidence. A responding capability gate followed by a measured zero-quality task is represented by `capability_measured_failure` and does not falsely claim retry exhaustion.

### Freeze

`freeze` records source hashes, task-contract hashes, ranking counts and per-model expectations, selected ranking artifacts, and SHA-256 checksums. This becomes the comparison anchor for Rankings V3.

## Regression evidence

The DeepSeek Coder V2 dynamic-offload profile is frozen as a fixture:

```text
4k:  10176 MiB VRAM, 0.000 offload
16k: 13754 MiB VRAM, 0.000 offload
32k: 15278 MiB VRAM, 0.364 offload
65k: 15526 MiB VRAM, 0.523 offload
```

The estimator must not treat GPU-VRAM slope as total resident-memory slope after offload changes.

## Offline validation target

```text
compileall: passed
pytest: 361 passed
selftest: SELFTEST: ALL GOOD
version: llm-modelbench 1.0.0rc11
```

## Real-host validation still required

Sandbox validation cannot prove live Ollama telemetry. One approved `context-profile` run is required on the RTX 5060 Ti host. It must confirm 64k telemetry fields and produce the operating card before the pre-V3 freeze is considered authoritative.
