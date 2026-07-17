# LLM ModelBench 1.0.0rc12 Pre-Rankings-V3 Truth Audit

## Purpose

RC12 corrects three host-proven RC11 defects before Rankings V3: misleading watcher state, a controlled 64k profile blocked by speculative memory accounting, and checksum verification that depended on the caller's working directory.

## Watcher truth

- Repair action completion and campaign/fixture lifecycle are reported separately.
- Linked repair children are collapsed during queue discovery.
- Repair restore state retains the last verified KV mode and reports the restored current/default state.
- Context profiles use a dedicated renderer in both standalone `llmb-watch` and the runner's inline UI.
- The context renderer shows the active tier, completed tier history, telemetry, behavior probe, and validation outcome instead of generic model/exclusion counters.

## Controlled 64k profiles

- Preflight estimates are advisory for the explicitly approved diagnostic profile.
- Dynamic-offload, PSS/RSS, and system-RAM slopes remain diagnostic and cannot hard-skip tiers.
- A live available-RAM floor remains a hard safety gate.
- Reusing a non-empty run directory is refused.
- A synthetic behavior probe records exact anchor recall, required ordering, repetition/shape warnings, TTFT, prompt/decode speed, and resource telemetry.
- The behavior probe explicitly records `agentic_readiness: not_assessed`.

## Freeze verification

Each freeze contains a repository-root-compatible `SHA256SUMS.txt`, a snapshot-local `SHA256SUMS.local.txt`, and a portable `VERIFY.sh`. `llmb freeze --out <snapshot> --verify` verifies independently of the current working directory.

## Offline validation target

```text
compileall: passed
pytest: 370 passed
selftest: SELFTEST: ALL GOOD
version: llm-modelbench 1.0.0rc12
```

No GPU, Ollama, sudo, or systemd operation is required for offline validation.
