# RC9 Context Truth, Capability Routing, and Repair Watch Audit

## Trigger

A real `deepseek-coder-v2:16b` repair demonstrated three separate correctness gaps:

1. GPU VRAM slope was treated as total KV-memory growth after Ollama began dynamic host offload. The resulting estimate decreased as context increased.
2. q8 and q4 both failed because quantized V cache required unavailable Flash Attention, while current/default KV completed the full 65k probe.
3. The repair watcher learned about a child run only after the phase ended, so it displayed the generic one-model renderer during the period the operator needed repair progress.

The final real-host evidence was: score 100, coverage 1.0, maximum verified effective context 64,773, with offload increasing from 0 at 16k to 0.364 at 32k and 0.523 at 65k.

## Corrections

### Offload-aware memory estimation

Measured GPU slope is accepted only while offload remains effectively zero. After offload changes, a hard pre-flight skip requires process-specific host-memory evidence, preferably PSS and then RSS. System-wide RAM delta is retained for model cards but is not trusted for hard skipping.

The estimator separates GPU and host components, enforces monotonic total estimates, and never projects total residency below observed GPU/process residency. Successful real probes supersede older speculative estimates.

### Per-context operating evidence

Each needle tier now stores best-effort:

- effective/requested context and retrieval result;
- prompt and output token counts;
- prompt throughput, decode throughput, TTFT, server phase durations, and wall time;
- GPU VRAM/utilisation/power/temperature;
- CPU utilisation/temperature;
- system RAM availability/peak/delta and swap;
- aggregate Ollama/llama-server RSS, PSS, swap, and process count;
- `/api/ps` model size, GPU-resident bytes, host-resident model bytes, context length, and offload fraction;
- exact/suspicious response diagnostics, repetition ratio, finish reason, and errors.

The ranking payload includes a 64k operating status. This is diagnostic, not a quality-score mutation. A successful needle proves retrieval at context; it does not by itself prove robust long-horizon agentic work.

### Current-first KV policy

Current/default KV runs before privileged service work. Sudo preflight, owner discovery, and q8/q4 mutation are lazy and occur only if current KV leaves guarded needle work unresolved.

Compatibility evidence is keyed to model digest and runtime identity. A quantized-V-cache failure requiring Flash Attention records `kv_quantization_requires_flash_attention`, prefers current KV, and blocks repeated q8/q4 attempts until relevant identity changes. q4 is not attempted when q8 proves the shared prerequisite is unavailable.

### Repair watcher

The child link is written before `runner.run()`. The watcher combines the parent repair phase with the child's live model/task/probe status. Standalone watch therefore follows current/default, q8, q4, and restore phases without falling back to the misleading generic child view.

### Capability resolver

Scored runs probe capabilities by default. Evidence combines operator profiles, runtime declarations, name hints, and functional probes. A probe that receives a valid response but misses the tiny semantic contract still routes the scored task, because that is quality evidence. Only definitive unsupported-build evidence excludes a lane. Ambiguous/transient probe failures are logged and withheld.

## Evidence boundaries

This release was built and validated without a live Ollama restart, GPU inference, or model benchmark. Real-host validation is still required for RAM/PSS availability, live per-depth watcher updates, current-first no-sudo behavior, and final ranking rescan.

## Validation requirements

```bash
python3 -m compileall -q llm_modelbench tests
python3 -m pytest -q
python3 -m llm_modelbench selftest
```

Then perform one operator-approved narrow current-KV probe and one repair-watcher smoke before broader fleet work.
