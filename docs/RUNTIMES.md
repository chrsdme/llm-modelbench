# Runtime architecture

LLM ModelBench treats runtimes as external, operator-managed evidence sources.
It supports Ollama and an external llama.cpp/`llama-server` endpoint through a
backend-neutral client boundary. A profile records a name, backend type,
endpoint, provenance, and optional physical GPU identities.

## Selection and identity

Use `runtime discover` to inspect bounded local candidates, `runtime save` to
store a profile, and `runtime select` to make the choice explicit. A non-mock
unattended operation with multiple viable candidates refuses before model work.
Run and campaign evidence record runtime identity; resume refuses when the
selected backend, endpoint, adapter identity, or GPU identity differs.

Endpoints must be HTTP(S), include a host, and contain no credentials. Local
discovery is bounded to local process and endpoint evidence; it does not scan
LAN hosts or persist raw process command lines or environments.

## Backend boundaries

Ollama supports inventory and its existing repair/KV workflow. llama.cpp uses
the externally running server's documented HTTP interface and is intentionally
one-model-at-a-time unless switching capability is proven. ModelBench never
starts, stops, restarts, or reconfigures `llama-server`. Service control and
KV repair report unsupported for llama.cpp rather than attempting an emulation.

## Hardware and telemetry

NVIDIA physical devices are identified by UUID, not CUDA ordinal. Multi-GPU
inventory distinguishes installed capacity from sample-time available capacity
and does not infer placement from aggregate memory. Process and runtime
telemetry are bounded, best-effort evidence: unavailable permissions or host
facilities are reported without changing quality scores or canonical rankings.

Telemetry is diagnostic. It does not prove GPU placement, mutate runtimes, or
authorize model work. See [SAFETY.md](SAFETY.md) for execution and privileged
Ollama repair boundaries.
