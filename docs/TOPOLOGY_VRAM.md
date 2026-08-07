# Topology-aware VRAM budgeting

`1.0.0rc21.post1` keys GPU memory evidence by stable physical GPU UUID, never a
CUDA ordinal. Each record carries PCI identity, installed capacity, live used
and free memory, an optional operator ceiling, selected-runtime reclaimable
residency, unrelated non-reclaimable residency, optional backend usable limit,
and derived effective-now/effective-after-reclaim capacity. `display_active` is
diagnostic evidence only; it never causes an automatic fixed reserve.

`memory.free` is already the immediate availability signal. Effective-now is the
minimum of available live-free, any policy ceiling, and an authoritative backend
usable limit. Selected-runtime residency may be represented separately as
reclaimable; unrelated processes are never reclaimable.

Placement includes known model weights, requested-context KV, runtime overhead,
and device overhead. Unknown components produce candidate/conditional results:
`single_gpu_fit`, `candidate_single_gpu_fit`, `multi_gpu_conditional_fit`,
`cpu_spill_required`, `confirmed_no_fit`, or `unknown`. The default operational
target remains 64K where supported. A suitable single GPU is always preferred;
only then is a full-layer/layer split proposed, biased toward the primary/faster
GPU. Tensor/row splitting is not the default.

For compatibility, `vram_budget_gb` remains accepted as an explicit manual cap
on a selected physical device. It is not silently interpreted as aggregate host
capacity. Optional `gpu_policy_ceilings_mib` maps UUIDs to caps; optional
`aggregate_policy_ceiling_mib` is an operator override (for example 26624 MiB
for a conservative 26 GiB aggregate), not a built-in fallback.

Runtime profiles may be explicitly configured as `ollama-gpu0`, `ollama-gpu1`,
`ollama-dual`, `llama_cpp-single-gpu`, or `llama_cpp-dual-gpu`; this release does
not create or modify services.
