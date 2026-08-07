# RC21 Stage 7 audit and plan

Stage 7 is an additive, read-only diagnostic lane. `RuntimeProfile` currently records backend, endpoint, and declarative physical GPU UUIDs; it has no persisted layer/tensor split, device-order, KV-distribution, or CPU-spill semantics. Consequently, aggregate VRAM never authorizes a multi-GPU fit decision. A Stage 7 request must explicitly declare a supported strategy before evaluating one.

`GPUDevice.total_vram_mb` is installed physical capacity. Stage 6B `GPUSample.memory_free_mib` is sample-time free capacity. These are reported separately. NVIDIA UUID is canonical identity; PCI corroborates it; physical index and CUDA ordinal are never runtime-fit identity.

Backend `model_size_bytes()` obtains an Ollama tag size or llama.cpp metadata size. It is declared/storage-size evidence, not measured loaded VRAM. Stage 7 calls it a lower-bound requirement and preserves provenance. Current generic model metadata provides context length but does not reliably provide all KV architecture inputs; KV remains unknown unless a caller supplies layer count, KV-head count, head dimension, dtype bytes, and parallel sequence count.

Stage 7 deliberately does not change model routing, benchmark behavior, reports, campaigns, rankings, repair, or service lifecycle. Stage 8 remains the possible integration boundary.

Implementation, fixture validation, and read-only real-host acceptance are complete. The authoritative evidence at `/tmp/llmb-rc21-stage7-acceptance-20260802T135500Z` confirms canonical UUID identity, separate installed/live-free capacity, conservative multi-device handling, deterministic JSON, and unchanged repository/runtime state. The host had no Ollama or llama.cpp model metadata and no UUID-bound llama.cpp profile, so optional real-model and configured-split scenarios are explicitly `environmentally_unavailable`, not failed. See `stage-07-live-acceptance-and-closeout.md` for the acceptance record and limitations. Stage 8 has not started.
