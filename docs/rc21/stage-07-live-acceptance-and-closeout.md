# RC21 Stage 7: Live acceptance and closeout

Stage 7 is complete. It adds a conservative runtime-fit diagnostic only; it does not select a runtime, start or reconfigure a service, load or remove a model, issue inference, alter benchmark/campaign semantics, or update reports, rankings, scoring, or repair behavior. Stage 8 remains not started.

## Scope and architecture

`llm_modelbench.runtime_fit` separates a pure evaluator from an opt-in, read-only collection wrapper. The pure evaluator accepts supplied physical inventory, live telemetry, backend-declared model metadata, and an invocation-local `RuntimeFitProfile`; it performs no subprocess, filesystem, procfs, or network work. The collection wrapper reads existing bounded inventory/live-telemetry and backend metadata interfaces only. `./llmb runtime-fit` requires an explicit selected runtime profile (except its deterministic offline mock path), reads model metadata, and emits either a concise assessment or deterministic JSON. It makes no generation request and exposes no lifecycle or host-code-execution option.

The version-1 result preserves a model-size lower-bound provenance, UUID-keyed device assessments, installed capacity, sample-time live-free capacity, reserve, KV evidence, reasons, bounded errors, and one of these fit states:

- `confirmed_fit`: all mandatory modeled requirements are known and fit;
- `candidate_fit`: the known weight lower bound fits, but mandatory overhead or KV evidence remains unknown;
- `conditional_fit`: an explicit condition such as layer-split overhead uncertainty or permitted CPU/RAM spill remains;
- `confirmed_no_fit`: a reliable model-weight lower bound exceeds reserved per-device capacity; or
- `unknown`: capacity, requirement, or required multi-device allocation evidence is absent.

Reason codes make the conservative condition explicit, including `lower_bound_exceeds_capacity`, `unknown_runtime_overhead`, `unknown_architecture`, `explicit_multi_device_strategy_required`, `per_device_allocation_unknown`, `explicit_layer_split`, `per_device_overhead_unknown`, `unsupported_tensor_split_memory_model`, and `cpu_spill_required_or_permitted`.

## Capacity and identity policy

NVIDIA UUID is the canonical physical identity. PCI bus ID is corroborating hardware evidence; physical index and CUDA ordinal are never runtime-fit identity. The evaluator supports arbitrary-N inventory and rejects malformed or duplicate canonical UUIDs, profile UUIDs absent from authoritative inventory, and allocation maps that do not exactly cover the declared UUIDs.

Installed physical capacity (`GPUDevice.total_vram_mb`) and current live-free capacity (`GPUSample.memory_free_mib`) are distinct fields and are never conflated. The deterministic default reserve is 512 MiB per device and can only be changed by an explicit CLI option. A single device receives a lower-bound assessment against reserved live-free capacity when available, otherwise reserved installed capacity.

For multiple devices, summed VRAM is informational only. It never authorizes a fit. A multi-device decision requires an explicit supported strategy plus UUID-keyed allocation weights; positional weights are rejected. Explicit `layer_split` evaluates every allocated physical device and stays conditional while per-device runtime overhead distribution is unknown. Tensor/row placement is not modeled and therefore stays unknown. Neither an assessment nor telemetry claims layer, tensor, KV-cache, or offload placement. CPU/RAM spill is a declared policy that produces a conditional result, never a silent fit.

Model-size evidence from `model_size_bytes()` is backend-declared storage/tag metadata and is preserved as a lower-bound provenance; it is not presented as measured loaded VRAM. KV cache is derived only when requested context and complete architecture inputs are available: layer count, KV-head count, head dimension, KV dtype bytes, and optional parallel-sequence count. Missing or invalid metadata remains unknown, never zero; context above a known model maximum is not silently clamped. Runtime overhead follows the same unknown-not-zero policy.

## Authoritative real-host acceptance

The read-only acceptance evidence is at `/tmp/llmb-rc21-stage7-acceptance-20260802T135500Z`. `acceptance-summary.json` reports `passed`. Its required evidence set is present and its checks confirm canonical NVIDIA UUID identity, no ordinal/index identity, distinct installed and live-free capacity, no aggregate multi-GPU authorization, per-device explicit layer-split evaluation, unknown overhead, explicit spill policy, deterministic finite JSON, no placement claims, and unchanged repository/runtime state.

`preflight.json` records `inference=false`, `runtime_lifecycle_operations=false`, `model_mutation=false`, `configuration_mutation=false`, and `host_code_execution=false`. The before/after git-status files are identical, as are the bounded runtime-state snapshots. The actual physical inventory contains the RTX 5060 Ti UUID `GPU-78077308-f4b2-3330-6d4e-19581d7b1511` and RTX 3060 UUID `GPU-5b99bce2-35ab-f6db-857b-72162069fa72`; each has separate installed and sample-time live-free evidence.

Ollama 0.32.5 was healthy but had zero installed models: `ollama list` and both bounded `/api/tags` snapshots contain `{"models":[]}`. The external llama.cpp `/v1/models` evidence contains `{"data":[]}`, and there is no configured UUID-bound llama.cpp runtime profile or explicit two-GPU layer-split profile. Consequently, the small installed model, near-or-above-12-GiB model, larger-than-either-single-GPU model, external llama.cpp model fit, configured real-host layer split, and real-model CLI human/JSON scenarios are correctly `environmentally_unavailable`. This is not a Stage 7 implementation failure: the real-host CLI model path had no model identifier or metadata source. Focused offline tests cover CLI behavior, and no model was installed merely to manufacture acceptance evidence.

## Arbitrary-N synthetic acceptance

The isolated pure-evaluator fixture in `arbitrary-n-synthetic-fit.json` uses three synthetic UUIDs only; it does not describe the two real host devices. It validates canonical UUID-keyed arbitrary-N ordering, a 3:3:1 explicit layer allocation, per-device lower-bound weight allocation and reserves, and `conditional_fit` while per-device overhead remains unknown. It separately shows that three-device aggregate capacity without a strategy is `unknown`, an undersized device yields `confirmed_no_fit` from a reliable lower bound, and an explicit spill policy is conditional. It makes no layer, tensor, KV-cache, or offload-placement claim.

## Validation and readiness

Offline fixture coverage includes UUID/invariant rejection, capacity and reserve boundaries, model-size provenance, KV/context derivation, single-device decisions, explicit layer split, spill policy, deterministic serialization, and CLI isolation. Final closeout validation passed: the focused Stage 7 evaluator/CLI suite had `9 passed`; the full suite had `743 passed`; compileall, selftest, release check, diff check, import-side-effect guard, and added-lines secret/environment-value review passed.

Known limitations are deliberate: a declared storage-size lower bound is not loaded-VRAM measurement; unknown runtime overhead/KV distribution is not estimated; unsupported tensor/row placement is not inferred; and real-model assessment requires an already available selected-runtime model and metadata. Stage 7 is complete and ready for Stage 8 planning/review only. Stage 8 has not started.
