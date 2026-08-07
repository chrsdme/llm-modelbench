# RC21 Stage 5: External llama-server Backend

## Scope and architecture

Stage 5 adds `llm_modelbench.llama_cpp.LlamaCppClient` and `LlamaCppBackendAdapter`, selected only after the existing runtime-profile policy chooses a healthy `llama_cpp` profile. The adapter uses standard-library HTTP and only an already-running endpoint. It never starts, stops, signals, reconfigures, loads, unloads, flushes, or switches the server model, and never sends `POST /props`.

The client bounds JSON responses at 4 MiB, reads one byte past the bound, and returns concise endpoint, timeout, status, JSON, shape, and size errors. Error bodies are capped at 8 KiB. No adapter error retries through Ollama.

## Captured server contract

Operator preflight captured llama-server `10086 (66e4bf7e5)` with `build_info` `b10086-66e4bf7e5`, one slot, served context `65536`, training context `262144`, and model metadata for one Q4_K - Medium, 27,320,697,856 parameter, 16,799,719,424 byte GGUF. The served ID and alias were the same `sha256-b0651e28555bde7d2459ce99f091319b1a547143463e8d49f2aa7f572675fe67` path. The committed tests use generic fixture paths derived from those responses, not host-local evidence.

`/v1/models.data` is authoritative whenever non-empty. The compatibility `models` array is used only when `data` is absent or empty; it is never concatenated with authoritative rows. Compatibility details merge only for an exact served ID or explicitly reported alias. Exactly one unique served model is required; zero-model and router/multi-model results fail closed. Requests must match the served ID or a reported alias. A digest is emitted only when a `sha256-<64 hex>` string is actually reported.

## Capability and request mapping

Supported endpoint facts are inventory, build/version, observed metadata, non-streaming chat, tokenization, slots, and bounded read-only metrics text. Chat uses `POST /v1/chat/completions` with `stream: false`, model, system/message list, temperature, seed, `max_tokens`, optional stop, and thinking control only when the reported template exposes it. Explicit `think=on|off` is sent as `chat_template_kwargs.enable_thinking`, never as a top-level request field. Supplied messages replace the simple prompt list; when both forms provide a system message, the first supplied system message wins and no duplicate is inserted. `num_ctx` is validation-only: values above active server context fail before inference, and accepted values do not change the server.

JSON mode maps to OpenAI `response_format: {"type": "json_object"}`; a raw schema maps to `json_schema`, while an already OpenAI-shaped response format is forwarded unchanged. Schema validation remains the existing downstream scorer responsibility. `message.reasoning_content` is retained separately; think-tag reasoning is also retained without removing the original content. Tool calls use reported template support for both tools and tool calls, preserve observed IDs/types/function data, normalize object or JSON-string arguments, and never execute a returned call.

`POST /tokenize` validates the returned integer token array and does not generate text. Usage and observed llama.cpp timing fields normalize prompt/completion counts, throughput, durations, finish reason, and length truncation only when returned. Global metrics are not used for per-request attribution.

## Unsupported boundaries

Vision, embeddings, suffix/FIM generation, unload, flush-all, Ollama service repair, Ollama KV repair, and unreported offload placement are unsupported or unavailable. Loaded-model evidence is limited to served ID, size, active context, and slot state. The adapter does not infer GPU layers, tensor splits, VRAM residency, or offload fraction.

One endpoint serves one model. A requested judge model must be that ID or alias; no model switching and no Ollama fallback occurs. Separate judge-runtime profile selection is deferred to a later stage.

## Validation and acceptance

Focused fixture validation:

```text
python3 -m pytest -q tests/test_llama_cpp.py tests/test_runner_lifecycle.py tests/test_backend.py tests/test_runtime_profiles.py tests/test_cli_subcommands.py tests/test_capability_workflow.py tests/test_repair.py
136 passed
```

Full validation passed:

```text
python3 -m compileall -q llm_modelbench tests
python3 -m pytest -q
613 passed
./llmb selftest
SELFTEST: ALL GOOD
python3 tools/release_check.py
RELEASE CHECK: PASS
git diff --check
passed
```

Direct operator acceptance on the AI-PC completed at `/tmp/llmb-rc21-stage5-live-20260730T221916`: one served model, build `b10086-66e4bf7e5`, active context `65536`, training context `262144`, one slot, tokenization, and direct timing evidence all passed. `think=off` returned no separate or visible reasoning; structured JSON returned `{"status":"ok","value":5}`; and a harmless `stage5_probe` tool call was normalized without execution. llama-server was healthy and idle afterward.

The first normal runner attempt exposed an unconditional `flush_all()` before generation. Runner and reachable generic repair lifecycle calls are now capability-gated before invocation. Unsupported external lifecycle skips are backend routing, not model-quality or harness failures; supported Ollama, Mock, and legacy direct clients retain their historical lifecycle behavior. Stage 5 does not erase llama.cpp slot cache or unload the served model: that remains operator-managed. No lifecycle or model-management operation was added.

Normal AI-PC runner acceptance then completed at `/tmp/llmb-rc21-stage5-runner-20260730T234849`. Run `rc21_stage5_runner_20260730T234849` selected one served model and `json_extract`, exited `0`, scored `100.0`, was valid, and measured approximately `23.1 tok/s`. The prompt-token metric changed from `380` to `455`; predicted-token metric changed from `59` to `88`. llama-server was healthy and idle afterward. Ollama remained unchanged with 57 installed models and zero loaded models, the canonical inventory digest remained `5f553c1450d7f2b1c3e52010bcd0db205a63a9ec42bf037072d795b7972a933c`, and isolated temporary profile stores ended empty.

## Human-review corrective iteration

Human review found that the first Stage 5 implementation combined differently named `data` and compatibility inventory rows, invented a GGUF format value, could advertise vision without image mapping, and left malformed nested responses and command errors insufficiently contained. The corrective iteration made `data` authoritative, emits only observed fields, fixes vision as unsupported, validates every used endpoint shape, rejects redirects, preserves bounded response/error sizes, and converts expected selected-llama.cpp failures to concise CLI exits without an Ollama retry.

Fixture transports record method, path, and payload. Tests assert the complete read-only allow-list: `GET /health`, `/v1/models`, `/props`, `/slots`, `/metrics`; and `POST /v1/chat/completions`, `/tokenize`. Unsupported lifecycle operations issue zero requests. The corrective tests do not claim live health, inference, thinking, JSON/schema, tool, tokenization, or timing acceptance. Stage 6 has not started.

### Follow-up review corrections

Explicit `think="on"` and `think="off"` now map only to `chat_template_kwargs.enable_thinking`; no top-level `enable_thinking` is sent. Caller template kwargs must be an object, are merged deterministically, and cannot contradict explicit ModelBench thinking mode. `think="auto"` adds no thinking override.

Unsupported suffix generation, embeddings, offload inspection, unload, and flush-all are transport-free on a cold client cache. Message validation now restricts roles, system placement, and content to the captured non-typed-content contract. Props validates `build_info`, `total_slots`, and interpreted context fields; slots validates every consumed field; tool calls validate ID/type/function/arguments; and forwarded OpenAI response formats are validated before a chat request. Interpreted integer request and endpoint fields reject Python booleans, including context/token controls, model size/parameter/context metadata, slot IDs/contexts, usage counts, and timing token counts. Fixture coverage includes bounded 500 bodies, redirects, malformed shapes, and selected-llama.cpp no-fallback containment. Fixture tests do not substitute for the completed operator acceptance above. Stage 6 remains not started.

Rollback is confined to removing the llama.cpp module and `_client()` llama.cpp construction branch; Ollama and Mock adapters retain their Stage 3 behavior.

## Known limitations

- One served model is supported per llama.cpp profile; router and multi-model endpoints are unsupported.
- The adapter cannot switch models, flush llama.cpp cache, or unload the served model; external lifecycle remains operator-managed.
- Embeddings and vision/OCR expansion are unsupported, and there is no separate judge runtime profile.
- Offload fraction and per-GPU placement are unavailable.
- Runtime identity is not yet frozen into campaign or report schemas. Existing capability evidence may retain legacy source terminology such as `ollama_metadata` until the later report/campaign integration stage.
- Stage 6 has not started.

## Proposed Stage 6 objective

Add backend-neutral per-GPU and server-process telemetry keyed by physical GPU identity, without changing benchmark scores or runtime lifecycle behavior.
