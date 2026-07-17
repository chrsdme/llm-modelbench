# Capability routing and model selection

LLM ModelBench separates **model selection** from **test selection**.

## Model selection

```bash
llmb run --models 'qwen2.5-coder:14b;qwen2.5-vl:7b' ...
llmb run --all ...
llmb run --select ...
```

- `--models` accepts exact installed Ollama names separated by semicolons.
- `--all` explicitly selects every model returned by `ollama list`. This is also the default when no model selector is supplied.
- `--select` opens an interactive **model-only** selector. It does not change level, category or task filters.
- There is deliberately no redundant `-all` alias.

Use `llmb wizard` when both models and test scope should be edited interactively. The wizard selects models, probes capabilities, proposes the routed tasks, and lets the operator edit level, categories, exact task IDs and judge mode before accepting.

## Evidence order

Routing uses, in order:

1. explicit `MODEL_PROFILES` operator overrides;
2. Ollama-declared capabilities;
3. opt-in functional probes (`--auto` or the wizard);
4. conservative name hints only when stronger evidence is absent.

Known fleet conversions with repeatedly incomplete Ollama metadata are kept as narrow explicit profiles. This currently includes InternVL3-8B, Garnet-OCR-7B and VL-1-Coder, whose community GGUF builds may report only `completion` even though they are visual models.

Hybrid models keep every supported lane. Current families are `text`, `vision`, `embedding`, `tools`, and `insert`. Display class and executable families are related but not identical: for example, a reasoning VLM can remain in the reasoning class while receiving both text and vision tests. An embedding-only route always wins over an incompatible name-based display class.

## Functional probes

`--auto` makes small pre-benchmark model requests:

- exact text response;
- OCR of a token present only inside a generated image; the probe also runs for conservative vision-name candidates even when Ollama returns partial `completion`-only metadata;
- embedding vector shape;
- native structured tool call (the tool is never executed);
- suffix-conditioned FIM completion for declared/coding candidates.

A failed weak name hint is removed. A failed operator-declared or Ollama-declared capability remains routed and is recorded as a warning so the real task failure remains visible.

## RC9 evidence policy

Actual `run` commands perform functional capability probes by default. Use
`--no-auto-probe` only when deliberately accepting metadata-only routing.
Read-only `plan` remains metadata-only unless `--auto` is explicit.

Capability decisions combine:

1. operator profiles;
2. Ollama declarations and model metadata;
3. conservative name hints as weak evidence;
4. functional endpoint responses;
5. task-equivalent probes where the generic label is insufficient, including
   `fim_suffix_assertion`;
6. persisted current-build capability repair evidence.

A successful endpoint response that gives the wrong tiny-probe answer is not
classified as capability-unavailable. The scored task is routed so the failure is
recorded as model quality. Only definitive unsupported-build evidence excludes a
lane. Transient or ambiguous probe failures are withheld and logged for review.
