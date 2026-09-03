# Safety Boundaries

## Model execution

Coding and file-operation tasks can execute model-generated code. Normal `run`
commands fail closed when such tasks are selected unless the operator supplies
`--allow-host-code-execution`. The scorer then uses a temporary directory, isolated
interpreter flags, a scrubbed environment, resource limits, and timeouts. These
controls are not a security boundary. Use the flag only inside a disposable
container or VM.

Do not start a benchmark, run model prompts, pull/delete models, or conduct a
broad fleet run without explicit operator approval. Documentation, reporting,
and cleanup work do not imply permission to execute models.

## Managed runtime child processes

For a benchmark it owns end to end, ModelBench may spawn its own ephemeral
`llama-server` child process and tear it down when the work completes. Such a
child is launched from an argv list (never a shell), bound to localhost, and
proven by PID plus `/proc` process-start-time identity before any teardown
signal; teardown is graceful then forced, re-proving ownership before each
signal, and never uses name-, port-, or process-group-based killing. This is
not a persistent daemon and not a system service. Ollama is reuse-only:
ModelBench never spawns, restarts, stops, or reconfigures Ollama, and it never
signals or reconfigures any runtime process it did not itself launch. The
privileged Ollama restart boundary below remains the *only* path that mutates a
system service.

## Read-only versus execution commands

`report`, `export-review`, `repeat-report`, `diff`, `coverage`, `gaps`, `dossier`, `sensitivity-report`, `simulate`, and `serve` read existing artifacts. `rankings` reads/rebuilds artifact databases. `runtime discover`, `inventory`, non-mock `plan`, and `doctor` inspect the selected local runtime or host metadata. `inventory/plan --auto` make small capability-probe model calls. `judge-dumps` calls only the selected judge model. `run` and `wizard` execute benchmark models.

## Ordinary operation: no sudo

Ordinary benchmarking, diagnostics, GPU/runtime discovery, telemetry, scoring,
rankings, reporting, Ollama API use, and external llama.cpp/`llama-server` use
have zero sudo involvement. Root is not an installation or normal-operation
requirement for LLM ModelBench.

## Results and review artifacts

Raw rows, subjective outputs, review packs, and generated reports may contain
model output. Inspect them before sharing. Retrieval diagnostics intentionally
store identifiers, ranks, and similarities rather than full query or document
text.

## Source-control safety

Commit locally, review changes, and tag only after release acceptance. Use protected branches where available. Never force-push a protected release branch. Public users should clone over HTTPS; maintainers may use authenticated SSH remotes.

## Fixture privacy

Public fixtures must follow [PRIVACY_FIXTURES.md](PRIVACY_FIXTURES.md). Do not
copy private operator material or unreviewed audit evidence into tasks, docs,
or review packs.


## Privileged Ollama restart boundary

`llmb repair --kv-cascade --restart-ollama` is the only benchmark path allowed to mutate a system service. It is opt-in and routes all mutation through the root-owned `llmb-ollama-kv-control` broker; the Python application cannot supply a unit, path, fragment, command, or environment value. The broker discovers and revalidates the Ollama listener owner, writes only its fixed one-variable drop-in atomically, and retains root-private recovery state. The operator must type `DISCOVER` and `RESTART` for supervised phases. Auto-confirm uses `sudo -n` only. Sudo handles the password directly; the benchmark never reads or stores it. This pathway is unsupported for llama.cpp/`llama-server`.

The bounded flow is: ModelBench -> `sudo llmb-ollama-kv-control` -> constrained
temporary Ollama KV configuration -> restart and verification -> eligible retry
-> restoration. The broker accepts only a bounded port, transaction token, and
the exact KV enum; it cannot receive an arbitrary service unit, filesystem
path, systemd content, command, executable, or environment value.
