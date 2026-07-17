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

## Read-only versus execution commands

`report`, `export-review`, `repeat-report`, `diff`, `coverage`, `gaps`, `dossier`, `sensitivity-report`, `simulate`, and `serve` read existing artifacts. `rankings` reads/rebuilds artifact databases. `inventory`, non-mock `plan`, and `doctor` may inspect Ollama metadata. `inventory/plan --auto` make small capability-probe model calls. `judge-dumps` calls only the selected judge model. `run` and `wizard` execute benchmark models.

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

`llmb repair --kv-cascade --restart-ollama` is the only benchmark path allowed to mutate a system service. It is opt-in, TTY-only, and limited to a dedicated Ollama systemd drop-in containing one environment variable. RC7 first discovers the unit that owns the configured Ollama port, or verifies an explicitly supplied unit, and refuses stale UUID-based CUDA bindings. The operator must type `DISCOVER` or `VERIFY` before privileged identification and `RESTART` before q8, q4, and restoration phases. Sudo handles the password directly; the benchmark never reads or stores it. Systemd's merged environment is checked before restart, the replacement process is checked after restart, and the original managed drop-in state is restored by default, including after a phase error when restoration remains possible.
