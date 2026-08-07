# Repository Guidance

## RC21 guardrails

- Keep `OllamaClient` and its existing behavior first-class while introducing backend-neutral boundaries.
- Treat all configured/running runtimes as external evidence. RC21 must not start, stop, restart, or reconfigure `llama-server`.
- Do not delete models automatically. Fleet classification and any removal workflow require RC22 and explicit operator approval.
- Preserve score calculation, canonical-ranking semantics, raw evidence, campaign immutability, and legacy-run readability.
- Runtime selection must be explicit and persisted. In unattended mode ambiguity is a pre-model-work failure.
- Keep backend-specific service/KV repair isolated to Ollama. llama.cpp reports those operations as unsupported.

## Evidence and hygiene

- Commit only durable source, tests, and documentation. Keep live logs, campaign outputs, scripts, and raw hardware evidence outside the repository.
- Use repository-relative paths in committed text; use `<repo>` for examples.
- Make additive, versioned schema migrations with compatibility readers before writing new fields.
- Exercise unit tests, offline selftest, release hygiene, and real hardware acceptance separately. Do not claim hardware evidence from mocks.

## Working practice

- Read affected source, tests, and historical audits before changing behavior.
- Do not alter version, release headings, scoring, canonical rankings, services, models, commits, tags, or remotes without an explicit request.
- Update `docs/rc21/RC21_PROGRESS.md` as stages are accepted. Stop at the approved stage boundary.
