# RC21 Stage 4: Runtime Profiles and Discovery

## Profile schema and storage

Reusable profiles are JSON objects with `name`, `backend` (`ollama` or `llama_cpp`), `endpoint`, `mode` (`external` only), `physical_gpu_uuids`, optional `description`, and `provenance` (`configured`, `discovered`, or `legacy-default`). The store is versioned and atomically replaced at `${XDG_CONFIG_HOME:-~/.config}/llm-modelbench/runtime_profiles.json`.

The existing `Config.ollama_url`, `LLM_MODELBENCH_OLLAMA_URL`, and default `http://127.0.0.1:11434` remain an unsaved implicit `legacy-ollama` profile. Existing Ollama commands therefore retain their legacy path when no saved/default profile or viable alternative requires a selection decision.

## Discovery and health boundaries

`llmb runtime discover` combines only saved profiles, the implicit configured Ollama endpoint, and local `/proc` command lines identified as Ollama or llama-server. It extracts an explicitly declared process port where present; only an identified Ollama process may justify port 11434 and only an identified llama-server process may justify port 8080. It does not scan port ranges, LAN hosts, or arbitrary remote endpoints.

Health is read-only and local-only: Ollama requires successful `/api/version` and `/api/tags` evidence; llama.cpp uses `/health`. Ollama health responses are bounded at `4 MiB`, read as `limit + 1`, and report an explicit bounded-size failure instead of attempting to parse a truncated response. `/api/tags` must decode to an object with a list-valued `models` field. Candidates report `healthy`, `unhealthy`, `unreachable`, `unsupported`, or `unknown`, and are deduplicated by normalized backend plus endpoint. No model-generation, load/unload, process, systemd, or configuration action occurs during discovery.

## Selection policy

Selection precedence is explicit `--runtime-profile`, saved default profile, exactly one healthy candidate, then an interactive choice. An unhealthy explicit/default profile fails rather than switching. Multiple healthy candidates in a non-interactive shell fail before model work and list the names requiring `--runtime-profile <name>`. The legacy implicit Ollama fallback is limited to the historical case with no healthy candidates, no explicit profile, no saved default, and no saved profiles; it cannot override ambiguity or an unhealthy explicit/default profile.

Interactive selection displays the candidates and a recommendation but never applies it silently. One physical GPU recommends healthy Ollama; multiple physical GPUs recommend healthy llama.cpp. The real inventory comes from `detect_gpus()`; profiles only retain explicitly provided physical GPU UUIDs and do not infer CUDA-visible indices, model placement, splits, KV, or batching.

`runtime save`, `show`, `list`, `select`, and `delete` manage only profile JSON. Replace/delete require confirmation or `--yes`. `runtime select --save-name <name>` can save a chosen discovery result and optionally make it default. Deletion cannot remove models, service files, processes, or systemd state.

Runtime subcommands convert `RuntimeProfileError` and `RuntimeSelectionError` to concise CLI exits. In particular, `runtime delete <unknown>` validates the profile before requesting confirmation and exits with `unknown runtime profile: <name>`.

## Stage 4 llama.cpp boundary

Stage 4 can discover, health-check, display, save, and select an external llama.cpp profile. Any inference command selecting one exits clearly before creating a client: llama.cpp inference is unavailable until Stage 5. It never falls back to Ollama silently.

## Files changed

- `llm_modelbench/runtime_profiles.py`
- `llm_modelbench/cli.py`
- `tests/test_runtime_profiles.py`
- `docs/rc21/RC21_PROGRESS.md`
- `docs/rc21/stage-04-runtime-profiles-discovery.md`

## Tests

Focused Stage 4 acceptance-fix validation passed:

```text
python3 -m pytest -q tests/test_runtime_profiles.py tests/test_backend.py tests/test_cli_subcommands.py
29 passed
```

It covers serialization, atomic writes, legacy compatibility, precedence, automatic/interactive selection, unattended ambiguity, unhealthy default failure, one/multi-GPU recommendation, deduplication, deletion containment, no-mutation discovery, and pre-Stage-5 llama.cpp rejection.

## Real-host acceptance fix iteration

The initial real-host acceptance found that `/api/tags` for 57 installed models exceeded the former 4096-byte read cap, causing a false `did not return JSON` health result. This iteration replaces that truncation with a deliberate 4 MiB bounded read and adds regression coverage for a valid response above 4096 bytes, a response exceeding the bound, non-object JSON, and an object without a list-valued `models` field.

It also adds regression coverage for clean invalid runtime command exits and for `_client()` remaining fail-closed when both Ollama and llama.cpp are healthy. The full real-host acceptance must be rerun by the operator outside Codex; this document does not claim it passed.

Full validation passed:

```text
python3 -m compileall -q llm_modelbench tests
python3 -m pytest -q
563 passed
./llmb selftest
SELFTEST: ALL GOOD
python3 tools/release_check.py
RELEASE CHECK: PASS
git diff --check
passed
```

## Real-host discovery evidence

Codex could not access the host loopback services, so final acceptance was performed separately on the real AI-PC.

### Ollama-only acceptance

The implicit `legacy-ollama` profile at `http://127.0.0.1:11434` was detected as healthy using Ollama version and tags endpoints. The host reported Ollama `0.30.7`, 57 installed models, and no loaded models. With exactly one viable runtime, `llmb runtime select` selected Ollama automatically without prompting.

A temporary Ollama profile was saved with the RTX 5060 Ti UUID, shown as the default, deleted, and verified absent. Deleting an unknown profile returned a concise CLI error without a Python traceback.

### Dual-runtime acceptance

An operator-started external llama-server was detected as healthy at `http://127.0.0.1:8081` while Ollama remained healthy at `http://127.0.0.1:11434`.

With two physical NVIDIA GPUs present, discovery marked llama.cpp as recommended and displayed both runtime candidates. Interactive selection respected the operator's choice and did not silently override it.

A non-interactive selection with both runtimes healthy failed closed with exit code 1 and required `--runtime-profile <name>`. An explicitly saved llama.cpp profile was accepted by the profile layer but inference exited clearly at the RC21 Stage 5 boundary rather than falling back to Ollama.

### Containment evidence

All profile lifecycle tests used the isolated store `/tmp/llmb-rc21-stage4-xdg/llm-modelbench/runtime_profiles.json`.

After discovery and selection tests:

- Ollama `/api/ps` still reported no loaded models.
- The canonical inventory contained 57 models before and after.
- The canonical inventory SHA-256 remained `5f553c1450d7f2b1c3e52010bcd0db205a63a9ec42bf037072d795b7972a933c`.
- No model, service, systemd unit, repository evidence, ranking, or benchmark result was modified by ModelBench.

## Real-host acceptance fix iteration

Initial host testing exposed three defects that were corrected before acceptance:

- Ollama `/api/tags` responses larger than 4096 bytes were truncated and falsely classified as invalid JSON. Health probes now use a bounded 4 MiB response limit and reject oversized responses explicitly.
- Unknown profile deletion previously confirmed first and then exposed an uncaught exception. Runtime-profile CLI errors now exit cleanly, and existence is checked before confirmation.
- The legacy Ollama fallback could absorb an unattended multiple-runtime ambiguity. It now applies only when no healthy candidate and no explicit, default, or saved profile exists.

Regression coverage was added for large and oversized health responses, non-object JSON, invalid tags schemas, clean CLI errors, ambiguity fail-closed behaviour, and the documented legacy fallback.

## Known limitations and rollback

- Discovery is limited to Linux `/proc` process inspection and loopback health probes in this stage.
- Saved non-local profiles remain storable but discovery marks them unsupported instead of probing them.
- A saved profile's GPU UUIDs are declarative only; CUDA-visible mapping and device split are deferred.
- Runtime identity is not yet frozen into campaigns, rows, reports, or rankings.
- Rollback removes the runtime profile module and CLI wiring; existing `ollama_url` construction remains intact.

## Proposed Stage 5 objective

Implement an external, read-only llama-server inference adapter behind the Stage 3 protocol. Support one already-served model per endpoint/profile, preserve Ollama behavior, and fail closed for unsupported model switching, service/KV operations, and unavailable judge profiles.
