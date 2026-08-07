# RC21 Stage 6C: Bounded Process-to-GPU Attribution

## Scope and compatibility

Stage 6C adds the standalone `llm_modelbench.process_telemetry` evidence layer. It is pure, read-only, bounded, and has no import-time collection side effects. No existing collector or consumer is redirected: `detect_gpu`, `detect_gpus`, `Telemetry`, `ProbeTelemetry`, `nvidia_live`, `live_snapshot`, runner, reports, status, doctor, watcher, context profiling, rankings, campaigns, and repair remain unchanged. No score, ranking, report-schema, campaign-schema, lifecycle, service, model, or runtime-profile behavior changes.

## Process and procfs evidence

`RuntimeProcessIdentity` records a positive PID plus optional Linux start-time ticks, parent PID, bounded executable/command evidence, bounded command arguments, listening ports, classification hint, and discovery sources. Its stable identity is `(pid, start_time_ticks)` when start ticks are available. Discovery reads and validates `stat` both before and after other evidence; a changed/disappeared identity is excluded rather than mixing metadata across PID reuse.

The production reader is bounded to `/proc` and only reads `stat`, `comm`, `cmdline`, `exe`, file-descriptor symlinks, and `net/tcp`/`net/tcp6`. It never reads `environ`, process memory, credentials, open regular-file contents, or unrelated user data. Named caps cover PIDs, proc-file bytes, command-line bytes and arguments, file descriptors, sockets, details, and errors. Process disappearance, permission failure, malformed stat, broken links, unreadable FD directories, malformed socket rows, and stale socket evidence become structured bounded diagnostics.

An empty `/proc/<pid>/cmdline` is valid unavailable command identity, as commonly reported by kernel threads: it produces no error, no command-line source, and no command-line classification. Only malformed or truncated non-empty command lines produce `proc_cmdline` evidence. This correction was identified in manual host acceptance at `/tmp/llmb-rc21-stage6c-manual-live-20260731T215257Z` and verified by the passing rerun below.

Listening ownership is evidence only: LISTEN rows in `/proc/net/tcp` and `/proc/net/tcp6` map local ports to the actual inode column, then FD symlinks map those inodes to zero or more processes. IPv4 and IPv6 are supported, multiple owners are retained, and malformed socket rows or incomplete FD/socket-table evidence make endpoint ownership incomplete. No port proves GPU placement.

## Classification and NVIDIA process evidence

Classification uses requested backend, endpoint ownership, executable basename, command name, and bounded command arguments. It recognizes compatible Ollama server/runner and llama-server forms without relying on installation paths. Generic shell, Python, or container processes are not classified merely because an argument contains a backend word. Endpoint ownership plus compatible process evidence can support confirmed attribution; compatible process evidence without it can support probable attribution.

`nvidia-smi` is queried once with the fixed argument vector:

```text
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory --format=csv,noheader,nounits
```

The query is shell-free, injected for tests, five-second bounded, and has bounded stdout/stderr. CSV requires exactly four fields and supports quoted process names. Empty output is a successful no-process result. A UUID/PID duplicate group is excluded completely, while one PID may remain associated with multiple GPU UUIDs and multiple PIDs with one GPU UUID. Missing executable, timeout, overflow, non-zero exit, invalid CSV/schema, malformed UUID/PID, and marker/malformed memory values are structured evidence. There is no fallback query and no CUDA-ordinal, memory-total, or command-line placement inference.

## Attribution contract

`GPUProcessSample` preserves direct `gpu_uuid` allocation evidence and explicit memory field states. A pure before/after reconciliation helper compares complete normalized process identity evidence, binds start ticks only when exactly one unchanged identity exists on both sides of the NVIDIA query, and clears any prior binding if reconciliation fails. Runtime observed UUIDs derive only from confirmed/probable records matched to an unambiguous compatible runtime process; explicit unequal start ticks are PID-reuse evidence and are non-attributable. Confirmed requires reconciled matching PID/start evidence, unique compatible endpoint ownership, and complete socket evidence; truncated command identity, stat, socket-table, or PID-cap evidence cannot promote confidence. Same stable identities with conflicting metadata are ambiguous, never input-order selected; direct result models reject ambiguous identity sets and relationship uniqueness is independent of confidence. Profile-only anchors are omitted under endpoint ambiguity or incomplete socket evidence. Declared profile UUIDs and observed process UUIDs stay separate. GPU allocation memory does not prove layer split, tensor split, KV placement, or offload fraction.

All structures are frozen and serialize deterministically: processes sort by stable identity, GPU process samples by UUID/PID, attribution records by UUID/PID/confidence, and ports, sources, errors, and UUID sets are normalized.

## Validation, acceptance, and rollback

Fixture validation passed: the required Stage 6C/Stage 6B/inventory focused set has `104 passed`; the full suite has `711 passed`; compileall, selftest, release check, import side-effect check, and `git diff --check` passed. Coverage includes proc stat edge cases, PID reuse, bounded command evidence, socket ownership, classification, fixed NVIDIA parsing/command failures, confidence combinations, declared-versus-observed sets, and absence of CUDA-ordinal or placement inference. Existing Stage 6B physical-GPU fixtures remain unchanged.

Manual host-shell before/query/after acceptance passed at `/tmp/llmb-rc21-stage6c-manual-live-20260731T220628Z` (`python_exit_code=0`, `finalization_exit_code=0`). The fixed NVIDIA compute-process query succeeded with zero rows and zero errors: this is valid idle evidence. Ollama (`127.0.0.1:11434`) was listening with two discovered processes but no observed GPU compute allocation; llama-server (`127.0.0.1:8081`) was unavailable and was not started. Reconciliation completed with no runtime samples or errors, both attribution results had zero relationships and observed UUIDs, and no GPU placement was invented.

Bounded `proc_fd` access limitations remained, so socket evidence was incomplete and confirmed attribution was correctly unavailable. Those limitations are non-blocking for this idle acceptance. Repository status was unchanged by collection. The earlier transient NVIDIA communication failures were not reproduced; no definitive cause is asserted.

Rollback is removal of `process_telemetry.py`, its Stage 6C fixtures, and this document. No existing caller needs reversal because no caller was redirected.

Stage 6C is complete. Stage 6 overall remains incomplete. Stage 6D has not started; its objective is an explicitly reviewed, additive snapshot boundary for these isolated evidence records, without integration into runner, report, status, doctor, watcher, campaigns, rankings, repair, or lifecycle behavior unless separately approved.
