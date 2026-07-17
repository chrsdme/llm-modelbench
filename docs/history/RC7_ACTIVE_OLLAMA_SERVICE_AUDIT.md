# LLM ModelBench 1.0.0rc7 active Ollama service audit

## Triggering real-host evidence

The host had two Ollama-related systemd units:

- `ollama-gpu0.service`: active, PID 2119, owner of `127.0.0.1:11434`;
- `ollama.service`: crash-looping in `activating (auto-restart)`.

RC6 assumed `ollama.service`. Its q8 drop-in therefore changed the wrong unit while the real API process continued unchanged. The active GPU0 unit also contained a stale UUID-based `CUDA_VISIBLE_DEVICES` value.

A second, independent RC6 issue was also confirmed: its managed `90-llmb-repair-kv.conf` could lose to a later-sorting unmanaged drop-in such as `override.conf`. That bug was real even though it occurred on the wrong unit in this host incident.

## RC7 corrections

RC7 no longer trusts a conventional service name. Before any service mutation it:

1. authenticates under an explicit human-supervised discovery or verification phase;
2. obtains the PID bound to the configured Ollama TCP port using `ss`;
3. enumerates Ollama-related systemd units;
4. requires exactly one unit whose `MainPID` equals the listener PID;
5. rejects an explicitly supplied `--ollama-service` that does not own the endpoint;
6. checks UUID-based `CUDA_VISIBLE_DEVICES` values against current `nvidia-smi` UUIDs;
7. writes a dedicated late-sorting `zzzz-llmb-repair-kv.conf` override;
8. asks systemd for the merged effective environment before restart;
9. restarts only the verified active unit;
10. verifies that the replacement unit still owns the configured port;
11. verifies the live process `OLLAMA_KV_CACHE_TYPE` after restart;
12. restores the original managed drop-in state after success or a post-mutation error.

RC7 never rewrites the stale GPU UUID or unrelated unmanaged drop-ins automatically. A stale explicit UUID is a hard pre-restart diagnostic so the operator can correct the unit deliberately.

## Additional safety correction

RC6 marked the service as changed before calling the controller. A pre-mutation guard failure could therefore trigger an unnecessary rollback restart. RC7 tracks whether a managed file mutation actually began. A wrong unit or stale GPU UUID now aborts without restarting any service.

## CLI behavior

The default is now:

```text
--ollama-service auto
```

Before automatic discovery the operator must type `DISCOVER`. For an explicit unit the operator must type `VERIFY`. The existing q8, q4 and restoration phases still require `RESTART`, and sudo retains exclusive password handling.

## Validation

Working-tree validation:

```text
pytest:      307 passed
compileall:  passed
selftest:    SELFTEST: ALL GOOD
version:     llm-modelbench 1.0.0rc7
git diff --check: passed
```

The final ZIP was independently extracted and the same validation was repeated from the packaged source.

No real service restart, Ollama request, model generation or GPU benchmark was performed while building this hotfix.

## Scope intentionally not included

This is a narrow service-control correctness release. The following remain separate outstanding work:

- repair-parent watcher redesign;
- production integration of Rankings V3 dark split-page output;
- capability-repair-aware ranking applicability;
- stronger FIM task-equivalent capability gate;
- explicit `think_ineffective` state;
- the real two-model q8/q4 needle smoke after the host service UUID is corrected.
