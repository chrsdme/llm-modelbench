# RC8 Service Readiness Hotfix Audit

## Trigger

On the real host, `ollama-gpu0.service` restarted successfully and systemd
reported `active`, but the ModelBench ownership check ran before `ollama serve`
had rebound `127.0.0.1:11434`. Seconds later the expected unit owned the port.
RC7 treated this normal startup interval as a service-ownership failure.

## Correction

RC8 changes the restart readiness gate from:

1. restart unit;
2. wait for `systemctl is-active`;
3. check endpoint ownership once;

to:

1. restart unit;
2. wait for `systemctl is-active`;
3. repeatedly require the unit's current `MainPID` to own the configured
   Ollama TCP endpoint;
4. proceed only after both conditions are true.

A missing MainPID or absent listener is treated as a temporary startup state.
A listener owned by another process/unit, ambiguous unit ownership, or another
non-transient mismatch remains a hard failure.

The same barrier is used by the restoration path.

## Evidence boundaries

No live Ollama restart, GPU inference, or model benchmark was performed while
building this hotfix. Validation uses injected command runners and real-host-
shaped fixtures.

## Validation

- Full pytest suite: 314 passed.
- Compileall: passed.
- Version: `llm-modelbench 1.0.0rc8`.
- Self-test: `SELFTEST: ALL GOOD`.
