# LLM ModelBench 1.0.0rc6 supervised KV cascade audit

## Purpose

RC6 adds the operator-approved service orchestration requested after RC5: the repair command can temporarily configure Ollama for q8, restart and verify it, run guarded needle repairs, repeat only unresolved actions under q4, and restore the original service state.

## Safety contract

- The feature is inactive unless both `--kv-cascade` and `--restart-ollama` are supplied with `--apply`.
- Dry-run never invokes sudo or systemctl.
- A real TTY is mandatory.
- Every privileged phase requires typing `RESTART`; `--yes` cannot bypass it.
- Sudo owns password input. No password enters Python memory, logs, plans, or artifacts.
- The controller writes only `/etc/systemd/system/<unit>.d/90-llmb-repair-kv.conf`.
- Existing unmanaged content at that exact path causes a fail-closed refusal.
- The running service is checked with `systemctl is-active` and the live process environment is checked for the exact KV value.
- The original drop-in state is restored by default.
- A failure after mutation triggers best-effort rollback before the error is re-raised.
- Service events are append-only in `repair_service_<plan-id>.jsonl`.

## Execution order

1. Standard non-needle repair actions under the original Ollama setting.
2. Supervised q8 service phase.
3. Guarded needle actions under verified q8.
4. Supervised q4 service phase only when q8 leaves actions unresolved.
5. Only unresolved needle actions under verified q4.
6. Supervised restoration of the original service state.
7. Rankings refresh.

This ordering prevents temporary KV changes from altering unrelated coding, tool, OCR, retrieval, or reasoning repair attempts.

## Additional correctness fix

RC5 considered a repaired task successful whenever it returned a numeric score without `error_kind`. For `needle`, a partial-depth row can still have a numeric score. RC6 requires `needle_coverage >= 1.0` before a needle repair is marked recovered, so q4 fallback selection is based on actual completed depth coverage.

## Permissions

A system-level `ollama.service` normally requires sudo. RC6 uses ordinary commands such as:

```text
sudo -v
sudo install ... /etc/systemd/system/ollama.service.d/90-llmb-repair-kv.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama.service
```

No blanket root shell or password piping is used.

## Verification scope

The automated test suite uses injected command runners and fake controllers. It verifies command construction, narrow drop-in use, sudo authentication flow, q8-to-q4 action filtering, rollback, and no false needle recovery. It does not restart the host's real Ollama service or execute real models.
