# RC10 Repair Truth and Watcher Hotfix Audit

## Trigger

A real InternVL3 repair under RC9 exposed three remaining defects:

1. Ordinary non-KV repairs did not publish a parent repair campaign, so `llmb-watch --follow-queue` still selected the generic one-model child renderer.
2. `--force` re-planned the vision capability gate even though the installed build already had terminal `capability_repair.json` evidence excluding vision.
3. The task-equivalent FIM probe responded and the scored FIM task produced a numeric zero with an empty insertion, but the result remained provisional because `empty_output` was still treated as unattempted evidence.

The same run also repeated all functional probes once per capability action instead of probing only the requested lanes once per model.

## Fixes

### Discoverable repair campaign

Every repair now writes atomic status to both:

- `runs/repair_status_<plan_id>.json` for compatibility;
- `runs/repair_campaign_<plan_id>/status.json` for normal watcher discovery.

The campaign exists during capability probing, before a child benchmark directory is created. When a child starts, its `status.json` is merged into the campaign view. Direct repairs and managed KV cascades share the same watcher model.

### Terminal capability evidence

`capability_repair.json` exclusions are consumed by repair planning. A lane already confirmed unavailable is not re-probed by generic `--force`; a future explicit invalidation/reprobe workflow is required when build/runtime identity changes.

### Scoped functional interrogation

Repair capability gates calculate the union of requested families per model, run one functional interrogation, cache it, and reuse it across actions. Unrelated lanes are not probed.

HTTP probe exceptions preserve status, reason, and response body so unsupported-build evidence can be classified from the server's actual message.

### Measured FIM failure

A task-equivalent insert probe that receives a response, followed by the scored insert task returning numeric zero with `empty_output`, is a measured zero-quality terminal result. It is not a transport failure and not an unattempted cell.

RC10 writes `measured_failure` explicitly. Rankings also adopt the exact RC9 legacy evidence pattern, allowing the existing InternVL3 run to resolve through a read-only rescan without another model call. Generic `--force` does not repeat it.

## Validation

- `python3 -m compileall -q llm_modelbench tests`
- `python3 -m pytest -q` -> 353 passed
- `python3 -m llm_modelbench selftest` -> SELFTEST: ALL GOOD
- clean patch application tested on the RC9 packaged source
- no Ollama, GPU, model, sudo, systemd, or ranking run performed while building this patch
