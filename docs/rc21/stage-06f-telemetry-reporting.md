# RC21 Stage 6F: Telemetry reporting

Reports read the optional runtime-telemetry artifact and show a concise backend, endpoint, and declared-versus-observed UUID summary. They explicitly state that allocation evidence does not prove layers, tensor split, KV-cache placement, or offload placement. Old runs render unchanged when the artifact is absent.

`doctor` adds a bounded observational capability summary for physical inventory, compute-process query, endpoint state, and socket-evidence completeness. It distinguishes idle/partial availability from failure and never starts, repairs, or reconfigures a runtime. There is no new watcher or daemon.

Stage 6F is implemented and fixture-tested. Stage 6G has not started; it remains the live acceptance and Stage 6 closeout boundary.

The consolidated review verified that an idle compute-process query remains supported/idle rather than failed, while failed queries, incomplete procfs/socket evidence, and unavailable schemas are labeled separately. The Stage 6G host script is prepared at `/tmp/llmb-rc21-stage6g-acceptance.sh`; it has not been run from this implementation batch.

For Ollama, a `llama-server` compute process may be attributed only after stable identity and bounded ancestor evidence prove descent from the uniquely selected endpoint owner. Incomplete socket evidence limits it to probable confidence; no lineage, loop, depth exhaustion, reuse, or external standalone server is attributed. Stage 6G remains pending rerun.

Stage 6G passed at `/tmp/llmb-rc21-stage6g-acceptance-20260801T001221Z`; report and doctor completed successfully without overclaiming the empty pre-run attribution. Stage 6 is complete; Stage 7 has not started.
