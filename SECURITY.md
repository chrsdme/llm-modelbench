# Security policy

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability that could expose user data, execute code unexpectedly, bypass approval gates, or alter benchmark evidence. Use GitHub's private security-advisory workflow for the repository.

Include the affected version, reproduction steps, expected behavior, observed behavior, and whether model execution or privileged service control is involved. Do not attach private run artifacts without first removing model output and local identifiers.

## Supported versions

Security fixes are applied to the latest release candidate or stable release.
Detailed internal development records are local-only and are not maintained
release branches.

## Execution boundary

LLM ModelBench can execute model-generated code for deterministic scoring. Normal benchmark runs fail closed for those tasks unless `--allow-host-code-execution` is explicit. Host-mode guards are not a complete sandbox. Use the flag only inside a container, VM, or disposable host, and review [docs/SAFETY.md](docs/SAFETY.md) before real execution.

## Optional Ollama service control

Normal benchmarking needs no root privileges. Root is required only for the
operator-enabled Ollama system-service KV repair, because it modifies one
dedicated systemd drop-in, performs daemon-reload and restart of the
independently discovered Ollama service, and verifies its resulting process
configuration. It is not needed for normal Ollama API use, llama.cpp use,
scoring, reporting, telemetry, or diagnostics.

The historical design exposed several generic privileged commands and crossed
the boundary with an unprivileged temporary configuration file. That design has
been removed. The supported boundary is one root-owned, standard-library
semantic broker at `/usr/local/libexec/llmb-ollama-kv-control`, `root:root`
mode `0755`, with non-writable trusted parent directories. The application can
request only bounded port, transaction, and `q4_0`/`q8_0` operations; the
broker independently finds the listener/service, constructs fixed content,
keeps root-private state, verifies ownership, and restores configuration.

No generic LLMB NOPASSWD policy is supported: never grant direct access to
`systemctl`, `install`, `rm`, `cat`, `test`, `ss`, a shell/interpreter, or the
old KV environment reader. Service restarts can disrupt in-flight Ollama work;
changed host state fails closed and can require explicit administrator recovery.
See [docs/auto_confirm_sudoers.md](docs/auto_confirm_sudoers.md) for optional
installation, recovery, and removal guidance.
