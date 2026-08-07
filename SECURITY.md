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

The optional KV-repair flow has one privileged boundary: the root-owned,
standard-library broker installed at `/usr/local/libexec/llmb-ollama-kv-control`.
It must be `root:root` mode `0755` and its parent directories must not be
writable by the invoking user. The sudoers policy may authorize that broker
only; it must not authorize generic file tools, `systemctl`, `ss`, a shell, or
an interpreter. See [docs/auto_confirm_sudoers.md](docs/auto_confirm_sudoers.md)
for installation, rollback/recovery, and removal guidance.
