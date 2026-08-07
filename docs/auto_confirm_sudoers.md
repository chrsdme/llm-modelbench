# Unattended repair sudo policy

This configuration is optional. It is needed only for unattended automatic
Ollama KV repair; it is not required to install or normally use LLM ModelBench.
Users who do not use automatic KV repair must not install the helper or create
an LLMB sudoers rule.

`--auto-confirm` uses one root-owned broker rather than a
collection of generic privileged commands. The broker accepts only its version,
bounded port, transaction token, and the `q8_0`/`q4_0` KV enum; it does not accept
paths, units, systemd fragments, environment values, or shell commands.

## One-time installation

```bash
sudo install -d -o root -g root -m 0755 /usr/local/libexec
sudo install -o root -g root -m 0755 scripts/libexec/llmb-ollama-kv-control \
  /usr/local/libexec/llmb-ollama-kv-control
sudo stat -c '%U:%G %a %n' /usr/local/libexec/llmb-ollama-kv-control
```

The expected final state is `/usr/local/libexec/llmb-ollama-kv-control`,
`root:root`, mode `0755`. The helper and trusted parent directories must not
be writable by the ModelBench user. The broker creates root-private transaction
recovery state beneath `/var/lib/llm-modelbench/kv-repair/`.

## Sudoers rule

Use `visudo` to grant the dedicated local benchmark account only the broker:

```sudoers
<username> ALL=(root) NOPASSWD: /usr/local/libexec/llmb-ollama-kv-control
```

Do not grant NOPASSWD access to `install`, `rm`, `cat`, `test`, `ss`,
`systemctl`, or a general shell. Sudoers authorizes the executable; the
root-owned broker validates its semantic protocol and independently identifies
the Ollama listener and owning systemd unit.

The selected port authorizes management only of the one system Ollama service
that the broker independently discovers listening on that port. It does not
authorize a caller-selected unit, path, fragment, executable, environment, or
systemctl operation.

## Behaviour and removal

Interactive repair still uses normal sudo authentication and typed phase
confirmation. Auto-confirm uses only `sudo -n`; a missing rule fails
immediately and never falls back to an interactive password prompt. A
transaction retains its original managed drop-in bytes until `restore`
succeeds, enabling recovery after an interrupted client process.

To remove unattended capability, remove the sudoers entry with `visudo`, then
remove `/usr/local/libexec/llmb-ollama-kv-control` after restoring any active
transaction. Normal benchmarking and supervised repair planning do not need
this broker.

If an interrupted transaction leaves Ollama down, an administrator logged in
as direct real root (not through sudo) may recover only the recorded
transaction:

```bash
/usr/local/libexec/llmb-ollama-kv-control recover --transaction <token>
```

Recovery validates root-private state and matching service lock, stored loaded
Ollama systemd metadata, and the managed-drop-in hash before restoring bytes.
It reloads and restarts the recorded service and requires the listener to
return before removing state and its lock. Sudo users remain owner-bound;
direct root cannot use this recovery override for `set`.

Older development installations may contain obsolete LLMB privileged helpers
or sudoers artifacts. Remove those manually before installing the current
broker policy. Do not restore the previous broad direct-command policy and do
not install the old KV environment reader.
