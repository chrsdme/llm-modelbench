# Unattended repair sudo policy

`--auto-confirm` is optional. It uses one root-owned broker rather than a
collection of generic privileged commands. The broker accepts only its version,
port, transaction token, and the `q8_0`/`q4_0` KV enum; it does not accept
paths, units, systemd fragments, environment values, or shell commands.

## One-time installation

```bash
sudo install -d -o root -g root -m 0755 /usr/local/libexec
sudo install -o root -g root -m 0755 scripts/libexec/llmb-ollama-kv-control \
  /usr/local/libexec/llmb-ollama-kv-control
sudo stat -c '%U:%G %a %n' /usr/local/libexec/llmb-ollama-kv-control
```

The expected final mode is `root:root 755`. The broker creates root-private
transaction recovery state beneath `/var/lib/llm-modelbench/kv-repair/`.

## Sudoers rule

Use `visudo` to grant the dedicated local benchmark account only the broker:

```sudoers
<username> ALL=(root) NOPASSWD: /usr/local/libexec/llmb-ollama-kv-control
```

Do not grant NOPASSWD access to `install`, `rm`, `cat`, `test`, `ss`,
`systemctl`, or a general shell. Sudoers authorizes the executable; the
root-owned broker validates its semantic protocol and independently identifies
the Ollama listener and owning systemd unit.

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
