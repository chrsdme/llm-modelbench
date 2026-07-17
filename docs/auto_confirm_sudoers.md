# Unattended repair sudo policy

`--auto-confirm` is an explicit local unattended mode for a supervised KV-cache repair. It does not read or store a password. It invalidates cached sudo credentials and uses `sudo -n`, so a missing or mismatched rule fails instead of prompting.

This feature is optional. Prefer normal interactive confirmation unless unattended recovery is genuinely required.

## 1. Resolve local values

Determine these values on the target host:

```bash
command -v sudo ss systemctl install test cat rm
systemctl list-units --type=service | grep -i ollama
id -un
```

Use placeholders below as follows:

- `<username>`: the dedicated local account running LLM ModelBench
- `<ss-path>`: the exact path returned by `command -v ss`
- `<ollama-service>`: the verified systemd unit that owns the configured Ollama port

Do not add wildcard service names or wildcard filesystem paths.

## 2. Install the narrow environment reader

From the repository root:

```bash
sudo install -d -m 0755 /usr/local/libexec
sudo install -o root -g root -m 0755 llmb-read-kv-env.sh \
  /usr/local/libexec/llmb-read-kv-env.sh
```

The helper accepts only a numeric PID and prints only `OLLAMA_KV_CACHE_TYPE` from that process environment.

## 3. Create a host-specific sudoers file

Open the file through `visudo`:

```bash
sudo visudo -f /etc/sudoers.d/llmb-repair
```

Generate the exact rule set for the one verified unit. The shape is:

```sudoers
<username> ALL=(root) NOPASSWD: <ss-path> -H -tlnp
<username> ALL=(root) NOPASSWD: /usr/bin/systemctl daemon-reload
<username> ALL=(root) NOPASSWD: /usr/bin/systemctl restart <ollama-service>
<username> ALL=(root) NOPASSWD: /usr/bin/install -d -m 0755 /etc/systemd/system/<ollama-service>.d
<username> ALL=(root) NOPASSWD: /usr/bin/install -m 0644 /tmp/llmb-ollama-kv-pending-<ollama-service>.conf /etc/systemd/system/<ollama-service>.d/zzzz-llmb-repair-kv.conf
<username> ALL=(root) NOPASSWD: /usr/bin/test -e /etc/systemd/system/<ollama-service>.d/zzzz-llmb-repair-kv.conf
<username> ALL=(root) NOPASSWD: /usr/bin/cat /etc/systemd/system/<ollama-service>.d/zzzz-llmb-repair-kv.conf
<username> ALL=(root) NOPASSWD: /usr/bin/rm -f /etc/systemd/system/<ollama-service>.d/zzzz-llmb-repair-kv.conf
<username> ALL=(root) NOPASSWD: /usr/local/libexec/llmb-read-kv-env.sh
```

Replace every placeholder before saving. Never paste the placeholder form into sudoers unchanged.

## 4. Validate without cached credentials

```bash
sudo visudo -cf /etc/sudoers.d/llmb-repair
sudo -k
sudo -n "$(command -v ss)" -H -tlnp >/dev/null
```

The final command must complete without a password prompt.

## 5. Apply deliberately

A previous unresolved repair is not repeated automatically. Use `--force` only after reviewing the plan:

```bash
./llmb repair --run-id RUN_ID --runs-dir runs --apply --force \
  --kv-cascade --restart-ollama --ollama-service auto --auto-confirm
```

`--auto-confirm` implies the normal compute `--yes` gate. It does not imply `--force`.
