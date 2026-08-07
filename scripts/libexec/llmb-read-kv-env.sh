#!/bin/sh
# /usr/local/libexec/llmb-read-kv-env.sh
#
# Reads exactly one environment variable (OLLAMA_KV_CACHE_TYPE) from a given
# process's /proc/<pid>/environ, and nothing else. Discards everything else
# in that process's environment -- never prints, logs, or stores any other
# variable, including secrets that might be present there.
#
# Usage: llmb-read-kv-env.sh <pid>

set -eu

pid="$1"

case "$pid" in
    ''|*[!0-9]*)
        echo "usage: $0 <pid>  (pid must be numeric)" >&2
        exit 2
        ;;
esac

if [ ! -r "/proc/$pid/environ" ]; then
    exit 1
fi

tr '\000' '\n' < "/proc/$pid/environ" | grep -m1 '^OLLAMA_KV_CACHE_TYPE=' || true
