#!/usr/bin/env bash
# One-command installer: creates a local virtualenv and installs the tool.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$here"
chmod 755 install.sh llmb llmb-run llmb-watch 2>/dev/null || true
PY="${PYTHON:-python3}"
echo "==> Creating virtualenv (.venv)"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
echo "==> Upgrading pip"
pip install --quiet --upgrade pip
echo "==> Installing llm-modelbench (with optional extras)"
if ! pip install -e ".[all]"; then
  echo "ERROR: extras install failed. Install aborted so optional dependency problems are visible." >&2
  echo "To install core only after reviewing the error, run: pip install -e ." >&2
  exit 1
fi
echo "==> Verifying"
llm-modelbench selftest
./llmb --version >/dev/null
cat <<'MSG'

Done. Installed: a local virtualenv (.venv) and the llm-modelbench package
with its optional extras (vision, pdf, yaml). Nothing outside this folder was
touched, and no models were downloaded or run.

Activate with:  source .venv/bin/activate
Then try:             ./llmb doctor
Plan first:           ./llmb plan --mock
Offline demo:         ./llmb-run --mock --allow-host-code-execution --yes --run-id demo
Watcher:              ./llmb-watch --run-id demo --layout compact --screen alternate
MSG
