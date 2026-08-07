#!/usr/bin/env bash
# Removes everything install.sh created (virtualenv, build artefacts, caches).
# Never touches benchmark evidence or local-only development material.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$here"

rm -rf .venv build dist ./*.egg-info _last_summary.json .pytest_cache
find . -type d -name "__pycache__" \
  -not -path "./runs/*" \
  -not -path "./campaigns/*" \
  -not -path "./rankings/*" \
  -not -path "./local_only/*" \
  -exec rm -rf {} + 2>/dev/null || true

cat <<'MSG'
Removed: .venv, build/, dist/, *.egg-info, __pycache__, .pytest_cache,
_last_summary.json.

NOT removed: runs/, campaigns/, rankings/, model_cards/, snapshots/, and
local_only/ (your benchmark and development material). Repo source files are
untouched.
Re-run ./install.sh to reinstall.
MSG
