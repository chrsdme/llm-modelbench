#!/usr/bin/env bash
set -euo pipefail

# Universal LLM ModelBench updater.
# - prefers the repository .venv so Debian/Ubuntu PEP 668 system Python is not touched
# - fetches/pulls fast-forward updates only
# - reinstalls the package editable
# - validates compileall, pytest, selftest, and version
# - never reads, writes, deletes, or cleans runs/

echo "LLM ModelBench updater"
echo "======================"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is not installed or not on PATH." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".git" ]; then
  echo "ERROR: update.sh must be run from the repository root." >&2
  exit 1
fi

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON_BIN="$SCRIPT_DIR/.venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "ERROR: no Python interpreter found. Create .venv first: python3 -m venv .venv" >&2
  exit 1
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$CURRENT_BRANCH" = "HEAD" ]; then
  echo "ERROR: detached HEAD. Checkout a branch before updating." >&2
  exit 1
fi

# Refuse tracked local edits only. Untracked run artefacts are intentionally ignored.
if ! git diff --quiet -- . ':(exclude)runs/**' || ! git diff --cached --quiet -- . ':(exclude)runs/**'; then
  echo "ERROR: tracked working tree changes detected outside runs/." >&2
  echo "Commit, stash, or discard them before updating." >&2
  git status --short -- . ':(exclude)runs/**'
  exit 1
fi

# Reject untracked source/configuration files. Generated evidence directories are allowed.
UNTRACKED_UNSAFE="$(git ls-files --others --exclude-standard | \
  grep -Ev '^(runs|rankings|rankings-separate|model_cards|snapshots)/' || true)"
if [ -n "$UNTRACKED_UNSAFE" ]; then
  echo "ERROR: untracked files outside generated evidence directories may alter imports or tests:" >&2
  printf '%s\n' "$UNTRACKED_UNSAFE" >&2
  echo "Move, commit, or remove them before updating." >&2
  exit 1
fi

REMOTE="${LLM_MODELBENCH_UPDATE_REMOTE:-origin}"
REMOTE_REF="$REMOTE/$CURRENT_BRANCH"

echo "Branch: $CURRENT_BRANCH"
echo "Current commit: $(git rev-parse --short HEAD)"
echo "Python: $PYTHON_BIN"

if git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo
  echo "Fetching remote..."
  git fetch --tags "$REMOTE"

  if git rev-parse --verify --quiet "$REMOTE_REF" >/dev/null; then
    LOCAL_HEAD="$(git rev-parse HEAD)"
    REMOTE_HEAD="$(git rev-parse "$REMOTE_REF")"
    BASE_HEAD="$(git merge-base HEAD "$REMOTE_REF")"

    if [ "$LOCAL_HEAD" = "$REMOTE_HEAD" ]; then
      echo "Already up to date with $REMOTE_REF."
    elif [ "$LOCAL_HEAD" = "$BASE_HEAD" ]; then
      echo "Updating from $REMOTE_REF..."
      git pull --ff-only "$REMOTE" "$CURRENT_BRANCH"
    elif [ "$REMOTE_HEAD" = "$BASE_HEAD" ]; then
      echo "Local branch is ahead of $REMOTE_REF; not pulling."
    else
      echo "ERROR: local and remote branches diverged. Manual git resolution required." >&2
      exit 1
    fi
  else
    echo "WARNING: remote ref $REMOTE_REF not found; skipping pull."
  fi
else
  echo "WARNING: remote '$REMOTE' not configured; skipping fetch/pull."
fi

echo
echo "Installing package in editable mode..."
if ! "$PYTHON_BIN" -m pip install -e ".[all]"; then
  echo "WARNING: optional extras install failed; retrying core editable install."
  "$PYTHON_BIN" -m pip install -e .
fi

echo
echo "Running validation..."
"$PYTHON_BIN" -m compileall -q llm_modelbench
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" -m llm_modelbench selftest

echo
echo "Version:"
"$PYTHON_BIN" -m llm_modelbench --version

if [ -x "./llmb" ]; then
  ./llmb --version || true
fi

echo
echo "Final git status, ignoring runs/:"
git status --short -- . ':(exclude)runs/**'

echo
echo "Update complete."
echo "Validation complete: compileall, pytest, selftest, and version checks passed."
