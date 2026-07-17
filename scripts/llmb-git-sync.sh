#!/usr/bin/env bash
set -euo pipefail

# LLM ModelBench Git sync helper for Linux.
# Usage:
#   scripts/llmb-git-sync.sh pull
#   scripts/llmb-git-sync.sh push
#   scripts/llmb-git-sync.sh sync
#   scripts/llmb-git-sync.sh status
#
# Safety:
# - Pull/rebase refuses to run with a dirty worktree.
# - Push first fetches origin/main and refuses if local main is behind.
# - Uses SSH key auth only; never switches to HTTPS.

MODE="${1:-sync}"
REMOTE="origin"
BRANCH="main"
REMOTE_URL="git@github.com:chrsdme/llm-modelbench.git"
SSH_KEY="${HOME}/.ssh/id_ed25519"

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  echo "ERROR: not inside a git repository." >&2
  exit 1
fi
cd "$ROOT"

echo "Repo: $ROOT"
echo "Mode: $MODE"
echo

ensure_ssh() {
  echo "Configuring SSH remote and repo-local SSH command..."
  git remote set-url "$REMOTE" "$REMOTE_URL"
  git remote set-url --push "$REMOTE" "$REMOTE_URL"
  git config core.sshCommand "ssh -i ${SSH_KEY} -o IdentitiesOnly=yes"

  if [[ ! -f "$SSH_KEY" ]]; then
    echo "WARNING: SSH key not found: $SSH_KEY" >&2
    echo "Push may fail until this key exists and its public key is registered with GitHub." >&2
  fi
}

require_clean() {
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: worktree is not clean. Commit/stash/discard local changes before pulling/rebasing." >&2
    git status --short
    exit 1
  fi
}

show_status() {
  echo
  echo "Remote:"
  git remote -v
  echo
  echo "HEAD:"
  git show --no-patch --format="%h %D %s" HEAD
  echo
  echo "Local status:"
  git status --short
  echo
  echo "Recent tags:"
  git tag --sort=-creatordate | head -20
}

pull_latest() {
  echo "Fetching $REMOTE..."
  git fetch "$REMOTE" --tags --prune

  require_clean

  current_branch="$(git branch --show-current)"
  if [[ "$current_branch" != "$BRANCH" ]]; then
    echo "ERROR: current branch is '$current_branch', expected '$BRANCH'." >&2
    exit 1
  fi

  echo "Rebasing $BRANCH onto $REMOTE/$BRANCH..."
  if ! git rebase "$REMOTE/$BRANCH"; then
    echo "ERROR: rebase failed. Resolve conflicts, then run:" >&2
    echo "  git rebase --continue" >&2
    echo "or:" >&2
    echo "  git rebase --abort" >&2
    exit 1
  fi

  echo "Pull sync complete."
  show_status
}

push_current() {
  echo "Fetching $REMOTE before push..."
  git fetch "$REMOTE" --tags --prune

  current_branch="$(git branch --show-current)"
  if [[ "$current_branch" != "$BRANCH" ]]; then
    echo "ERROR: current branch is '$current_branch', expected '$BRANCH'." >&2
    exit 1
  fi

  if git show-ref --verify --quiet "refs/remotes/$REMOTE/$BRANCH"; then
    if ! git merge-base --is-ancestor "$REMOTE/$BRANCH" HEAD; then
      echo "ERROR: local $BRANCH is not based on $REMOTE/$BRANCH." >&2
      echo "Run:" >&2
      echo "  scripts/llmb-git-sync.sh pull" >&2
      echo "then resolve any rebase conflicts before pushing." >&2
      exit 1
    fi
  else
    echo "Remote branch $REMOTE/$BRANCH does not exist; allowing initial publication."
  fi

  echo "Pushing $BRANCH..."
  git push -u "$REMOTE" "$BRANCH"

  echo "Pushing tags..."
  git push "$REMOTE" --tags

  echo "Push sync complete."
  show_status
}

ensure_ssh

case "$MODE" in
  status)
    show_status
    ;;
  pull)
    pull_latest
    ;;
  push)
    push_current
    ;;
  sync)
    pull_latest
    push_current
    ;;
  *)
    echo "ERROR: unknown mode '$MODE'." >&2
    echo "Usage: scripts/llmb-git-sync.sh [pull|push|sync|status]" >&2
    exit 1
    ;;
esac
