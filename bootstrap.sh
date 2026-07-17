#!/usr/bin/env bash
# Fetch + install in one line (for: curl -fsSL <raw-url>/bootstrap.sh | bash)
set -euo pipefail
REPO="${LLM_MODELBENCH_REPO:-https://github.com/chrsdme/llm-modelbench.git}"
DIR="${LLM_MODELBENCH_DIR:-llm-modelbench}"
command -v git >/dev/null || { echo "git is required"; exit 1; }
[ -d "$DIR" ] || git clone "$REPO" "$DIR"
cd "$DIR"
bash install.sh
