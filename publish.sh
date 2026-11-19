#!/usr/bin/env bash
# Create the public repository and publish the current project using GitHub CLI.
set -euo pipefail
OWNER="Tejeshwini89"
REPO="multimodal-whisper-vision"
DESCRIPTION="Multimodal AI: Whisper speech-to-text with vision (BLIP, CLIP)"

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is required. Install it and run: gh auth login" >&2
  exit 1
fi
if [[ "$(gh api user --jq .login)" != "$OWNER" ]]; then
  echo "Please authenticate GitHub CLI as $OWNER first: gh auth login" >&2
  exit 1
fi

git init -b main
git add -A
git commit -m "Initial commit: multimodal Whisper + vision system"
gh repo create "$OWNER/$REPO" --public --description "$DESCRIPTION" --source=. --remote=origin --push
