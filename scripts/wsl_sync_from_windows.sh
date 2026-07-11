#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-/mnt/c/Users/Dark Hacker/Desktop/POLYFORGE}"
DEST="${2:-$HOME/POLYFORGE}"

mkdir -p "$DEST"
rsync -a --delete \
  --exclude '.git/' \
  --exclude '.wsl_env' \
  --exclude 'vendor/vortex/.git/' \
  --exclude 'vendor/vortex/build/' \
  --exclude 'vendor/vortex/third_party/softfloat/' \
  --exclude 'vendor/vortex/third_party/ramulator/' \
  "$SRC/" "$DEST/"

echo "[synced] $SRC -> $DEST"
echo "Next:"
echo "  cd \"$DEST\""
echo "  bash scripts/wsl_setup_vortex.sh"
echo "  source .wsl_env"
echo "  bash scripts/wsl_run_agent_artifacts.sh"
