#!/bin/bash
set -e
cd ~/hackathon-project
source .wsl_env

ORIG="vendor/vortex/build/sim/simx/simx"
WIDE="/tmp/vortex_wide/simx"
BACKUP="/tmp/simx_base_backup"

echo "[1] Backing up original simx and swapping to wide (8T/2W/2C) binary..."
cp "$ORIG" "$BACKUP"
cp "$WIDE" "$ORIG"
echo "    Wide binary in place."

echo "[2] Running discovery agent with wide binary..."
python3 discovery_agent.py --simt --sim simx > /tmp/wide_probe_output.txt 2>&1
EXIT=$?

echo "[3] Restoring original simx..."
cp "$BACKUP" "$ORIG"
echo "    Done."

echo "[4] Result (exit=$EXIT):"
cat /tmp/wide_probe_output.txt
