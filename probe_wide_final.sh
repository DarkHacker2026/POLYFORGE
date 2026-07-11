#!/bin/bash
# probe_wide_final.sh -- probe wide Vortex (8T/2W/2C) and write hardware_facts.vortex_wide.json
set -e
cd ~/hackathon-project
source .wsl_env

ORIG="vendor/vortex/build/sim/simx/simx"
WIDE="$HOME/vortex_wide_build/simx"
BACKUP="$HOME/simx_base_backup"

echo "=== Wide Vortex Probe (8T/2W/2C) ==="
echo "Wide binary: $WIDE"

# Backup original and swap
cp "$ORIG" "$BACKUP"
cp "$WIDE" "$ORIG"
echo "[OK] Wide binary in place. Running discovery..."

# Run discovery against the wide binary
python3 discovery_agent.py --simt --sim simx

RC=$?
echo "Discovery exit code: $RC"

# Restore original
cp "$BACKUP" "$ORIG"
echo "[OK] Original simx restored."

# The discovery agent writes to data/hardware_facts.vortex.json
# Copy it to vortex_wide.json and tag it
python3 - <<'PYEOF'
import json, datetime
with open("data/hardware_facts.vortex.json") as f:
    facts = json.load(f)
facts["_build_config"] = {
    "VX_CFG_NUM_THREADS": 8,
    "VX_CFG_NUM_WARPS": 2,
    "VX_CFG_NUM_CORES": 2,
    "simx_binary": "$HOME/vortex_wide_build/simx",
    "probed_at_utc": datetime.datetime.utcnow().isoformat() + "Z"
}
with open("data/hardware_facts.vortex_wide.json", "w") as f:
    json.dump(facts, f, indent=4)
print("Wide facts written to data/hardware_facts.vortex_wide.json")
sf = facts.get("simt_facts", {})
print(f"  threads={sf.get('num_threads_per_warp')} warps={sf.get('num_warps_per_core')} cores={sf.get('num_cores')}")
PYEOF

echo "=== Done ==="
