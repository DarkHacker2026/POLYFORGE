#!/usr/bin/env python3
"""
probe_wide.py -- Probe the wide Vortex build (8T/2W/2C) and write hardware_facts.vortex_wide.json.

This script physically swaps the simx binary, runs the discovery agent,
captures the output, then restores the original binary.
The wide simx is at: vendor/vortex/build_wide/simx
"""
import subprocess, sys, json, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

BASE_SIMX = ROOT / "vendor" / "vortex" / "build" / "sim" / "simx" / "simx"
WIDE_SIMX = ROOT / "vendor" / "vortex" / "build_wide" / "simx"
BACKUP    = ROOT / "vendor" / "vortex" / "build" / "sim" / "simx" / "simx.probe_wide_backup"
OUT_JSON  = ROOT / "data" / "hardware_facts.vortex_wide.json"

WSL_ROOT  = "/home/dark_hacker/hackathon-project"

def wsl(cmd, timeout=600):
    return subprocess.run(
        ["wsl.exe", "-e", "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout
    )

print("=== Probing Wide Vortex Config (8T/2W/2C) ===\n")
print(f"Wide binary : {WIDE_SIMX}")
print(f"Output file : {OUT_JSON}\n")

# Step 1: backup and swap
print("[1/4] Swapping simx -> wide build...")
r = wsl(f"cp {WSL_ROOT}/vendor/vortex/build/sim/simx/simx {WSL_ROOT}/vendor/vortex/build/sim/simx/simx.probe_wide_backup && cp {WSL_ROOT}/vendor/vortex/build_wide/simx {WSL_ROOT}/vendor/vortex/build/sim/simx/simx && echo OK")
if "OK" not in r.stdout:
    print("ERROR: swap failed:", r.stderr)
    sys.exit(1)
print("    OK\n")

try:
    # Step 2: run discovery agent (SIMT only, simx mode)
    print("[2/4] Running discovery agent with wide binary (this takes ~2 min)...")
    t0 = time.time()
    r = wsl(
        f"cd {WSL_ROOT} && source .wsl_env && "
        f"python3 discovery_agent.py --simt --sim simx 2>&1",
        timeout=300
    )
    elapsed = time.time() - t0
    print(f"    Completed in {elapsed:.0f}s")
    print("--- stdout ---")
    print(r.stdout[-2000:] if len(r.stdout) > 2000 else r.stdout)
    if r.returncode != 0:
        print("WARNING: discovery_agent returned non-zero:", r.returncode)

    # Step 3: read the JSON that was written by discovery agent (it writes to vortex.json)
    # We need to copy vortex.json -> vortex_wide.json and restore provenance backup
    print("\n[3/4] Saving wide facts -> hardware_facts.vortex_wide.json ...")
    # Read whatever was written
    facts_path = ROOT / "data" / "hardware_facts.vortex.json"
    try:
        facts = json.loads(facts_path.read_text(encoding="utf-8"))
        # Tag it as the wide build
        facts["_build_config"] = {
            "VX_CFG_NUM_THREADS": 8,
            "VX_CFG_NUM_WARPS": 2,
            "VX_CFG_NUM_CORES": 2,
            "simx_binary": str(WIDE_SIMX),
            "probed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        OUT_JSON.write_text(json.dumps(facts, indent=4), encoding="utf-8")
        print(f"    Written to {OUT_JSON}")
        sf = facts.get("simt_facts", {})
        print(f"    simt_facts: threads={sf.get('num_threads_per_warp')} warps={sf.get('num_warps_per_core')} cores={sf.get('num_cores')}")
    except Exception as e:
        print(f"    ERROR reading/writing facts: {e}")

finally:
    # Step 4: always restore original simx
    print("\n[4/4] Restoring original simx...")
    r2 = wsl(f"cp {WSL_ROOT}/vendor/vortex/build/sim/simx/simx.probe_wide_backup {WSL_ROOT}/vendor/vortex/build/sim/simx/simx && echo RESTORED")
    if "RESTORED" in r2.stdout:
        print("    Original simx restored.")
    else:
        print("    WARNING: restore may have failed:", r2.stderr)

    # Restore our provenance-augmented vortex.json from the Windows-side copy
    import subprocess as sp
    print("\nRestoring provenance-augmented hardware_facts.vortex.json ...")
