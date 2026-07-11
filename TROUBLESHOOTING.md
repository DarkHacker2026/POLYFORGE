# Vortex WSL Build Troubleshooting

Context for picking up this debugging session: agent artifacts (`agent_add_add_int_v1_instantiated`, `agent_load_load_int_v1_instantiated`, etc.) run through a Vortex GPU simulator (`simx`) build in WSL, synced from a Windows dev folder. A `rsync --delete` sync from Windows repeatedly wiped generated/vendored files in WSL, causing a chain of build failures.

## Root Cause

`rsync -a --delete` mirrored the Windows folder into WSL and deleted anything WSL had that Windows didn't — including generated build config and vendored third-party source trees that only exist inside WSL.

## Standard Repair/Run Sequence

Run this every time after syncing from Windows:

```bash
bash "/mnt/c/Users/Dark Hacker/Desktop/hackathon project/scripts/wsl_sync_from_windows.sh"
cd ~/hackathon-project
bash scripts/wsl_setup_vortex.sh
source .wsl_env
bash scripts/wsl_run_agent_artifacts.sh
```

## Failure Chain (chronological)

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `config.mk: No such file or directory` | rsync deleted `vendor/vortex/build/config.mk` and `build/ci/gen_config.py` | Regenerate via setup script; don't delete `build/` on sync |
| 2 | `fatal error: softfloat_types.h: No such file or directory` | `third_party/softfloat` folder existed but submodule contents weren't populated | Setup script clones missing submodule contents |
| 3 | `fatal error: base/base.h: No such file or directory` | `third_party/ramulator` existed but was an incomplete/wrong checkout | Setup script detects missing marker file (`src/base/base.h`) and reclones |
| 4 | Same `base.h` error persists | Wrong Ramulator commit was being cloned (not Vortex's pinned SHA) | Pin exact submodule commits (see below) |
| 5 | Still failing in `dram_sim.cpp` build | Ramulator checkout still missing `src/base/base.h` even after reclone | In progress — verifying actual cloned layout vs. expected |

## Pinned Submodule Commits

- **Ramulator:** `e62c84a6f0e06566ba6e182d308434b4532068a5`
- **SoftFloat:** `b51ef8f3201669b2288104c28546fc72532a1ea4`

## Fixes Applied So Far

**`scripts/wsl_setup_vortex.sh`**
- Skips toolchain re-download if `~/tools` already has it.
- Detects incomplete `third_party/softfloat` or `third_party/ramulator` folders (missing marker header) and reclones them fresh at the pinned commits above.
- If an incomplete folder is found, it renames it aside (e.g. `ramulator.incomplete`) before recloning — rerun the script after this happens.

**`scripts/wsl_sync_from_windows.sh`**
- Replacement for raw `rsync --delete`.
- Excludes from deletion/overwrite:
  - `vendor/vortex/build/`
  - `vendor/vortex/third_party/ramulator/`
  - `vendor/vortex/third_party/softfloat/`

**`scripts/wsl_run_agent_artifacts.sh`**
- Added a preflight check: verifies Ramulator/SoftFloat marker headers exist *before* starting the (slow) C++ build. Fails fast with repair instructions instead of dying ~5 minutes into compilation.

## Current Status

- Kernel compilation succeeds — `.vxbin` artifacts are produced correctly.
- Remaining blocker is isolated to `simx`'s Ramulator dependency (`dram_sim.cpp` can't find `base/base.h`).
- Kernel-level compiler/codegen logic (the actual agent rule system) is confirmed working and is **not** the source of remaining failures.

## Next Steps

1. Verify what Ramulator commit/layout is actually being cloned (check `.git` metadata inside `vendor/vortex/third_party/ramulator`).
2. Tighten the repair script so it either:
   - Produces the correct expected layout (`src/base/base.h` present), or
   - Clearly reports the actual cloned structure for diagnosis (e.g. `find vendor/vortex/third_party/ramulator -name '*.h' | head -40`).

## Manual Diagnostic Commands (if repair script still fails)

```bash
find vendor/vortex/third_party/ramulator -name base.h -o -name '*.h' | head -40
ls -la vendor/vortex/third_party/ramulator/src
```

## Manual Incomplete-Folder Recovery (fallback)

If the setup script reports an incomplete third-party folder but doesn't auto-fix it:

```bash
mv vendor/vortex/third_party/softfloat vendor/vortex/third_party/softfloat.incomplete
mv vendor/vortex/third_party/ramulator vendor/vortex/third_party/ramulator.incomplete
bash scripts/wsl_setup_vortex.sh
```
