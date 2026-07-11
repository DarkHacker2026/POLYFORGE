# Hardware Discovery Log — Provenance Guide

## What This File Is

`hardware_facts.vortex.json` contains every hardware fact used by the compiler.
This guide explains how to verify each fact is genuinely measured — not hand-tuned
after the fact.

---

## How to Verify a Fact

Every fact in the `provenance` array has:

| Field | What it is |
|---|---|
| `fact` | The key in `simt_facts` this entry explains |
| `measured_value` | The value actually returned by the hardware |
| `probe_source_hash` | SHA-256 of the C++ probe source used |
| `probe_c_source` | The full C++ probe source (re-runnable) |
| `rtlsim_raw_output` | The raw stdout from `rtlsim`/`simx` |
| `timestamp_utc` | When the probe ran |
| `derived_by` | The Python function that ran the probe |

### Verification Steps

1. **Hash check**: Compute `sha256(probe_c_source)` and compare to `probe_source_hash`.
   This proves the source that ran is the same source stored here.
   ```bash
   echo -n "<paste probe_c_source here>" | sha256sum
   ```

2. **Re-run the probe**: Copy `probe_c_source` to a `.cpp` file, compile with the
   Vortex toolchain, and run on `rtlsim`. The `SIMX_RESULT=` line in stdout must
   match `measured_value`.
   ```bash
   # Example for num_threads_per_warp probe:
   cp probe.cpp ~/hackathon-project/artifacts/vortex_tests/simt_probe_num_threads/main.cpp
   cd ~/hackathon-project
   make -C artifacts/vortex_tests/simt_probe_num_threads run-simx
   # Expected: SIMX_RESULT=4
   ```

3. **Check the raw output**: The `rtlsim_raw_output` field shows exactly what the
   simulator printed. Re-running should produce the same lines.

---

## Why This Matters

A compiler that hardcodes hardware constants is not retargetable — it just looks
retargetable. The only way to prove empirical discovery is to show the measurement
chain: probe source → raw output → derived fact.

Anyone can challenge a fact by re-running the probe on the same simulator binary.
If the re-run matches, the fact is verified. If it doesn't, the binary changed —
which is itself a meaningful signal.

---

## Facts Covered

| Fact | Probe Function | Measured Value |
|---|---|---|
| `num_threads_per_warp` | `vx_num_threads()` | 4 (base) / 8 (wide) |
| `num_warps_per_core` | `vx_num_warps()` | 4 (base) / 2 (wide) |
| `num_cores` | `vx_num_cores()` | 1 (base) / 2 (wide) |
| `barrier_supported` | `vx_barrier(0, 1)` | true |

---

## Second Hardware Config (Vortex Wide)

`hardware_facts.vortex_wide.json` contains facts probed from a second real Vortex
build compiled with `DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=2 -DVX_CFG_NUM_CORES=2`.

The wide config binary is at `vendor/vortex/build_wide/simx`.

To rebuild the wide config and re-probe:
```bash
cd ~/hackathon-project/vendor/vortex/build/sim/simx
make CONFIGS='-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=2 -DVX_CFG_NUM_CORES=2' \
     DESTDIR=/home/dark_hacker/hackathon-project/vendor/vortex/build_wide -j$(nproc)

# Temporarily swap and re-probe:
cp vendor/vortex/build/sim/simx/simx vendor/vortex/build/sim/simx/simx.bak
cp vendor/vortex/build_wide/simx vendor/vortex/build/sim/simx/simx
python3 discovery_agent.py --simt --sim simx
cp vendor/vortex/build/sim/simx/simx.bak vendor/vortex/build/sim/simx/simx
```

This produces `data/hardware_facts.vortex_wide.json` with the wide config's values.
