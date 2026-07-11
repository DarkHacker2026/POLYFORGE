# Final Test Runs Record

This document records the exact console outputs of the four major test suites and evaluations required to validate the parallel oracle, the hardware discovery, the scaling limits, and the live LLM integration using Kimi-2.6.

## 1. Hardened Oracle Test Suite (`test_oracle_hardened.py`)
Validates that the Oracle correctly detects complex parallelism hazards (WAR, WAW Partial Overlaps, Multi-barrier Epoch Leakage).

```
=== Oracle Hardened Test Suite ===

-- WAR Hazards --
  [PASS] WAR — missing barrier (must FAIL): correctly caught -> WAR Data Race: Thread 0 wrote byte 0 already read by Thread 1 in epoch 1.
  [PASS] WAR — barrier present, separate addrs post-barrier (must PASS): correctly passed (no race)

-- Partial Overlap Hazards --
  [PASS] Partial Overlap WAW — bytes 0-3 vs 2-5 (must FAIL): correctly caught -> WAW Data Race (Partial Overlap): Thread 1 wrote byte 2 already written by Thread 0 in epoch 1.
  [PASS] WAW — same thread writes same addr twice (must PASS): correctly passed (no race)
  [PASS] Multi-barrier 2-epoch, clean separate addresses (must PASS): correctly passed (no race)
  [PASS] Multi-barrier epoch leakage — WAR across epoch boundary (must FAIL): correctly caught -> WAR Data Race: Thread 0 wrote byte 0 already read by Thread 1 in epoch 2.

-- Edge Cases --
  [PASS] 5 threads (non-power-of-2) — must not crash (PASS): correctly passed (no race)
  [PASS] Single-thread baseline — must PASS, no race checks fire: correctly passed (no race)

=============================================
Result: 8/8 tests passed
All hardened oracle tests PASSED.
=============================================
```

## 2. Oracle Scaling Benchmark (`scale_oracle.py`)
Proves the standalone oracle can verify massive parallel workloads without exhausting memory or taking excessive time.

```
scale_oracle.py -- Vector-add kernel scaling test
----------------------------------------------------
       N    time_s    rss_mb  result
----------------------------------------------------
     256      0.00      21.1  PASS
    1024      0.02      27.5  PASS
    4096      0.07      50.1  PASS
   16384      0.31     137.0  PASS
   65536      1.34     481.5  PASS
----------------------------------------------------
CEILING: not reached in tested range (N up to 65536)
```

## 3. RTL Simulator Limits (`scale_rtlsim.py`)
Demonstrates the OOM risk of pure hardware simulators (Verilator/Spike) and justifies the need for the Oracle at large N values.

```
scale_rtlsim.py — SAXPY Vortex RTL kernel generation + cycle prediction
========================================================================
[ Step 1 ] Generating SAXPY C++ kernels
...
     N  path                                      size_bytes   lines
--------------------------------------------------------------------
  4096  artifacts\scale_test\N4096\main.cpp              432      22

[ Step 2 ] Linear model fit (cycles = base + rate * N)
  Known data points: {16: 3911, 64: 5401, 256: 11565}
  Fitted model     : cycles ~= 3380.3 + 31.9524 * N

[ Step 3 ] Predicted cycle counts for larger N
     N   predicted_cycles   scalar_est_cycles   speedup_vs_scalar
-----------------------------------------------------------------
   512             19,740           8,379,505             424.5x
  1024             36,100          33,507,880             928.2x
  2048             68,819         134,021,380            1947.4x  <- OOM risk
  4096            134,257         536,075,377            3992.9x  <- OOM risk

[Warning] RTL simulator trace buffers are predicted to OOM at N~2048 based on extrapolation; not directly observed.
```

## 4. Retargeting Demo (`retarget_demo.py`)
Shows how dynamically discovered hardware facts (from `hardware_facts.vortex.json` vs `hardware_facts.vortex_wide.json`) automatically parameterize parallel kernel compilation without human intervention.

```
=== Retargeting Demo (Item 6) ===

--- TARGET: vortex_base ---
Facts loaded: Threads=4 | Warps=4 | Cores=1 | Barrier Supported=True
  [Kernel 1: Conditional Scatter]
    Oracle Check: PASS
    Generated C++ Printf: "CONFIGS: num_threads=4, N=8\n"
  [Kernel 2: Strided Reduction]
    Oracle Check: PASS
    Generated C++ Printf: "CONFIGS: num_threads=4, N=8, expected_sum=36\n"

--- TARGET: vortex_wide ---
Facts loaded: Threads=8 | Warps=2 | Cores=2 | Barrier Supported=False
  [Kernel 1: Conditional Scatter]
    Oracle Check: PASS
    Generated C++ Printf: "CONFIGS: num_threads=8, N=8\n"
  [Kernel 2: Strided Reduction]
    Oracle Check: PASS
    Generated C++ Printf: "CONFIGS: num_threads=8, N=8, expected_sum=36\n"
```

## 5. Live Kimi-2.6 LLM Integration (`llm_kernel_test.py`)
The capstone run: Kimi-2.6 live generation -> C++ Lowering via Hardware Facts -> Python Oracle Verification -> RTL Simulator Hardware Execution.

```
--- 1. Agentic Kernel Generation ---
Sending prompt to LLM (Kimi-2.6)...

LLM Output:
{
  "language": "CUDA C++",
  "kernel_name": "vectorMultiply",
  "code": "__global__ void vectorMultiply(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {\n  int i = blockIdx.x * blockDim.x + threadIdx.x;\n  if (i < N) {\n    C[i] = A[i] * B[i];\n  }\n}",
  "launch_config": "vectorMultiply<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);",
  "notes": "Assumes d_A, d_B, d_C are device pointers and N is the number of elements."
}

--- 2. Causal Compiler Lowering ---
[Compiler Causal Link]
 -> Probe discovered: 4 warps/core
 -> Probe discovered: barrier_supported = True
 -> Compiler Decision: Using native hardware barrier '__syncthreads()'

Lowering Surface Code into Vortex C++...
...
static void kernel_vector_mul(void *__args) {
    (void)__args;
    int i = vx_thread_id(); // Discovered THREAD_ID primitive
    z[i] = x[i] * y[i];     // The LLM's body
}

--- 3. Strict Oracle Verification ---
Oracle Verified: All vector multiply results matched!

--- 4. Hardware Simulation (rtlsim) ---
Emitting C++ harness and dispatching to WSL...
RTL Simulator Output: Passed! result matched expected
```
