#!/usr/bin/env python3
import sys, time
sys.path.insert(0, ".")
from reference_isa import ParallelReferenceISA

try:
    import psutil, os
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

print("\nscale_oracle.py -- Vector-add kernel scaling test")
print("-"*52)
print(f"  {'N':>6}  {'time_s':>8}  {'rss_mb':>8}  result")
print("-"*52)

ceiling = None
for N in [256, 1024, 4096, 16384, 65536]:
    mem_size = N * 12 + 256
    insts = [
        {"op": "THREAD_ID", "dst": "r10"},
        {"op": "SLLI", "dst": "r11", "src1": "r10", "imm": 2},
        {"op": "ADD",  "dst": "r12", "src1": "r1",  "src2": "r11"},
        {"op": "ADD",  "dst": "r13", "src1": "r2",  "src2": "r11"},
        {"op": "ADD",  "dst": "r14", "src1": "r3",  "src2": "r11"},
        {"op": "LW",   "dst": "r15", "base": "r12", "offset": 0},
        {"op": "LW",   "dst": "r16", "base": "r13", "offset": 0},
        {"op": "ADD",  "dst": "r17", "src1": "r15", "src2": "r16"},
        {"op": "SW",   "src2": "r17", "base": "r14", "offset": 0},
    ]
    initial_mem = {}
    for i in range(N):
        initial_mem[i*4]       = i + 1   # x[i]
        initial_mem[N*4+i*4]   = 10      # y[i]
        initial_mem[N*8+i*4]   = 0       # z[i]
    init_regs = {"r1": 0, "r2": N*4, "r3": N*8}

    try:
        t0 = time.perf_counter()
        oracle = ParallelReferenceISA(num_threads=N, memory_size=mem_size)
        results = oracle.execute_parallel(insts, init_regs, initial_mem)
        elapsed = time.perf_counter() - t0
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024**2 if HAS_PSUTIL else float("nan")
        print(f"  {N:>6}  {elapsed:>8.2f}  {rss:>8.1f}  PASS")
        if elapsed > 30:
            print(f"  TIME CEILING HIT at N={N}")
            ceiling = N
            break
        if HAS_PSUTIL and rss > 2048:
            print(f"  RAM CEILING HIT at N={N}")
            ceiling = N
            break
    except (RuntimeError, MemoryError, Exception) as e:
        print(f"  {N:>6}  {'N/A':>8}  {'N/A':>8}  FAIL ({type(e).__name__}: {e})")
        ceiling = N
        break

print("-"*52)
if ceiling:
    print(f"CEILING: first failure at N={ceiling}")
else:
    print("CEILING: not reached in tested range (N up to 65536)")
