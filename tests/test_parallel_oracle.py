#!/usr/bin/env python3
"""
test_parallel_oracle.py

Proves the ParallelReferenceISA oracle is correct on a hand-written
multi-threaded example — same style as the Phase 1 bit-for-bit test
against rtlsim.

Test program: 4-thread parallel vector-add.
  Thread i:  out[i] = in_a[i] + in_b[i]

Memory layout:
  in_a[] at byte 0:   [10, 20, 30, 40]
  in_b[] at byte 64:  [1,  2,  3,  4]
  out[]  at byte 128: [0,  0,  0,  0]  (written by threads)

Expected output:
  out = [11, 22, 33, 44]

Instructions per thread (same kernel on every thread):
  THREAD_ID  r10          -- i = vx_thread_id()
  SLLI       r11, r10, 2  -- byte_offset = i << 2 = i * 4
  ADD        r12, r1, r11 -- a_ptr = base_a + offset
  ADD        r13, r2, r11 -- b_ptr = base_b + offset
  ADD        r14, r3, r11 -- out_ptr = base_out + offset
  LW         r15, 0(r12)  -- a_val = in_a[i]
  LW         r16, 0(r13)  -- b_val = in_b[i]
  ADD        r17, r15, r16 -- sum = a_val + b_val
  SW         r17, 0(r14)  -- out[i] = sum

Initial regs (same for all threads):
  r1 = 0   (base_a)
  r2 = 64  (base_b)
  r3 = 128 (base_out)
"""

import ctypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))

from reference_isa import ParallelReferenceISA

NUM_THREADS = 4
BASE_A   = 0
BASE_B   = 64
BASE_OUT = 128

def build_initial_memory():
    mem = {}
    for i in range(NUM_THREADS):
        mem[BASE_A   + i * 4] = (i + 1) * 10   # in_a: [10,20,30,40]
        mem[BASE_B   + i * 4] = i + 1            # in_b: [1,2,3,4]
        mem[BASE_OUT + i * 4] = 0                # out:  [0,0,0,0]
    return mem

INSTRUCTIONS = [
    {"op": "THREAD_ID", "dst": "r10"},                               # i = tid
    {"op": "SLLI",  "dst": "r11", "src1": "r10", "imm": 2},          # byte offset
    {"op": "ADD",   "dst": "r12", "src1": "r1",  "src2": "r11"},     # a_ptr
    {"op": "ADD",   "dst": "r13", "src1": "r2",  "src2": "r11"},     # b_ptr
    {"op": "ADD",   "dst": "r14", "src1": "r3",  "src2": "r11"},     # out_ptr
    {"op": "LW",    "dst": "r15", "base": "r12", "offset": 0},       # a_val
    {"op": "LW",    "dst": "r16", "base": "r13", "offset": 0},       # b_val
    {"op": "ADD",   "dst": "r17", "src1": "r15", "src2": "r16"},     # sum
    {"op": "SW",    "src2": "r17", "base": "r14", "offset": 0},      # out[i] = sum
]

def run_oracle_test():
    print("=" * 60)
    print("Parallel Oracle Test: 4-thread vector-add")
    print("=" * 60)

    oracle = ParallelReferenceISA(num_threads=NUM_THREADS, memory_size=256)
    initial_mem  = build_initial_memory()
    initial_regs = {"r1": BASE_A, "r2": BASE_B, "r3": BASE_OUT}

    results = oracle.execute_parallel(INSTRUCTIONS, initial_regs, initial_mem)

    # Verify per-thread register state
    print("\nPer-thread register results:")
    reg_ok = True
    for tid in range(NUM_THREADS):
        r17 = results[tid].get("r17", None)
        expected = (tid + 1) * 10 + (tid + 1)  # 10*(i+1) + (i+1) = 11*(i+1)
        status = "OK" if r17 == expected else f"FAIL (got {r17} expected {expected})"
        print(f"  Thread {tid}: r17={r17}  expected={expected}  [{status}]")
        if r17 != expected:
            reg_ok = False

    # Verify shared memory
    print("\nShared memory out[] results:")
    mem_ok = True
    for i in range(NUM_THREADS):
        addr     = BASE_OUT + i * 4
        got      = int.from_bytes(oracle.memory[addr:addr+4], byteorder='little', signed=True)
        expected = (i + 1) * 10 + (i + 1)
        status = "OK" if got == expected else f"FAIL (got {got} expected {expected})"
        print(f"  out[{i}] at byte {addr}: {got}  expected={expected}  [{status}]")
        if got != expected:
            mem_ok = False

    # Thread ID verification
    print("\nThread ID (THREAD_ID pseudo-op) check:")
    tid_ok = True
    for tid in range(NUM_THREADS):
        r10 = results[tid].get("r10", -1)
        status = "OK" if r10 == tid else f"FAIL (got {r10} expected {tid})"
        print(f"  Thread {tid}: r10={r10}  [{status}]")
        if r10 != tid:
            tid_ok = False

    print("\n" + "=" * 60)
    all_ok = reg_ok and mem_ok and tid_ok
    if all_ok:
        print("RESULT: ALL CHECKS PASSED")
        print("ParallelReferenceISA correctly executes a 4-thread kernel with:")
        print("  - THREAD_ID pseudo-op  (vx_thread_id() equivalent)")
        print("  - Shared bytearray memory (read and written by all threads)")
        print("  - Per-thread independent register files")
        print("  - SLLI, ADD, LW, SW, ADD instructions")
    else:
        print("RESULT: ONE OR MORE CHECKS FAILED")
    print("=" * 60)
    return all_ok


def run_barrier_test():
    """Verify that BARRIER is a no-op that doesn't disrupt execution."""
    print("\n" + "=" * 60)
    print("Parallel Oracle Test: BARRIER pseudo-op (no-op verification)")
    print("=" * 60)

    # 2 threads: each reads its cell, adds 1, writes back, hits barrier, reads again
    instructions = [
        {"op": "THREAD_ID", "dst": "r10"},
        {"op": "SLLI",  "dst": "r11", "src1": "r10", "imm": 2},
        {"op": "ADD",   "dst": "r12", "src1": "r1",  "src2": "r11"},
        {"op": "LW",    "dst": "r13", "base": "r12", "offset": 0},
        {"op": "ADDI",  "dst": "r14", "src1": "r13", "imm": 1},
        {"op": "BARRIER"},  # should be a no-op
        {"op": "ADD",   "dst": "r15", "src1": "r14", "src2": "r14"},  # r15 = 2*(val+1)
    ]

    oracle = ParallelReferenceISA(num_threads=2, memory_size=64)
    initial_mem  = {0: 5, 4: 10}   # cell[0]=5, cell[1]=10
    initial_regs = {"r1": 0}

    results = oracle.execute_parallel(instructions, initial_regs, initial_mem)
    # Thread 0: r14 = 5+1 = 6, r15 = 12
    # Thread 1: r14 = 10+1 = 11, r15 = 22
    ok = (results[0]["r15"] == 12 and results[1]["r15"] == 22)
    print(f"  Thread 0: r15={results[0]['r15']} (expected 12)  "
          f"{'OK' if results[0]['r15'] == 12 else 'FAIL'}")
    print(f"  Thread 1: r15={results[1]['r15']} (expected 22)  "
          f"{'OK' if results[1]['r15'] == 22 else 'FAIL'}")
    print(f"BARRIER no-op test: {'PASSED' if ok else 'FAILED'}")
    return ok


def test_parallel_oracle():
    ok1 = run_oracle_test()
    ok2 = run_barrier_test()
    assert ok1 and ok2


if __name__ == "__main__":
    ok1 = run_oracle_test()
    ok2 = run_barrier_test()
    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL TESTS PASSED' if ok1 and ok2 else 'SOME TESTS FAILED'}")
    print("=" * 60)
    sys.exit(0 if (ok1 and ok2) else 1)
