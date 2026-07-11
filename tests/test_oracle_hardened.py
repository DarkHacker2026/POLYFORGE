#!/usr/bin/env python3
"""
test_oracle_hardened.py

8 test cases proving the hardened ParallelReferenceISA catches hazards
that the original oracle missed: WAR, partial overlaps, multi-barrier chains.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_isa import ParallelReferenceISA

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def run_test(name: str, instructions, num_threads, initial_mem, initial_regs=None, expect_race=False):
    oracle = ParallelReferenceISA(num_threads=num_threads, memory_size=65536)
    try:
        oracle.execute_parallel(instructions, initial_regs, initial_mem)
        if expect_race:
            print(f"  [{FAIL}] {name}: expected a race but oracle PASSED")
            return False
        else:
            print(f"  [{PASS}] {name}: correctly passed (no race)")
            return True
    except RuntimeError as e:
        if expect_race:
            print(f"  [{PASS}] {name}: correctly caught -> {e}")
            return True
        else:
            print(f"  [{FAIL}] {name}: unexpected race -> {e}")
            return False

def main():
    results = []
    print("\n=== Oracle Hardened Test Suite ===\n")

    # ── Test 1: WAR — missing barrier (Thread 0 reads, Thread 1 writes same addr) ──
    print("-- WAR Hazards --")
    insts_war_bad = [
        {"op": "THREAD_ID", "dst": "r1"},
        # Thread 0: read addr 0
        {"op": "LW", "dst": "r2", "base": "r0", "offset": 0},
        # Thread 1: write addr 0  (WAR: Thread 0 already read it, Thread 1 writes it)
        {"op": "SW", "src2": "r1", "base": "r0", "offset": 0},
    ]
    results.append(run_test(
        "WAR — missing barrier (must FAIL)",
        insts_war_bad, num_threads=2,
        initial_mem={0: 42},
        expect_race=True
    ))

    # ── Test 2: WAR — barrier present (safe): T0 writes addr 0, T1 writes addr 4 in next epoch ──
    # Each thread writes to its own separate address → no overlap at all
    insts_war_good = [
        {"op": "THREAD_ID", "dst": "r1"},
        # Thread 0: read addr 0
        {"op": "LW", "dst": "r2", "base": "r0", "offset": 0},
        {"op": "BARRIER"},
        # After barrier: Thread 0 writes addr 0, Thread 1 writes addr 4 (separate addrs)
        {"op": "SLLI", "dst": "r3", "src1": "r1", "imm": 2},  # r3 = tid*4
        {"op": "SW",   "src2": "r1", "base": "r3", "offset": 0},
    ]
    results.append(run_test(
        "WAR — barrier present, separate addrs post-barrier (must PASS)",
        insts_war_good, num_threads=2,
        initial_mem={0: 42, 4: 0},
        expect_race=False
    ))

    # ── Test 3: Partial Overlap WAW — Thread 0 writes bytes 0-3, Thread 1 writes bytes 2-5 ──
    print("\n-- Partial Overlap Hazards --")
    insts_overlap_bad = [
        {"op": "THREAD_ID", "dst": "r1"},
        # Thread 0 writes to addr 0 (bytes 0-3)
        # Thread 1 writes to addr 2 (bytes 2-5) — overlaps bytes 2-3
        {"op": "ADDI", "dst": "r3", "src1": "r1", "imm": 0},   # offset = tid * 2
        {"op": "SLLI", "dst": "r3", "src1": "r1", "imm": 1},
        {"op": "SW",   "src2": "r1", "base": "r3", "offset": 0},
    ]
    results.append(run_test(
        "Partial Overlap WAW — bytes 0-3 vs 2-5 (must FAIL)",
        insts_overlap_bad, num_threads=2,
        initial_mem={},
        expect_race=True
    ))

    # ── Test 4: WAW — aligned, same thread writes same addr twice (safe, no cross-thread race) ──
    insts_same_thread = [
        {"op": "THREAD_ID", "dst": "r1"},
        {"op": "SLLI", "dst": "r2", "src1": "r1", "imm": 2},   # addr = tid * 4 (no overlap)
        {"op": "SW",   "src2": "r1", "base": "r2", "offset": 0},
        {"op": "SW",   "src2": "r1", "base": "r2", "offset": 0},  # same thread, same addr — safe
    ]
    results.append(run_test(
        "WAW — same thread writes same addr twice (must PASS)",
        insts_same_thread, num_threads=4,
        initial_mem={},
        expect_race=False
    ))

    # ── Test 5: Multi-barrier 2-epoch, clean — each thread writes to its own addr ──
    # T0 writes addr 0, T1 writes addr 4, T2 writes addr 8 — no overlap ever
    insts_2epoch_clean = [
        {"op": "THREAD_ID", "dst": "r1"},
        {"op": "SLLI",   "dst": "r2",  "src1": "r1", "imm": 2},   # r2 = tid*4
        # epoch 1: each thread writes its own slot
        {"op": "SW",     "src2": "r1",  "base": "r2", "offset": 0},
        {"op": "BARRIER"},
        # epoch 2: each thread reads the *next* thread's slot (stride=4)
        {"op": "ADDI",   "dst": "r3",  "src1": "r2", "imm": 4},
        {"op": "LW",     "dst": "r4",   "base": "r3", "offset": 0},
        {"op": "BARRIER"},
        # epoch 3: each thread writes addr 64+tid*4 (fresh addresses)
        {"op": "ADDI",   "dst": "r5",  "src1": "r2", "imm": 64},
        {"op": "SW",     "src2": "r1",  "base": "r5", "offset": 0},
    ]
    results.append(run_test(
        "Multi-barrier 2-epoch, clean separate addresses (must PASS)",
        insts_2epoch_clean, num_threads=3,
        initial_mem={0: 0, 4: 0, 8: 0, 12: 0},
        expect_race=False
    ))

    # ── Test 6: Multi-barrier epoch leakage — WAR across 3 epochs without proper sync ──
    # epoch1: T0 reads addr 0
    # barrier → epoch2
    # epoch2: T1 reads addr 0
    # NO barrier → still epoch2
    # T0 writes addr 0 → WAR with T1's read in epoch2
    insts_3epoch_leak = [
        {"op": "THREAD_ID", "dst": "r1"},
        {"op": "LW",  "dst": "r2",  "base": "r0", "offset": 0},     # T0,T1 read addr 0
        {"op": "BARRIER"},
        {"op": "LW",  "dst": "r2",  "base": "r0", "offset": 0},     # T0,T1 read again in epoch2
        # T1 now writes addr 0 in epoch2 — WAR with T0's read in epoch2
        {"op": "SW",  "src2": "r1", "base": "r0", "offset": 0},
    ]
    results.append(run_test(
        "Multi-barrier epoch leakage — WAR across epoch boundary (must FAIL)",
        insts_3epoch_leak, num_threads=2,
        initial_mem={0: 99},
        expect_race=True
    ))

    # ── Test 7: Non-power-of-2 thread count (5 threads) — must not crash ──
    print("\n-- Edge Cases --")
    insts_5threads = [
        {"op": "THREAD_ID", "dst": "r1"},
        {"op": "SLLI", "dst": "r2", "src1": "r1", "imm": 2},
        {"op": "SW",   "src2": "r1", "base": "r2", "offset": 0},
    ]
    results.append(run_test(
        "5 threads (non-power-of-2) — must not crash (PASS)",
        insts_5threads, num_threads=5,
        initial_mem={},
        expect_race=False
    ))

    # ── Test 8: Single thread baseline — no race checks should fire ──
    insts_single = [
        {"op": "LW",  "dst": "r1", "base": "r0", "offset": 0},
        {"op": "ADDI","dst": "r1", "src1": "r1", "imm": 1},
        {"op": "SW",  "src2": "r1","base": "r0", "offset": 0},
        {"op": "LW",  "dst": "r2", "base": "r0", "offset": 0},
    ]
    results.append(run_test(
        "Single-thread baseline — must PASS, no race checks fire",
        insts_single, num_threads=1,
        initial_mem={0: 41},
        expect_race=False
    ))

    # ── Summary ──
    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*45}")
    print(f"Result: {passed}/{total} tests passed")
    if passed == total:
        print("All hardened oracle tests PASSED.")
    else:
        print(f"WARNING: {total - passed} test(s) FAILED.")
    print(f"{'='*45}\n")
    return 0 if passed == total else 1


def test_hardened_suite():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
