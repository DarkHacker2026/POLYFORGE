#!/usr/bin/env python3
import sys
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))

from reference_isa import ParallelReferenceISA

def run_test(name, instructions, should_fail):
    print(f"--- Running: {name} ---")
    oracle = ParallelReferenceISA(num_threads=2, memory_size=256)
    try:
        oracle.execute_parallel(instructions, initial_regs=None, initial_mem={0: 10, 4: 20})
        if should_fail:
            print("FAIL: Expected a data race exception, but it passed!")
            return False
        else:
            print("PASS: Execution completed successfully as expected.")
            return True
    except Exception as e:
        if should_fail and "Data Race" in str(e):
            print(f"PASS: Caught expected data race -> {e}")
            return True
        else:
            print(f"FAIL: Unexpected error -> {e}")
            traceback.print_exc()
            return False

def main():
    # 1. RAW Race (Missing Barrier)
    # Thread 0 writes to addr 0. Thread 1 reads from addr 0.
    # Because there is no barrier, this is a RAW race.
    raw_race_insts = [
        {"op": "THREAD_ID", "dst": "r10"},
        # Addr = r10 * 4
        {"op": "SLLI", "dst": "r11", "src1": "r10", "imm": 2},
        # Write to memory[addr]
        {"op": "SW", "src2": "r10", "base": "r11", "offset": 0},
        # Read from memory[ (1 - r10) * 4 ] -> Thread 0 reads addr 4, Thread 1 reads addr 0
        # Compute 1 - r10
        {"op": "ADDI", "dst": "r12", "src1": "r0", "imm": 1},
        {"op": "SUB", "dst": "r13", "src1": "r12", "src2": "r10"},
        {"op": "SLLI", "dst": "r14", "src1": "r13", "imm": 2},
        {"op": "LW", "dst": "r15", "base": "r14", "offset": 0},
    ]

    # 2. RAW Safe (With Barrier)
    raw_safe_insts = [
        {"op": "THREAD_ID", "dst": "r10"},
        {"op": "SLLI", "dst": "r11", "src1": "r10", "imm": 2},
        {"op": "SW", "src2": "r10", "base": "r11", "offset": 0},
        {"op": "BARRIER"},  # <--- synchronizes memory writes
        {"op": "ADDI", "dst": "r12", "src1": "r0", "imm": 1},
        {"op": "SUB", "dst": "r13", "src1": "r12", "src2": "r10"},
        {"op": "SLLI", "dst": "r14", "src1": "r13", "imm": 2},
        {"op": "LW", "dst": "r15", "base": "r14", "offset": 0},
    ]

    # 3. WAW Race (Missing Barrier)
    # Both threads write to address 100
    waw_race_insts = [
        {"op": "THREAD_ID", "dst": "r10"},
        {"op": "ADDI", "dst": "r11", "src1": "r0", "imm": 100},
        {"op": "SW", "src2": "r10", "base": "r11", "offset": 0},
    ]

    results = [
        run_test("RAW Data Race (Missing Barrier)", raw_race_insts, should_fail=True),
        run_test("RAW Safe (With Barrier)", raw_safe_insts, should_fail=False),
        run_test("WAW Data Race (Missing Barrier)", waw_race_insts, should_fail=True),
    ]
    passed = sum(results)
    total = len(results)
    print(f"\nResult: {passed}/{total} tests passed")
    return 0 if passed == total else 1


def test_race_suite():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
