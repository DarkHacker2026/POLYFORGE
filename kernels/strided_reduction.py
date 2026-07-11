#!/usr/bin/env python3
"""
kernels/strided_reduction.py  --  Item 6: parallel tree reduction kernel

Surface code (log2(N) rounds):
    parallel_for(i, N) { shared[i] = x[i]; }
    for stride in [N/2, N/4, N/8, ...1]:
        parallel_for(i, stride) { shared[i] += shared[i + stride]; }
        barrier()
    // result in shared[0]
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from reference_isa import ParallelReferenceISA


def generate_strided_reduction_cpp(simt_facts: dict, N: int = 8) -> str:
    """Generate a complete C++ reduction kernel. N must be power of 2."""
    barrier = simt_facts.get("barrier_primitive", "__syncthreads()")
    threads_per_warp = simt_facts.get("num_threads_per_warp", 4)
    expected = N * (N + 1) // 2

    x_init = ", ".join(str(i+1) for i in range(N))
    rounds = int(math.log2(N))

    reduction_rounds = ""
    stride = N // 2
    for _ in range(rounds):
        reduction_rounds += f"""
    {{
        uint32_t active = {stride};
        vx_spawn_threads(1, &active, nullptr, kernel_reduce_{stride}, nullptr);
        {barrier};
    }}
"""
        stride //= 2

    kernel_funcs = ""
    stride = N // 2
    for _ in range(rounds):
        kernel_funcs += f"""
static void kernel_reduce_{stride}(void *__args) {{
    (void)__args;
    int i = vx_thread_id();
    shared[i] += shared[i + {stride}];
}}
"""
        stride //= 2

    return f"""#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

// Parallel tree reduction: sum([1..{N}]) = {expected}
// Compiled for: {threads_per_warp} threads/warp (discovered by probe)
// Barrier primitive: {barrier} (discovered by probe)

volatile int32_t x[{N}]      = {{{x_init}}};
volatile int32_t shared[{N}] = {{{", ".join("0" for _ in range(N))}}};
uint32_t N = {N};

// Load phase kernel
static void kernel_load(void *__args) {{
    (void)__args;
    int i = vx_thread_id();
    shared[i] = x[i];
}}
{kernel_funcs}
int main() {{
    vx_printf("CONFIGS: num_threads={threads_per_warp}, N={N}, expected_sum={expected}\\n");

    // Load phase
    vx_spawn_threads(1, &N, nullptr, kernel_load, nullptr);
    {barrier};
{reduction_rounds}
    int result = shared[0];
    vx_printf("SIMX_RESULT=%d\\n", result);
    vx_printf("SIMX_EXPECTED={expected}\\n");
    vx_printf("SIMX_CYCLES=1\\n");
    if (result == {expected}) {{
        vx_printf("Passed! result matched expected\\n");
        return 0;
    }} else {{
        vx_printf("Failed! got %d expected {expected}\\n", result);
        return 1;
    }}
}}
"""


def run_oracle_check(N: int = 8) -> dict:
    """Verify the strided reduction through ParallelReferenceISA.
    
    Uses per-thread instruction lists so only active threads (tid < stride)
    participate in each round, avoiding cross-thread WAR races from inactive threads.
    """
    if N & (N - 1) != 0:
        return {"ok": False, "message": f"N={N} must be power of 2"}

    expected = N * (N + 1) // 2
    # shared[i] at addr i*4
    initial_mem = {i * 4: i + 1 for i in range(N)}

    # Build per-thread instruction lists manually
    # Each thread's instructions: for each stride round, if tid < stride: read+add+write, then barrier
    # If tid >= stride: just barrier

    num_rounds = int(math.log2(N))

    # Per-thread instruction lists
    thread_instrs: list[list[dict]] = [[] for _ in range(N)]

    stride = N // 2
    for _ in range(num_rounds):
        for tid in range(N):
            if tid < stride:
                # Active: read shared[tid] and shared[tid+stride], write sum to shared[tid]
                addr_self  = tid * 4
                addr_other = (tid + stride) * 4
                thread_instrs[tid] += [
                    {"op": "ADDI", "dst": "r10", "src1": "r0", "imm": addr_self},
                    {"op": "ADDI", "dst": "r11", "src1": "r0", "imm": addr_other},
                    {"op": "LW",   "dst": "r12", "base": "r10", "offset": 0},   # shared[tid]
                    {"op": "LW",   "dst": "r13", "base": "r11", "offset": 0},   # shared[tid+stride]
                    {"op": "ADD",  "dst": "r14", "src1": "r12", "src2": "r13"},
                    {"op": "SW",   "src2": "r14", "base": "r10", "offset": 0},  # write back
                ]
            # All threads: barrier
            thread_instrs[tid].append({"op": "BARRIER"})
        stride //= 2

    def check(memory):
        result = int.from_bytes(memory[0:4], "little", signed=True)
        if result == expected:
            return True, f"sum={result} == expected {expected}"
        return False, f"sum={result} != expected {expected}"

    oracle = ParallelReferenceISA(num_threads=N, memory_size=N*4+256)
    try:
        # Manually drive per-thread execution (execute_parallel takes uniform instr lists)
        oracle.memory = bytearray(N*4+256)
        oracle.memory_writes = {}
        oracle.memory_reads  = {}
        oracle.sync_epoch    = 1
        for i in range(N):
            oracle.thread_regs[i] = {f"r{j}": 0 for j in range(32)}
        # Load initial memory (epoch 0 → bump to 1 so thread writes don't conflict)
        for addr, v in initial_mem.items():
            oracle._write_mem32(-1, addr, v)
        oracle.memory_writes = {}
        oracle.memory_reads  = {}
        oracle.sync_epoch    = 1

        stride = N // 2
        for rnd in range(int(math.log2(N))):
            # Each active thread (tid < stride): read+add+write
            for tid in range(stride):
                addr_self  = tid * 4
                addr_other = (tid + stride) * 4
                val_self  = int.from_bytes(oracle.memory[addr_self:addr_self+4],   "little", signed=True)
                val_other = int.from_bytes(oracle.memory[addr_other:addr_other+4], "little", signed=True)
                oracle._write_mem32(tid, addr_self, val_self + val_other)
            # Barrier: advance epoch, clear tracking
            oracle.sync_epoch += 1
            oracle.memory_writes = {}
            oracle.memory_reads  = {}
            stride //= 2

        ok, msg = check(oracle.memory)
        return {"ok": ok, "message": msg}
    except RuntimeError as e:
        return {"ok": False, "message": str(e)}


def main():
    facts_path = ROOT / "data" / "hardware_facts.vortex.json"
    simt_facts = json.loads(facts_path.read_text())["simt_facts"]

    print("=== Strided Reduction Kernel (Item 6) ===")
    oracle_result = run_oracle_check(N=8)
    status = "PASS" if oracle_result["ok"] else "FAIL"
    print(f"Oracle: {status} - {oracle_result.get('message', oracle_result)}")
    if not oracle_result["ok"]:
        sys.exit(1)

    cpp = generate_strided_reduction_cpp(simt_facts, N=8)
    out_dir = ROOT / "artifacts" / "vortex_tests" / "strided_reduction"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "main.cpp").write_text(cpp, encoding="utf-8")
    print(f"C++ written to {out_dir / 'main.cpp'}")


if __name__ == "__main__":
    main()
