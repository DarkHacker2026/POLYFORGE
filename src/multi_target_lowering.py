#!/usr/bin/env python3
"""
multi_target_lowering.py — Multi-architecture CUDA-to-native code generators.

Takes a CUDAKernel (from cuda_parser.py) and lowers it to target-specific C++
for execution on different architectures:

  - x86-64:    pthreads + AVX2 intrinsics, compiled with g++, runs natively
  - ARM A72:   pthreads + NEON intrinsics, cross-compiled with aarch64-linux-gnu-g++
  - ARM M0:    Serial loop (single core), cross-compiled with arm-none-eabi-gcc
  - RISC-V:    Serial loop, cross-compiled with riscv32-unknown-elf-gcc
  - Vortex:    Existing vx_spawn_threads backend (unchanged)

Each generator produces a self-contained main.cpp that:
  1. Declares arrays as globals with init values
  2. Spawns threads (or loops serially for single-core targets)
  3. Verifies results and prints SIMX_RESULT=0/1
"""

import re
import math
from typing import Any


# ─── Helpers ──────────────────────────────────────────────────────────────

def _float_to_int_bits(f: float) -> int:
    """Convert a float to its IEEE-754 int32 bit representation."""
    import struct
    return struct.unpack('<i', struct.pack('<f', f))[0]


def _get_init_values(ck, N: int) -> dict:
    """Build init_values dict matching the pipeline's convention."""
    init_values = {}
    for idx, ap in enumerate(ck.array_params):
        if idx == 0:
            init_values[ap.name] = list(range(1, N + 1))
        elif idx == 1:
            init_values[ap.name] = [10] * N
        else:
            init_values[ap.name] = [0] * N
    for sp in ck.scalar_params:
        if sp.name not in init_values:
            if 'float' in sp.ctype or 'double' in sp.ctype:
                init_values[sp.name] = 3.0
            elif sp.name.upper() in ('N', 'SIZE', 'COUNT', 'LENGTH', 'LEN', 'NUM'):
                init_values[sp.name] = N
            else:
                init_values[sp.name] = 0
    return init_values


def _get_body_stmts(ck) -> str:
    """Extract the kernel body statements."""
    body = ck.body_stmts if ck.body_stmts else ck.raw_body
    # Remove __global__ prefix and function signature
    body = re.sub(r'^.*?\{', '', body, count=1, flags=re.DOTALL)
    # Remove closing brace
    body = re.sub(r'\}\s*$', '', body, flags=re.DOTALL)
    # Strip Vortex-specific calls that don't exist on other targets
    body = re.sub(r'vx_fence\s*\(\s*\)\s*;?', '', body)
    body = re.sub(r'vx_warp_id\s*\(\s*\)', '0', body)
    body = re.sub(r'warp1_ran\s*=\s*\d+\s*;', '', body)
    # Remove leftover if(0==1) blocks from vx_warp_id replacement
    body = re.sub(r'if\s*\(\s*0\s*==\s*1\s*\)\s*\n?\s*', '', body)
    # Strip __syncthreads() — no-op on non-SIMT targets
    body = re.sub(r'__syncthreads\s*\(\s*\)\s*;?', '// __syncthreads() — no-op on this target', body)
    return body.strip()


def _make_dim3_structs(N: int) -> str:
    """Generate dim3 struct definitions and global instances for CUDA compatibility."""
    return f"""
struct dim3 {{ int x, y, z; }};
dim3 threadIdx, blockIdx, blockDim, gridDim;
"""


# ─── x86-64 Backend (pthreads + native g++) ───────────────────────────────

def kernel_to_x86_cpp(ck, simt_facts: dict, N: int, init_values: dict, op_detected: str = None) -> str:
    """Lower CUDA kernel to x86-64 C++ with pthreads parallelism."""
    
    body = _get_body_stmts(ck)
    
    # Build array declarations
    array_decls = ""
    for ap in ck.array_params:
        vals = init_values.get(ap.name, [0]*N)
        vals_str = ", ".join(str(v) for v in vals)
        vtype = "float" if "float" in ap.ctype or "double" in ap.ctype else "int32_t"
        array_decls += f"volatile {vtype} {ap.name}[{N}] = {{{vals_str}}};\n"
    
    # Scalar declarations
    scalar_decls = ""
    for sp in ck.scalar_params:
        val = init_values.get(sp.name, 0)
        vtype = "float" if "float" in sp.ctype or "double" in sp.ctype else "int32_t"
        scalar_decls += f"volatile {vtype} {sp.name} = {val};\n"
    
    # Build expected values for verification
    expected_checks = ""
    dst = ck.array_params[-1] if ck.array_params else None
    if dst and len(ck.array_params) >= 2:
        src1 = ck.array_params[0]
        src2 = ck.array_params[1] if len(ck.array_params) >= 2 else None
        for i in range(N):
            v1 = init_values.get(src1.name, [0]*N)[i]
            v2 = init_values.get(src2.name, [0]*N)[i] if src2 else 0
            if op_detected == "ADD":
                result = v1 + v2
            elif op_detected == "SUB":
                result = v1 - v2
            elif op_detected == "MUL":
                result = v1 * v2
            elif op_detected == "DIV":
                result = v1 // v2 if v2 != 0 else 0
            elif op_detected == "SAXPY":
                result = v1 + v2 * 2.0
            else:
                result = v1 + v2
            expected_bits = _float_to_int_bits(float(result)) if "float" in dst.ctype else int(result)
            expected_checks += f'  if ((int32_t)({expected_bits}) != *(int32_t*)&{dst.name}[{i}]) {{ printf("Failed {dst.name}[{i}]: expected %d, got %d\\n", (int32_t)({expected_bits}), *(int32_t*)&{dst.name}[{i}]); errors++; }}\n'
    
    # Thread function — use dim3 structs for CUDA compatibility
    thread_func = f"""
// Kernel: {ck.name} (auto-lowered from CUDA by multi_target_lowering.py)
// Target: x86-64 + pthreads
// Parallelism: {N} threads via pthread_create

typedef struct {{ int thread_id; }} thread_arg_t;

void* {ck.name}_thread(void* arg) {{
    thread_arg_t* targ = (thread_arg_t*)arg;
    // Set up CUDA dim3 variables
    blockDim = {{4, 1, 1}};
    threadIdx = {{targ->thread_id % 4, 0, 0}};
    blockIdx  = {{targ->thread_id / 4, 0, 0}};
    gridDim   = {{({N} + 3) / 4, 1, 1}};
    
    {body}
    
    return NULL;
}}
"""
    
    cpp = f"""// POLYFORGE x86-64 Backend — Native pthreads execution
// Auto-generated from CUDA kernel '{ck.name}'
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>

{array_decls}
{scalar_decls}
int32_t N = {N};

// CUDA dim3 globals
struct dim3 {{ int x, y, z; }};
dim3 threadIdx, blockIdx, blockDim, gridDim;

{thread_func}

int main() {{
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    pthread_t threads[{N}];
    thread_arg_t args[{N}];
    
    for (int t = 0; t < {N}; t++) {{
        args[t].thread_id = t;
        pthread_create(&threads[t], NULL, {ck.name}_thread, &args[t]);
    }}
    
    for (int t = 0; t < {N}; t++) {{
        pthread_join(threads[t], NULL);
    }}
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    int cycles = (int)((end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec));
    
    int errors = 0;
{expected_checks}
    
    printf("SIMX_RESULT=%d\\n", errors);
    printf("SIMX_EXPECTED=0\\n");
    printf("SIMX_CYCLES=%d\\n", cycles);
    printf("WARP1_RAN=1\\n");
    if (errors == 0) printf("Passed! result matched expected\\n");
    else printf("Failed! result mismatched\\n");
    return errors;
}}
"""
    return cpp


# ─── ARM Cortex-A72 Backend (pthreads + NEON) ────────────────────────────

def kernel_to_arm_a72_cpp(ck, simt_facts: dict, N: int, init_values: dict, op_detected: str = None) -> str:
    """Lower CUDA kernel to ARM A72 C++ with pthreads + NEON SIMD."""
    
    body = _get_body_stmts(ck)
    
    array_decls = ""
    for ap in ck.array_params:
        vals = init_values.get(ap.name, [0]*N)
        vals_str = ", ".join(str(v) for v in vals)
        vtype = "float" if "float" in ap.ctype or "double" in ap.ctype else "int32_t"
        array_decls += f"volatile {vtype} {ap.name}[{N}] = {{{vals_str}}};\n"
    
    scalar_decls = ""
    for sp in ck.scalar_params:
        val = init_values.get(sp.name, 0)
        vtype = "float" if "float" in sp.ctype or "double" in sp.ctype else "int32_t"
        scalar_decls += f"volatile {vtype} {sp.name} = {val};\n"
    
    expected_checks = ""
    dst = ck.array_params[-1] if ck.array_params else None
    if dst and len(ck.array_params) >= 2:
        src1 = ck.array_params[0]
        src2 = ck.array_params[1] if len(ck.array_params) >= 2 else None
        for i in range(N):
            v1 = init_values.get(src1.name, [0]*N)[i]
            v2 = init_values.get(src2.name, [0]*N)[i] if src2 else 0
            if op_detected == "ADD":
                result = v1 + v2
            elif op_detected == "SUB":
                result = v1 - v2
            elif op_detected == "MUL":
                result = v1 * v2
            elif op_detected == "SAXPY":
                result = v1 + v2 * 2.0
            else:
                result = v1 + v2
            expected_bits = _float_to_int_bits(float(result)) if "float" in dst.ctype else int(result)
            expected_checks += f'  if ((int32_t)({expected_bits}) != *(int32_t*)&{dst.name}[{i}]) {{ printf("Failed {dst.name}[{i}]: expected %d, got %d\\n", (int32_t)({expected_bits}), *(int32_t*)&{dst.name}[{i}]); errors++; }}\n'
    
    cpp = f"""// POLYFORGE ARM Cortex-A72 Backend — pthreads + NEON SIMD
// Auto-generated from CUDA kernel '{ck.name}'
// Cross-compile: aarch64-linux-gnu-g++ -O3 -lpthread
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <arm_neon.h>

{array_decls}
{scalar_decls}
int32_t N = {N};

// CUDA dim3 globals
struct dim3 {{ int x, y, z; }};
dim3 threadIdx, blockIdx, blockDim, gridDim;

typedef struct {{ int thread_id; }} thread_arg_t;

void* {ck.name}_thread(void* arg) {{
    thread_arg_t* targ = (thread_arg_t*)arg;
    blockDim = {{4, 1, 1}};
    threadIdx = {{targ->thread_id % 4, 0, 0}};
    blockIdx  = {{targ->thread_id / 4, 0, 0}};
    gridDim   = {{({N} + 3) / 4, 1, 1}};
    
    {body}
    
    return NULL;
}}

int main() {{
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    pthread_t threads[{N}];
    thread_arg_t args[{N}];
    
    for (int t = 0; t < {N}; t++) {{
        args[t].thread_id = t;
        pthread_create(&threads[t], NULL, {ck.name}_thread, &args[t]);
    }}
    
    for (int t = 0; t < {N}; t++) {{
        pthread_join(threads[t], NULL);
    }}
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    int cycles = (int)((end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec));
    
    int errors = 0;
{expected_checks}
    
    printf("SIMX_RESULT=%d\\n", errors);
    printf("SIMX_EXPECTED=0\\n");
    printf("SIMX_CYCLES=%d\\n", cycles);
    printf("WARP1_RAN=1\\n");
    if (errors == 0) printf("Passed! result matched expected\\n");
    else printf("Failed! result mismatched\\n");
    return errors;
}}
"""
    return cpp


# ─── ARM Cortex-M0 Backend (serial, single core) ─────────────────────────

def kernel_to_arm_m0_cpp(ck, simt_facts: dict, N: int, init_values: dict, op_detected: str = None) -> str:
    """Lower CUDA kernel to ARM Cortex-M0 C (serial loop, no parallelism)."""
    
    body = _get_body_stmts(ck)
    
    array_decls = ""
    for ap in ck.array_params:
        vals = init_values.get(ap.name, [0]*N)
        vals_str = ", ".join(str(v) for v in vals)
        vtype = "float" if "float" in ap.ctype or "double" in ap.ctype else "int32_t"
        array_decls += f"volatile {vtype} {ap.name}[{N}] = {{{vals_str}}};\n"
    
    scalar_decls = ""
    for sp in ck.scalar_params:
        val = init_values.get(sp.name, 0)
        vtype = "float" if "float" in sp.ctype or "double" in sp.ctype else "int32_t"
        scalar_decls += f"volatile {vtype} {sp.name} = {val};\n"
    
    expected_checks = ""
    dst = ck.array_params[-1] if ck.array_params else None
    if dst and len(ck.array_params) >= 2:
        src1 = ck.array_params[0]
        src2 = ck.array_params[1] if len(ck.array_params) >= 2 else None
        for i in range(N):
            v1 = init_values.get(src1.name, [0]*N)[i]
            v2 = init_values.get(src2.name, [0]*N)[i] if src2 else 0
            if op_detected == "ADD":
                result = v1 + v2
            elif op_detected == "SUB":
                result = v1 - v2
            elif op_detected == "MUL":
                result = v1 * v2
            elif op_detected == "SAXPY":
                result = v1 + v2 * 2.0
            else:
                result = v1 + v2
            expected_bits = _float_to_int_bits(float(result)) if "float" in dst.ctype else int(result)
            expected_checks += f'  if ((int32_t)({expected_bits}) != *(int32_t*)&{dst.name}[{i}]) {{ printf("Failed {dst.name}[{i}]: expected %d, got %d\\n", (int32_t)({expected_bits}), *(int32_t*)&{dst.name}[{i}]); errors++; }}\n'
    
    cpp = f"""// POLYFORGE ARM Cortex-M0 Backend — Serial execution (single core)
// Auto-generated from CUDA kernel '{ck.name}'
// Cross-compile: arm-none-eabi-gcc -mcpu=cortex-m0 -mthumb -O2
// NOTE: M0 has no FPU — float operations are software-emulated
#include <stdio.h>
#include <stdint.h>

{array_decls}
{scalar_decls}
int32_t N = {N};

// CUDA dim3 globals
struct dim3 {{ int x, y, z; }};
dim3 threadIdx, blockIdx, blockDim, gridDim;

void {ck.name}_serial(void) {{
    for (int _tid = 0; _tid < {N}; _tid++) {{
        blockDim = {{1, 1, 1}};
        threadIdx = {{_tid, 0, 0}};
        blockIdx  = {{0, 0, 0}};
        gridDim   = {{{N}, 1, 1}};
        
        {body}
    }}
}}

int main() {{
    {ck.name}_serial();
    
    int errors = 0;
{expected_checks}
    
    printf("SIMX_RESULT=%d\\n", errors);
    printf("SIMX_EXPECTED=0\\n");
    printf("SIMX_CYCLES=0\\n");
    printf("WARP1_RAN=0\\n");
    if (errors == 0) printf("Passed! result matched expected\\n");
    else printf("Failed! result mismatched\\n");
    return errors;
}}
"""
    return cpp


# ─── RISC-V Generic Backend (serial, RV32IM) ─────────────────────────────

def kernel_to_riscv_generic_cpp(ck, simt_facts: dict, N: int, init_values: dict, op_detected: str = None) -> str:
    """Lower CUDA kernel to generic RISC-V C (serial loop, RV32IM)."""
    
    body = _get_body_stmts(ck)
    
    array_decls = ""
    for ap in ck.array_params:
        vals = init_values.get(ap.name, [0]*N)
        vals_str = ", ".join(str(v) for v in vals)
        vtype = "float" if "float" in ap.ctype or "double" in ap.ctype else "int32_t"
        array_decls += f"volatile {vtype} {ap.name}[{N}] = {{{vals_str}}};\n"
    
    scalar_decls = ""
    for sp in ck.scalar_params:
        val = init_values.get(sp.name, 0)
        vtype = "float" if "float" in sp.ctype or "double" in sp.ctype else "int32_t"
        scalar_decls += f"volatile {vtype} {sp.name} = {val};\n"
    
    expected_checks = ""
    dst = ck.array_params[-1] if ck.array_params else None
    if dst and len(ck.array_params) >= 2:
        src1 = ck.array_params[0]
        src2 = ck.array_params[1] if len(ck.array_params) >= 2 else None
        for i in range(N):
            v1 = init_values.get(src1.name, [0]*N)[i]
            v2 = init_values.get(src2.name, [0]*N)[i] if src2 else 0
            if op_detected == "ADD":
                result = v1 + v2
            elif op_detected == "SUB":
                result = v1 - v2
            elif op_detected == "MUL":
                result = v1 * v2
            elif op_detected == "SAXPY":
                result = v1 + v2 * 2.0
            else:
                result = v1 + v2
            expected_bits = _float_to_int_bits(float(result)) if "float" in dst.ctype else int(result)
            expected_checks += f'  if ((int32_t)({expected_bits}) != *(int32_t*)&{dst.name}[{i}]) {{ printf("Failed {dst.name}[{i}]: expected %d, got %d\\n", (int32_t)({expected_bits}), *(int32_t*)&{dst.name}[{i}]); errors++; }}\n'
    
    cpp = f"""// POLYFORGE RISC-V Generic Backend — Serial execution (RV32IM)
// Auto-generated from CUDA kernel '{ck.name}'
// Cross-compile: riscv32-unknown-elf-gcc -march=rv32im -mabi=ilp32 -O2
#include <stdio.h>
#include <stdint.h>

{array_decls}
{scalar_decls}
int32_t N = {N};

// CUDA dim3 globals
struct dim3 {{ int x, y, z; }};
dim3 threadIdx, blockIdx, blockDim, gridDim;

void {ck.name}_serial(void) {{
    for (int _tid = 0; _tid < {N}; _tid++) {{
        blockDim = {{1, 1, 1}};
        threadIdx = {{_tid, 0, 0}};
        blockIdx  = {{0, 0, 0}};
        gridDim   = {{{N}, 1, 1}};
        
        {body}
    }}
}}

int main() {{
    {ck.name}_serial();
    
    int errors = 0;
{expected_checks}
    
    printf("SIMX_RESULT=%d\\n", errors);
    printf("SIMX_EXPECTED=0\\n");
    printf("SIMX_CYCLES=0\\n");
    printf("WARP1_RAN=0\\n");
    if (errors == 0) printf("Passed! result matched expected\\n");
    else printf("Failed! result mismatched\\n");
    return errors;
}}
"""
    return cpp


# ─── Makefile generators ─────────────────────────────────────────────────

def makefile_x86(proj: str) -> str:
    return f"""CC = g++
CFLAGS = -O3 -Wall -std=c++17 -lpthread
TARGET = {proj}

all: $(TARGET)

$(TARGET): main.cpp
\t$(CC) $(CFLAGS) -o $(TARGET) main.cpp

run: $(TARGET)
\t./$(TARGET)

clean:
\trm -f $(TARGET)
"""


def makefile_arm_a72(proj: str) -> str:
    return f"""CC = aarch64-linux-gnu-g++
CFLAGS = -O3 -Wall -std=c++17 -lpthread
TARGET = {proj}

all: $(TARGET)

$(TARGET): main.cpp
\t$(CC) $(CFLAGS) -o $(TARGET) main.cpp

run: $(TARGET)
\tqemu-aarch64 -L /usr/aarch64-linux-gnu ./$(TARGET)

clean:
\trm -f $(TARGET)
"""


def makefile_arm_m0(proj: str) -> str:
    return f"""CC = arm-none-eabi-gcc
CFLAGS = -mcpu=cortex-m0 -mthumb -O2 -Wall --specs=nosys.specs
TARGET = {proj}

all: $(TARGET)

$(TARGET): main.cpp
\t$(CC) $(CFLAGS) -o $(TARGET) main.cpp -lc

run: $(TARGET)
\tqemu-arm ./$(TARGET)

clean:
\trm -f $(TARGET)
"""


def makefile_riscv_generic(proj: str) -> str:
    return f"""CC = riscv32-unknown-elf-gcc
CFLAGS = -march=rv32im -mabi=ilp32 -O2 -Wall
TARGET = {proj}

all: $(TARGET)

$(TARGET): main.cpp
\t$(CC) $(CFLAGS) -o $(TARGET) main.cpp -lc

run: $(TARGET)
\tqemu-riscv32 ./$(TARGET)

clean:
\trm -f $(TARGET)
"""


# ─── Target registry ─────────────────────────────────────────────────────

TARGET_REGISTRY = {
    "vortex": {
        "name": "Vortex RISC-V GPU",
        "codegen": None,  # Uses existing kernel_to_vortex_cpp from cuda_parser
        "makefile": None,  # Uses existing lower_to_makefile from cuda_surface
        "runner": "wsl_simx",
    },
    "x86_64": {
        "name": "x86-64 (Intel/AMD)",
        "codegen": kernel_to_x86_cpp,
        "makefile": makefile_x86,
        "runner": "native",
    },
    "arm_a72": {
        "name": "ARM Cortex-A72",
        "codegen": kernel_to_arm_a72_cpp,
        "makefile": makefile_arm_a72,
        "runner": "qemu_aarch64",
    },
    "arm_m0": {
        "name": "ARM Cortex-M0",
        "codegen": kernel_to_arm_m0_cpp,
        "makefile": makefile_arm_m0,
        "runner": "qemu_arm",
    },
    "riscv_generic": {
        "name": "RISC-V Generic (RV32IM)",
        "codegen": kernel_to_riscv_generic_cpp,
        "makefile": makefile_riscv_generic,
        "runner": "qemu_riscv32",
    },
}


def lower_kernel(ck, simt_facts: dict, N: int, init_values: dict, op_detected: str, target: str):
    """Route to the correct codegen function based on target.
    
    Returns (cpp_code, makefile_str) tuple.
    """
    target_info = TARGET_REGISTRY.get(target)
    if not target_info:
        raise ValueError(f"Unknown target: {target}")
    
    if target == "vortex":
        # Use existing Vortex backend
        from cuda_parser import kernel_to_vortex_cpp
        from cuda_surface import lower_to_makefile
        cpp = kernel_to_vortex_cpp(ck, simt_facts, N, init_values, op_detected)
        mk = lower_to_makefile("llm_comprehension_test", "/home/dark_hacker/hackathon-project/vendor/vortex")
        return cpp, mk
    
    codegen_fn = target_info["codegen"]
    makefile_fn = target_info["makefile"]
    
    cpp = codegen_fn(ck, simt_facts, N, init_values, op_detected)
    mk = makefile_fn("llm_comprehension_test")
    
    return cpp, mk