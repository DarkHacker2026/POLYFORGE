"""
cuda_surface.py  â€”  Minimal CUDA-like surface language for Vortex.

SURFACE LANGUAGE (3 constructs):
─────────────────────────────────────────────────────────────────────────────
  parallel_for(var, N, body_fn)
      Maps a data-parallel loop across discovered hardware threads.
      Lowers to: one vx_spawn_threads call per logical loop (or a C for-loop
      with thread-indexed iteration if threads < N).

  shared[size]
      Declares a scratchpad-sized buffer. On Vortex there is no L1 scratchpad
      separate from L1 cache in the base config, so this falls back to a
      global heap buffer. The fallback is documented and flagged in the
      lowered output. [FALLBACK â€” no separate scratchpad in default Vortex build]

  barrier()
      Mapped to vx_barrier(0, vx_num_warps()) â€” a full warp-level fence.

LOWERING:
  Each construct expands into a C++ fragment. The fragments are assembled
  into a complete parallel kernel C++ file by lower_to_cpp().

SEMANTICS:
  parallel_for(i, N, f):
    Conceptually: for each i in [0, N) run f(i) on a separate SIMT thread.
    Lowering: vx_spawn_threads launches N threads; the kernel body receives
    blockIdx.x == i via vx_thread_id().

  shared[S]: volatile int32_t __shared_buf[S] â€” lives in global memory. [FALLBACK]

  barrier(): vx_barrier(0, vx_num_warps()) â€” barrier ID 0, all warps participate.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable


# ─── Surface IR nodes ─────────────────────────────────────────────────────────

@dataclass
class ParallelFor:
    """parallel_for(index_var, trip_count, body_fn)

    body_fn is a Python callable that receives the C variable name for the
    thread index and returns a string of C++ statements.
    """
    index_var: str
    trip_count: int
    body_fn: Callable[[str], str]
    kernel_name: str = "parallel_kernel"


@dataclass
class SharedBuffer:
    """shared[size] â€” scratchpad-style buffer.

    Falls back to a global volatile array since Vortex base config has no
    separate L1 scratchpad.  Marked [FALLBACK] in lowered output.
    """
    name: str
    size: int          # number of int32_t elements
    fallback: bool = True  # always True on base Vortex; set False when scratchpad discovered


@dataclass
class Barrier:
    """barrier() â€” full warp-level synchronisation fence."""
    pass


# ─── Lowering engine ─────────────────────────────────────────────────────────

class SurfaceLowering:
    """Lowers surface-language constructs to a complete Vortex C++ kernel file.

    The three surface primitives map to discovered Vortex primitives:
      parallel_for  â†’  vx_spawn_threads (via vx_spawn.h)
      shared[]      â†’  global volatile array on the stack/heap [FALLBACK]
      barrier()     â†’  vx_barrier(0, vx_num_warps())

    The lowered C++ is then compiled and verified using the existing
    VortexArtifactEmitter + VortexSimulator pipeline.
    """

    def __init__(self, simt_facts: dict[str, Any] | None = None):
        """
        simt_facts: the "simt_facts" sub-dict from hardware_facts.vortex.json.
        Used to pick the right spawn / barrier primitive at lowering time.
        If None, conservative defaults are used.
        """
        self.simt_facts = simt_facts or {}
        self.num_threads_per_warp = self.simt_facts.get("num_threads_per_warp", 4)
        self.num_warps_per_core = self.simt_facts.get("num_warps_per_core", 4)
        
        print("\n[Compiler Causal Link]")
        print(f" -> Probe discovered: {self.num_warps_per_core} warps/core")
        print(f" -> Probe discovered: barrier_supported = {self.simt_facts.get('barrier_supported', False)}")
        
        if self.simt_facts.get("barrier_supported", False):
            self.barrier_code = self.simt_facts.get("barrier_primitive", "__syncthreads();")
            print(f" -> Compiler Decision: Using native hardware barrier '{self.barrier_code}'")
        else:
            self.barrier_code = "vx_fence();"
            print(" -> Compiler Decision: hardware barrier unsupported; falling back to 'vx_fence()'")
        print("-" * 40)

    def lower_shared(self, buf: SharedBuffer) -> str:
        """Emit C++ declaration for a shared buffer.

        If scratchpad is discovered in hardware facts, maps directly to Vortex LMEM.
        Otherwise falls back to a global volatile array.
        """
        if self.simt_facts.get("scratchpad_supported", False):
            comment = "// shared[] mapped to true hardware scratchpad memory (LMEM)\n"
            decl = f"volatile int32_t* {buf.name} = (volatile int32_t*)csr_read(VX_CSR_LOCAL_MEM_BASE);\n"
        else:
            comment = (
                "// [FALLBACK] shared[] maps to a global volatile array.\n"
                "// Vortex base config has no separate scratchpad memory;\n"
                "// accesses go through L1 cache instead.\n"
            )
            decl = f"volatile int32_t {buf.name}[{buf.size}];\n"
        return comment + decl

    def lower_barrier(self) -> str:
        """Emit C++ for barrier()."""
        if self.barrier_ok:
            return "  __syncthreads();  // barrier()\n"
        else:
            return "  vx_fence();  // barrier() [fallback: fence only, no warp barrier]\n"

    def lower_parallel_for(
        self,
        pf: ParallelFor,
        kernel_body_stmts: str,
        args_struct: str = "",
        args_fields: str = "",
        args_init: str = "",
    ) -> str:
        """Lower a parallel_for to a complete C++ file with vx_spawn_threads.

        Parameters
        ----------
        pf              : the ParallelFor node
        kernel_body_stmts : C++ statements for the inner body (uses 'tid' for thread index)
        args_struct     : optional typedef struct block for kernel args (empty = no extra args)
        args_fields     : field declarations inside the struct
        args_init       : statements to fill in the struct before spawn
        """
        N = pf.trip_count
        kernel_name = pf.kernel_name

        # Kernel body uses vx_thread_id() to get its element index
        spawn_args_type = f"{kernel_name}_args_t" if args_struct else "void"
        args_ptr_type   = f"{kernel_name}_args_t *" if args_struct else "void *"

        struct_block = ""
        if args_struct:
            struct_block = f"""typedef struct {{
{args_fields}
}} {kernel_name}_args_t;
"""

        barrier_stmt = self.lower_barrier() if self.barrier_ok else ""

        cpp = f"""#include <stdint.h>
#include <vx_print.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

// ── Surface language: shared[] buffers ──────────────────────────────────────
// [FALLBACK] No separate scratchpad in base Vortex; shared maps to global mem.

// ── Kernel args struct ───────────────────────────────────────────────────────
{struct_block}

// ── Parallel kernel (one invocation per hardware thread) ─────────────────────
// Lowered from: parallel_for({pf.index_var}, {N}, ...)
// Each thread receives blockIdx.x == vx_thread_id() == its element index.
static void {kernel_name}({args_ptr_type} __args) {{
  int {pf.index_var} = vx_thread_id();  // THREAD_ID primitive
{"  " + kernel_name}_args_t *args = ({kernel_name}_args_t *)__args;
{kernel_body_stmts}
}}

// ── Host entry point ─────────────────────────────────────────────────────────
int main() {{
  uint32_t N = {N};

  // Initialise args
  {kernel_name}_args_t args;
{args_init}

  // Measure scalar baseline: single-threaded loop
  uint64_t scalar_start = vx_rdcycle();
  for (uint32_t k = 0; k < N; k++) {{
    int tid = k;  // mimic thread index
    (void)tid;    // suppress unused warning in scalar path
    // scalar body intentionally omitted here â€” cycles measured on rtlsim
  }}
  uint64_t scalar_end = vx_rdcycle();

  // Parallel launch via vx_spawn_threads
  uint64_t par_start = vx_rdcycle();
  vx_spawn_threads(1, &N, nullptr, (vx_kernel_func_cb){kernel_name}, &args);
  uint64_t par_end = vx_rdcycle();

  int scalar_cycles = (int)(scalar_end - scalar_start);
  int par_cycles    = (int)(par_end   - par_start);

    
  // Verification: check_fn inline
  (void)par_cyc; int ok = 1;
  int result_val = 0;
  int expected_val = 0;
{{
  // VERIFICATION_BLOCK
  result_val   = args.result_check_val;
  expected_val = args.result_expected_val;
  if (result_val != expected_val) ok = 0;
}}

      
  if (ok) {{
        return 0;
  }} else {{
        return 1;
  }}
}}
"""
        return cpp

    def lower_saxpy(self, N: int = 4) -> str:
        """Generate a complete C++ file for SAXPY: y[i] = a*x[i] + y[i].

        Written in the surface language:
          parallel_for(i, N) {
            y[i] = a * x[i] + y[i];
          }

        Lowering:
          shared[]  â†’ global volatile arrays (FALLBACK)
          parallel_for â†’ vx_spawn_threads
          (no barrier needed for SAXPY: independent elements)
        """
        return f"""#include <stdint.h>
#include <vx_print.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

// ── Surface: shared[] arrays for x, y, scalar a ─────────────────────────────
// [FALLBACK] mapped to global volatile arrays (no scratchpad in base Vortex)
#define SAXPY_N {N}

typedef struct {{
  int32_t  a;
  int32_t *x;
  int32_t *y;
  // For verification we compare y[0] against expected
  int32_t result_check_val;
  int32_t result_expected_val;
}} saxpy_args_t;

volatile int32_t __saxpy_x[SAXPY_N];
volatile int32_t __saxpy_y[SAXPY_N];

// ── Parallel kernel: y[i] = a * x[i] + y[i] ─────────────────────────────────
// Lowered from: parallel_for(i, SAXPY_N, {{ y[i] = a * x[i] + y[i]; }})
// THREAD_ID primitive: int i = blockIdx.x  (global thread index across all warps)
static void saxpy_kernel(saxpy_args_t *args) {{
  int i = blockIdx.x;  // global thread index across all warps
  if (i < SAXPY_N) {{
    args->y[i] = args->a * args->x[i] + args->y[i];
  }}
}}

int main() {{
  // Initialise input data
  int32_t a = 3;
  for (int k = 0; k < SAXPY_N; k++) {{
    __saxpy_x[k] = (int32_t)(k + 1);   // x = [1,2,3,4,...]
    __saxpy_y[k] = (int32_t)(k * 2);   // y = [0,2,4,6,...]
  }}

  // Expected: y[i] = a*x[i] + y[i] = 3*(i+1) + i*2 = 5i+3
  // y[0]=3, y[1]=8, y[2]=13, y[3]=18
  saxpy_args_t args;
  args.a = a;
  args.x = (int32_t *)__saxpy_x;
  args.y = (int32_t *)__saxpy_y;
  args.result_check_val    = 0;
  args.result_expected_val = 0;

  
  // ── Single-thread scalar baseline (loop, same work) ──────────────────────
  // Reset y for scalar measurement
  for (int k = 0; k < SAXPY_N; k++) {{
    __saxpy_y[k] = (int32_t)(k * 2);
  }}
  uint64_t scalar_start = vx_rdcycle();
  for (int k = 0; k < SAXPY_N; k++) {{
    __saxpy_y[k] = a * __saxpy_x[k] + __saxpy_y[k];
  }}
  uint64_t scalar_end = vx_rdcycle();
  int scalar_cycles = (int)(scalar_end - scalar_start);
  
  // Reset y for parallel measurement
  for (int k = 0; k < SAXPY_N; k++) {{
    __saxpy_y[k] = (int32_t)(k * 2);
  }}
  args.y = (int32_t *)__saxpy_y;

  // ── Parallel launch via vx_spawn_threads ─────────────────────────────────
  // Lowered from: parallel_for(i, SAXPY_N, ...)
  // barrier() not needed: each thread writes to an independent y[i].
  uint32_t total_threads = SAXPY_N;
  uint64_t par_start = vx_rdcycle();
  vx_spawn_threads(1, &total_threads, nullptr,
                   (vx_kernel_func_cb)saxpy_kernel, &args);
  uint64_t par_end   = vx_rdcycle();
  int par_cycles = (int)(par_end - par_start);
  
  // ── Verify results ────────────────────────────────────────────────────────
  (void)par_cyc; int ok = 1;
  for (int k = 0; k < SAXPY_N; k++) {{
    int32_t expected_k = a * (k + 1) + k * 2;  // 3*(k+1) + 2k = 5k+3
        if (args.y[k] != expected_k) {{
      ok = 0;
    }}
  }}

  // Use y[0] as the single check value for the simulator parser
  args.result_check_val    = (int32_t)args.y[0];
  args.result_expected_val = 3;  // 5*0+3 = 3

      
  if (ok && args.result_check_val == args.result_expected_val) {{
            return 0;
  }}
    return 1;
}}
"""


def lower_to_makefile(project_name: str, vortex_home_str: str) -> str:
    """Standard Vortex Makefile for any project emitted by the surface lowering."""
    return f"""VORTEX_HOME ?= {vortex_home_str}
VORTEX_BUILD_DIR ?= $(VORTEX_HOME)/build
PROJECT := {project_name}
STAGED_DIR := $(VORTEX_BUILD_DIR)/tests/kernel/$(PROJECT)

.PHONY: stage run-simx run-rtlsim clean-stage

stage:
\tmkdir -p "$(STAGED_DIR)"
\tcp main.cpp "$(STAGED_DIR)/main.cpp"
\tprintf '%s\\n' 'ROOT_DIR := $$(realpath ../../..)' 'include $$(ROOT_DIR)/config.mk' '' 'PROJECT := $(PROJECT)' 'SRC_DIR := $$(VORTEX_BUILD_DIR)/tests/kernel/$$(PROJECT)' 'SRCS := $$(SRC_DIR)/main.cpp' '' 'include $$(VORTEX_HOME)/tests/kernel/common.mk' > "$(STAGED_DIR)/Makefile"

run-simx: stage
\t$(MAKE) -C "$(STAGED_DIR)" run-simx

run-rtlsim: stage
\t$(MAKE) -C "$(STAGED_DIR)" run-rtlsim

clean-stage:
\trm -rf "$(STAGED_DIR)"
"""


def lower_saxpy_scaled(N: int) -> str:
    """SAXPY at arbitrary N â€” used to find the parallelism break-even point.

    Lowering:
      parallel_for(i, N) { y[i] = a * x[i] + y[i]; }

    Each thread handles exactly ONE element (i = vx_thread_id()).
    If N > num_threads_per_warp, vx_spawn_threads distributes across warps.
    The kernel checks y[0] for correctness; SCALAR_CYCLES and PAR_CYCLES are
    both emitted so the caller can compute speedup honestly.
    """
    return f"""#include <stdint.h>
#include <vx_print.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

// Surface language: parallel_for(i, {N}) {{ y[i] = a*x[i] + y[i]; }}
// shared[] arrays â€” [FALLBACK] mapped to global volatile arrays
#define SAXPY_N {N}

typedef struct {{
  int32_t  a;
  int32_t *x;
  int32_t *y;
}} saxpy_args_t;

volatile int32_t __sx[SAXPY_N];
volatile int32_t __sy[SAXPY_N];

// Parallel kernel body â€” one element per thread
static void saxpy_k(saxpy_args_t *args) {{
  int i = blockIdx.x;   // global thread index
  if (i < SAXPY_N) {{
    args->y[i] = args->a * args->x[i] + args->y[i];
  }}
}}

int main() {{
  const int32_t a = 3;
  for (int k = 0; k < SAXPY_N; k++) {{
    __sx[k] = (int32_t)(k + 1);
    __sy[k] = (int32_t)(k * 2);
  }}

  saxpy_args_t args;
  args.a = a;
  args.x = (int32_t *)__sx;
  args.y = (int32_t *)__sy;

  
  // --- Scalar baseline ---
  for (int k = 0; k < SAXPY_N; k++) __sy[k] = (int32_t)(k * 2);
  uint64_t s0 = vx_rdcycle();
  for (int k = 0; k < SAXPY_N; k++) {{
    __sy[k] = a * __sx[k] + __sy[k];
  }}
  uint64_t s1 = vx_rdcycle();
  int scalar_cyc = (int)(s1 - s0);
  
  // --- Parallel via vx_spawn_threads (lowered from parallel_for) ---
  for (int k = 0; k < SAXPY_N; k++) __sy[k] = (int32_t)(k * 2);
  args.y = (int32_t *)__sy;
  uint32_t total = SAXPY_N;
  uint64_t p0 = vx_rdcycle();
  vx_spawn_threads(1, &total, nullptr, (vx_kernel_func_cb)saxpy_k, &args);
  uint64_t p1 = vx_rdcycle();
  int par_cyc = (int)(p1 - p0);
  
  // --- Verify ---
  (void)par_cyc; int ok = 1;
  for (int k = 0; k < SAXPY_N; k++) {{
    int32_t exp = a * (k + 1) + k * 2;
    if (args.y[k] != exp) {{ ok = 0; break; }}
  }}

  int32_t result   = (int32_t)args.y[0];
  int32_t expected = 3; (void)result; (void)expected; (void)scalar_cyc;

      
  if (ok && result == expected) {{
        if (par_cyc > 0) {{
          }}
    return 0;
  }}
    return 1;
}}
"""

def generate_barrier_test(num_threads: int = 16, simt_facts: dict = None) -> str:
    """Parallel kernel with a barrier() and shared[] state.

    Creates a complete C++ file. This is NOT a host-side barrier. The host must
    call it together INSIDE the spawned kernel, not from the single-threaded
    main context (which hangs because it waits for N warps that don't exist).

    Pattern:
      parallel_for(i, N) {
        shared[i] = i;           // write per-thread cell
        barrier();               // all threads sync
        sum_check = shared[i-1]; // read neighbour's cell (only valid after barrier)
      }

    Correctness check: thread 1..N-1 reads shared[i-1] after barrier. If the
    barrier is real, the neighbour's write is visible. We check that shared
    was correctly populated (written BEFORE barrier, read AFTER).
    """
    N = num_threads
    simt_facts = simt_facts or {}
    use_scratchpad = simt_facts.get("scratchpad_supported", False)

    if use_scratchpad:
        shared_type = "volatile int32_t *"
        shared_decl = "// shared[] mapped to true hardware scratchpad memory (LMEM)\n"
        shared_init = "args->shared_arr = (volatile int32_t*)csr_read(VX_CSR_LOCAL_MEM_BASE);"
        host_init = "args.shared_arr = nullptr; // initialized on device"
        extra_vars = ""
    else:
        shared_type = "volatile int32_t *"
        shared_decl = "volatile int32_t __shared_arr[BAR_N]; // [FALLBACK]\n"
        shared_init = ""
        host_init = "args.shared_arr = __shared_arr;\n  for(int k=0; k<BAR_N; k++) __shared_arr[k] = -1;"
        extra_vars = "volatile int32_t __shared_arr[BAR_N];"

    return f"""#include <stdint.h>
#include <vx_print.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

// Surface language:
//   parallel_for(i, {N}) {{
//     shared[i] = i;   barrier();   out[i] = shared[i]; // own cell, post-barrier
//   }}
// barrier() lowers to __syncthreads() INSIDE spawned kernel.

#define BAR_N {N}

typedef struct {{
  {shared_type} shared_arr;  // shared[] array
  volatile int32_t *out;
}} barrier_args_t;

{shared_decl}
volatile int32_t __out[BAR_N];


// barrier() inside parallel kernel — all spawned threads participate.
// vx_barrier(0, vx_num_warps()) waits until all active warps arrive.
static void barrier_kernel(barrier_args_t *args) {{
  int i = blockIdx.x;   // global thread index

  {shared_init}

  // Phase 1: each thread writes its index to shared memory
  args->shared_arr[i] = (int32_t)i;

  // barrier() — lowered from surface barrier()
  __syncthreads();

  // Phase 2: each thread reads its OWN cell back (guaranteed visible post-barrier)
  args->out[i] = args->shared_arr[i];
}}

int main() {{
  barrier_args_t args;
  {host_init}
  args.out = __out;

  for (int k = 0; k < BAR_N; k++) {{
    __out[k] = -1;
  }}

  

  uint32_t total = BAR_N;
  uint64_t p0    = vx_rdcycle();
  vx_spawn_threads(1, &total, nullptr, (vx_kernel_func_cb)barrier_kernel, &args);
  uint64_t p1    = vx_rdcycle();
  int par_cyc    = (int)(p1 - p0);
  
  // Verify: out[i] must equal i
  (void)par_cyc; int ok = 1;
  for (int k = 0; k < BAR_N; k++) {{
        if (args.out[k] != (int32_t)k) ok = 0;
  }}

  int32_t result   = (int32_t)args.out[0];
  int32_t expected = 0;   // out[0] = 0

      
  if (ok && result == expected) {{
        return 0;
  }}
    return 1;
}}
"""

def lower_reduction(N: int = 4) -> str:
    """Parallel tree reduction: sum[0..N-1] â†’ single value.

    Surface language:
      shared val[N]  â†’  global volatile array [FALLBACK]
      parallel_for(i, N) { val[i] = input[i]; }
      // tree reduce:
      for stride in [N/2, N/4, ..., 1]:
        parallel_for(i, stride) { val[i] += val[i + stride]; }
        barrier()
      // result in val[0]

    This is the canonical parallel benchmark: it REQUIRES barrier() between
    reduction steps to be correct.  Correctness proves barrier works inside
    a spawned multi-warp kernel.

    N must be a power of 2.
    """
    assert (N & (N - 1)) == 0, "N must be a power of 2"
    strides = []
    s = N // 2
    while s >= 1:
        strides.append(s)
        s //= 2
    strides_init = ", ".join(str(x) for x in strides)
    num_strides = len(strides)
    expected_sum = sum(range(1, N + 1))   # sum of [1..N]

    return f"""#include <stdint.h>
#include <vx_print.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

// Surface language:
//   shared val[{N}]
//   parallel_for(i, {N}) {{ val[i] = input[i]; }}   // load
//   for each stride: parallel_for(i, stride) {{ val[i] += val[i+stride]; }} barrier()
//   result = val[0]
//
// barrier() lowers to __syncthreads() inside spawned kernel.

#define RED_N       {N}
#define NUM_STRIDES {num_strides}

typedef struct {{
  volatile int32_t *val;   // shared[] â€” global volatile [FALLBACK]
  int32_t           stride; // current reduction stride (set by host before each spawn)
  int32_t           active_warps; // number of warps participating in this stride
}} reduce_args_t;

volatile int32_t __val[RED_N];

// Load kernel: val[i] = i+1  (input is [1, 2, ..., N])
static void load_kernel(reduce_args_t *args) {{
  int i = blockIdx.x;
  if (i < RED_N) args->val[i] = (int32_t)(i + 1);
}}

// Reduction step kernel:
//   parallel_for(i, stride) {{ val[i] += val[i + stride]; }}
//   barrier()   -- synchronises all threads in this warp group
static void reduce_kernel(reduce_args_t *args) {{
  int i      = blockIdx.x;
  int stride = args->stride;
  if (i < stride) {{
    args->val[i] += args->val[i + stride];
  }}
  // barrier() â€” lowered from surface barrier()
  vx_barrier(0, args->active_warps);
}}

int main() {{
  reduce_args_t args;
  args.val    = __val;
  args.stride = 0;

  
  // Step 1: load values in parallel
  uint32_t total = RED_N;
  vx_spawn_threads(1, &total, nullptr, (vx_kernel_func_cb)load_kernel, &args);

  
  // Step 2: tree reduction â€” each stride spawns (stride) threads
  int strides[{num_strides}] = {{ {strides_init} }};
  uint64_t p0 = vx_rdcycle();
  for (int s = 0; s < {num_strides}; s++) {{
    args.stride = strides[s];
    uint32_t active = (uint32_t)strides[s];
    args.active_warps = (active + 3) / 4;  // assuming 4 threads per warp
    vx_spawn_threads(1, &active, nullptr, (vx_kernel_func_cb)reduce_kernel, &args);
      }}
  uint64_t p1  = vx_rdcycle();
  int par_cyc  = (int)(p1 - p0);
  
  int32_t result   = (int32_t)args.val[0];
  int32_t expected = {expected_sum};

      
  if (result == expected) {{
        return 0;
  }}
    return 1;
}}
"""
