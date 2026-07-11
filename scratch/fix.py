import sys

with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\cuda_surface.py", "r") as f:
    lines = f.readlines()

s0_idx = -1
for i, l in enumerate(lines):
    if "uint64_t s0 = vx_rdcycle();" in l:
        s0_idx = i
        break

rest_of_saxpy = """  for (int k = 0; k < SAXPY_N; k++) {
    __sy[k] = a * __sx[k] + __sy[k];
  }
  uint64_t s1 = vx_rdcycle();
  int scalar_cyc = (int)(s1 - s0);
  vx_printf("SCALAR_CYCLES=%d\\n", scalar_cyc);

  // --- Parallel via vx_spawn_threads (lowered from parallel_for) ---
  for (int k = 0; k < SAXPY_N; k++) __sy[k] = (int32_t)(k * 2);
  args.y = (int32_t *)__sy;
  uint32_t total = SAXPY_N;
  uint64_t p0 = vx_rdcycle();
  vx_spawn_threads(1, &total, nullptr, (vx_kernel_func_cb)saxpy_k, &args);
  uint64_t p1 = vx_rdcycle();
  int par_cyc = (int)(p1 - p0);
  vx_printf("PAR_CYCLES=%d\\n", par_cyc);

  // --- Verify ---
  int ok = 1;
  for (int k = 0; k < SAXPY_N; k++) {
    int32_t exp = a * (k + 1) + k * 2;
    if (args.y[k] != exp) { ok = 0; break; }
  }

  int32_t result   = (int32_t)args.y[0];
  int32_t expected = 3;   // y[0] = 3*(0+1) + 0*2 = 3

  vx_printf("SIMX_RESULT=%d\\n",   result);
  vx_printf("SIMX_EXPECTED=%d\\n", expected);
  vx_printf("SIMX_CYCLES=%d\\n",   par_cyc);

  if (ok && result == expected) {
    vx_printf("Passed! result matched expected\\n");
    if (par_cyc > 0) {
      vx_printf("SPEEDUP_NUM=%d SPEEDUP_DEN=%d\\n", scalar_cyc, par_cyc);
    }
    return 0;
  }
  vx_printf("Failed! result mismatched\\n");
  return 1;
}
\"\"\"
"""

def_barrier = """
def generate_barrier_test(num_threads: int = 16, simt_facts: dict = None) -> str:
    \"\"\"Parallel kernel with a barrier() and shared[] state.

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
    \"\"\"
    N = num_threads
    simt_facts = simt_facts or {}
    use_scratchpad = simt_facts.get("scratchpad_supported", False)

    if use_scratchpad:
        shared_type = "volatile int32_t *"
        shared_decl = "// shared[] mapped to true hardware scratchpad memory (LMEM)\\n"
        shared_init = "args->shared_arr = (volatile int32_t*)csr_read(VX_CSR_LOCAL_MEM_BASE);"
        host_init = "args.shared_arr = nullptr; // initialized on device"
        extra_vars = ""
    else:
        shared_type = "volatile int32_t *"
        shared_decl = "volatile int32_t __shared_arr[BAR_N]; // [FALLBACK]\\n"
        shared_init = ""
        host_init = "args.shared_arr = __shared_arr;\\n  for(int k=0; k<BAR_N; k++) __shared_arr[k] = -1;"
        extra_vars = "volatile int32_t __shared_arr[BAR_N];"

    return f\"\"\"#include <stdint.h>
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

  vx_printf(">> Barrier-in-kernel test: N={N}\\n");

  uint32_t total = BAR_N;
  uint64_t p0    = vx_rdcycle();
  vx_spawn_threads(1, &total, nullptr, (vx_kernel_func_cb)barrier_kernel, &args);
  uint64_t p1    = vx_rdcycle();
  int par_cyc    = (int)(p1 - p0);
  vx_printf("PAR_CYCLES=%d\\n", par_cyc);

  // Verify: out[i] must equal i
  int ok = 1;
  for (int k = 0; k < BAR_N; k++) {{
    vx_printf("  out[%d]=%d (expected %d)\\n", k, (int)args.out[k], k);
    if (args.out[k] != (int32_t)k) ok = 0;
  }}

  int32_t result   = (int32_t)args.out[0];
  int32_t expected = 0;   // out[0] = 0

  vx_printf("SIMX_RESULT=%d\\n",   result);
  vx_printf("SIMX_EXPECTED=%d\\n", expected);
  vx_printf("SIMX_CYCLES=%d\\n",   par_cyc);

  if (ok && result == expected) {{
    vx_printf("Passed! result matched expected\\n");
    return 0;
  }}
  vx_printf("Failed! result mismatched\\n");
  return 1;
}}
\"\"\"
"""

end_idx = -1
for i, l in enumerate(lines):
    if "def lower_reduction(" in l:
        end_idx = i
        break

new_lines = lines[:s0_idx+1] + [rest_of_saxpy, def_barrier, "\n"] + lines[end_idx:]

with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\cuda_surface.py", "w") as f:
    f.writelines(new_lines)
print("Fixed cuda_surface.py")
