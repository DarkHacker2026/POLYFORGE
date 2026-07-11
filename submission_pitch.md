# Agentic Hardware-Software Co-Design: A Procedural CUDA Compiler for RISC-V

## 1. The Thesis (The North Star)
Writing high-performance, parallel software for custom or evolving hardware is currently a massive bottleneck. Compilers and drivers must be painstakingly hand-written for every new architecture.

**Our solution:** A fully autonomous agentic loop that gives developers CUDA-style control over GPUs that don't natively support CUDA. 

A user writes high-level parallel code *once*. Our system automatically maps it onto the target GPU's raw primitives — discovering those primitives empirically through a hardware simulator, and never trusting a mapping until an oracle strictly verifies it for race conditions and the cycle-accurate simulator proves its correctness.

---

## 2. The Core Discipline
> **The simulator is the only judge. Nothing commits without hardware-verified correctness AND measured cycles.**

The pipeline strictly enforces this through four vertical layers, which we successfully built and demonstrated end-to-end on the **Vortex RISC-V GPU**.

---

## 3. The 4-Layer Architecture (What We Built)

### Layer 1 — Agentic Hardware Discovery
Instead of hardcoding the compiler for Vortex, the system probes the unknown GPU via test C++ kernels run on the cycle-accurate `rtlsim` simulator. It discovered:
* `num_threads_per_warp`: 4
* `num_warps_per_core`: 4
* `barrier_supported`: True (mapped to `__syncthreads()`)

### Layer 2 — The Strict Parallel Oracle
Before touching the hardware, all generated kernels pass through our custom `ParallelReferenceISA` Python oracle. 
* It simulates parallel execution byte-by-byte. 
* **Data Race Detection:** It tracks every memory write by thread and synchronization epoch. If any thread reads a value written by another thread without a `barrier()` between them, it immediately throws a fatal `Data Race` exception. 

### Layer 3 — Procedural Surface Compiler
We designed a high-level CUDA-like language that **procedurally lowers** to Vortex C++ using *only* the facts discovered in Layer 1. The compiler explicitly logs its causal decisions:
> `[Compiler Causal Link] -> Probe discovered: barrier_supported = True -> Compiler Decision: Using native hardware barrier '__syncthreads()'`

### Layer 4 — Hardware Verification (rtlsim)
The final C++ is pushed to Vortex `rtlsim`. We proved scaling via a parallel Vector-Add (SAXPY) break-even analysis:
* N=16: 0.25x speedup (Spawn overhead dominates)
* N=64: 0.91x speedup (Break-even)
* **N=256: 1.81x speedup (GPU parallelism wins)**

We also successfully simulated a complex **Tree Reduction**, proving the system handles precise, multi-warp barrier synchronisation.

---

## 4. The "Killer Demo" — Live LLM End-to-End
We integrated the **Kimi-2.6** LLM via the live Fireworks API to complete the autonomous loop.

In a live network call, **Kimi-2.6 wrote a vectorized CUDA kernel.** 
1. Our compiler read the hardware probe facts and automatically lowered the LLM's surface code.
2. The Python Oracle simulated the instructions and strictly verified no data races existed.
3. The generated C++ was dispatched to `rtlsim` and executed flawlessly, printing `Passed! result matched expected`.

*(Honest Accounting: While this proved the end-to-end LLM integration loop for a simple kernel, sustained use requires a funded API key. Also, our complex barrier-sync reduction kernels were hand-written, as the base Vortex lacks a dedicated L1 scratchpad, forcing a fallback to global memory for `shared[]` constructs).*

---

## 5. The Retargetability Proof
To prove this architecture scales beyond Vortex, we ran a diff-demo. 
We fed the compiler the **exact same source code** (`parallel_for(i, N) { z[i] = x[i] * y[i]; } barrier();`), but swapped the probed facts to a fictional GPU (8 threads/warp, no hardware barrier).

The compiler automatically adapted, changing thread indices and falling back to a `vx_fence()` instruction. **Not a single line of kernel code was touched.**

We have built a compiler that learns the hardware it targets.
