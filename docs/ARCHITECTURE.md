# POLYFORGE Architecture

```
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CUDA .cu  │───▶│  LLM Comprehend │───▶│  Zero-Trust     │───▶│  Hardware       │───▶│  SIMX / RTL     │───▶│  PASS / FAIL    │
│   file      │    │  (Kimi-2.6)      │    │  Oracle          │    │  Lowering       │    │  Execution      │    │  Verdict        │
└─────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
      │                    │                      │                      │                      │
      ▼                    ▼                      ▼                      ▼                      ▼
 cuda_parser.py      grow_compiler.py      reference_isa.py        cuda_surface.py       vortex_compile.py
```

## Stage 1: LLM Comprehension (`grow_compiler.py`, `cuda_parser.py`)
The raw CUDA kernel is sent to Kimi-2.6 via Fireworks AI with a strict JSON schema prompt. The LLM extracts kernel name, parameters, thread indexing, bounds checks, operations, shared memory, and local variables. `cuda_parser.py` then converts this JSON IR into a `CUDAKernel` dataclass and lowers the body to Vortex C++.

## Stage 2: Zero-Trust Oracle (`reference_isa.py`)
The LLM output is **never trusted**. The `ParallelReferenceISA` runs the extracted kernel semantics on a software SIMT model with per-thread registers and shared memory. It detects RAW, WAW, and WAR data races at the byte level. If the oracle rejects the kernel, the pipeline stops before hardware compilation.

## Stage 3: Hardware Lowering (`cuda_surface.py`, `cuda_parser.py`)
The verified `CUDAKernel` is lowered to target-specific C++ with inline assembly. `cuda_surface.py` handles surface-language constructs (`parallel_for`, `shared[]`, `barrier()`) and maps them to Vortex intrinsics (`vx_spawn_threads`, `__local_mem`, `__syncthreads`).

## Stage 4: Hardware Execution (`vortex_compile.py`, `discovery_agent.py`)
The generated C++ is cross-compiled with `riscv32-unknown-elf-clang`, linked against the Vortex runtime, and executed in the `simx` simulator via WSL2. `discovery_agent.py` probes the target hardware at runtime to discover register counts, ISA latencies, SIMT dimensions, and barrier support.

## Stage 5: Rule Learning (`grow_compiler.py`)
On success, the LLM abstracts the verified instruction sequence into a reusable, parameterized rule. The `RuleDatabase` stores it in `data/rules.json`. Replacement is deterministic: a new rule overwrites an old one **only** if its hardware cycle count is strictly lower.

## Stage 6: Offline Verification (`oracle_standalone.py`, `demo_offline.py`)
The entire oracle engine is available as a standalone library. `demo_offline.py` runs the 8-test race-detection suite and a SAXPY kernel without any API keys or WSL.

---

## Zero-Trust Philosophy

POLYFORGE treats the LLM as an untrusted agent. Every output is verified before it reaches hardware:

1. **JSON schema validation** — malformed responses are retried or rejected.
2. **Static checker** — illegal registers or unsupported opcodes are caught before compilation.
3. **Parallel oracle** — byte-level race detection proves the kernel is safe to run in parallel.
4. **Hardware assertion** — `SIMX_RESULT` must equal `SIMX_EXPECTED` or the kernel fails.

The LLM has zero authority to overwrite a learned rule. Only a lower hardware cycle count can trigger replacement.

---

## Adding a New Hardware Target

To retarget POLYFORGE to a new architecture (e.g., a custom FPGA or another RISC-V GPU):

1. **Probe the hardware** — modify `discovery_agent.py`:
   - Add new CSR probes or micro-benchmarks to `discover_isa_and_latencies()`.
   - Emit a `hardware_facts.<target>.json` with register files, ISA latencies, and SIMT dimensions.

2. **Write a lowering backend** — modify `cuda_surface.py`:
   - Implement a new `SurfaceLowering` subclass (or extend the existing one).
   - Map `parallel_for` → your spawn primitive, `shared[]` → your scratchpad, `barrier()` → your fence.

3. **Wire the CLI** — modify `vortex_compile.py`:
   - Add a `--target` flag and load the correct `hardware_facts` JSON.
   - Replace the WSL/Vortex-specific `run_wsl()` calls with your target's compilation and execution hooks.

No changes to the LLM prompt, the oracle, or the CUDA parser are required.
