# POLYFORGE Oracle

> A race-detecting oracle that verifies parallel kernel correctness — before you touch hardware.

---

## What It Is

**POLYFORGE** includes a standalone, hardware-agnostic oracle that catches data races in parallel kernels at the IR level, before any silicon is involved. It detects all three classes of hazard:

- **RAW** (Read-After-Write) — a thread reads a location before a writing thread's result is made visible via a barrier.
- **WAW** (Write-After-Write) — two threads write the same location in the same sync epoch with no ordering guarantee.
- **WAR** (Write-After-Read) — a thread overwrites a location that another thread is still reading within the same epoch.

The oracle is completely decoupled from any specific compiler, surface language, or ISA. It consumes a simple JSON IR and produces a deterministic pass/fail report.

---

## Why It Exists

Most parallel compilers targeting custom hardware — RISC-V GPUs, FPGAs, domain-specific accelerators — have no reference model to validate against. CUDA and OpenCL tools assume their own memory models. This oracle fills that gap: it gives any custom compiler team a fast, free, language-neutral correctness check grounded in the actual memory-access semantics of their kernel, not in assumptions borrowed from another architecture.

---

## How It Works

The oracle simulates all threads executing a minimal READ/WRITE/BARRIER IR, tracking every byte-level memory access annotated with thread ID and synchronization epoch. If any cross-thread hazard is detected without a separating BARRIER between the two conflicting accesses, the oracle raises a fatal race error with a full diagnostic showing the conflicting threads, addresses, access types, and epochs.

---

## IR Format

Kernels are described as a JSON object. Example: 2 threads, thread 0 writes then signals a barrier, thread 1 waits at the barrier then reads the written value.

```json
{
  "num_threads": 2,
  "shared_memory_bytes": 256,
  "initial_memory": {"0": 99},
  "threads": [
    {"tid": 0, "instructions": [
      {"op": "WRITE", "addr": 0, "value": 42, "width": 4},
      {"op": "BARRIER"}
    ]},
    {"tid": 1, "instructions": [
      {"op": "BARRIER"},
      {"op": "READ", "addr": 0, "width": 4}
    ]}
  ]
}
```

| Field | Description |
|---|---|
| `num_threads` | Total number of parallel threads in the kernel. |
| `shared_memory_bytes` | Size of the shared memory region in bytes. |
| `initial_memory` | Optional map of `"addr": value` pairs to pre-seed memory state. |
| `threads[].tid` | Zero-based thread identifier. |
| `instructions[].op` | One of `READ`, `WRITE`, or `BARRIER`. |
| `addr` / `value` / `width` | Byte address, integer value (writes only), access width in bytes. |

---

## Installation & CLI Usage

```bash
pip install polyforge
```

Run a kernel JSON through the oracle:

```bash
python oracle_standalone.py my_kernel.json
# or, after pip install:
polyforge my_kernel.json
```

---

## Python API

```python
from oracle_standalone import OracleInput, StandaloneOracle

kernel = OracleInput.from_json("my_kernel.json")
result = StandaloneOracle().run(kernel)

if not result.passed:
    print(result.error)   # full race diagnostic
```

---

## Examples

See the [`oracle_examples/`](oracle_examples/) directory for a set of annotated kernel JSON files covering correct barriers, missing barriers (RAW), double-write races (WAW), and write-after-read hazards (WAR).

---

## Who Should Use This

Any team writing a **custom parallel compiler or kernel scheduler** for hardware that lacks an existing CUDA/OpenCL reference model:

- **University accelerator labs** building research RISC-V or VLIW processors.
- **RISC-V GPU projects** (e.g., Vortex, Ventus) wanting a fast software-level correctness check.
- **FPGA teams** compiling parallel workloads to custom datapaths.
- **Compiler engineers** who want a regression oracle in CI before taping out or running RTL simulation.

If your compiler can emit a sequence of READ, WRITE, and BARRIER operations per thread, you can use this tool today.
