# How to Generate Oracle IR From Your Parallel Compiler Output

This guide walks any parallel compiler team through the steps to convert their existing compiler output into the **parallel-oracle** IR — a simple JSON format that the oracle uses to detect RAW, WAW, and WAR data races.

---

## The IR Specification

The oracle takes a single JSON object with the following fields:

- **`num_threads`** — An integer declaring how many parallel threads will execute the kernel. All threads are assumed to run concurrently with no implicit ordering between them.
- **`shared_memory_bytes`** — The total size of the shared memory region being analyzed, in bytes. Accesses outside this range are flagged as out-of-bounds.
- **`initial_memory`** — An optional object mapping string byte addresses to integer values, used to pre-seed memory before any thread begins execution. Omit it or leave it empty if memory starts zeroed.
- **`threads`** — An array of per-thread objects. Each has a `tid` (zero-based integer) and an `instructions` array. Each instruction has an `op` field (`"READ"`, `"WRITE"`, or `"BARRIER"`), plus `addr` and `width` for READ/WRITE, and `value` for WRITE. A `BARRIER` instruction represents any global synchronization point (thread-group barrier, fence, etc.) where all threads must arrive before any may proceed.

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

---

## Worked Example: Parallel Vector-Add Kernel

**C pseudocode (2 threads, each adds its element):**

```c
// Thread 0                         // Thread 1
shared[0] = a[0] + b[0];           shared[1] = a[1] + b[1];
__syncthreads();                    __syncthreads();
result[0] = shared[0];             result[1] = shared[1];
```

**Annotated memory accesses:**

| Thread | Op    | Addr | Value          | Notes                        |
|--------|-------|------|----------------|------------------------------|
| 0      | WRITE | 0    | a[0]+b[0] = 7  | Writes element 0 to shared   |
| 1      | WRITE | 4    | a[1]+b[1] = 11 | Writes element 1 to shared   |
| —      | BARRIER | — | —             | Both threads sync here       |
| 0      | READ  | 0    | —              | Reads back its own element   |
| 1      | READ  | 4    | —              | Reads back its own element   |

**Equivalent Oracle IR:**

```json
{
  "num_threads": 2,
  "shared_memory_bytes": 64,
  "initial_memory": {},
  "threads": [
    {"tid": 0, "instructions": [
      {"op": "WRITE", "addr": 0,  "value": 7,  "width": 4},
      {"op": "BARRIER"},
      {"op": "READ",  "addr": 0,  "width": 4}
    ]},
    {"tid": 1, "instructions": [
      {"op": "WRITE", "addr": 4,  "value": 11, "width": 4},
      {"op": "BARRIER"},
      {"op": "READ",  "addr": 4,  "width": 4}
    ]}
  ]
}
```

No races — each thread writes a distinct address, and the barrier correctly separates the write phase from the read phase.

---

## If Your Compiler Emits LLVM IR

- **`store` → WRITE**: Map every `store <width> <value>, ptr <addr>` inside a parallel region to a WRITE instruction. Use the pointer's numeric offset into shared memory as `addr`, the stored constant or computed value as `value`, and the type width (e.g., `i32` → `width: 4`) as `width`.
- **`load` → READ**: Map every `load <width>, ptr <addr>` to a READ instruction. Extract `addr` and `width` the same way as stores; `value` is omitted for reads.
- **`@llvm.nvvm.barrier0` / `fence syncscope` → BARRIER**: Any LLVM synchronization intrinsic that represents a thread-group or global fence maps directly to a BARRIER instruction at the same program-order position in the thread's instruction list.

---

## If Your Compiler Emits RISC-V Assembly

- **`sw` / `sh` / `sb` → WRITE**: The store word/halfword/byte instructions map to WRITE. The destination register + offset encodes `addr`; the source register value (or its known constant) is `value`; the mnemonic determines `width` (`sw`→4, `sh`→2, `sb`→1).
- **`lw` / `lh` / `lb` → READ**: Load instructions map to READ. Compute `addr` from base register + immediate offset; set `width` from the mnemonic (`lw`→4, `lh`→2, `lb`→1).
- **`fence` / `fence.tso` → BARRIER**: Any RISC-V fence instruction, regardless of predecessor/successor operands, maps to a BARRIER. If your target uses a custom CSR-based barrier, treat any write to that CSR as a BARRIER at that program point.

---

## What the Oracle Does NOT Check

Be honest about tool scope to avoid wasted debugging time:

- **Control flow** — The oracle does not simulate branches, loops, or conditional execution. Every instruction in a thread's list is assumed to execute exactly once, in order. Variable-trip loops must be manually unrolled to a fixed iteration count before generating IR.
- **Cache coherence** — The oracle models a flat, coherent shared memory. It does not simulate cache hierarchies, MSHR conflicts, cache-line granularity, or non-uniform latency. Hardware-specific coherence bugs require RTL simulation or hardware probing.
- **Hardware timing and microarchitecture** — The oracle is a static access-pattern checker, not a cycle-accurate simulator. It cannot detect races that only manifest due to specific pipeline timing, out-of-order execution, or memory-system reordering below the barrier level.
- **Replacing hardware testing** — A passing oracle result means the *access pattern* is race-free at the IR level. It does not guarantee correct hardware behavior. Always follow oracle verification with RTL simulation and real hardware runs.

---

## Contact

To run your kernels through our oracle and get a **free race analysis report**, open a GitHub Issue at **[your-repo-url]** with your kernel JSON attached (or pasted inline). We will return a full diagnostic within 48 hours.
