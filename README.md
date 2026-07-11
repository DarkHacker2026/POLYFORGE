# 🧠 Agentic Compiler Growth Loop for Vortex RISC-V GPU

An autonomous, self-improving compiler that learns hardware-verified assembly rules from scratch — no human annotations, no hand-written patterns. It generates RISC-V assembly candidates via a large language model, compiles and ships them directly to the **Vortex GPU simulator**, extracts real hardware cycle counts, and permanently commits only the candidates that pass correctness verification.

---

## What We Built

### The Core Idea

Traditional compilers ship with hand-written instruction selection patterns. We asked: *what if a compiler could grow its own rule database by running candidates on real hardware and learning from the results?*

This project implements exactly that — a closed, autonomous loop:

```
Mini-IR Program
      │
      ▼
 LLM Candidate Generation  ──────────────────────────────────────┐
      │                                                           │
      ▼                                                     (failure feedback)
 Static Checker                                                   │
 (legal ISA ops, regs, structure)                                 │
      │                                                           │
      ▼                                                           │
 VortexSimulator                                                  │
 (GCC RISC-V cross-compile → WSL → simx)                         │
      │                                                           │
      ▼                                                           │
 Correctness Check + Cycle Count                                  │
      │                                                           │
      ├── FAIL ──────────────────────────────────────────────────┘
      │
      └── PASS ──▶ Rule Extraction (LLM/Gemma)
                        │
                        ▼
                   RuleDatabase
                   (deterministic cycle-based replacement)
```

### Key Components

| Component | Description |
|---|---|
| **Mini-IR Parser** | Parses a simple 3-address IR like `r3 = add(r1, r2)` into structured `IROperation` objects |
| **LLM Provider** | Calls Fireworks AI (Kimi-2.6) to generate RISC-V instruction candidates as strict JSON |
| **Static Checker** | Validates that every instruction uses legal ops, defined registers, and correct operand structure before wasting a simulation slot |
| **VortexSimulator** | Cross-compiles the candidate to a C++ inline-assembly kernel, syncs it to WSL, and runs it natively in Vortex `simx` |
| **VortexArtifactEmitter** | Translates the LLM's JSON instruction list into a compilable `main.cpp` with `vx_rdcycle()` hardware counters and correctness assertions |
| **RuleDatabase** | Persists learned rules to `data/rules.json`. Replacement is **deterministic**: a new rule only wins if `proof["cycles"] < old["actual_cycles"]` — never an LLM judgment call |
| **Rule Extractor** | On passing simulation, calls the LLM again to abstract the verified candidate into a reusable, parameterized rule pattern |

---

## Results

### Full Integration Test — All 4 Operations

Running the full integration test (`examples/full_integration_test.ir`) against the real Vortex RISC-V hardware simulator produced the following verified results — **zero human intervention, zero manual assembly writing**:

```
[start] provider=fireworks program=examples\full_integration_test.ir
[rule] reused ADD_INT_v2 for r3 = add(r1, r2)
[sim] reused rule passed cycles=88
[emit] vortex test artifact: artifacts/vortex_tests/agent_add_add_int_v2_instantiated
[rule] reused MUL_INT_v1 for r4 = mul(r1, r2)
[sim] reused rule passed cycles=88
[emit] vortex test artifact: artifacts/vortex_tests/agent_mul_mul_int_v1_instantiated
[rule] reused LOAD_INT_v1 for r5 = load(r6)
[sim] reused rule passed cycles=57
[emit] vortex test artifact: artifacts/vortex_tests/agent_load_load_int_v1_instantiated
[rule] reused LOOP_DEC_v1 for r1 = loop(r1)
[sim] reused rule passed cycles=350
[emit] vortex test artifact: artifacts/vortex_tests/agent_loop_loop_dec_v1_instantiated
[done] learned/reused operations:
  - r3 = add(r1, r2) => reused (88 cycles)
  - r4 = mul(r1, r2) => reused (88 cycles)
  - r5 = load(r6)    => reused (57 cycles)
  - r1 = loop(r1)    => reused (350 cycles)
```

### Learned Operations Summary

| IR Operation | Learned Rule | Hardware Cycles | Instruction Pattern |
|---|---|---|---|
| `r3 = add(r1, r2)` | `ADD_INT_v2` | **88** | `ADD rd, rs1, rs2` |
| `r4 = mul(r1, r2)` | `MUL_INT_v1` | **88** | `MUL rd, rs1, rs2` |
| `r5 = load(r6)` | `LOAD_INT_v1` | **57** | `LW rd, 0(rs1)` |
| `r1 = loop(r1)` | `LOOP_DEC_v1` | **350** | `ADDI r, r, -1` + `BNE r, zero, label` |

### Deterministic Rule Replacement (Proved)

We ran `--force-relearn add loop` to force the LLM to re-generate candidates for already-learned operations. The pipeline correctly applied deterministic hardware-cycle-based decisions:

```
[llm] candidate ADD_INT_v2 for r3 = add(r1, r2)
[sim] passed cycles=88 expected=0
[gemma] kept_existing: ADD_INT_v2        ← new == old (88 cycles), no overwrite

[llm] candidate LOOP_DEC_v2 for r1 = loop(r1)
[sim] passed cycles=350 expected=0
[gemma] kept_existing: LOOP_DEC_v1      ← new == old (350 cycles), no overwrite
```

**The LLM had zero authority** to overwrite an existing rule. Only a numerically faster cycle count triggers replacement — that comparison is a single `<` in plain Python, no model call involved.

---

## How to Run

### Prerequisites
- Python 3.10+
- WSL2 with Vortex GPU simulator built (`simx` available)
- Fireworks AI API key

### Setup
```bash
cd "hackathon project"
echo "FIREWORKS_API_KEY=your_key_here" > .env
```

### Learn from scratch (first time)
```bash
python grow_compiler.py \
  --provider fireworks \
  --program examples/full_integration_test.ir \
  --simulator vortex \
  --reset-rules
```

### Re-run with cached rules (fast replay)
```bash
python grow_compiler.py \
  --provider fireworks \
  --program examples/full_integration_test.ir \
  --simulator vortex
```

### Force re-learn specific operations (dev only)
```bash
# DEV-ONLY: bypasses the rule cache for the specified operations
python grow_compiler.py \
  --provider fireworks \
  --program examples/demo_loop_and_replace.ir \
  --simulator vortex \
  --force-relearn add loop
```

### Swap LLM model
```powershell
$env:CANDIDATE_MODEL="accounts/fireworks/models/glm-5p2"
$env:RULE_MODEL="accounts/fireworks/models/glm-5p2"
python grow_compiler.py --provider fireworks --program examples/full_integration_test.ir --simulator vortex
```

---

## Architecture Deep-Dive

### Vortex Hardware Pipeline

Each candidate is compiled and run on a real RISC-V GPU simulator:

1. `VortexArtifactEmitter.emit()` — writes `main.cpp` with GCC inline assembly + `vx_rdcycle()` hardware counters
2. `wsl_sync_from_windows.sh` — rsyncs the Windows artifacts directory into the WSL Linux filesystem
3. `make -C artifacts/vortex_tests/<candidate> run-simx` — cross-compiles with `riscv64-unknown-elf-g++`, links against Vortex runtime, and runs `simx`
4. Stdout is parsed for `SIMX_CYCLES=<n>` and `Passed!` / `Failed!`
5. A hard 10-second `timeout` in the WSL subprocess prevents infinite loops from hanging the agent

### Multi-Instruction Control Flow

For operations requiring multiple instructions (e.g. loop countdown), the `_multi_asm_body()` emitter:
- Scans all registers referenced across every LLM instruction
- Automatically declares them all in the GCC inline-asm constraint list (`"+r"` for written, `"r"` for read-only)
- Normalizes label syntax to use GCC's `%=` uniquifier (e.g. `1%=:` / `1%=b`) to prevent symbol collisions
- Works for any future multi-instruction operations, not just loops

### Deterministic Replacement Decision

```python
# In RuleDatabase.add_or_replace()
assert proof["ok"], "only correctness-verified candidates reach this point"
if old and proof["cycles"] < old["actual_cycles"]:
    # Hardware says new is strictly faster — replace
    rules[op_name] = new_rule
else:
    # Tie or regression — keep existing, reject new
    pass
```

No model, no heuristic, no human — just hardware numbers.

---

## IR Format

The mini-IR is intentionally minimal — one operation per line:

```
# Comments are supported
r3 = add(r1, r2)     # binary integer addition
r4 = mul(r1, r2)     # integer multiply (RISC-V M extension)
r5 = load(r6)        # word load from memory
r1 = loop(r1)        # fixed 10-iteration countdown loop
```

---

## Project Structure

```
hackathon project/
├── grow_compiler.py              # Main compiler growth loop
├── data/
│   ├── rules.json                # Learned rule database (grows over time)
│   └── hardware_facts.vortex.json # Vortex ISA constraints fed to the LLM
├── examples/
│   ├── full_integration_test.ir  # 4-operation integration test
│   └── demo_loop_and_replace.ir  # Rule replacement demo
├── scripts/
│   └── wsl_sync_from_windows.sh  # Windows→WSL artifact sync
├── artifacts/
│   └── vortex_tests/             # Generated C++ kernel artifacts
│       ├── agent_add_*/
│       ├── agent_mul_*/
│       ├── agent_load_*/
│       └── agent_loop_*/
└── README.md
```
