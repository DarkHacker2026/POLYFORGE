# 🚀 POLYFORGE

> **Live Web App**: [https://polyforge-web.vercel.app](https://polyforge-web.vercel.app)

**POLYFORGE** is a universal parallel kernel compiler that uses AI (LLM comprehension) combined with a Zero-Trust Oracle to automatically compile CUDA kernels into RISC-V machine code and execute them on the Vortex SIMX GPU simulator.

---

## 📋 Table of Contents

- [What is POLYFORGE?](#what-is-polyforge)
- [How it Works](#how-it-works)
- [How to Use It](#how-to-use-it)
- [Verified Results](#verified-results)
- [Project Structure](#project-structure)
- [Documentation](#documentation)

---

## What is POLYFORGE?

POLYFORGE is an AI-powered compiler pipeline that:

1. **Comprehends** raw CUDA kernel source code using a lite LLM model (Gemma-2-9B)
2. **Verifies** the LLM extraction with an independent Clang AST Oracle (Zero-Trust architecture)
3. **Lowers** the verified IR to Vortex C++ for RISC-V compilation
4. **Executes** the compiled binary on the Vortex SIMX GPU simulator

The system is designed to be **unbounded** — if the LLM fails or produces imperfect output, a robust regex fallback parser extracts the kernel IR directly from the source code, ensuring the pipeline never breaks.

### Key Features

- **AI Comprehension**: Lite LLM model (Gemma-2-9B) for fast, cheap IR extraction
- **Zero-Trust Oracle**: Independent Clang AST verification — never trusts the LLM blindly
- **Unbounded Parser**: Regex fallback handles any LLM failure gracefully
- **Full Execution Transparency**: All subprocess output streamed live to the terminal
- **Native NVIDIA-Style Output**: Clean CUDA-style stdout on success, debug logs on failure
- **Race Detection**: Built-in parallel oracle detects WAR/WAW/RAR data races

---

## How it Works

The POLYFORGE pipeline consists of 4 stages:

### Stage 1: LLM Comprehension
```
CUDA Source → Lite LLM (Gemma-2-9B) → JSON IR
                    ↓ (if LLM fails)
              Regex Fallback → JSON IR
```
A lite LLM model extracts the kernel's semantic structure (parameters, thread indexing, operations, shared memory) into a JSON IR. If the LLM is unavailable or produces garbage, a pure regex fallback extracts the same IR directly from the CUDA source.

### Stage 2: Oracle Verification
```
JSON IR + Raw Source → Clang AST Oracle → PASS/FAIL
```
An independent Clang AST evaluator verifies the LLM's IR against ground truth extracted from the raw CUDA source. This Zero-Trust architecture ensures the LLM cannot silently introduce errors.

### Stage 3: Hardware Lowering
```
Verified IR → Vortex C++ Code Generator → main.cpp + Makefile
```
The verified IR is lowered to Vortex C++ with `vx_spawn_threads`, global volatile arrays, and hardware-specific barrier primitives. The output is a complete, compilable C++ file.

### Stage 4: SIMX Execution
```
main.cpp → clang (RISC-V) → .vxbin → Vortex SIMX → Results
```
The C++ is compiled with `clang --target=riscv32-unknown-elf` and executed on the Vortex SIMX simulator, which models a real GPU with warps, threads, barriers, and caches.

---

## How to Use It

### Prerequisites

- Python 3.10+
- WSL2 with Vortex SIMX installed
- Fireworks API key (for LLM comprehension)
- LLVM/libclang (for Oracle AST evaluation)

### Installation

```bash
# Clone the repository
git clone https://github.com/DarkHacker2026/POLYFORGE.git
cd POLYFORGE

# Create a virtual environment
python -m venv aider_env
source aider_env/bin/activate  # Linux: aider_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Fireworks API key
FIREWORKS_API_KEY=fw_your_key_here

# The lite parser model (optional, defaults to Gemma-2-9B)
PARSER_MODEL=accounts/fireworks/models/gemma-2-9b-it
```

### Running the Pipeline

```bash
# Full pipeline: CUDA → LLM → Oracle → Vortex C++ → SIMX
python vortex_compile.py examples/vectorAdd.cu

# Offline demo (no WSL or API key required)
python demo_offline.py

# Run the test suite
python -m pytest tests/ -v
```

---

## Verified Results

| Kernel | Status | SIMX Result | Cycles | Notes |
|--------|--------|-------------|--------|-------|
| `vectorAdd.cu` | ✅ PASSED | SIMX_RESULT=0 | 2037 | C[i] = A[i] + B[i], 8 elements verified |
| `mandelbrot_shader.cu` | ✅ PASSED | SIMX_RESULT=0 | 1862 | 2D kernel, complex shader, regex fallback |
| `saxpy_demo.cu` | ✅ PASSED | Oracle verified | — | SAXPY: y[i] = a*x[i] + y[i] |
| Oracle Race Tests | ✅ 8/8 PASSED | — | — | WAR, WAW, partial overlap, multi-barrier |

---

## Project Structure

```
POLYFORGE/
├── vortex_compile.py          # Main CLI: CUDA → Vortex pipeline
├── demo_offline.py            # Offline demo (no WSL/API key needed)
├── retarget_demo.py           # Kernel retargeting demo
├── saxpy_demo.cu              # Example SAXPY kernel
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project configuration
├── Dockerfile                 # Container support
├── .env.example               # Environment variable template
│
├── src/                       # Core compiler source
│   ├── __init__.py
│   ├── cuda_parser.py         # CUDA parser + IR extraction + regex fallback
│   ├── grow_compiler.py       # LLM provider + compiler growth loop
│   ├── cuda_surface.py        # Surface language lowering (parallel_for, shared, barrier)
│   ├── reference_isa.py       # Reference ISA simulator + parallel oracle
│   ├── oracle_standalone.py   # Standalone oracle for offline verification
│   ├── discovery_agent.py     # Hardware discovery agent
│   ├── clang_evaluator.py     # Clang AST expression evaluator
│   └── block_compiler.py      # Block-level compiler
│
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── conftest.py            # Pytest configuration (sets up src/ path)
│   ├── test_grow_compiler.py  # Compiler growth loop tests
│   ├── test_oracle_hardened.py# Hardened oracle race detection tests
│   ├── test_oracle_races.py   # Race condition detection tests
│   ├── test_parallel_oracle.py# Parallel oracle verification tests
│   ├── test_llm_comprehension.py # LLM comprehension pipeline test
│   ├── llm_kernel_test.py     # LLM kernel extraction test
│   ├── run_parallel_kernel.py # Parallel kernel runner
│   ├── scale_oracle.py        # Oracle scaling tests
│   └── scale_rtlsim.py        # RTL simulation scaling
│
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md        # System architecture
│   ├── CONVENTIONS.md         # Coding conventions
│   ├── TROUBLESHOOTING.md     # Troubleshooting guide
│   ├── README_ORACLE.md       # Oracle documentation
│   └── parallel_oracle_whitepaper.*  # Oracle whitepaper
│
├── examples/                  # Example CUDA kernels
│   ├── vectorAdd.cu           # Vector addition
│   ├── mandelbrot_shader.cu   # Mandelbrot fractal shader
│   └── *.ir                   # Example IR files
│
├── kernels/                   # Kernel implementations
│   ├── conditional_scatter.py
│   └── strided_reduction.py
│
├── data/                      # Hardware facts & rules
│   ├── hardware_facts.*.json  # Hardware configuration files
│   ├── rules.json             # Compiler rules database
│   └── rules.fireworks.json   # Fireworks-specific rules
│
├── oracle_examples/           # Oracle test cases
│   ├── saxpy.json
│   ├── reduction.json
│   └── war_race.json
│
└── vendor/vortex/             # Vortex GPU simulator (submodule)
```

---

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — System design and component overview
- [Quick Start](QUICKSTART.md) — Getting started guide
- [Troubleshooting](docs/TROUBLESHOOTING.md) — Common issues and solutions
- [Oracle Whitepaper](docs/parallel_oracle_whitepaper.pdf) — Parallel oracle design

---

## License

This project is part of the POLYFORGE hackathon submission.

## Links

- 🌐 [Live Web App](https://polyforge-web.vercel.app)
- 📦 [GitHub Repository](https://github.com/DarkHacker2026/POLYFORGE)
