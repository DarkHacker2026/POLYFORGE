# POLYFORGE
> Any CUDA kernel. Any parallel hardware. Verified.

POLYFORGE is a universal parallel kernel compiler powered by LLMs + a zero-trust race-detecting oracle. Drop in a CUDA kernel — POLYFORGE understands it, proves it correct, and compiles it to run on any parallel hardware target.

Today: Vortex RISC-V GPU. Tomorrow: anything.

## Quick Start (3 commands)
```bash
pip install -r requirements.txt
cp .env.example .env    # paste your FIREWORKS_API_KEY
python vortex_compile.py examples/vectorAdd.cu
```

## No WSL? Run the offline demo:
```bash
python demo_offline.py
```

## How It Works
```
.cu file → [LLM Comprehension] → [Zero-Trust Oracle] → [Hardware Lowering] → [SIMX Execution] → PASS/FAIL
```

## Verified Results
| Kernel      | Result | Cycles | Oracle |
|-------------|--------|--------|--------|
| vectorAdd   | PASS   | 2,057  | PASS   |
| SAXPY (N=4) | PASS   | 3,911  | PASS   |
| Oracle suite| —      | —      | 8/8    |

## Why POLYFORGE Exists
CUDA is locked to NVIDIA. Every custom hardware team (RISC-V GPUs, FPGAs, research accelerators) has to reinvent the compiler stack from scratch — with no reference model to verify against. POLYFORGE breaks that lock.

## Project Structure
| File | Purpose |
|------|---------|
| `vortex_compile.py` | Main CLI: CUDA → Vortex pipeline |
| `cuda_parser.py` | CUDA AST parser + lowering engine |
| `reference_isa.py` | Zero-trust race-detecting oracle |
| `cuda_surface.py` | Hardware abstraction layer / lowering |
| `grow_compiler.py` | LLM provider + rule database |
| `discovery_agent.py` | Hardware capability auto-discovery |
| `oracle_standalone.py` | Standalone oracle library |
| `demo_offline.py` | Offline demo (no WSL/API key) |

## Requirements
- Python 3.10+
- FIREWORKS_API_KEY (get free credits at fireworks.ai)
- WSL2 + Vortex SIMX (for hardware execution — see QUICKSTART.md)
