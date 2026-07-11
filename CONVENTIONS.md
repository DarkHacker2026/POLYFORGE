# POLYFORGE
**POLYFORGE** — Universal parallel kernel compiler. Any CUDA kernel, any hardware, verified.

## Technology Stack
- **Core pipeline:** Python 3.10+
- **LLM:** Kimi-2.6 via Fireworks AI (`accounts/fireworks/models/kimi-k2p6`)
- **AST evaluation:** libclang (Python bindings)
- **Hardware target:** Vortex RISC-V GPU (simx), via WSL2
- **Race detection:** ParallelReferenceISA (reference_isa.py)

## Key Architecture
- `vortex_compile.py` — main CLI entry point, accepts any `.cu` file
- `cuda_parser.py` — parses CUDA `__global__` kernels
- `reference_isa.py` — zero-trust parallel oracle (WAR/WAW/RAW detection)
- `cuda_surface.py` — lowers IR to Vortex C++
- `grow_compiler.py` — builds and runs on SIMX
- `discovery_agent.py` — probes hardware facts via rtlsim
- `oracle_standalone.py` — standalone oracle library

## Coding Conventions
- Python 3.10+ only, type hints preferred
- No hardcoded hardware constants — always read from hardware_facts.vortex.json
- All oracle errors must raise RuntimeError with explicit thread/address/epoch info
- WSL calls always via `wsl.exe -e bash -c` subprocess

## Known Gotchas
- **DO NOT TOUCH** `vendor/vortex/` — it is a git submodule
- The LLM output is NEVER trusted directly — always goes through oracle verification
- `clang.cindex` (libclang) must be installed separately — it is NOT in pip as `clang`
- Hardware execution requires WSL2 — offline demo via `demo_offline.py` works without it
