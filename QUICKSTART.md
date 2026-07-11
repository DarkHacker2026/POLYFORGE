# POLYFORGE Quick Start

## Path 1: Windows + WSL2 (Full Pipeline)

1. Install WSL2 and the Vortex RISC-V GPU simulator.
2. `pip install -r requirements.txt`
3. `cp .env.example .env`  # paste your FIREWORKS_API_KEY
4. `python vortex_compile.py examples/vectorAdd.cu`

## Path 2: Linux / Mac (No WSL)

Hardware execution requires WSL2 + Vortex SIMX, but you can run the offline demo immediately:

```bash
python demo_offline.py
```

This runs the zero-trust oracle and the 8-test race-detection suite with no API keys.

## Path 3: Docker

```bash
docker build -t polyforge .
docker run polyforge
```

The Docker image runs Stages 1-5 (Oracle + Lowering demo). Hardware execution (Stage 6) requires WSL2 on the Windows host.

## Path 4: Just the Oracle

```bash
python oracle_standalone.py oracle_examples/saxpy.json
```

Standalone race-detection on any parallel IR JSON. No LLM, no hardware, no dependencies beyond Python 3.10.
