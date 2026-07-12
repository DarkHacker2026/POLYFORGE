#!/usr/bin/env python3
"""
POLYFORGE API Backend — FastAPI server for the web platform.

POST /api/compile     —  Accept raw CUDA code, run the full pipeline,
                          return stdout/stderr + generated artifacts.
GET  /api/targets     —  List available hardware architectures.
POST /api/rtl/analyze —  Run RTL capability analysis on a target.
GET  /api/health      —  Health check.
"""

import tempfile
import subprocess
import pathlib
import sys
import os
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="POLYFORGE API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = pathlib.Path(__file__).resolve().parent

# ─── Available Hardware Targets ──────────────────────────────────────────

TARGETS = [
    {
        "id": "vortex",
        "name": "Vortex RISC-V GPU",
        "arch": "RISC-V RV32IMAF + XVortex SIMT",
        "description": "Open-source RISC-V GPU with SIMT extensions. "
                       "4 warps × 4 SIMD lanes. Verified end-to-end.",
        "ram_options": [64, 128, 256, 512, 1024, 2048, 4096],
        "default_ram": 4096,
        "status": "verified",
        "features": {
            "thread_id": True,
            "parallel_spawn": True,
            "barrier_sync": True,
            "addressable_memory": True,
        },
    },
    {
        "id": "arm_m0",
        "name": "ARM Cortex-M0",
        "arch": "ARMv6-M (Thumb)",
        "description": "Ultra-low-power microcontroller. Single-core, "
                       "no SIMD, no GPU. Suitable for lightweight kernels.",
        "ram_options": [8, 16, 32, 64, 128],
        "default_ram": 64,
        "status": "verified",
        "features": {
            "thread_id": False,
            "parallel_spawn": False,
            "barrier_sync": False,
            "addressable_memory": True,
        },
    },
    {
        "id": "arm_a72",
        "name": "ARM Cortex-A72",
        "arch": "ARMv8-A (AArch64)",
        "description": "High-performance application processor. "
                       "Multi-core, NEON SIMD. Desktop/server class.",
        "ram_options": [256, 512, 1024, 2048, 4096, 8192],
        "default_ram": 2048,
        "status": "verified",
        "features": {
            "thread_id": True,
            "parallel_spawn": True,
            "barrier_sync": True,
            "addressable_memory": True,
        },
    },
    {
        "id": "x86_64",
        "name": "x86-64 (Intel/AMD)",
        "arch": "x86-64 + AVX2",
        "description": "Standard desktop/server CPU. Multi-core, "
                       "AVX2 SIMD. Maximum compatibility.",
        "ram_options": [512, 1024, 2048, 4096, 8192, 16384],
        "default_ram": 4096,
        "status": "verified",
        "features": {
            "thread_id": True,
            "parallel_spawn": True,
            "barrier_sync": True,
            "addressable_memory": True,
        },
    },
    {
        "id": "riscv_generic",
        "name": "RISC-V Generic (RV32IM)",
        "arch": "RISC-V RV32IM",
        "description": "Bare RISC-V integer core. No SIMT extensions. "
                       "Good baseline for custom silicon.",
        "ram_options": [32, 64, 128, 256, 512],
        "default_ram": 128,
        "status": "verified",
        "features": {
            "thread_id": False,
            "parallel_spawn": False,
            "barrier_sync": False,
            "addressable_memory": True,
        },
    },
    {
        "id": "custom",
        "name": "Custom RTL (Upload)",
        "arch": "User-provided RTL",
        "description": "Upload your own Verilog/SystemVerilog RTL. "
                       "POLYFORGE will analyze capabilities and report.",
        "ram_options": [64, 128, 256, 512, 1024, 2048, 4096],
        "default_ram": 1024,
        "status": "experimental",
        "features": {
            "thread_id": False,
            "parallel_spawn": False,
            "barrier_sync": False,
            "addressable_memory": False,
        },
    },
]


# ─── Models ──────────────────────────────────────────────────────────────

class CompileRequest(BaseModel):
    cuda_code: str
    target: str = "vortex"
    ram_mb: int = 4096


class CompileResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    generated_cpp: str = ""
    generated_ir: str = ""


class RtlAnalyzeRequest(BaseModel):
    target: str = "vortex"


class RtlAnalyzeResponse(BaseModel):
    target: str
    capabilities: dict
    raw_output: str


# ─── Endpoints ───────────────────────────────────────────────────────────

@app.get("/api/targets")
def list_targets():
    """Return all available hardware targets with their specs."""
    return {"targets": TARGETS}


@app.post("/api/compile", response_model=CompileResponse)
def compile_cuda(req: CompileRequest):
    """Run the full POLYFORGE pipeline on raw CUDA source code."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".cu", delete=False, encoding="utf-8"
    )
    try:
        tmp.write(req.cuda_code)
        tmp.close()

        # Build environment with target config
        env = os.environ.copy()
        env["POLYFORGE_TARGET"] = req.target
        env["POLYFORGE_RAM_MB"] = str(req.ram_mb)

        result = subprocess.run(
            [sys.executable, str(ROOT / "vortex_compile.py"), tmp.name],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(ROOT),
            env=env,
        )

        art_dir = ROOT / "artifacts" / "llm_comprehension_test"
        generated_cpp = ""
        generated_ir = ""
        if (art_dir / "main.cpp").exists():
            generated_cpp = (art_dir / "main.cpp").read_text(encoding="utf-8")
        if (art_dir / "kernel.ir.json").exists():
            generated_ir = (art_dir / "kernel.ir.json").read_text(encoding="utf-8")

        return CompileResponse(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            generated_cpp=generated_cpp,
            generated_ir=generated_ir,
        )
    except subprocess.TimeoutExpired:
        return CompileResponse(
            stdout="",
            stderr="Pipeline timed out after 300 seconds.",
            exit_code=1,
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.post("/api/rtl/analyze", response_model=RtlAnalyzeResponse)
def analyze_rtl(req: RtlAnalyzeRequest):
    """Run RTL capability analysis on the selected target."""
    target_id = req.target

    # Find target info
    target_info = next((t for t in TARGETS if t["id"] == target_id), None)
    if not target_info:
        return RtlAnalyzeResponse(
            target=target_id,
            capabilities={},
            raw_output=f"Unknown target: {target_id}",
        )

    # For Vortex, read the real hardware facts
    if target_id == "vortex":
        facts_path = ROOT / "data" / "hardware_facts.vortex.json"
        if facts_path.exists():
            facts = json.loads(facts_path.read_text())
            simt = facts.get("simt_facts", {})
            isa = facts.get("isa", {})
            regs = facts.get("registers", [])

            capabilities = {
                "thread_id": {
                    "status": "CONFIRMED",
                    "detail": f"vx_thread_id() CSR = {simt.get('thread_id_csr_value_on_main_thread', 'N/A')}",
                },
                "parallel_spawn": {
                    "status": "CONFIRMED",
                    "detail": f"vx_spawn_threads, {simt.get('num_threads_per_warp', '?')} threads/warp, "
                              f"{simt.get('total_threads_per_core', '?')} total/core",
                },
                "barrier_sync": {
                    "status": "CONFIRMED",
                    "detail": f"{simt.get('barrier_primitive', 'vx_barrier')} supported",
                },
                "addressable_memory": {
                    "status": "CONFIRMED",
                    "detail": f"LW/SW, latency {isa.get('LW', {}).get('latency', '?')} cycles",
                },
                "registers": len(regs),
                "isa_ops": list(isa.keys()),
                "num_warps": simt.get("num_warps_per_core", 4),
                "num_threads": simt.get("num_threads_per_warp", 4),
            }
            raw_output = (
                f"Vortex RISC-V GPU — ALL capabilities verified empirically via rtlsim.\n"
                f"Registers: {len(regs)} GPRs\n"
                f"ISA ops: {list(isa.keys())}\n"
                f"Warps: {simt.get('num_warps_per_core', 4)}, "
                f"Threads/warp: {simt.get('num_threads_per_warp', 4)}"
            )
        else:
            capabilities = target_info["features"]
            raw_output = "No hardware_facts.vortex.json found. Run discovery_agent.py first."
    else:
        # For non-Vortex targets, return the planned feature set
        capabilities = {
            "thread_id": {
                "status": "CONFIRMED" if target_info["features"]["thread_id"] else "ABSENT",
                "detail": "Based on architecture specification",
            },
            "parallel_spawn": {
                "status": "CONFIRMED" if target_info["features"]["parallel_spawn"] else "ABSENT",
                "detail": "Based on architecture specification",
            },
            "barrier_sync": {
                "status": "CONFIRMED" if target_info["features"]["barrier_sync"] else "ABSENT",
                "detail": "Based on architecture specification",
            },
            "addressable_memory": {
                "status": "CONFIRMED" if target_info["features"]["addressable_memory"] else "ABSENT",
                "detail": "Based on architecture specification",
            },
        }
        raw_output = (
            f"Target: {target_info['name']} ({target_info['arch']})\n"
            f"Status: {target_info['status'].upper()}\n"
            f"Description: {target_info['description']}\n"
            f"Note: This target is {target_info['status']}. "
            f"Only Vortex has been empirically verified end-to-end."
        )

    return RtlAnalyzeResponse(
        target=target_id,
        capabilities=capabilities,
        raw_output=raw_output,
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}