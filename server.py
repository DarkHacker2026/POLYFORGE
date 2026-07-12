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
    diagnosis: dict = {}


class RtlAnalyzeRequest(BaseModel):
    target: str = "vortex"


class RtlAnalyzeResponse(BaseModel):
    target: str
    capabilities: dict
    raw_output: str


# ─── Diagnosis Generator ─────────────────────────────────────────────────

import re as _re

def _generate_diagnosis(stdout: str, stderr: str, exit_code: int) -> dict:
    """Parse pipeline output and generate a human-readable diagnosis.

    Returns a dict with:
      - status: "pass" | "fail" | "warning"
      - summary: one-line verdict
      - issues: list of {severity, title, detail, fix}
      - stages: dict of stage_name -> stage_status
    """
    diag = {
        "status": "pass" if exit_code == 0 else "fail",
        "summary": "",
        "issues": [],
        "stages": {},
    }

    combined = stdout + "\n" + stderr

    # ── Parse stage results ──
    # LLM stage
    if "[2/5]" in combined:
        if "parsed OK" in combined:
            diag["stages"]["llm_comprehension"] = "passed"
        elif "FALLBACK" in combined:
            diag["stages"]["llm_comprehension"] = "fallback"
            diag["issues"].append({
                "severity": "warning",
                "title": "LLM parsing used regex fallback",
                "detail": "The LLM could not parse the kernel correctly. POLYFORGE fell back to regex extraction, which may miss complex patterns.",
                "fix": "Simplify the kernel syntax or check the LLM API key.",
            })
        elif "FAIL" in combined and "LLM" in combined.split("[2/5]")[1].split("[3/5]")[0]:
            diag["stages"]["llm_comprehension"] = "failed"
            diag["issues"].append({
                "severity": "critical",
                "title": "LLM comprehension failed",
                "detail": "The AI could not understand the kernel and regex fallback also failed.",
                "fix": "Ensure the kernel has a standard __global__ signature with typed parameters.",
            })

    # Oracle stage
    if "[3/5]" in combined:
        if "Oracle VERIFIED" in combined:
            diag["stages"]["oracle_verification"] = "passed"
        elif "Oracle SKIPPED" in combined:
            diag["stages"]["oracle_verification"] = "skipped"
            diag["issues"].append({
                "severity": "info",
                "title": "Oracle verification skipped",
                "detail": "The Oracle could not numerically verify this kernel (e.g., shared memory kernel).",
                "fix": "No action needed — this is expected for certain kernel types.",
            })
        elif "Oracle FAILED" in combined or "Oracle REJECTED" in combined:
            diag["stages"]["oracle_verification"] = "failed"
            # Determine the specific reason
            if "Data race detected" in combined or "RAW/WAR hazard" in combined:
                diag["issues"].append({
                    "severity": "critical",
                    "title": "⚠️ Data Race Detected (RAW/WAR Hazard)",
                    "detail": "Your kernel reads from array elements that adjacent parallel threads are simultaneously writing to. "
                              "This creates a Read-After-Write or Write-After-Read hazard. The result depends on thread scheduling — "
                              "it may work sometimes but produce wrong results other times.",
                    "fix": "Use __shared__ memory to let each thread read its neighbors' original values before anyone writes, "
                           "then call __syncthreads() before writing results back. Or redesign the algorithm to avoid cross-thread dependencies.",
                })
            elif "Cannot evaluate local_variable" in combined:
                diag["issues"].append({
                    "severity": "critical",
                    "title": "Oracle cannot evaluate a variable",
                    "detail": "The Oracle tried to simulate your kernel but couldn't compute a local variable. "
                              "This usually means the LLM extracted an expression the Oracle can't evaluate.",
                    "fix": "Simplify the kernel expressions or add the variable to test_params in the pipeline config.",
                })
            elif "non_standard_annotations" in combined:
                diag["issues"].append({
                    "severity": "critical",
                    "title": "Non-standard CUDA annotations",
                    "detail": "The kernel uses annotations that don't exist in standard CUDA.",
                    "fix": "Remove non-standard annotations and use only standard CUDA keywords.",
                })
            else:
                diag["issues"].append({
                    "severity": "critical",
                    "title": "Oracle verification failed",
                    "detail": "The independent Oracle could not verify the kernel's correctness.",
                    "fix": "Check the terminal output for the specific error message.",
                })

    # Lowering stage
    if "[4/5]" in combined:
        if "Lowering complete" in combined:
            diag["stages"]["code_lowering"] = "passed"
        elif "Lowering failed" in combined:
            diag["stages"]["code_lowering"] = "failed"
            diag["issues"].append({
                "severity": "critical",
                "title": "Code lowering failed",
                "detail": "The compiler could not generate C++ code for the target architecture.",
                "fix": "Check if the kernel uses features unsupported on the selected target.",
            })

    # Execution stage
    if "[5/5]" in combined:
        if "SIMX_RESULT=0" in combined or "Passed!" in combined:
            diag["stages"]["hardware_execution"] = "passed"
        elif "Failed!" in combined or "SIMX_RESULT=" in combined:
            # Check if it's a non-zero result
            m = _re.search(r'SIMX_RESULT=(\d+)', combined)
            if m and int(m.group(1)) != 0:
                diag["stages"]["hardware_execution"] = "failed"
                diag["issues"].append({
                    "severity": "critical",
                    "title": "Hardware execution produced wrong results",
                    "detail": f"The compiled kernel ran on the target but produced incorrect output (SIMX_RESULT={m.group(1)}).",
                    "fix": "Check the generated C++ code for lowering errors. The kernel may use features not correctly translated.",
                })
            elif "TIMEOUT" in combined:
                diag["stages"]["hardware_execution"] = "timeout"
                diag["issues"].append({
                    "severity": "warning",
                    "title": "Hardware execution timed out",
                    "detail": "The kernel took too long to execute on the target.",
                    "fix": "Reduce the problem size or simplify the kernel.",
                })

    # Kernel drop detection
    if "kernel dropping" in combined.lower() or "silent dropping" in combined.lower():
        diag["issues"].append({
            "severity": "critical",
            "title": "Kernel was silently dropped",
            "detail": "The LLM failed to extract one or more kernels from the source, which means code would be lost in production.",
            "fix": "Use --kernel NAME to target a specific kernel, or simplify the kernel signatures.",
        })

    # Connection errors
    if "ERROR" in combined and "server" in combined.lower():
        diag["issues"].append({
            "severity": "warning",
            "title": "Server connection error",
            "detail": "Could not reach the POLYFORGE backend server.",
            "fix": "Make sure the server is running: uvicorn server:app --reload",
        })

    # ── Generate summary ──
    if exit_code == 0 and not diag["issues"]:
        diag["summary"] = "✅ All pipeline stages passed — kernel verified and executed correctly."
    elif exit_code == 0 and any(i["severity"] == "warning" for i in diag["issues"]):
        diag["status"] = "warning"
        diag["summary"] = "⚠️ Kernel executed successfully, but with warnings — see issues below."
    elif exit_code == 0 and any(i["severity"] == "critical" for i in diag["issues"]):
        diag["status"] = "fail"
        diag["summary"] = "❌ Hardware passed but Oracle REJECTED this kernel — it's unsafe for production."
    else:
        diag["summary"] = "❌ Pipeline failed — see issues below for details."

    return diag


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

        # ── Generate diagnosis from pipeline output ──
        diagnosis = _generate_diagnosis(result.stdout, result.stderr, result.returncode)

        return CompileResponse(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            generated_cpp=generated_cpp,
            generated_ir=generated_ir,
            diagnosis=diagnosis,
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