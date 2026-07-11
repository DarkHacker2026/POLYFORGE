#!/usr/bin/env python3
"""
run_extended_kernels.py

Executes the extended CUDA-on-Vortex checklist:
1. Scaled SAXPY (N=16, N=64, N=256) to find the spawn overhead break-even point.
2. Barrier-in-kernel (N=16) to prove multi-warp synchronization works.
3. Parallel tree reduction (N=16) as the canonical synchronization benchmark.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from grow_compiler import VortexArtifactEmitter, VortexSimulator
from cuda_surface import SurfaceLowering, lower_to_makefile

ARTIFACT_DIR = ROOT / "artifacts" / "vortex_tests"
VORTEX_HOME  = ROOT / "vendor" / "vortex"
FACTS_FILE   = ROOT / "data" / "hardware_facts.vortex.json"

def get_simt_facts() -> dict:
    facts = json.loads(FACTS_FILE.read_text(encoding="utf-8"))
    return facts.get("simt_facts", {})

def run_kernel(project_name: str, cpp_content: str) -> dict:
    """Emits the C++ code, syncs to WSL, runs on rtlsim, and parses cycles."""
    project_dir = ARTIFACT_DIR / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    vortex_home_str = str(VORTEX_HOME).replace("\\", "/")
    (project_dir / "main.cpp").write_text(cpp_content, encoding="utf-8")
    (project_dir / "Makefile").write_text(
        lower_to_makefile(project_name, vortex_home_str), encoding="utf-8")
    
    # Empty candidate JSON to satisfy the directory structure expectations if any
    (project_dir / "candidate.json").write_text("{}", encoding="utf-8")

    # Sync to WSL
    win_path   = project_dir.absolute().as_posix()
    wsl_src    = win_path.replace("C:/", "/mnt/c/")
    wsl_parent = "~/hackathon-project/artifacts/vortex_tests"
    sync_cmd   = f"mkdir -p {wsl_parent} && cp -r '{wsl_src}' {wsl_parent}/"
    subprocess.run(["wsl.exe", "-e", "bash", "-c", sync_cmd],
                   check=True, capture_output=True, text=True, timeout=30)

    # Run on rtlsim
    run_cmd = (
        f"cd ~/hackathon-project && source .wsl_env && "
        f"timeout 300 make -C artifacts/vortex_tests/{project_name} run-rtlsim"
    )
    result = subprocess.run(
        ["wsl.exe", "-e", "bash", "-c", run_cmd],
        capture_output=True, text=True, timeout=350)

    stdout = result.stdout
    
    passed        = "Passed!" in stdout
    m_result      = re.search(r"SIMX_RESULT=(-?\d+)",  stdout)
    m_expected    = re.search(r"SIMX_EXPECTED=(-?\d+)", stdout)
    m_par         = re.search(r"PAR_CYCLES=(\d+)",      stdout)
    m_scalar      = re.search(r"SCALAR_CYCLES=(\d+)",   stdout)

    par_cycles    = int(m_par.group(1))    if m_par    else -1
    scalar_cycles = int(m_scalar.group(1)) if m_scalar else -1
    result_val    = int(m_result.group(1)) if m_result else -1
    expected_val  = int(m_expected.group(1)) if m_expected else -1

    return {
        "ok":           passed,
        "result_val":   result_val,
        "expected_val": expected_val,
        "par_cycles":   par_cycles,
        "scalar_cycles": scalar_cycles,
        "stdout":       stdout
    }

def main():
    print("=" * 70)
    print("Running Extended Kernels: Scaling, Barriers, and Reductions")
    print("=" * 70)

    simt_facts = get_simt_facts()
    # We will FORCE barrier_supported to true for the in-kernel test because
    # the probe only failed on the single-warp main thread.
    simt_facts["barrier_supported"] = True
    
    import cuda_surface
    
    # 1. Scaled SAXPY
    for N in [16, 64, 256]:
        print(f"\\n--- 1. Scaled SAXPY (N={N}) ---")
        cpp = cuda_surface.lower_saxpy_scaled(N)
        res = run_kernel(f"saxpy_scaled_{N}", cpp)
        if res["ok"]:
            print(f"  [PASS] SAXPY N={N} completed successfully.")
            print(f"         Scalar Cycles: {res['scalar_cycles']}")
            print(f"         Spawn+Kernel Cycles: {res['par_cycles']}")
            if res['par_cycles'] > 0 and res['scalar_cycles'] > 0:
                speedup = res['scalar_cycles'] / res['par_cycles']
                print(f"         Speedup vs Scalar: {speedup:.2f}x")
        else:
            print(f"  [FAIL] SAXPY N={N} failed or timed out.")
            print(res["stdout"][-1000:])

    # 2. Barrier In Kernel
    print("\\n--- 2. Multi-warp Barrier Test (N=16) ---")
    cpp = cuda_surface.lower_barrier_in_kernel(16)
    res = run_kernel("barrier_test_16", cpp)
    if res["ok"]:
         print(f"  [PASS] Barrier test completed successfully.")
         print(f"         Parallel Cycles: {res['par_cycles']}")
    else:
         print(f"  [FAIL] Barrier test failed.")
         print(res["stdout"][-1000:])

    # 3. Reduction Kernel
    print("\\n--- 3. Parallel Tree Reduction (N=16) ---")
    cpp = cuda_surface.lower_reduction(16)
    res = run_kernel("reduction_16", cpp)
    if res["ok"]:
         print(f"  [PASS] Reduction test completed successfully.")
         print(f"         Expected Sum: {res['expected_val']}, Got: {res['result_val']}")
         print(f"         Parallel Cycles: {res['par_cycles']}")
    else:
         print(f"  [FAIL] Reduction test failed.")
         print(res["stdout"][-1000:])

    print("\\n" + "=" * 70)
    print("Extended tests complete.")

if __name__ == "__main__":
    main()
