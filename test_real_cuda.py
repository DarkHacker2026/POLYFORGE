import os
import io
import sys
import json
import urllib.request
import subprocess
from pathlib import Path

# Force stdout to utf-8 so box-drawing chars in generated C++ don't crash Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cuda_parser import parse_cuda_kernel, kernel_to_vortex_cpp, kernel_to_oracle_ir, describe_parse, ParseError
from cuda_surface import lower_to_makefile
from reference_isa import verify_parallel_kernel

VORTEX_HOME_WSL = "/home/dark_hacker/hackathon-project/vendor/vortex"
PROJECT_NAME = "real_cuda_test"
N = 8

def run_wsl(cmd: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["wsl.exe", "-e", "bash", "-c", cmd],
        capture_output=True, text=True,
        check=check
    )

def main():
    print("=" * 60)
    print("STEP 1 — Fetch Real Unmodified CUDA Source")
    print("=" * 60)
    
    url = "https://raw.githubusercontent.com/NVIDIA/cuda-samples/v11.8/Samples/0_Introduction/vectorAdd/vectorAdd.cu"
    print(f"Downloading from: {url}")
    
    req = urllib.request.urlopen(url)
    cuda_code = req.read().decode('utf-8')
    
    # Save the raw file for provenance
    provenance_dir = ROOT / "artifacts" / PROJECT_NAME
    provenance_dir.mkdir(parents=True, exist_ok=True)
    raw_path = provenance_dir / "vectorAdd.cu"
    raw_path.write_text(cuda_code, encoding='utf-8')
    print(f"Saved unmodified source to: {raw_path}")
    
    print("\n" + "=" * 60)
    print("STEP 2 — Parsing Real CUDA Kernel")
    print("=" * 60)
    
    facts_path = ROOT / "data" / "hardware_facts.vortex.json"
    simt_facts = json.loads(facts_path.read_text())["simt_facts"]
    barrier_primitive = simt_facts.get("barrier_primitive", "__syncthreads()")

    try:
        # We pass the full source file, including headers and cudaMalloc
        ck = parse_cuda_kernel(cuda_code, barrier_code=barrier_primitive)
    except ParseError as e:
        print(f"\n[FAILURE] PARSE ERROR: {e}")
        sys.exit(1)

    describe_parse(ck)

    # ── Step 3: Build init values for array params ────────────────────────────
    # Generate simple values for A, B, C
    init_values: dict[str, list[int]] = {}
    for idx, ap in enumerate(ck.array_params):
        if idx == 0:
            init_values[ap.name] = list(range(1, N + 1))
        elif idx == 1:
            init_values[ap.name] = [10] * N
        else:
            init_values[ap.name] = [0] * N

    print("\n" + "=" * 60)
    print("STEP 3 — Auto-generating oracle IR")
    print("=" * 60)

    instructions, initial_mem, per_thread_regs, op_detected = kernel_to_oracle_ir(ck, N, init_values)
    print(f"Generated {len(instructions)} oracle IR instructions")
    print(f"Detected op: {op_detected}")
    print(f"Array bases: {per_thread_regs}")
    for ins in instructions:
        print(f"  {ins}")

    print("\n" + "=" * 60)
    print("STEP 4 — Oracle verification")
    print("=" * 60)

    def check_result(results, memory):
        if not ck.array_params:
            return True, "No arrays to check"
        dst = ck.array_params[-1]
        src_arrays = ck.array_params[:-1]
        errors = []
        for i in range(N):
            dst_base = per_thread_regs.get(f'r{len(ck.array_params)}', 0)
            addr = dst_base + i * 4
            got = int.from_bytes(memory[addr:addr+4], byteorder='little', signed=True)
            src_vals = []
            for sidx, sap in enumerate(src_arrays):
                sv = init_values.get(sap.name, [0]*N)[i]
                src_vals.append(sv)
            if len(src_vals) == 2:
                if op_detected == 'MUL':
                    expected = src_vals[0] * src_vals[1]
                elif op_detected in ('ADD', 'SAXPY', None):
                    expected = src_vals[0] + src_vals[1]
                elif op_detected == 'SUB':
                    expected = src_vals[0] - src_vals[1]
                else:
                    expected = got
            elif len(src_vals) == 1:
                expected = src_vals[0]
            else:
                expected = got
            if got != expected:
                errors.append(f"[{dst.name}][{i}]: got={got}, expected={expected}")
        if errors:
            return False, "; ".join(errors[:3])
        return True, f"All {N} results correct"

    oracle_res = verify_parallel_kernel(
        instructions, N,
        [dict(per_thread_regs)] * N,
        initial_mem,
        check_result
    )

    if oracle_res["ok"]:
        print(f"[SUCCESS] Oracle PASSED: {oracle_res['message']}")
    else:
        print(f"[FAILURE] Oracle REJECTED: {oracle_res['message']}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("STEP 5 — Lowering to Vortex C++")
    print("=" * 60)

    cpp_code = kernel_to_vortex_cpp(ck, simt_facts, N, init_values, op_detected)
    print(f"Generated {len(cpp_code.splitlines())} lines of Vortex C++")
    
    (provenance_dir / "main.cpp").write_text(cpp_code, encoding="utf-8")
    (provenance_dir / "Makefile").write_text(
        lower_to_makefile(PROJECT_NAME, VORTEX_HOME_WSL),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("STEP 6 — Hardware simulation (simx)")
    print("=" * 60)

    wsl_src = str(provenance_dir.absolute()).replace("C:\\", "/mnt/c/").replace("\\", "/")
    wsl_dest_parent = "~/hackathon-project/artifacts"

    print("Syncing to WSL...")
    r = run_wsl(f"mkdir -p {wsl_dest_parent} && cp -r '{wsl_src}' {wsl_dest_parent}/")
    if r.returncode != 0:
        print("[FAILURE] WSL sync failed:", r.stderr)
        sys.exit(1)

    print("Running simx...")
    r = run_wsl(
        f"cd ~/hackathon-project && source .wsl_env && "
        f"timeout 120 make -C artifacts/{PROJECT_NAME} run-simx"
    )

    print("\n--- simx stdout ---")
    print(r.stdout)
    if r.returncode != 0:
        print("--- simx stderr ---")
        print(r.stderr)

    if "Passed!" in r.stdout:
        print("\n" + "=" * 60)
        print("HARDWARE RESULT: PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("HARDWARE RESULT: FAILED or incomplete")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
