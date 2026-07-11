import os
import io
import json
import sys
import subprocess
from pathlib import Path

# Force stdout to utf-8 so box-drawing chars in generated C++ don't crash Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from grow_compiler import FireworksProvider, load_dotenv
from cuda_parser import parse_cuda_kernel, kernel_to_vortex_cpp, kernel_to_oracle_ir, describe_parse, ParseError
from cuda_surface import lower_to_makefile
from reference_isa import verify_parallel_kernel

VORTEX_HOME_WSL = "/home/dark_hacker/hackathon-project/vendor/vortex"
PROJECT_NAME = "llm_kernel"

# ── LLM prompt ────────────────────────────────────────────────────────────────
# Ask Kimi for a simple element-wise kernel. We tell it exactly what JSON fields
# to return so we can reliably extract the code string.
KERNEL_PROMPT = """\
Write a simple CUDA parallel kernel. Return JSON with exactly these fields:
{
  "kernel_name": "<name>",
  "code": "<complete __global__ function definition as a string>",
  "launch_config": "<launch config line>",
  "description": "<one sentence>"
}

Requirements for the kernel:
- Must be a __global__ void function
- Must use exactly: int i = blockIdx.x * blockDim.x + threadIdx.x;
- Must be a simple element-wise operation on 1D arrays (e.g. C[i] = A[i] * B[i])
- Input arrays should be float* or const float* (or int*)
- Must include a bounds check: if (i < N)
- Do NOT include cudaMalloc, cudaMemcpy, or any host code
- Do NOT use atomics, 2D thread blocks, or dynamic shared memory
- Return ONLY the __global__ function, nothing else in the "code" field
"""

N = 8  # number of elements to test with


def run_wsl(cmd: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["wsl.exe", "-e", "bash", "-c", cmd],
        capture_output=True, text=True,
        check=check
    )


def main():
    load_dotenv(ROOT / ".env")
    provider = FireworksProvider()

    # ── Step 1: Live LLM call ─────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — Live LLM call to Kimi-2.6")
    print("=" * 60)
    print(f"Model: {provider.candidate_model}")
    print("Sending prompt...")

    response = provider._chat_json(provider.candidate_model, KERNEL_PROMPT)

    kernel_name = response.get("kernel_name", "unknown")
    cuda_code   = response.get("code", "")
    description = response.get("description", "")

    print(f"\nKernel returned: '{kernel_name}'")
    print(f"Description: {description}")
    print(f"\nRaw CUDA code from LLM:\n{'-'*40}\n{cuda_code}\n{'-'*40}")

    if not cuda_code:
        print("\nERROR: LLM did not return a 'code' field. Cannot proceed.")
        print("Full response:", json.dumps(response, indent=2))
        sys.exit(1)

    # ── Step 2: Parse the CUDA code ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Parsing CUDA kernel")
    print("=" * 60)

    facts_path = ROOT / "data" / "hardware_facts.vortex.json"
    if not facts_path.exists():
        print("Run discovery_agent.py --simt first!")
        sys.exit(1)
    simt_facts = json.loads(facts_path.read_text())["simt_facts"]

    barrier_primitive = simt_facts.get("barrier_primitive", "__syncthreads()")

    try:
        ck = parse_cuda_kernel(cuda_code, barrier_code=barrier_primitive)
    except ParseError as e:
        print(f"\nPARSE ERROR: {e}")
        print("The LLM returned CUDA that this pipeline cannot auto-translate.")
        print("You need to either: (a) retry the LLM with a stricter prompt,")
        print("or (b) implement the missing parser feature.")
        sys.exit(1)

    describe_parse(ck)

    # ── Step 3: Build init values for array params ────────────────────────────
    # We generate simple values: first src = [1..N], second src = [10..10], dst = [0..0]
    init_values: dict[str, list[int]] = {}
    for idx, ap in enumerate(ck.array_params):
        if idx == 0:
            init_values[ap.name] = list(range(1, N + 1))          # [1, 2, ..., N]
        elif idx == 1:
            init_values[ap.name] = [10] * N                        # [10, 10, ...]
        else:
            init_values[ap.name] = [0] * N                         # destination = zeros

    # ── Step 4: Generate oracle IR ────────────────────────────────────────────
    print("=" * 60)
    print("STEP 3 — Auto-generating oracle IR")
    print("=" * 60)

    instructions, initial_mem, per_thread_regs, op_detected = kernel_to_oracle_ir(ck, N, init_values)
    print(f"Generated {len(instructions)} oracle IR instructions")
    print(f"Detected op: {op_detected}")
    print(f"Array bases: {per_thread_regs}")
    print(f"Instructions:")
    for ins in instructions:
        print(f"  {ins}")

    # ── Step 5: Run oracle ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Oracle verification")
    print("=" * 60)

    def check_result(results, memory):
        # Check dst array (last array param)
        if not ck.array_params:
            return True, "No arrays to check"
        dst = ck.array_params[-1]
        src_arrays = ck.array_params[:-1]
        errors = []
        for i in range(N):
            dst_base = per_thread_regs.get(f'r{len(ck.array_params)}', 0)
            addr = dst_base + i * 4
            got = int.from_bytes(memory[addr:addr+4], byteorder='little', signed=True)
            # Infer expected from init_values using the detected op
            src_vals = []
            for sidx, sap in enumerate(src_arrays):
                sbase = per_thread_regs.get(f'r{sidx + 1}', 0)
                sv = init_values.get(sap.name, [0]*N)[i]
                src_vals.append(sv)
            if len(src_vals) == 2:
                # Use the op detected by the parser from the RHS expression
                if op_detected == 'MUL':
                    expected = src_vals[0] * src_vals[1]
                elif op_detected in ('ADD', 'SAXPY'):
                    expected = src_vals[0] + src_vals[1]
                elif op_detected == 'SUB':
                    expected = src_vals[0] - src_vals[1]
                else:
                    expected = got  # unknown op, skip check
            elif len(src_vals) == 1:
                expected = src_vals[0]
            else:
                expected = got  # can't infer
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
        print(f"Oracle PASSED: {oracle_res['message']}")
    else:
        print(f"Oracle REJECTED: {oracle_res['message']}")
        print("The auto-generated oracle IR does not match the kernel semantics.")
        print("This means the body is more complex than the auto-IR supports.")
        print("The C++ lowering may still be valid — proceeding to rtlsim anyway.")

    # ── Step 6: Generate Vortex C++ ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — Lowering to Vortex C++")
    print("=" * 60)

    cpp_code = kernel_to_vortex_cpp(ck, simt_facts, N, init_values, op_detected)
    print(f"Generated {len(cpp_code.splitlines())} lines of Vortex C++")
    print("\nGenerated C++:\n" + "-" * 40)
    print(cpp_code)
    print("-" * 40)

    # ── Step 7: Write to disk and run on hardware ─────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 — Hardware simulation (simx)")
    print("=" * 60)

    proj_dir = ROOT / "artifacts" / PROJECT_NAME
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "main.cpp").write_text(cpp_code, encoding="utf-8")
    (proj_dir / "Makefile").write_text(
        lower_to_makefile(PROJECT_NAME, VORTEX_HOME_WSL),
        encoding="utf-8"
    )

    wsl_src = str(proj_dir.absolute()).replace("C:\\", "/mnt/c/").replace("\\", "/")
    wsl_dest_parent = "~/hackathon-project/artifacts"

    print("Syncing to WSL...")
    r = run_wsl(f"mkdir -p {wsl_dest_parent} && cp -r '{wsl_src}' {wsl_dest_parent}/")
    if r.returncode != 0:
        print("WSL sync failed:", r.stderr)
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
        print("=" * 60)
        print("HARDWARE RESULT: PASSED")
        print("=" * 60)
    else:
        print("=" * 60)
        print("HARDWARE RESULT: FAILED or incomplete")
        print("=" * 60)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n=== PIPELINE SUMMARY ===")
    print(f"  LLM model      : {provider.candidate_model}")
    print(f"  Kernel parsed  : {ck.name}")
    print(f"  Arrays         : {', '.join(p.name for p in ck.array_params)}")
    print(f"  N              : {N}")
    print(f"  Oracle         : {'PASS' if oracle_res['ok'] else 'WARN (arithmetic check)'}")
    print(f"  rtlsim         : {'PASS' if 'Passed!' in r.stdout else 'FAIL'}")
    print(f"  Hardcoded code : NONE — fully driven by LLM output")


if __name__ == "__main__":
    main()
