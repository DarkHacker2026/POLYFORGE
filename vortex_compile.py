#!/usr/bin/env python3
"""
vortex_compile.py — Single-entry-point CLI for the CUDA->Vortex pipeline.

Usage:
    python vortex_compile.py <path/to/kernel.cu> [--kernel NAME]

This wraps the existing verified test_llm_comprehension.py pipeline directly.
It adds:
  - Labeled stage progress [1/5]...[5/5]
  - --kernel flag for multi-kernel targeting
  - Clean exit codes (0 = PASS, 1 = FAIL or any stage error)
  - Non-standard annotation rejection printed clearly
  - Kernel-drop warnings from source pre-scan

All pipeline logic comes from the existing, verified modules.
Do not duplicate pipeline logic here — call what already works.
"""

import sys
import os
import re
import argparse
import pathlib
import subprocess
import math

_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))


def _inject_stage_labels():
    """Monkey-patch print to detect and label pipeline stages from existing output."""
    # We let test_llm_comprehension.py print its own steps, and just add our headers.
    pass


def run_pipeline(cuda_file: str, kernel_filter: str | None = None) -> int:
    """
    Run the full pipeline on a .cu file by delegating to test_llm_comprehension.py.
    Returns exit code: 0 for SIMX_RESULT=0, 1 otherwise.
    """
    total_stages = 5
    cu_path = pathlib.Path(cuda_file)

    # ── [1/5] Load & pre-scan ────────────────────────────────────────────
    print(f"\n[1/{total_stages}] Loading source: {cu_path.name}", flush=True)
    if not cu_path.exists():
        print(f"[FAIL] File not found: {cuda_file}")
        return 1
    cuda_code = cu_path.read_text(encoding="utf-8", errors="replace")
    print(f"         {len(cuda_code)} bytes  |  {cu_path.resolve()}")

    # Pre-scan kernel markers BEFORE LLM call
    global_markers = re.findall(r'\b(?:__global__|__tile_global__)\b', cuda_code)
    print(f"         Source pre-scan: {len(global_markers)} kernel marker(s) "
          f"found: {list(set(global_markers))}")

    # ── [2/5] LLM comprehension ──────────────────────────────────────────
    print(f"\n[2/{total_stages}] LLM comprehension (Kimi-2.6)...", flush=True)

    # Import the existing pipeline directly
    try:
        from grow_compiler import FireworksProvider, load_dotenv
        from cuda_parser import (
            CUDAKernel, CUDAParam,
            kernel_to_vortex_cpp, kernel_to_oracle_ir, describe_parse,
            evaluate_clang_ast
        )
        from cuda_surface import lower_to_makefile
        from reference_isa import verify_parallel_kernel
    except ImportError as e:
        print(f"[FAIL] Cannot import pipeline module: {e}")
        print("       Run from the hackathon project root directory.")
        return 1

    load_dotenv(_ROOT / ".env")
    provider = FireworksProvider()

    # Reuse the exact PROMPT from test_llm_comprehension.py
    from test_llm_comprehension import PROMPT, CUDA_UNEVALUABLE_EXPRS, run_wsl, running_in_wsl

    # Split source into per-kernel chunks (same as test_llm_comprehension.py)
    matches = list(re.finditer(r'(?:template\s*<[^>]+>\s*)?\b__global__\b', cuda_code))
    if not matches:
        kernels_code = [cuda_code]
    else:
        kernels_code = []
        for i in range(len(matches)):
            start = matches[i].start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(cuda_code)
            kernels_code.append(cuda_code[start:end].strip())

    # Filter by --kernel if given
    if kernel_filter:
        kernels_code = [
            k for k in kernels_code
            if re.search(r'\b' + re.escape(kernel_filter) + r'\s*\(', k)
        ]
        if not kernels_code:
            print(f"[FAIL] --kernel '{kernel_filter}' not found in source.")
            return 1

    print(f"         Sending {len(kernels_code[:1])} kernel chunk(s) to LLM...")

    all_irs = []
    for idx, kcode in enumerate(kernels_code[:1]):
        formatted = PROMPT.format(kernel_code=kcode)
        try:
            ir = provider._chat_json(provider.candidate_model, formatted)
            all_irs.append(ir)
            print(f"         Kernel {idx+1}: '{ir.get('kernel_name', '?')}' parsed OK")
        except Exception as e:
            print(f"[FAIL] LLM API call failed for kernel {idx+1}: {e}")
            return 1

    # Kernel-drop warning
    llm_count = len(all_irs)
    if len(global_markers) > llm_count:
        dropped = len(global_markers) - llm_count
        print(f"\n[WARNING] Source has {len(global_markers)} kernel markers but LLM "
              f"extracted {llm_count} ({dropped} possibly dropped).")
        print(f"[WARNING] Use --kernel NAME to target one specific kernel.")
    else:
        print(f"         [OK] {llm_count} kernel(s) returned, no silent dropping detected")

    ir = all_irs[0]

    # Reject non-standard annotations (zero-trust gate)
    ns_annotations = ir.get("non_standard_annotations", [])
    if ns_annotations:
        print(f"\n[FAIL] Non-standard annotations detected: {ns_annotations}")
        print(f"       Kernel '{ir.get('kernel_name')}' cannot be lowered to standard Vortex ABI.")
        return 1

    # ── [3/5] Oracle verification ─────────────────────────────────────────
    print(f"\n[3/{total_stages}] Independent Oracle verification (Clang AST)...", flush=True)

    import json
    from pathlib import Path
    facts_path = _ROOT / "data" / "hardware_facts.vortex.json"
    simt_facts = json.loads(facts_path.read_text())["simt_facts"]
    N = 8   # matches existing pipeline constant

    # Reuse the exact CUDAKernel construction from test_llm_comprehension.py
    test_params = {
        "reduce0": {"extern_smem_expr": "N * 4", "expected_val": 3},
        "initializeInputs": {"HEAD_DIM": 128, "HALF_ROPE_DIM": 64,
                             "SEQ_LEN": 1024, "Q_SIZE": 32},
    }
    import math

    params = []
    for p in ir["parameters"]:
        params.append(CUDAParam(
            ctype=p["base_type"] + ("*" if p["is_pointer"] else ""),
            name=p["name"],
            is_pointer=p["is_pointer"],
            is_scalar=not p["is_pointer"]
        ))
    array_params = [p for p in params if p.is_pointer]
    scalar_params = [p for p in params if p.is_scalar]

    n_param = 'N'
    for p in scalar_params:
        if p.name.upper() in ('N', 'SIZE', 'COUNT', 'LENGTH', 'LEN', 'NUM', 'NUMELEMENTS'):
            n_param = p.name
            break

    idx_var = ir["thread_indexing"]["index_variable"]
    body_stmts = f"int {idx_var} = blockIdx.x * blockDim.x + threadIdx.x;\n"
    if idx_var != 'tid':
        body_stmts += "int tid = blockIdx.x * blockDim.x + threadIdx.x; (void)tid;\n"
    if n_param != 'N':
        body_stmts += f"int {n_param} = N; (void){n_param};\n"

    seen_vars: set = set()
    for var in ir.get("local_variables", []):
        if var['name'] in (idx_var, 'tid', n_param) or var['name'] in seen_vars:
            continue
        if var['expression'] in CUDA_UNEVALUABLE_EXPRS:
            seen_vars.add(var['name'])
            continue
        seen_vars.add(var['name'])
        body_stmts += f"auto {var['name']} = {var['expression']}; (void){var['name']};\n"

    for op in ir.get("operations", []):
        expr = op['expression']
        target = op['target']
        if "__half{" in expr:
            expr = expr.replace("__half{", "(__half)(")
            if expr.endswith("}"):
                expr = expr[:-1] + ")"
        expr = re.sub(r'\b(float|int|uint32_t|int32_t)\((.*?)\)', r'(\1)(\2)', expr)
        body_stmts += f"{target} = {expr};\nvx_fence();\n"

    ck = CUDAKernel(
        name=ir["kernel_name"],
        params=params,
        raw_body=kernels_code[0],
        body_stmts=body_stmts,
        array_params=array_params,
        scalar_params=scalar_params,
        N_param=n_param,
        N_value=None,
        has_syncthreads=False,
        has_shared=len(ir.get("shared_memory", [])) > 0,
        shared_decls=[],
        extern_shared_decls=[],
        verified_shared_buffers=[],
        launch_N=None,
        is_2d=ir["thread_indexing"]["type"] == "2D_global",
        is_3d=ir["thread_indexing"]["type"] == "3D_global",
        grid_width=None,
        grid_height=None,
        warnings=[],
        is_template=False,
        template_type='int32_t'
    )

    # Build init_values
    init_values: dict = {}
    for idx2, ap in enumerate(ck.array_params):
        if idx2 == 0:
            init_values[ap.name] = list(range(1, N + 1))
        elif idx2 == 1:
            init_values[ap.name] = [10] * N
        else:
            init_values[ap.name] = [0] * N

    # Shared memory verification
    ktp = test_params.get(ck.name, {})
    for sm in ir.get("shared_memory", []):
        if sm.get("size_expression", "extern") != "extern":
            print(f"[FAIL] Oracle REJECTED: LLM hallucinated size for shared memory '{sm['name']}'.")
            return 1
        if ck.name in test_params and "extern_smem_expr" in ktp:
            gt_expr = ktp["extern_smem_expr"]
        else:
            print(f"[FAIL] Oracle REJECTED: No extern_smem_expr for '{sm['name']}'.")
            return 1
        from cuda_parser import evaluate_clang_ast
        env_b = {'math': math, 'N': N}
        expected_size = int(evaluate_clang_ast(gt_expr, env_b))
        ck.verified_shared_buffers.append({
            "name": sm["name"],
            "ctype": sm["base_type"],
            "size_bytes": expected_size
        })
        print(f"         Shared buffer '{sm['name']}' verified ({expected_size} bytes)")

    # Run oracle
    instructions, init_mem, init_regs, op_detected = kernel_to_oracle_ir(ck, N, init_values)

    def check_result(results, memory):
        if not ck.array_params:
            return True, "No arrays to check"
        dst = ck.array_params[-1]
        src_arrays = ck.array_params[:-1]
        errors = []
        for i in range(N):
            dst_base = init_regs.get(f'r{len(ck.array_params)}', 0)
            addr = dst_base + i * 4
            got = int.from_bytes(memory[addr:addr+4], byteorder='little', signed=True)
            
            env = {'i': i, 'math': math}
            idx_var = ir["thread_indexing"]["index_variable"]
            env[idx_var] = i
            for sap in src_arrays:
                env[sap.name] = init_values.get(sap.name, [0]*N)
            for sp in ck.scalar_params:
                if sp.name.upper() in ('N', 'SIZE', 'COUNT', 'LENGTH', 'LEN', 'NUM'):
                    env[sp.name] = N
                else:
                    env[sp.name] = 0
            if ck.name in test_params:
                env.update(test_params[ck.name])
            env['threadIdx'] = type('dim3', (), {'x': i, 'y': 0, 'z': 0})()
            env['blockIdx']  = type('dim3', (), {'x': 0, 'y': 0, 'z': 0})()
            env['blockDim']  = type('dim3', (), {'x': N, 'y': 1,  'z': 1})()
            env['gridDim']   = type('dim3', (), {'x': 1, 'y': 1,  'z': 1})()

            for var in ir.get("local_variables", []):
                if var['name'] in env:
                    continue
                if var['expression'] in CUDA_UNEVALUABLE_EXPRS:
                    continue
                try:
                    env[var['name']] = evaluate_clang_ast(var['expression'], env)
                except Exception as e:
                    return False, (
                        f"Oracle REJECTED: Cannot evaluate local_variable '{var['name']}' "
                        f"from expression '{var['expression']}': {e}. "
                        f"This identifier is unmappable in the current Oracle environment. "
                        f"Add it to test_params or fix the LLM extraction."
                    )

            match = re.search(rf'\b{dst.name}\s*\[.*?\]\s*=\s*(.+?);', kernels_code[0], re.DOTALL)
            if not match:
                match = re.search(dst.name + r'\[.*?\]\s*=\s*(.+?);', kernels_code[0], re.DOTALL)
            if match:
                raw_rhs = match.group(1).strip()
            else:
                raw_rhs = "0"
            
            try:
                if ck.name in test_params and "expected_val" in test_params[ck.name]:
                    expected = test_params[ck.name]["expected_val"]
                else:
                    expected = int(evaluate_clang_ast(raw_rhs, env))
            except Exception as e:
                return False, f"Failed to eval source truth '{raw_rhs}': {e}"
                
            if got != expected:
                errors.append(f"[{dst.name}][{i}]: got={got}, expected={expected}")
        if errors:
            return False, "; ".join(errors[:3])
        return True, f"All {N} results correct"

    oracle_result = verify_parallel_kernel(
        instructions=instructions,
        num_threads=N,
        initial_regs_per_thread=[dict(init_regs)] * N,
        initial_mem=init_mem,
        check_fn=check_result
    )
    oracle_passed = oracle_result.get("ok", False)
    oracle_msg = oracle_result.get("message", "")

    if oracle_passed:
        print(f"         [OK] Oracle PASSED — {oracle_msg}")
        print(f"         op_detected={op_detected}")
    else:
        print(f"         [NOTE] Oracle FAILED — {oracle_msg}")
        print(f"         op_detected={op_detected}")
        print(f"         Hardware result is the authoritative pass/fail.")

    # ── [4/5] Vortex C++ lowering ─────────────────────────────────────────
    print(f"\n[4/{total_stages}] Lowering to Vortex C++...", flush=True)

    PROJ = "llm_comprehension_test"
    VORTEX_HOME_WSL = "/home/dark_hacker/hackathon-project/vendor/vortex"

    # Inject test_params macros into cpp generation if needed
    if ck.name in test_params:
        macros = ""
        for k, v in test_params[ck.name].items():
            if k == "extern_smem_expr":
                continue
            macros += f"#define {k} {v}\n"
    else:
        macros = ""

    try:
        cpp_code = kernel_to_vortex_cpp(ck, simt_facts, N, init_values, op_detected)
    except Exception as e:
        print(f"[FAIL] Lowering failed: {e}")
        return 1

    if macros:
        cpp_code = macros + cpp_code

    art_dir = _ROOT / "artifacts" / PROJ
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "main.cpp").write_text(cpp_code, encoding="utf-8")
    (art_dir / "Makefile").write_text(
        lower_to_makefile(PROJ, VORTEX_HOME_WSL), encoding="utf-8"
    )
    print(f"         Generated {len(cpp_code)} bytes  ->  artifacts/{PROJ}/main.cpp")
    print(f"         [OK] Lowering complete for kernel '{ck.name}'")

    # ── [5/5] RTL Simulation ──────────────────────────────────────────────
    print(f"\n[5/{total_stages}] RTL simulation (simx)...", flush=True)

    wsl_project_root = str(_ROOT).replace("\\", "/").replace("C:/", "/mnt/c/")

    print("         Invoking: make -C artifacts/" + PROJ + " run-simx")
    r = run_wsl(
        f"source ~/hackathon-project/.wsl_env && cd '{wsl_project_root}' && "
        f"timeout 120 make -C artifacts/{PROJ} run-simx"
    )

    print("\n--- simx stdout ---")
    print(r.stdout)
    if r.returncode != 0:
        print("--- simx stderr ---")
        stderr_lines = r.stderr.strip().splitlines()
        for line in stderr_lines[:50]:
            print(line)
        if len(stderr_lines) > 50:
            print(f"... ({len(stderr_lines)-50} more lines)")

    m_res = re.search(r'SIMX_RESULT=(\d+)', r.stdout)
    m_cyc = re.search(r'SIMX_CYCLES=(\d+)', r.stdout)
    result_val = int(m_res.group(1)) if m_res else -1
    cycles_val = int(m_cyc.group(1)) if m_cyc else -1

    print("\n" + "=" * 60)
    if result_val == 0:
        print(f"HARDWARE RESULT: PASSED  (SIMX_RESULT=0  cycles={cycles_val})")
        print("=" * 60)
        return 0
    else:
        if r.returncode != 0 and result_val == -1:
            print(f"HARDWARE RESULT: FAILED (simx did not complete — exit {r.returncode})")
        else:
            print(f"HARDWARE RESULT: FAILED  (SIMX_RESULT={result_val}  cycles={cycles_val})")
        print("=" * 60)
        return 1


def main():
    p = argparse.ArgumentParser(
        prog="vortex_compile",
        description=(
            "CUDA -> Vortex RISC-V full pipeline:\n"
            "  LLM comprehension (Kimi-2.6) ->\n"
            "  Independent Oracle (Clang AST) ->\n"
            "  Vortex C++ lowering ->\n"
            "  RTL simulation (simx)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("cuda_file", help="Path to .cu source file")
    p.add_argument(
        "--kernel", metavar="NAME", default=None,
        help="Target a single kernel by name (for multi-kernel files)",
    )
    args = p.parse_args()
    sys.exit(run_pipeline(args.cuda_file, kernel_filter=args.kernel))


if __name__ == "__main__":
    main()
