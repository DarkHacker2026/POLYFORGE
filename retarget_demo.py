import json
import os
from pathlib import Path

from kernels.conditional_scatter import run_oracle_check as scatter_oracle, generate_conditional_scatter_cpp
from kernels.strided_reduction import run_oracle_check as reduce_oracle, generate_strided_reduction_cpp

ROOT = Path(__file__).resolve().parent

targets = {
    "vortex_base": ROOT / "data" / "hardware_facts.vortex.json",
    "vortex_wide": ROOT / "data" / "hardware_facts.vortex_wide.json"
}

def load_simt_facts(path):
    with open(path) as f:
        return json.load(f).get("simt_facts", {})

def main():
    print("=== Retargeting Demo (Item 6) ===\n")
    for target_name, path in targets.items():
        if not path.exists():
            print(f"Skipping {target_name} (file not found)")
            continue
            
        facts = load_simt_facts(path)
        print(f"--- TARGET: {target_name} ---")
        print(f"Facts loaded: Threads={facts.get('num_threads_per_warp')} | "
              f"Warps={facts.get('num_warps_per_core')} | "
              f"Cores={facts.get('num_cores')} | "
              f"Barrier Supported={facts.get('barrier_supported')}")

        # Kernel 1: Conditional Scatter
        print("  [Kernel 1: Conditional Scatter]")
        try:
            scatter_res = scatter_oracle(N=8, threshold=4, scale=10)
            print("    Oracle Check:", "PASS" if scatter_res["ok"] else f"FAIL ({scatter_res.get('error')})")
        except Exception as e:
            print("    Oracle Check:", f"ERROR ({e})")
        cpp = generate_conditional_scatter_cpp(facts, N=8, threshold=4, scale=10)
        # Verify the generated code used the target's threads_per_warp
        lines = [line.strip() for line in cpp.split('\n') if "CONFIGS:" in line]
        if lines:
            print("    Generated C++ Printf:", lines[0].replace('vx_printf(', '').replace(');', ''))
        else:
            print("    Generated C++ Printf: (None)")

        # Kernel 2: Strided Reduction
        print("  [Kernel 2: Strided Reduction]")
        try:
            # Need to adapt strided_reduction to take simt facts if we were truly parameterizing the oracle,
            # but we'll use the default args which assume N=8
            reduce_res = reduce_oracle(N=8)
            print("    Oracle Check:", "PASS" if reduce_res["ok"] else f"FAIL ({reduce_res.get('error')})")
        except Exception as e:
            print("    Oracle Check:", f"ERROR ({e})")
        cpp2 = generate_strided_reduction_cpp(facts, N=8)
        lines2 = [line.strip() for line in cpp2.split('\n') if "CONFIGS:" in line]
        if lines2:
            print("    Generated C++ Printf:", lines2[0].replace('vx_printf(', '').replace(');', ''))
        else:
            print("    Generated C++ Printf: (None)")
        print()

if __name__ == "__main__":
    main()
