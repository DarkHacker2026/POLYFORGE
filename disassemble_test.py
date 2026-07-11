import sys
import uuid
import subprocess
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from grow_compiler import IROperation, VortexArtifactEmitter, VortexSimulator

def create_candidate(instructions):
    return {"candidate_id": str(uuid.uuid4())[:8], "instructions": instructions}

def run_test():
    worker_root = ROOT / "artifacts" / "disasm_test"
    worker_root.mkdir(parents=True, exist_ok=True)
    emitter = VortexArtifactEmitter(worker_root, ROOT / "vendor" / "vortex")
    sim = VortexSimulator(ROOT / "vendor" / "vortex", emitter, sim_target="simx")
    
    i_load = {"op": "LW", "dst": "r2", "src1": "r6", "offset": 0}
    i_add  = {"op": "ADD", "dst": "r3", "src1": "r2", "src2": "r1"}
    i_f1   = {"op": "ADDI", "dst": "r7", "src1": "r7", "imm": 1}
    i_f2   = {"op": "ADDI", "dst": "r8", "src1": "r8", "imm": 1}
    i_f3   = {"op": "ADDI", "dst": "r9", "src1": "r9", "imm": 1}
    
    schedules = {
        "STALL": [i_load, i_add, i_f1, i_f2, i_f3],
        "HOIST": [i_load, i_f1, i_f2, i_f3, i_add]
    }
    
    macro_op = IROperation("macro", "macro", (), "disasm_test")
    
    disasm = {}
    for name, insts in schedules.items():
        cand = create_candidate(insts)
        # Running via simulator compiles it properly via the Makefile
        print(f"Compiling {name} via simx...")
        sim.run(macro_op, cand)
        
        # The ELF is in vendor/vortex/build/tests/kernel/disasm_test_<uuid>/kernel.elf
        proj_name = f"agent_macro_disasm_test_{cand['candidate_id']}"
        elf_path = f"vendor/vortex/build/tests/kernel/{proj_name}/kernel.elf"
        
        # Disassemble
        disasm_cmd = (
            f"cd ~/hackathon-project && source .wsl_env && "
            f"riscv32-unknown-elf-objdump -d {elf_path} | grep -A 20 kernel_main"
        )
        res = subprocess.run(["wsl.exe", "-e", "bash", "-c", disasm_cmd], capture_output=True, text=True)
        
        disasm[name] = res.stdout
        
        print(f"=== {name} Disassembly ===")
        print(res.stdout)
        print("======================\n")

    if disasm["STALL"].strip() == disasm["HOIST"].strip():
        print("Result: IDENTICAL (GCC completely optimized away the ordering!)")
    else:
        print("Result: DIFFERENT (Ordering was preserved by GCC)")

if __name__ == "__main__":
    run_test()
