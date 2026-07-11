import sys
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from grow_compiler import IROperation, VortexArtifactEmitter, VortexSimulator

def create_candidate(instructions):
    return {"candidate_id": str(uuid.uuid4())[:8], "instructions": instructions}

def run_test():
    # Setup
    worker_root = ROOT / "artifacts" / "vortex_tests"
    worker_root.mkdir(parents=True, exist_ok=True)
    emitter = VortexArtifactEmitter(worker_root, ROOT / "vendor" / "vortex")
    sim = VortexSimulator(ROOT / "vendor" / "vortex", emitter, sim_target="rtlsim")
    
    # We create a STALL schedule and 3 HOIST schedules
    # Load chain: LW r2, 0(r6) -> ADD r3, r2, r1
    # Fillers:
    # F1: ADDI r7, r7, 1
    # F2: ADDI r8, r8, 1
    # F3: ADDI r9, r9, 1
    
    i_load = {"op": "LW", "dst": "r2", "src1": "r6", "offset": 0}
    i_add  = {"op": "ADD", "dst": "r3", "src1": "r2", "src2": "r1"}
    i_f1   = {"op": "ADDI", "dst": "r7", "src1": "r7", "imm": 1}
    i_f2   = {"op": "ADDI", "dst": "r8", "src1": "r8", "imm": 1}
    i_f3   = {"op": "ADDI", "dst": "r9", "src1": "r9", "imm": 1}
    
    schedules = {
        "STALL (gap=0)": [i_load, i_add, i_f1, i_f2, i_f3],
        "HOIST (gap=1)": [i_load, i_f1, i_add, i_f2, i_f3],
        "HOIST (gap=2)": [i_load, i_f1, i_f2, i_add, i_f3],
        "HOIST (gap=3)": [i_load, i_f1, i_f2, i_f3, i_add]
    }
    
    # We will validate data-flow equivalence by construction (the schedules are just shuffled independent instructions)
    
    # We need a macro_op to pass to simulator, it doesn't matter what it is
    macro_op = IROperation("macro", "macro", (), "discrimination_test")
    
    print("Running Discrimination Test on rtlsim...\n")
    
    results = {}
    for name, insts in schedules.items():
        print(f"Testing {name}...")
        cand = create_candidate(insts)
        proof = sim.run(macro_op, cand)
        if not proof["ok"]:
            print(f"  Failed to run {name}: {proof.get('error')}")
            return
        
        cycles = proof["cycles"]
        results[name] = cycles
        print(f"  {name} -> {cycles} cycles")
        
    print("\n=== Interpretation ===")
    stall_cycles = results["STALL (gap=0)"]
    hoist_cycles = results["HOIST (gap=3)"]
    
    if stall_cycles == hoist_cycles:
        print("OUTCOME: STALL and HOIST tie (same cycles).")
        print("CONCLUSION: The Vortex rtlsim model does not penalize load-use stalls at this level, so instruction ordering is cycle-neutral on this target.")
        print("Scheduling-based cost modeling is unnecessary here. Shelve the entire scheduling-optimization line.")
    elif stall_cycles > hoist_cycles:
        print("OUTCOME: STALL costs MORE than HOIST.")
        print("CONCLUSION: Scheduling DOES matter. The earlier dataset was the problem. Proceed to regenerate a dataset containing load-use latency gaps.")
    else:
        print(f"OUTCOME: Unexpected result (STALL < HOIST). STALL={stall_cycles}, HOIST={hoist_cycles}")

if __name__ == "__main__":
    run_test()
