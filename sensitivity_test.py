import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from grow_compiler import IROperation, VortexArtifactEmitter, VortexSimulator

def create_candidate(instructions):
    return {"candidate_id": str(uuid.uuid4())[:8], "instructions": instructions}

def run_test():
    worker_root = ROOT / "artifacts" / "vortex_tests"
    worker_root.mkdir(parents=True, exist_ok=True)
    emitter = VortexArtifactEmitter(worker_root, ROOT / "vendor" / "vortex")
    sim = VortexSimulator(ROOT / "vendor" / "vortex", emitter, sim_target="rtlsim")
    
    # Pathological STALL: 8 back-to-back load-use pairs
    # Wait, the load is from r6, we just repeatedly load and add.
    # LW r2, 0(r6) -> ADD r2, r2, r1
    stall_insts = []
    for i in range(8):
        stall_insts.append({"op": "LW", "dst": "r2", "src1": "r6", "offset": 0})
        stall_insts.append({"op": "ADD", "dst": "r2", "src1": "r2", "src2": "r1"})
        
    # Zero out the checked register so it passes validation (expected=0)
    stall_insts.append({"op": "SUB", "dst": "r2", "src1": "r2", "src2": "r2"})
        
    # SPREAD: 8 independent loads, then 8 independent adds
    # We will use registers r8 to r15
    spread_insts = []
    for i in range(8):
        reg = f"r{8+i}"
        spread_insts.append({"op": "LW", "dst": reg, "src1": "r6", "offset": 0})
    for i in range(8):
        reg = f"r{8+i}"
        spread_insts.append({"op": "ADD", "dst": reg, "src1": reg, "src2": "r1"})
        
    # Zero out the checked register (r8)
    spread_insts.append({"op": "SUB", "dst": "r8", "src1": "r8", "src2": "r8"})
        
    schedules = {
        "PATHOLOGICAL_STALL": stall_insts,
        "MAXIMAL_SPREAD": spread_insts
    }
    
    macro_op = IROperation("macro", "macro", (), "sensitivity_test")
    
    print("Running Sensitivity Test on rtlsim...\n")
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
        
    stall = results["PATHOLOGICAL_STALL"]
    spread = results["MAXIMAL_SPREAD"]
    
    print("\n=== Interpretation ===")
    if stall == spread:
        print("OUTCOME: Pathological Stall and Maximal Spread TIE exactly!")
        print("CONCLUSION: The RTL model truly has no load-use penalty. Even extreme back-to-back stalls don't increase cycles.")
    elif stall > spread:
        print(f"OUTCOME: Stall ({stall}) costs more than Spread ({spread}). Difference = {stall - spread} cycles.")
        diff_per_unroll = (stall - spread) / 64.0
        print(f"This is a difference of {diff_per_unroll} cycles per block iteration.")
        if diff_per_unroll < 1.0:
            print("CONCLUSION: The difference is so small that the N=64 averaging and fixed overhead are rounding it away in smaller blocks.")
        else:
            print("CONCLUSION: Load-use penalty exists and is measurable at large scales!")

if __name__ == "__main__":
    run_test()
