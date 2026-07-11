import sys
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from reference_isa import ReferenceISA, get_initial_state
from grow_compiler import IROperation, VortexArtifactEmitter, VortexSimulator, RuleDatabase
from block_compiler import BlockCompiler

def generate_random_block(length=4):
    ops = []
    # Available base variables
    vars = ["r1", "r2", "r6"]
    
    for i in range(length):
        op_type = random.choice(["add", "mul", "load"])
        dst = f"r{10+i}"
        
        if op_type == "add":
            src1 = random.choice(vars)
            src2 = random.choice(vars)
            ops.append(IROperation(dst, "add", (src1, src2), f"{dst} = add({src1}, {src2})"))
        elif op_type == "mul":
            src1 = random.choice(vars)
            src2 = random.choice(vars)
            ops.append(IROperation(dst, "mul", (src1, src2), f"{dst} = mul({src1}, {src2})"))
        elif op_type == "load":
            src1 = "r6" # safe address base
            ops.append(IROperation(dst, "load", (src1,), f"{dst} = load({src1})"))
            
        vars.append(dst)
        
    return ops

def main():
    print("Generating candidate_log.jsonl dataset using simx...")
    db = RuleDatabase(ROOT / "data" / "rules.json")
    compiler = BlockCompiler(db)
    emitter = VortexArtifactEmitter(ROOT / "artifacts" / "vortex_tests", ROOT / "vendor" / "vortex")
    sim = VortexSimulator(ROOT / "vendor" / "vortex", emitter, sim_target="simx")
    
    log_file = ROOT / "data" / "candidate_log.jsonl"
    
    generated_count = 0
    for block_idx in range(500):
        ops = generate_random_block(length=random.randint(4, 8))
        try:
            baseline_insts = compiler.instantiate_block(ops)
            allocated = compiler.allocate_registers(baseline_insts)
        except Exception:
            continue
            
        candidates = [allocated]
        for _ in range(10):
            shuffled = allocated.copy()
            random.shuffle(shuffled)
            if compiler.validate_schedule(allocated, shuffled):
                candidates.append(shuffled)
                
        for cand_idx, cand in enumerate(candidates):
            macro_op = IROperation("macro", "macro", (), "macro_block")
            candidate_dict = {
                "candidate_id": f"gen_b{block_idx}_c{cand_idx}",
                "instructions": cand
            }
            proof = sim.run(macro_op, candidate_dict)
            
            log_entry = {
                "ts": int(__import__("time").time()),
                "op_type": "macro_block",
                "simulator_target": proof.get("simulator", "unknown"),
                "candidate_json": candidate_dict,
                "proof_cycles": proof.get("cycles", -1),
                "passed": proof.get("ok", False),
                "error": proof.get("error"),
                "is_dag_valid": compiler.validate_schedule(allocated, cand)
            }
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
                
            generated_count += 1
            if generated_count % 50 == 0:
                print(f"Generated {generated_count} records...")
                
    print(f"Done! Generated {generated_count} records.")

if __name__ == "__main__":
    main()
