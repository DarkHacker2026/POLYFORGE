import sys
import json
import random
import multiprocessing
import shutil
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from grow_compiler import IROperation, VortexArtifactEmitter, VortexSimulator
from block_compiler import BlockCompiler

def generate_realistic_block(idx: int) -> list[IROperation]:
    """Generates a realistic basic block of IR operations with parallel independent chains."""
    ops = []
    chains = 6
    depth = 4
    # Create loads (genuine load-use chains)
    for c in range(chains):
        ops.append(IROperation(f"v{idx}_{c}_0", "load", (f"r6",), f"v{idx}_{c}_0 = load(r6)"))
    
    # Create dependent math ops for each chain
    for d in range(1, depth):
        for c in range(chains):
            src1 = f"v{idx}_{c}_{d-1}"
            if c > 0 and random.random() > 0.7:
                src2 = f"v{idx}_{c-1}_{d-1}"
            else:
                src2 = src1
            
            op_type = "add" if random.random() > 0.5 else "mul"
            ops.append(IROperation(f"v{idx}_{c}_{d}", op_type, (src1, src2), f"v{idx}_{c}_{d} = {op_type}({src1}, {src2})"))
            
    # Add independent filler instructions (so hoisting can hide latency without colliding)
    for f in range(2):
        # use r20..r27 to guarantee they don't collide with each other or the chains
        ops.append(IROperation(f"v{idx}_f{f}", "add", (f"r{20+f}", f"r{20+f}"), f"v{idx}_f{f} = add(r{20+f}, r{20+f})"))
        
    return ops

def process_block(block_idx: int):
    # Set up isolated workspace for this concurrent process!
    worker_root = ROOT / "artifacts" / f"vortex_tests_{block_idx}"
    worker_root.mkdir(parents=True, exist_ok=True)
    emitter = VortexArtifactEmitter(worker_root, ROOT / "vendor" / "vortex")
    # Real harness: rtlsim
    sim = VortexSimulator(ROOT / "vendor" / "vortex", emitter, sim_target="rtlsim")
    
    hardware_facts = json.loads((ROOT / "data" / "hardware_facts.vortex.json").read_text())
    from grow_compiler import RuleDatabase
    rules_db = RuleDatabase(ROOT / "data" / "rules.json")
    compiler = BlockCompiler(rules_db, hardware_facts)
    
    ops = generate_realistic_block(block_idx)
    try:
        baseline_insts = compiler.instantiate_block(ops)
        allocated = compiler.allocate_registers(baseline_insts)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return []
        
    # To find diverse valid schedules in a large DAG, we perform random topological sorts.
    # To find diverse valid schedules in a large DAG, we'll explicitly generate DFS and BFS-like schedules.
    # Actually, we can just randomly pick ready nodes.
    
    # 1. Build DAG of indices
    in_degree = {i: 0 for i in range(len(allocated))}
    children = {i: [] for i in range(len(allocated))}
    
    for i in range(len(allocated)):
        for j in range(i + 1, len(allocated)):
            if not compiler.validate_schedule(allocated, [allocated[i], allocated[j]]):
                # If i must come before j, it means j depends on i.
                # Actually, validate_schedule checks a whole sequence, but we know the baseline is valid.
                pass
                
    # A much simpler way to get STALL vs SPREAD:
    # Just reverse the chains!
    candidates = [allocated]
    
    # Let's generate 14 random valid schedules using a random topological sort!
    # To build the graph properly:
    for i in range(len(allocated)):
        writes_i = allocated[i].get("dst")
        for j in range(i + 1, len(allocated)):
            reads_j = [allocated[j].get("src1"), allocated[j].get("src2")]
            writes_j = allocated[j].get("dst")
            if (writes_i and writes_i in reads_j) or (writes_i and writes_i == writes_j) or (writes_j and writes_j in [allocated[i].get("src1"), allocated[i].get("src2")]):
                children[i].append(j)
                in_degree[j] += 1
                
    attempts = 0
    while len(candidates) < 15 and attempts < 100:
        attempts += 1
        ready = [i for i in range(len(allocated)) if in_degree[i] == 0]
        curr_in_degree = in_degree.copy()
        sched_indices = []
        
        while ready:
            # For STALLs, we want to pick children of the LAST picked node if possible (DFS)
            # For SPREAD, we want to pick completely random or BFS.
            if random.random() > 0.5 and sched_indices:
                last = sched_indices[-1]
                # find children of last that are ready
                ready_children = [c for c in children[last] if curr_in_degree[c] == 0 and c in ready]
                if ready_children:
                    nxt = random.choice(ready_children)
                else:
                    nxt = random.choice(ready)
            else:
                nxt = random.choice(ready)
                
            ready.remove(nxt)
            sched_indices.append(nxt)
            for c in children[nxt]:
                curr_in_degree[c] -= 1
                if curr_in_degree[c] == 0:
                    ready.append(c)
                    
        shuffled = [allocated[i] for i in sched_indices]
        if compiler.validate_schedule(allocated, shuffled):
            if shuffled not in candidates:
                candidates.append(shuffled)
                
    num_insts = len(allocated)
    
    # Calculate independent instructions (those that don't consume a register produced by a previous instruction in this block)
    produced_regs = set()
    independent_count = 0
    for inst in allocated:
        inputs = [inst.get("src1"), inst.get("src2")]
        if not any(inp in produced_regs for inp in inputs if inp):
            independent_count += 1
        if inst.get("dst"):
            produced_regs.add(inst["dst"])
            
    print(f"Block {block_idx} Stats: {num_insts} instructions, {independent_count} independent (reorderable) instructions. Generated {len(candidates)} valid topological schedules.")
                
    results = []
    for cand_idx, cand in enumerate(candidates):
        macro_op = IROperation("macro", "macro", (), "macro_block")
        candidate_dict = {
            "candidate_id": f"real_b{block_idx}",
            "cand_variant": cand_idx,
            "instructions": cand
        }
        
        # LABEL DEFINITION: Whole-block marginal cycles (cycles / N_unroll)
        # Because we run the entire block as a macro operation!
        proof = sim.run(macro_op, candidate_dict)
        if "cycles" in proof and proof["cycles"] > 0:
            log_entry = {
                "ts": int(__import__("time").time()),
                "op_type": "macro_block",
                "simulator_target": "rtlsim",
                "candidate_json": candidate_dict,
                "proof_cycles": proof["cycles"],
                "passed": True
            }
            results.append(log_entry)
        else:
            print(f"Candidate {cand_idx} failed: {proof.get('error', 'unknown')}")
            
    # Cleanup isolated workspace
    shutil.rmtree(worker_root, ignore_errors=True)
    return results

if __name__ == "__main__":
    log_file = ROOT / "data" / "candidate_log.jsonl"
        
    print("Generating real Phase 2 dataset CONCURRENTLY to scale up n...")
    
    blocks_to_generate = 150
    valid_count = 0
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # We need process_block to return the block results
        results = pool.map(process_block, range(blocks_to_generate))
        
    with open(log_file, "w", encoding="utf-8") as f:
        for block_results in results:
            for res in block_results:
                f.write(json.dumps(res) + "\n")
                valid_count += 1
                
    print(f"Done! Generated {valid_count} real candidate records.")
