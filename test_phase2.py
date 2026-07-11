import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from reference_isa import ReferenceISA, get_initial_state
from grow_compiler import IROperation, VortexArtifactEmitter, VortexSimulator, RuleDatabase, parse_ir
from block_compiler import BlockCompiler
import copy

def run_tests():
    # Setup mock rules for add, mul, load, store
    import json
    # Mocking rule database
    rules_data = {
        "rules": {
            "add": {"pattern": [{"op": "ADD", "dst": "$dst", "src1": "$src1", "src2": "$src2"}]},
            "mul": {"pattern": [{"op": "MUL", "dst": "$dst", "src1": "$src1", "src2": "$src2"}]},
            "load": {"pattern": [{"op": "LW", "dst": "$dst", "src1": "$src1", "offset": 0}]},
            "store": {"pattern": [{"op": "SW", "dst": "r0", "src1": "$src2", "src2": "$src1", "offset": 0}]}
        }
    }
    with open("mock_rules_phase2.json", "w") as f:
        json.dump(rules_data, f)
        
    db = RuleDatabase(Path("mock_rules_phase2.json"))
    compiler = BlockCompiler(db)
    
    print("--- Test (a): 3-4 op dependent kernel compiled and allocated ---")
    # r5 = load(r6); r3 = add(r5, r1); r4 = mul(r3, r2); store(r4, r7)
    ops = [
        IROperation("r5", "load", ("r6",), ""),
        IROperation("r3", "add", ("r5", "r1"), ""),
        IROperation("r4", "mul", ("r3", "r2"), ""),
        IROperation("", "store", ("r4", "r7"), "")
    ]
    insts = compiler.instantiate_block(ops)
    allocated = compiler.allocate_registers(insts)
    print("Naive allocated sequence:")
    for inst in allocated:
        print(f"  {inst}")
        
    # We should see physical registers r1-r31 used instead of v_*
    has_virtual = any(str(v).startswith("v_") for inst in allocated for v in inst.values() if isinstance(v, str))
    if not has_virtual:
        print("PASS: Virtual registers killed, only physical remains!\n")
    else:
        print("FAIL: Virtual registers leaked!\n")

    print("--- Test (b): Reordered schedule validation ---")
    # Let's say baseline has 2 independent ops:
    # 1. r3 = add(r1, r2)
    # 2. r4 = mul(r6, r7)
    # They can be reordered!
    baseline = [
        {"op": "ADD", "dst": "r3", "src1": "r1", "src2": "r2"},
        {"op": "MUL", "dst": "r4", "src1": "r6", "src2": "r7"}
    ]
    reordered = [
        {"op": "MUL", "dst": "r4", "src1": "r6", "src2": "r7"},
        {"op": "ADD", "dst": "r3", "src1": "r1", "src2": "r2"}
    ]
    valid = compiler.validate_schedule(baseline, reordered)
    print(f"Independent reordering valid? {valid}")
    if valid:
        print("PASS: DAG validator approved valid reordering!\n")
    else:
        print("FAIL: DAG validator rejected valid reordering!\n")
        
    # Now invalid (RAW violation)
    invalid_reorder = [
        {"op": "MUL", "dst": "r4", "src1": "r3", "src2": "r7"},
        {"op": "ADD", "dst": "r3", "src1": "r1", "src2": "r2"}
    ]
    valid = compiler.validate_schedule(baseline, invalid_reorder)
    print(f"Dependent reordering valid? {valid}")
    if not valid:
        print("PASS: DAG validator rejected invalid reordering!\n")
    else:
        print("FAIL: DAG validator allowed invalid reordering!\n")

    print("--- Test (c): Forced register pressure (spill) ---")
    # We artificially limit free_regs in compiler to 2 registers (r1, r2)
    # Then try to allocate 4 virtual registers.
    class SpillCompiler(BlockCompiler):
        def allocate_registers(self, insts):
            import block_compiler
            old_free = block_compiler.BlockCompiler.allocate_registers
            # We will just hack the method to force a spill by injecting variables, 
            # or just test it natively if we create a block with 35 live variables!
            pass
            
    ops_spill = []
    # Write 33 variables so they are all live
    for i in range(33):
        ops_spill.append(IROperation(f"r{i+100}", "load", ("r1",), ""))
    # Read them all at the end so they stay live!
    for i in range(33):
        ops_spill.append(IROperation(f"r{i+200}", "add", (f"r{i+100}", "r2"), ""))
    insts_spill = compiler.instantiate_block(ops_spill)
    allocated_spill = compiler.allocate_registers(insts_spill)
    
    spill_count = sum(1 for x in allocated_spill if x["op"] == "SW" and x.get("dst") == "r0")
    print(f"Spill count (SW emitted): {spill_count}")
    if spill_count > 0:
        print("PASS: Spills successfully emitted under pressure!\n")
    else:
        print("FAIL: No spills emitted!\n")

if __name__ == "__main__":
    run_tests()
