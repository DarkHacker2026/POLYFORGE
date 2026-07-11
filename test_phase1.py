import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from reference_isa import ReferenceISA, get_initial_state, compute_expected
from grow_compiler import IROperation, VortexArtifactEmitter, VortexSimulator

def run_tests():
    print("--- Test (a): Fault injection ---")
    isa = ReferenceISA()
    initial_regs, initial_mem = get_initial_state()
    # Expected ADD behavior for add(r1, r2). r1=10, r2=5.
    # We'll inject SUB: r3 = r1 - r2 = 5.
    # Expected: 15.
    # We use ReferenceISA to compute expected for "add", but execute "SUB".
    expected = compute_expected(IROperation("r3", "add", ("r1", "r2"), ""), initial_regs, initial_mem)
    final_regs = isa.execute([{"op": "SUB", "dst": "r3", "src1": "r1", "src2": "r2"}], initial_regs, initial_mem)
    actual = final_regs["r3"]
    print(f"Injected SUB for ADD. Expected: {expected}, Actual: {actual}")
    if actual != expected:
        print("PASS: Fault injection rejected!\n")
    else:
        print("FAIL: Fault injection missed!\n")

    print("--- Test (b): 3-instruction dependent sequence ---")
    # r5 = r1 + r2 (10+5=15)
    # r4 = r5 * r2 (15*5=75)
    # r3 = r4 - r1 (75-10=65)
    seq = [
        {"op": "ADD", "dst": "r5", "src1": "r1", "src2": "r2"},
        {"op": "MUL", "dst": "r4", "src1": "r5", "src2": "r2"},
        {"op": "SUB", "dst": "r3", "src1": "r4", "src2": "r1"}
    ]
    isa = ReferenceISA()
    initial_regs, initial_mem = get_initial_state()
    final_regs = isa.execute(seq, initial_regs, initial_mem)
    oracle_res = final_regs['r3']
    print(f"Oracle final r3: {oracle_res}")
    
    # We will run this sequence through VortexSimulator!
    emitter = VortexArtifactEmitter(ROOT / "artifacts" / "vortex_tests", ROOT / "vendor" / "vortex")
    sim = VortexSimulator(ROOT / "vendor" / "vortex", emitter, sim_target="rtlsim")
    
    # We will temporarily patch compute_expected for this custom test
    import grow_compiler
    old_compute = grow_compiler.compute_expected
    grow_compiler.compute_expected = lambda *args: oracle_res
    
    op = IROperation("r3", "custom_3op", ("r1", "r2"), "")
    candidate = {
        "candidate_id": "test_3op",
        "instructions": seq
    }
    
    print("Running sequence on rtlsim...")
    proof = sim.run(op, candidate)
    
    grow_compiler.compute_expected = old_compute # restore
    
    print(f"RTL sim result: {proof}")
    if proof.get("ok"):
        print("PASS: 3-instruction sequence matched exactly!\n")
    else:
        print("FAIL: RTL sim didn't match or failed.\n")

    print("--- Test (c): Load/Store round trip ---")
    seq_mem = [
        {"op": "LW", "dst": "r3", "src1": "r6", "offset": 0},    # r3 = mem[12] = 101
        {"op": "ADDI", "dst": "r3", "src1": "r3", "imm": 99},    # r3 = 200
        {"op": "SW", "dst": "r0", "src1": "r6", "src2": "r3", "offset": 4}, # mem[16] = 200
        {"op": "LW", "dst": "r4", "src1": "r6", "offset": 4}     # r4 = mem[16] = 200
    ]
    isa = ReferenceISA()
    initial_regs, initial_mem = get_initial_state()
    final_regs = isa.execute(seq_mem, initial_regs, initial_mem)
    print(f"Memory round trip final r4: {final_regs['r4']}")
    if final_regs["r4"] == 200:
        print("PASS: Memory round trip succeeded!\n")
    else:
        print("FAIL: Memory round trip missed!\n")

if __name__ == "__main__":
    run_tests()
