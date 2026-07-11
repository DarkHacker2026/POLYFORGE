import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grow_compiler import IROperation, VortexArtifactEmitter, VortexSimulator

def measure_naive():
    op = IROperation("r3", "mul_by_const_8", ("r1",), "r3 = mul_by_const_8(r1)")
    
    slli_candidate = {
        "candidate_id": "MUL_BY_CONST_8_SHIFT_v1",
        "instructions": [
            {"op": "SLLI", "dst": "r3", "src1": "r1", "imm": 3}
        ]
    }
    
    emitter = VortexArtifactEmitter(ROOT / "artifacts" / "vortex_tests", ROOT / "vendor" / "vortex")
    sim = VortexSimulator(ROOT / "vendor" / "vortex", emitter, sim_target="rtlsim")
    
    print("Measuring SLLI candidate on rtlsim...")
    proof = sim.run(op, slli_candidate)
    print("Result:", proof)
    
    if proof["ok"]:
        print(f"Honest rtlsim cycle count: {proof['cycles']}")

if __name__ == "__main__":
    measure_naive()
