import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grow_compiler import IROperation, VortexArtifactEmitter, VortexSimulator

def test_fault():
    op = IROperation("r3", "add", ("r1", "r2"), "r3 = add(r1, r2)")
    # Bad candidate: uses SUB instead of ADD
    bad_candidate = {
        "candidate_id": "bad_sub_for_add",
        "instructions": [
            {"op": "SUB", "dst": "r3", "src1": "r1", "src2": "r2"}
        ]
    }
    
    emitter = VortexArtifactEmitter(ROOT / "artifacts" / "vortex_tests", ROOT / "vortex")
    sim = VortexSimulator(ROOT / "vortex", emitter)
    
    print("Running bad candidate...")
    proof = sim.run(op, bad_candidate)
    print("Result:", proof)

if __name__ == "__main__":
    test_fault()
