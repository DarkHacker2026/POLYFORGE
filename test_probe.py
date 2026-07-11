import sys
from pathlib import Path
from discovery_agent import DiscoveryAgent, VortexArtifactEmitter
from grow_compiler import VortexSimulator, IROperation

ROOT = Path(".")
emitter = VortexArtifactEmitter(ROOT / "artifacts" / "vortex_tests", ROOT / "vendor" / "vortex")
sim = VortexSimulator(ROOT / "vendor" / "vortex", emitter, sim_target="simx")

cand = {"candidate_id": "probe", "instructions": [{"op": "ADDI", "dst": "r0", "src1": "r0", "imm": 0}]}
macro_op = IROperation("probe", "probe", (), "probe")
proof = sim.run(macro_op, cand)
print("PROOF:", proof)
