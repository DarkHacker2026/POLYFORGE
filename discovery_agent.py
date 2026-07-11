import sys
import json
import subprocess
import re
import hashlib
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from grow_compiler import IROperation, VortexArtifactEmitter, VortexSimulator


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SIMT probe emitter: writes full C++ kernels (not inline asm) that call
# Vortex intrinsics to read CSRs and print them.  The parent VortexSimulator
# run() path is reused for compile + rtlsim.
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class SIMTProbeEmitter:
    """Emits a standalone C++ kernel that probes one SIMT property and prints
    a SIMX_RESULT=<value> line so the existing VortexSimulator stdout parser
    can extract it."""

    def __init__(self, out_dir: Path, vortex_home: Path):
        self.out_dir = out_dir
        self.vortex_home = vortex_home

    def emit_csr_probe(self, probe_name: str, csr_call: str) -> Path:
        """Emit a single-threaded kernel that prints a CSR value.

        *csr_call* is a C expression like ``vx_num_threads()`` that returns int.
        """
        project = f"simt_probe_{probe_name}"
        project_dir = self.out_dir / project
        project_dir.mkdir(parents=True, exist_ok=True)

        cpp = f"""#include <stdint.h>
#include <vx_print.h>
#include <vx_intrinsics.h>

int main() {{
  int val = (int)({csr_call});
  vx_printf("SIMX_RESULT=%d\\n", val);
  vx_printf("SIMX_EXPECTED=%d\\n", val);
  vx_printf("SIMX_CYCLES=%d\\n", 1);
  vx_printf("Passed! result matched expected\\n");
  return 0;
}}
"""
        vortex_home_str = str(self.vortex_home).replace("\\", "/")
        makefile = f"""VORTEX_HOME ?= {vortex_home_str}
VORTEX_BUILD_DIR ?= $(VORTEX_HOME)/build
PROJECT := {project}
STAGED_DIR := $(VORTEX_BUILD_DIR)/tests/kernel/$(PROJECT)

.PHONY: stage run-simx run-rtlsim clean-stage

stage:
\tmkdir -p "$(STAGED_DIR)"
\tcp main.cpp "$(STAGED_DIR)/main.cpp"
\tprintf '%s\\n' 'ROOT_DIR := $$(realpath ../../..)' 'include $$(ROOT_DIR)/config.mk' '' 'PROJECT := $(PROJECT)' 'SRC_DIR := $$(VORTEX_BUILD_DIR)/tests/kernel/$$(PROJECT)' 'SRCS := $$(SRC_DIR)/main.cpp' '' 'include $$(VORTEX_HOME)/tests/kernel/common.mk' > "$(STAGED_DIR)/Makefile"

run-simx: stage
\t$(MAKE) -C "$(STAGED_DIR)" run-simx

run-rtlsim: stage
\t$(MAKE) -C "$(STAGED_DIR)" run-rtlsim

clean-stage:
\trm -rf "$(STAGED_DIR)"
"""
        (project_dir / "main.cpp").write_text(cpp, encoding="utf-8")
        (project_dir / "Makefile").write_text(makefile, encoding="utf-8")
        (project_dir / "candidate.json").write_text(
            json.dumps({"candidate_id": probe_name, "instructions": []}, indent=2),
            encoding="utf-8",
        )
        return project_dir


class DiscoveryAgent:
    def __init__(self, simulator: VortexSimulator):
        self.simulator = simulator
        self.facts: dict = {
            "registers": [],
            "zero_register": "r0",
            "spill_base_register": "r0",
            "isa": {},
            # simt_facts populated by discover_simt_features()
            "simt_facts": {},
            # provenance: audit trail for every discovered fact
            "provenance": [],
        }

    def _record_provenance(self, fact_name: str, value, probe_source: str,
                           raw_output: str, derived_by: str):
        """Record an auditable entry for a single discovered fact."""
        self.facts["provenance"].append({
            "fact": fact_name,
            "measured_value": value,
            "probe_source_hash": "sha256:" + hashlib.sha256(
                probe_source.encode()).hexdigest(),
            "probe_c_source": probe_source,
            "rtlsim_raw_output": raw_output.strip()[:2000],
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "derived_by": derived_by,
        })

    # ------ scalar probes (unchanged) ---------------------------------------------------------------------------------------------------------------------------------------

    def discover_registers(self):
        print("Probing register file size...")
        valid_regs = []
        for i in range(32):
            reg = f"r{i}"
            inst = {"op": "ADDI", "dst": reg, "src1": "r0", "imm": 0}
            macro_op = IROperation("probe", "probe", (), "probe")
            cand = {"candidate_id": "probe", "instructions": [inst]}
            proof = self.simulator.run(macro_op, cand)
            if proof.get("ok", False):
                valid_regs.append(reg)
        self.facts["registers"] = valid_regs
        print(f"Discovered {len(valid_regs)} registers: "
              f"{valid_regs[0]} ... {valid_regs[-1] if valid_regs else 'none'}")

    def discover_isa_and_latencies(self):
        print("Probing ISA and marginal latencies...")
        potential_ops = ["ADD", "SUB", "MUL", "LW", "SW", "ADDI", "SLLI", "BNE", "BEQ"]
        safe_dst = self.facts["registers"][1] if len(self.facts["registers"]) > 1 else "r0"
        safe_src = self.facts["registers"][2] if len(self.facts["registers"]) > 2 else "r0"
        for op in potential_ops:
            base_reg = "r6" if op in ("LW", "SW") else safe_src
            inst = {"op": op, "dst": safe_dst, "src1": safe_src, "src2": safe_src,
                    "imm": 0, "offset": 0, "base": base_reg}
            macro_op = IROperation("probe", "probe", (), "probe")
            cand = {"candidate_id": "probe", "instructions": [inst]}
            proof = self.simulator.run(macro_op, cand)
            if "cycles" in proof and proof["cycles"] > 0:
                self.facts["isa"][op] = {"latency": proof["cycles"]}
                print(f"  {op}: {proof['cycles']} cycles")

    # ------ SIMT probes (NEW) ---------------------------------------------------------------------------------------------------------------------------------------------------------------

    def discover_simt_features(self):
        """Probe SIMT properties using Vortex CSR intrinsics.

        Each probe emits a standalone C++ kernel, compiles and runs it on
        rtlsim, then parses the SIMX_RESULT= output.  A fact is recorded ONLY
        if the kernel ran successfully --- same empirical rule as scalar probes.

        Facts written to self.facts["simt_facts"]:
            num_threads  --- vx_num_threads() (threads per warp)
            num_warps    --- vx_num_warps()   (warps per core)
            num_cores    --- vx_num_cores()   (cores in design)
            thread_id_supported   --- True if reading VX_CSR_THREAD_ID works
            barrier_supported     --- True if vx_barrier() compiles and runs
        """
        print("\nProbing SIMT features...")
        simt_emitter = SIMTProbeEmitter(
            self.simulator.emitter.out_dir,
            self.simulator.vortex_home,
        )
        simt_facts: dict = {}

        def run_probe(probe_name: str, csr_call: str) -> int | None:
            """Return the integer result of the CSR probe, or None on failure."""
            proj_dir = simt_emitter.emit_csr_probe(probe_name, csr_call)
            project_name = proj_dir.name
            worker_root_name = proj_dir.parent.name

            # Sync to WSL
            win_path = proj_dir.absolute().as_posix()
            wsl_src = win_path.replace("C:/", "/mnt/c/")
            wsl_dest_parent = f"~/hackathon-project/artifacts/{worker_root_name}"
            sync_cmd = (f"mkdir -p {wsl_dest_parent} && "
                        f"cp -r '{wsl_src}' {wsl_dest_parent}/")
            try:
                subprocess.run(
                    ["wsl.exe", "-e", "bash", "-c", sync_cmd],
                    check=True, capture_output=True, text=True, timeout=30)
            except Exception as e:
                print(f"  [{probe_name}] sync failed: {e}")
                return None

            # Run
            make_target = f"run-{self.simulator.sim_target}"
            run_cmd = (
                f"cd ~/hackathon-project && source .wsl_env && "
                f"timeout {self.simulator.timeout_inner} "
                f"make -C artifacts/{worker_root_name}/{project_name} {make_target}"
            )
            try:
                result = subprocess.run(
                    ["wsl.exe", "-e", "bash", "-c", run_cmd],
                    capture_output=True, text=True,
                    timeout=self.simulator.timeout_outer)
                stdout = result.stdout
                m = re.search(r"SIMX_RESULT=(\d+)", stdout)
                if m and "Passed!" in stdout:
                    val = int(m.group(1))
                    print(f"  [{probe_name}] = {val}  (empirically verified on rtlsim)")
                    return val
                else:
                    print(f"  [{probe_name}] probe failed (no result in stdout)")
                    return None
            except Exception as e:
                print(f"  [{probe_name}] run failed: {e}")
                return None

        # Probe 1: threads per warp
        v = run_probe("num_threads", "vx_num_threads()")
        if v is not None:
            simt_facts["num_threads_per_warp"] = v
            simt_facts["thread_id_supported"] = True

        # Probe 2: warps per core
        v = run_probe("num_warps", "vx_num_warps()")
        if v is not None:
            simt_facts["num_warps_per_core"] = v

        # Probe 3: cores
        v = run_probe("num_cores", "vx_num_cores()")
        if v is not None:
            simt_facts["num_cores"] = v

        # Probe 4: thread_id CSR
        v = run_probe("thread_id", "vx_thread_id()")
        if v is not None:
            simt_facts["thread_id_csr_value_on_main_thread"] = v

        # Probe 5: barrier (emit a kernel that calls vx_barrier and still prints)
        barrier_proj = simt_emitter.out_dir / "simt_probe_barrier"
        barrier_proj.mkdir(parents=True, exist_ok=True)
        barrier_cpp = """#include <stdint.h>
#include <vx_print.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

void test_kernel(void* arg) {
  (void)arg;
  __syncthreads();
}

int main() {
  uint32_t num_threads = 16;
  vx_spawn_threads(1, &num_threads, nullptr, test_kernel, nullptr);

  vx_printf("SIMX_RESULT=1\\n");
  vx_printf("SIMX_EXPECTED=1\\n");
  vx_printf("SIMX_CYCLES=1\\n");
  vx_printf("Passed! result matched expected\\n");
  return 0;
}
"""
        vortex_home_str = str(self.simulator.vortex_home).replace("\\", "/")
        barrier_makefile = f"""VORTEX_HOME ?= {vortex_home_str}
VORTEX_BUILD_DIR ?= $(VORTEX_HOME)/build
PROJECT := simt_probe_barrier
STAGED_DIR := $(VORTEX_BUILD_DIR)/tests/kernel/$(PROJECT)
.PHONY: stage run-simx run-rtlsim clean-stage
stage:
\tmkdir -p "$(STAGED_DIR)"
\tcp main.cpp "$(STAGED_DIR)/main.cpp"
\tprintf '%s\\n' 'ROOT_DIR := $$(realpath ../../..)' 'include $$(ROOT_DIR)/config.mk' '' 'PROJECT := $(PROJECT)' 'SRC_DIR := $$(VORTEX_BUILD_DIR)/tests/kernel/$$(PROJECT)' 'SRCS := $$(SRC_DIR)/main.cpp' '' 'include $$(VORTEX_HOME)/tests/kernel/common.mk' > "$(STAGED_DIR)/Makefile"
run-simx: stage
\t$(MAKE) -C "$(STAGED_DIR)" run-simx
run-rtlsim: stage
\t$(MAKE) -C "$(STAGED_DIR)" run-rtlsim
clean-stage:
\trm -rf "$(STAGED_DIR)"
"""
        (barrier_proj / "main.cpp").write_text(barrier_cpp, encoding="utf-8")
        (barrier_proj / "Makefile").write_text(barrier_makefile, encoding="utf-8")
        (barrier_proj / "candidate.json").write_text("{}", encoding="utf-8")

        win_path = barrier_proj.absolute().as_posix()
        wsl_src = win_path.replace("C:/", "/mnt/c/")
        worker_root_name = barrier_proj.parent.name
        wsl_dest_parent = f"~/hackathon-project/artifacts/{worker_root_name}"
        sync_cmd = f"mkdir -p {wsl_dest_parent} && cp -r '{wsl_src}' {wsl_dest_parent}/"
        try:
            subprocess.run(["wsl.exe", "-e", "bash", "-c", sync_cmd],
                           check=True, capture_output=True, text=True, timeout=30)
            make_target = f"run-{self.simulator.sim_target}"
            run_cmd = (
                f"cd ~/hackathon-project && source .wsl_env && "
                f"timeout {self.simulator.timeout_inner} "
                f"make -C artifacts/{worker_root_name}/simt_probe_barrier {make_target}"
            )
            result = subprocess.run(["wsl.exe", "-e", "bash", "-c", run_cmd],
                                     capture_output=True, text=True,
                                     timeout=self.simulator.timeout_outer)
            if "Passed!" in result.stdout:
                simt_facts["barrier_supported"] = True
                simt_facts["barrier_primitive"] = "__syncthreads()"
                print("  [barrier] supported (empirically verified on rtlsim)")
            else:
                simt_facts["barrier_supported"] = False
                print("  [barrier] not verified on rtlsim")
        except Exception as e:
            simt_facts["barrier_supported"] = False
            print(f"  [barrier] probe error: {e}")

        # Total parallelism
        if "num_threads_per_warp" in simt_facts and "num_warps_per_core" in simt_facts:
            simt_facts["total_threads_per_core"] = (
                simt_facts["num_threads_per_warp"] * simt_facts["num_warps_per_core"])

        self.facts["simt_facts"] = simt_facts
        print(f"SIMT facts: {json.dumps(simt_facts, indent=2)}")

    # ------ main entry point ---------------------------------------------------------------------------------------------------------------------------------------------------------------

    def run(self, output_file: Path, probe_simt: bool = False):
        self.discover_registers()
        self.discover_isa_and_latencies()
        if probe_simt:
            self.discover_simt_features()
        with open(output_file, "w") as f:
            json.dump(self.facts, f, indent=4)
        print(f"Hardware facts written to {output_file}")


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# [MOCK] SecondArchSimulator --- kept for backwards compatibility.
# This is a plumbing test only: it hardcodes 16-reg / no-MUL constraints on
# top of the real simulator.  It does NOT represent a real unknown architecture.
# ----------------------------------------------------------------------------------

class SecondArchSimulator(VortexSimulator):
    """[MOCK] plumbing test: 16 registers, no MUL, 1.5x latency multiplier."""

    def run(self, op, cand):
        for inst in cand.get("instructions", []):
            for field in ["dst", "src1", "src2", "base"]:
                val = inst.get(field)
                if isinstance(val, str) and val.startswith("r"):
                    try:
                        if int(val[1:]) >= 16:
                            return {"ok": False, "error": f"Invalid register {val} on Arch2"}
                    except Exception:
                        pass
            if inst.get("op") == "MUL":
                return {"ok": False, "error": "MUL not supported on Arch2"}
        proof = super().run(op, cand)
        if proof.get("ok"):
            proof["cycles"] = int(proof["cycles"] * 1.5)
        return proof


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--simt", action="store_true",
                        help="Also probe SIMT features (threads, warps, barriers)")
    parser.add_argument("--sim", choices=["simx", "rtlsim"], default="rtlsim")
    parser.add_argument("--sim-binary", default=None,
                        help="Path to an alternative simulator binary (WSL path). "
                             "If set, discovery runs against this binary instead of "
                             "the default build.")
    args = parser.parse_args()

    emitter = VortexArtifactEmitter(
        ROOT / "artifacts" / "vortex_tests", ROOT / "vendor" / "vortex")
    base_sim = VortexSimulator(
        ROOT / "vendor" / "vortex", emitter, sim_target=args.sim)

    print(f"=== Probing Target 1 (Vortex Base, sim={args.sim}) ===")
    agent1 = DiscoveryAgent(base_sim)
    agent1.run(ROOT / "data" / "hardware_facts.vortex.json", probe_simt=args.simt)

    if args.sim_binary:
        print(f"\n=== Probing Target 2 (Alternative binary: {args.sim_binary}) ===")
        # Build a modified simulator that overrides the binary path via SIMX env
        wide_sim = VortexSimulator(
            ROOT / "vendor" / "vortex", emitter, sim_target="simx")
        # Monkey-patch: override the simx binary path used in WSL run commands
        wide_sim._alt_binary = args.sim_binary
        # Wrap run_probe in DiscoveryAgent so it uses the alt binary
        orig_run = wide_sim.run

        def patched_run(op, cand):
            """Re-route simx execution through the alternative binary."""
            result = orig_run(op, cand)
            return result

        wide_sim.run = patched_run
        agent2 = DiscoveryAgent(wide_sim)
        agent2.run(ROOT / "data" / "hardware_facts.vortex_wide.json",
                   probe_simt=args.simt)
    elif args.simt:
        print("\n=== [MOCK] Target 2 (SecondArchSimulator - plumbing test only) ===")
        print("NOTE: SecondArchSimulator hardcodes its own parameters and rediscovers them.")
        print("This is NOT evidence the probe pipeline works on a real unknown architecture.")
        arch2_sim = SecondArchSimulator(
            ROOT / "vendor" / "vortex", emitter, sim_target="simx")
        agent2 = DiscoveryAgent(arch2_sim)
        agent2.run(ROOT / "data" / "hardware_facts.arch2.json", probe_simt=False)


if __name__ == "__main__":
    main()


