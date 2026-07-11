import json
from pathlib import Path
import subprocess
from cuda_surface import generate_barrier_test

def main():
    root = Path(__file__).resolve().parent
    facts_file = root / "data" / "hardware_facts.vortex.json"
    with open(facts_file, "r") as f:
        facts = json.load(f)
    simt_facts = facts.get("simt_facts", {})

    print(f"Scratchpad supported: {simt_facts.get('scratchpad_supported', False)}")

    cpp_code = generate_barrier_test(num_threads=16, simt_facts=simt_facts)
    
    staged_dir = root / "vendor" / "vortex" / "build" / "tests" / "kernel" / "test_scratchpad"
    staged_dir.mkdir(parents=True, exist_ok=True)
    
    with open(staged_dir / "main.cpp", "w", newline="\n", encoding="utf-8") as f:
        f.write(cpp_code)
    
    makefile = f"""ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk
PROJECT := test_scratchpad
SRC_DIR := $(VORTEX_BUILD_DIR)/tests/kernel/$(PROJECT)
SRCS    := $(SRC_DIR)/main.cpp
include $(VORTEX_HOME)/tests/kernel/common.mk
"""
    with open(staged_dir / "Makefile", "w", newline="\n", encoding="utf-8") as f:
        f.write(makefile)
    
    import subprocess
    subprocess.run(["wsl", "bash", "-c", "cp -r '/mnt/c/Users/Dark Hacker/Desktop/hackathon project/vendor/vortex/build/tests/kernel/test_scratchpad' ~/hackathon-project/vendor/vortex/build/tests/kernel/"])
    print(f"Staged test to {staged_dir}")

    cmd = "cd ~/hackathon-project && source .wsl_env && timeout 300 make -C vendor/vortex/build/tests/kernel/test_scratchpad run-rtlsim"
    print(f"Running: {cmd}")
    result = subprocess.run(["wsl.exe", "bash", "-c", cmd], capture_output=True, text=True)
    
    print("\n--- STDOUT ---")
    print(result.stdout)
    if result.stderr:
        print("\n--- STDERR ---")
        print(result.stderr)
        
if __name__ == "__main__":
    main()
