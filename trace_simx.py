import subprocess
import os

wsl_root = "/home/dark_hacker/hackathon-project"
try:
    out = subprocess.check_output(
        f"wsl bash -c \"cd {wsl_root} && source .wsl_env && "
        f"{wsl_root}/vendor/vortex/build/sim/simx/simx -d 3 "
        f"{wsl_root}/artifacts/llm_comprehension_test/llm_comprehension_test.vxbin 2>&1\"",
        shell=True,
        text=True
    )
except subprocess.CalledProcessError as e:
    out = e.output

for line in out.splitlines():
    if "wrote" in line or "Failed" in line or "SIMX" in line:
        print(line)
    if "Mem Store" in line:
        print(line)
