import sys
with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\test_scratchpad.py", "r", encoding="utf-8") as f:
    text = f.read()

replacement = """import subprocess
    subprocess.run(["wsl", "bash", "-c", "cp -r '/mnt/c/Users/Dark Hacker/Desktop/hackathon project/vendor/vortex/build/tests/kernel/test_scratchpad' ~/hackathon-project/vendor/vortex/build/tests/kernel/"])
    print(f"Staged test to {staged_dir}")"""

text = text.replace('print(f"Staged test to {staged_dir}")', replacement)

with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\test_scratchpad.py", "w", encoding="utf-8") as f:
    f.write(text)
