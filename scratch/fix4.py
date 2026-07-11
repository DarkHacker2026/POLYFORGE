import sys

with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\cuda_surface.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace('vx_printf(">> Barrier-in-kernel test: N={N}\\n");', 'vx_printf(">> Barrier-in-kernel test: N={N}\\\\n");')

with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\cuda_surface.py", "w", encoding="utf-8") as f:
    f.write(text)
