import sys
import re

with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\cuda_surface.py", "r", encoding="utf-8") as f:
    text = f.read()

# I will just replace ALL literal newlines that are inside a vx_printf call!
# Regex to match vx_printf("...\n"); where \n is a literal newline
text = re.sub(r'vx_printf\(([^)]*?)\n([^)]*?)\);', r'vx_printf(\1\\n\2);', text)
text = re.sub(r'vx_printf\(([^)]*?)\n([^)]*?)\);', r'vx_printf(\1\\n\2);', text) # run twice in case of multiple newlines

# And I will ensure N=16 has \n at the end
text = text.replace('vx_printf(">> Barrier-in-kernel test: N={N} ");', 'vx_printf(">> Barrier-in-kernel test: N={N}\\n");')

with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\cuda_surface.py", "w", encoding="utf-8") as f:
    f.write(text)
