import sys

with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\cuda_surface.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("{{{", "{{").replace("}}}", "}}")
text = text.replace("â”€", "─")
text = text.replace("volatile int32_t __out[BAR_N];", "volatile int32_t __out[BAR_N];\n")

with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\cuda_surface.py", "w", encoding="utf-8") as f:
    f.write(text)
