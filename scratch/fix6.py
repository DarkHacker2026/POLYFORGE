import sys

with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\cuda_surface.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace any literal newlines inside strings with \\n
text = text.replace('vx_printf("  out[%d]=%d (expected %d)\n", k, (int)args.out[k], k);', 'vx_printf("  out[%d]=%d (expected %d)\\\\n", k, (int)args.out[k], k);')
text = text.replace('vx_printf("SIMX_RESULT=%d\n",   result);', 'vx_printf("SIMX_RESULT=%d\\\\n",   result);')
text = text.replace('vx_printf("SIMX_EXPECTED=%d\n", expected);', 'vx_printf("SIMX_EXPECTED=%d\\\\n", expected);')
text = text.replace('vx_printf("SIMX_CYCLES=%d\n",   par_cyc);', 'vx_printf("SIMX_CYCLES=%d\\\\n",   par_cyc);')
text = text.replace('vx_printf("Passed! result matched expected\n");', 'vx_printf("Passed! result matched expected\\\\n");')
text = text.replace('vx_printf("Failed! result mismatched\n");', 'vx_printf("Failed! result mismatched\\\\n");')
text = text.replace('vx_printf(">> Barrier-in-kernel test: N={N}\n");', 'vx_printf(">> Barrier-in-kernel test: N={N}\\\\n");')
text = text.replace('vx_printf("PAR_CYCLES=%d\n", par_cyc);', 'vx_printf("PAR_CYCLES=%d\\\\n", par_cyc);')

with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\cuda_surface.py", "w", encoding="utf-8") as f:
    f.write(text)
