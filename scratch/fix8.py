import sys
import re

with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\cuda_surface.py", "r", encoding="utf-8") as f:
    text = f.read()

# I will find all vx_printf calls and remove their newlines completely.
text = text.replace('vx_printf(">> Barrier-in-kernel test: N={N}\\n");', 'vx_printf(">> Barrier-in-kernel test: N={N}");')
text = text.replace('vx_printf(">> Barrier-in-kernel test: N={N}\n");', 'vx_printf(">> Barrier-in-kernel test: N={N}");')
text = text.replace('vx_printf("  out[%d]=%d (expected %d)\\n", k, (int)args.out[k], k);', 'vx_printf("  out[%d]=%d (expected %d)", k, (int)args.out[k], k);')
text = text.replace('vx_printf("  out[%d]=%d (expected %d)\n", k, (int)args.out[k], k);', 'vx_printf("  out[%d]=%d (expected %d)", k, (int)args.out[k], k);')
text = text.replace('vx_printf("SIMX_RESULT=%d\\n",   result);', 'vx_printf("SIMX_RESULT=%d",   result);')
text = text.replace('vx_printf("SIMX_RESULT=%d\n",   result);', 'vx_printf("SIMX_RESULT=%d",   result);')
text = text.replace('vx_printf("SIMX_EXPECTED=%d\\n", expected);', 'vx_printf("SIMX_EXPECTED=%d", expected);')
text = text.replace('vx_printf("SIMX_EXPECTED=%d\n", expected);', 'vx_printf("SIMX_EXPECTED=%d", expected);')
text = text.replace('vx_printf("SIMX_CYCLES=%d\\n",   par_cyc);', 'vx_printf("SIMX_CYCLES=%d",   par_cyc);')
text = text.replace('vx_printf("SIMX_CYCLES=%d\n",   par_cyc);', 'vx_printf("SIMX_CYCLES=%d",   par_cyc);')
text = text.replace('vx_printf("Passed! result matched expected\\n");', 'vx_printf("Passed! result matched expected");')
text = text.replace('vx_printf("Passed! result matched expected\n");', 'vx_printf("Passed! result matched expected");')
text = text.replace('vx_printf("Failed! result mismatched\\n");', 'vx_printf("Failed! result mismatched");')
text = text.replace('vx_printf("Failed! result mismatched\n");', 'vx_printf("Failed! result mismatched");')
text = text.replace('vx_printf("PAR_CYCLES=%d\\n", par_cyc);', 'vx_printf("PAR_CYCLES=%d", par_cyc);')
text = text.replace('vx_printf("PAR_CYCLES=%d\n", par_cyc);', 'vx_printf("PAR_CYCLES=%d", par_cyc);')

# Also for saxpy parallel
text = text.replace('vx_printf("SCALAR_CYCLES=%d\\n", scalar_cyc);', 'vx_printf("SCALAR_CYCLES=%d", scalar_cyc);')
text = text.replace('vx_printf("SCALAR_CYCLES=%d\n", scalar_cyc);', 'vx_printf("SCALAR_CYCLES=%d", scalar_cyc);')
text = text.replace('vx_printf("SPEEDUP_NUM=%d SPEEDUP_DEN=%d\\n", scalar_cyc, par_cyc);', 'vx_printf("SPEEDUP_NUM=%d SPEEDUP_DEN=%d", scalar_cyc, par_cyc);')
text = text.replace('vx_printf("SPEEDUP_NUM=%d SPEEDUP_DEN=%d\n", scalar_cyc, par_cyc);', 'vx_printf("SPEEDUP_NUM=%d SPEEDUP_DEN=%d", scalar_cyc, par_cyc);')

# Use regex for anything left over
text = re.sub(r'vx_printf\(([^)]*?)\n([^)]*?)\);', r'vx_printf(\1 \2);', text)

with open(r'C:\Users\Dark Hacker\Desktop\hackathon project\cuda_surface.py', 'w', encoding='utf-8') as f:
    f.write(text)
