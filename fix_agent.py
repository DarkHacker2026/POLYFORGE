"""Fix corrupted discovery_agent.py"""
lines = open('discovery_agent.py', encoding='latin-1').readlines()

fixed_lines = lines[:329]  # everything up to line 329 (the good part)

# Insert the missing SecondArchSimulator class + correct main()
fixed_lines.append('# ----------------------------------------------------------------------------------\n')
fixed_lines.append('\n')
fixed_lines.append('class SecondArchSimulator(VortexSimulator):\n')
fixed_lines.append('    """[MOCK] plumbing test: 16 registers, no MUL, 1.5x latency multiplier."""\n')
fixed_lines.append('\n')
fixed_lines.append('    def run(self, op, cand):\n')
fixed_lines.append('        for inst in cand.get("instructions", []):\n')
fixed_lines.append('            for field in ["dst", "src1", "src2", "base"]:\n')
fixed_lines.append('                val = inst.get(field)\n')
fixed_lines.append('                if isinstance(val, str) and val.startswith("r"):\n')
fixed_lines.append('                    try:\n')
fixed_lines.append('                        if int(val[1:]) >= 16:\n')
fixed_lines.append('                            return {"ok": False, "error": f"Invalid register {val} on Arch2"}\n')
fixed_lines.append('                    except Exception:\n')
fixed_lines.append('                        pass\n')
fixed_lines.append('            if inst.get("op") == "MUL":\n')
fixed_lines.append('                return {"ok": False, "error": "MUL not supported on Arch2"}\n')
fixed_lines.append('        proof = super().run(op, cand)\n')
fixed_lines.append('        if proof.get("ok"):\n')
fixed_lines.append('            proof["cycles"] = int(proof["cycles"] * 1.5)\n')
fixed_lines.append('        return proof\n')
fixed_lines.append('\n')
fixed_lines.append('\n')

# The main() + if __name__ block is at lines 330-383 (idx 330-382 before junk)
# Find the end of if __name__ block and stop before duplicate
for line in lines[330:]:
    fixed_lines.append(line)
    if line.strip() == 'main()' and lines.index(line) > 380:
        break  # stop after the real main() call

fixed_lines.append('\n')

content = ''.join(fixed_lines)
with open('discovery_agent.py', 'w', encoding='ascii', errors='replace') as f:
    f.write(content)
print('Fixed, total lines:', content.count('\n'))

import ast
try:
    ast.parse(content)
    print('Syntax OK')
except SyntaxError as e:
    print('SyntaxError:', e)
