import sys
with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\test_llm_comprehension.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(len(lines)):
    if 'op_detected = ir["operations"][0]["op_type"] if ir["operations"] else None' in lines[i]:
        lines[i] = '        if ir.get("operations"):\n            print("\\n*** INTENTIONAL SABOTAGE: Corrupted op_type to MUL ***\\n")\n            ir["operations"][0]["op_type"] = "MUL"\n        op_detected = ir["operations"][0]["op_type"] if ir["operations"] else None\n'
        break

with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\test_llm_comprehension.py", "w", encoding="utf-8") as f:
    f.writelines(lines)
