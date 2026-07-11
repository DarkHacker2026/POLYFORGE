import sys
with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\test_llm_comprehension.py", "r", encoding="utf-8") as f:
    text = f.read()

replacement = """        if ir.get("operations"):
            print("\\n*** INTENTIONAL SABOTAGE: Corrupted op_type to MUL ***\\n")
            ir["operations"][0]["op_type"] = "MUL"
        op_detected = ir["operations"][0]["op_type"] if ir["operations"] else None"""

text = text.replace('op_detected = ir["operations"][0]["op_type"] if ir["operations"] else None', replacement)

with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\test_llm_comprehension.py", "w", encoding="utf-8") as f:
    f.write(text)
