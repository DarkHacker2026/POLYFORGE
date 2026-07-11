with open("cuda_parser.py", "r") as f:
    content = f.read()

old_str = r"""        # Re-extract RHS specifically for C++ generator
        match = re.search(r'\b\w+\[.*?\]\s*=\s*(.+?);', ck.raw_body, re.DOTALL)
        if match:
            raw_rhs = match.group(1).strip()
        else:
            raw_rhs = "0" # Fallback if unparseable"""

new_str = r"""        # Re-extract RHS specifically for C++ generator
        import re
        match = re.search(r'\b' + dst_param.name + r'\[.*?\]\s*=\s*(.+?);', ck.raw_body, re.DOTALL)
        if not match:
            match = re.search(dst_param.name + r'\[.*?\]\s*=\s*(.+?);', ck.raw_body, re.DOTALL)
        if match:
            raw_rhs = match.group(1).strip()
        else:
            raw_rhs = "0" # Fallback if unparseable"""

content = content.replace(old_str, new_str)
with open("cuda_parser.py", "w") as f:
    f.write(content)
