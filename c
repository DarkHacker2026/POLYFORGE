f = 'cuda_parser.py'
content = open(f, encoding='utf-8').read()

old = """    scalar_decls = []
    for sp in ck.scalar_params:
        if sp.name in init_values:
            val = init_values[sp.name]
            if isinstance(val, float):
                val_str = f"{val}f"
            else:
                val_str = str(val)
        elif sp.name.upper() in ('N', 'SIZE', 'COUNT', 'LENGTH', 'LEN', 'NUM'):
            val_str = str(N)
        else:
            if 'float' in sp.ctype or 'double' in sp.ctype:
                val_str = "0.0f"
            else:
                val_str = "0"
        scalar_decls.append(f'{sp.ctype} {sp.name} = {val_str};')"""

new = """    scalar_decls = []
    for sp in ck.scalar_params:
        # Skip N/size/count params -- already declared as uint32_t N above
        if sp.name.upper() in ('N', 'SIZE', 'COUNT', 'LENGTH', 'LEN', 'NUM'):
            continue
        if sp.name in init_values:
            val = init_values[sp.name]
            if isinstance(val, float):
                val_str = f"{val}f"
            else:
                val_str = str(val)
        else:
            if 'float' in sp.ctype or 'double' in sp.ctype:
                val_str = "0.0f"
            else:
                val_str = "0"
        scalar_decls.append(f'{sp.ctype} {sp.name} = {val_str};')"""

if old in content:
    content = content.replace(old, new, 1)
    open(f, 'w', encoding='utf-8').write(content)
    print("Fixed duplicate N declaration")
else:
    print("ERROR: Could not find the target text")
    # Try to find a partial match
    if "scalar_decls = []" in content:
        idx = content.index("scalar_decls = []")
        print(f"Found 'scalar_decls = []' at index {idx}")
        print("Context:")
        print(content[idx:idx+500])