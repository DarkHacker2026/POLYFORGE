import sys

with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\cuda_surface.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("for (int k = 0; k < SAXPY_N; k++) {\n    __sy[k] = a * __sx[k] + __sy[k];\n  }", "for (int k = 0; k < SAXPY_N; k++) {{\n    __sy[k] = a * __sx[k] + __sy[k];\n  }}")
text = text.replace("for (int k = 0; k < SAXPY_N; k++) {\n    int32_t exp = a * (k + 1) + k * 2;\n    if (args.y[k] != exp) { ok = 0; break; }\n  }", "for (int k = 0; k < SAXPY_N; k++) {{\n    int32_t exp = a * (k + 1) + k * 2;\n    if (args.y[k] != exp) {{ ok = 0; break; }}\n  }}")
text = text.replace("if (ok && result == expected) {\n    vx_printf(\"Passed! result matched expected\\n\");\n    if (par_cyc > 0) {\n      vx_printf(\"SPEEDUP_NUM=%d SPEEDUP_DEN=%d\\n\", scalar_cyc, par_cyc);\n    }\n    return 0;\n  }", "if (ok && result == expected) {{\n    vx_printf(\"Passed! result matched expected\\n\");\n    if (par_cyc > 0) {{\n      vx_printf(\"SPEEDUP_NUM=%d SPEEDUP_DEN=%d\\n\", scalar_cyc, par_cyc);\n    }}\n    return 0;\n  }}")

text = text.replace("static void barrier_kernel(barrier_args_t *args) {", "static void barrier_kernel(barrier_args_t *args) {{")
text = text.replace("int main() {", "int main() {{")
text = text.replace("for (int k = 0; k < BAR_N; k++) {", "for (int k = 0; k < BAR_N; k++) {{")
text = text.replace("if (args.out[k] != (int32_t)k) ok = 0;", "if (args.out[k] != (int32_t)k) ok = 0;")
text = text.replace("if (ok && result == expected) {", "if (ok && result == expected) {{")
# Be sure to double any remaining trailing } at the end of the functions that weren't caught
text = text.replace("  }\n  vx_printf(\"Failed! result mismatched\\n\");\n  return 1;\n}\n", "  }}\n  vx_printf(\"Failed! result mismatched\\n\");\n  return 1;\n}}\n")
text = text.replace("  // Phase 2: each thread reads its OWN cell back (guaranteed visible post-barrier)\n  args->out[i] = args->shared_arr[i];\n}\n", "  // Phase 2: each thread reads its OWN cell back (guaranteed visible post-barrier)\n  args->out[i] = args->shared_arr[i];\n}}\n")


with open(r"C:\Users\Dark Hacker\Desktop\hackathon project\cuda_surface.py", "w", encoding="utf-8") as f:
    f.write(text)

