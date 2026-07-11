#!/usr/bin/env python3
"""
demo_offline.py — POLYFORGE offline demo.

Runs WITHOUT WSL or a Fireworks API key.
- Loads oracle_examples/saxpy.json using oracle_standalone.py's OracleInput
- Runs StandaloneOracle().run() on it
- Runs the 8 hardened oracle test cases inline (import and call them)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from oracle_standalone import OracleInput, StandaloneOracle
from test_oracle_hardened import main as run_hardened_tests


def run_saxpy_demo():
    inp = OracleInput.from_json(ROOT / "oracle_examples" / "saxpy.json")
    oracle = StandaloneOracle()
    result = oracle.run(inp)
    return result.passed


def main():
    print("Running POLYFORGE offline demo...\n")

    saxpy_ok = run_saxpy_demo()
    print(f"SAXPY kernel: {'PASS' if saxpy_ok else 'FAIL'}\n")

    # Run hardened tests and capture result
    import io
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    hardened_exit = run_hardened_tests()
    sys.stdout = old_stdout
    hardened_output = buffer.getvalue()
    hardened_ok = (hardened_exit == 0)

    # Parse pass count from output
    passed_count = 0
    for line in hardened_output.splitlines():
        if "tests passed" in line:
            try:
                passed_count = int(line.split("/")[0].strip().split()[-1])
            except Exception:
                pass

    print(hardened_output)

    box = f"""\
╔══════════════════════════════════════════╗
║     POLYFORGE — Offline Demo             ║
╠══════════════════════════════════════════╣
║  Oracle Race Detection:    {passed_count}/8 PASS {'✓' if hardened_ok else '✗'}   ║
║  SAXPY Kernel Verified:       {'PASS' if saxpy_ok else 'FAIL'} {'✓' if saxpy_ok else '✗'}   ║
║  No WSL or API key required             ║
╚══════════════════════════════════════════╝"""
    print(box)
    return 0 if (saxpy_ok and hardened_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
