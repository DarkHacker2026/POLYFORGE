#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CHECKS = [
    ("make", ["make", "--version"], "required to build Vortex runtime/tests"),
    ("clang", ["clang", "--version"], "required unless using Vortex LLVM path directly"),
    ("riscv32-unknown-elf-gcc", ["riscv32-unknown-elf-gcc", "--version"], "baseline RISC-V GNU toolchain"),
    ("llvm-objcopy", ["llvm-objcopy", "--version"], "Vortex vxbin conversion uses LLVM objcopy"),
    ("verilator", ["verilator", "--version"], "required for RTL simulation"),
]


def run_check(name: str, command: list[str], reason: str) -> bool:
    exe = shutil.which(command[0])
    if not exe:
        print(f"[missing] {name}: {reason}")
        return False
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )
    except OSError as exc:
        print(f"[error]   {name}: {exc}")
        return False
    first_line = proc.stdout.splitlines()[0] if proc.stdout.splitlines() else "found"
    print(f"[ok]      {name}: {first_line}")
    return proc.returncode == 0


def main() -> int:
    print(f"[root] {ROOT}")
    ok = True
    for name, command, reason in CHECKS:
        ok = run_check(name, command, reason) and ok
    vortex = ROOT / "vendor" / "vortex"
    if vortex.exists():
        print(f"[ok]      vortex repo: {vortex}")
    else:
        print(f"[missing] vortex repo: expected {vortex}")
        ok = False
    if ok:
        print("[ready] Vortex build/sim prerequisites look available.")
        return 0
    print("[not-ready] Install Vortex prerequisites or run inside a prepared Linux/WSL environment.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
