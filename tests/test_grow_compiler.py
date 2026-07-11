#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grow_compiler import is_structural_ir_op, parse_ir


def run_cmd(args: list[str]) -> str:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "grow_compiler.py"), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return proc.stdout


def test_full_mock_demo_learns_and_reuses() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        rules = Path(tmp) / "rules.json"
        artifacts = Path(tmp) / "vortex_tests"
        args = [
            "--provider", "mock",
            "--program", str(ROOT / "examples" / "demo_program_full.ir"),
            "--rules", str(rules),
            "--reset-rules",
        ]
        first = run_cmd(args)
        assert "ADD_INT_v1" in first
        assert "MUL_INT_v1" in first
        assert "LOAD_INT_v1" in first

        second = run_cmd([
            "--provider", "mock",
            "--program", str(ROOT / "examples" / "demo_program_full.ir"),
            "--rules", str(rules),
        ])
        assert "reused ADD_INT_v1" in second
        assert "reused MUL_INT_v1" in second
        assert "reused LOAD_INT_v1" in second

        data = json.loads(rules.read_text(encoding="utf-8"))
        assert sorted(data["rules"].keys()) == ["add", "load", "mul"]

        emit = run_cmd([
            "--provider", "mock",
            "--program", str(ROOT / "examples" / "demo_program_full.ir"),
            "--rules", str(rules),
            "--emit-vortex-tests", str(artifacts),
        ])
        assert "vortex test artifact" in emit
        for project in (
            "agent_add_add_int_v1_instantiated",
            "agent_mul_mul_int_v1_instantiated",
            "agent_load_load_int_v1_instantiated",
        ):
            assert (artifacts / project / "Makefile").exists()
            assert (artifacts / project / "main.cpp").exists()


def test_boundary_ir_parses_and_skips_structural_ops() -> None:
    ops = parse_ir(ROOT / "examples" / "demo_boundaries.ir")
    assert [op.op for op in ops] == ["label", "add", "br_if_ne", "mul", "label", "load"]
    assert [is_structural_ir_op(op) for op in ops] == [True, False, True, False, True, False]


def test_mixed_arch_lift_syntax_is_tolerated() -> None:
    ops = parse_ir(ROOT / "examples" / "demo_mixed_arch_lift.ir")
    assert [op.op for op in ops] == [
        "label",
        "add",
        "mul",
        "add",
        "load",
        "bne",
        "unknown_vendor_op",
        "label",
        "ret",
    ]
    assert ops[1].dst == "r3"
    assert ops[1].args == ("r1", "r2")
    assert ops[3].dst == "r7"
    assert ops[3].args == ("r3", "r4")
    assert ops[4].dst == "r8"
    assert ops[4].args == ("r6",)
    assert is_structural_ir_op(ops[5])


if __name__ == "__main__":
    test_full_mock_demo_learns_and_reuses()
    print("tests passed")
