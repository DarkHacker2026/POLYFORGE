#!/usr/bin/env python3
"""Automated compiler-growth loop for the hackathon prototype.

The first milestone uses a local semantic simulator so the pipeline is runnable
before the heavier Vortex SimX/RTL harness is connected.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from reference_isa import ReferenceISA, get_initial_state, compute_expected


ROOT = Path(__file__).resolve().parent
DEFAULT_FACTS = ROOT / "data" / "hardware_facts.vortex.json"
DEFAULT_RULES = ROOT / "data" / "rules.json"


class CompilerError(RuntimeError):
    pass


@dataclass(frozen=True)
class IROperation:
    dst: str
    op: str
    args: tuple[str, ...]
    source: str

    def to_prompt_json(self) -> dict[str, Any]:
        return {
            "dst": self.dst,
            "op": self.op,
            "args": list(self.args),
            "source": self.source,
        }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    # Aggressively try to parse just the JSON block by finding the first '{' and last '}'
    start = stripped.find('{')
    end = stripped.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_str = stripped[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
    # Fallback to the original attempt
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise CompilerError(f"Failed to parse LLM JSON response: {exc}. RAW CONTENT: {text}") from exc


def render_template(path: Path, values: dict[str, str]) -> str:
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def strip_inline_comment(line: str) -> str:
    in_quote: str | None = None
    i = 0
    while i < len(line):
        ch = line[i]
        if in_quote:
            if ch == in_quote and (i == 0 or line[i - 1] != "\\"):
                in_quote = None
            i += 1
            continue
        if ch in {"'", '"'}:
            in_quote = ch
            i += 1
            continue
        if ch == "#":
            return line[:i]
        if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
            return line[:i]
        i += 1
    return line


def split_args(args_text: str) -> tuple[str, ...]:
    args: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in args_text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}" and depth:
            depth -= 1
        if ch == "," and depth == 0:
            arg = "".join(current).strip()
            if arg:
                args.append(arg)
            current = []
            continue
        current.append(ch)
    arg = "".join(current).strip()
    if arg:
        args.append(arg)
    return tuple(args)


def normalize_ir_line(line: str) -> str:
    replacements = {
        "\u00a0": " ",
        "\u2217": "*",
        "\u00d7": "*",
        "\u2212": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2190": "=",
        "\u27f5": "=",
    }
    for old, new in replacements.items():
        line = line.replace(old, new)
    return re.sub(r"\s+", " ", line).strip()


def normalize_memory_operand(text: str) -> tuple[str, ...]:
    text = text.strip()
    bracket_match = re.match(r"^\[(?P<base>[A-Za-z_][A-Za-z0-9_]*)(?:\s*\+\s*(?P<offset>[-+]?\d+))?\]$", text)
    if bracket_match:
        args = [bracket_match.group("base")]
        if bracket_match.group("offset"):
            args.append(bracket_match.group("offset"))
        return tuple(args)
    deref_match = re.match(r"^\*\s*(?P<base>[A-Za-z_][A-Za-z0-9_]*)$", text)
    if deref_match:
        return (deref_match.group("base"),)
    return (text,)


def parse_ir_line(line: str) -> IROperation:
    original = line
    line = normalize_ir_line(line)
    label_match = re.match(r"^([A-Za-z_.$][A-Za-z0-9_.$]*):$", line)
    if label_match:
        return IROperation(dst="", op="label", args=(label_match.group(1),), source=original)

    call_match = re.match(r"^(?:(?P<dst>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*)?(?P<op>[A-Za-z_][A-Za-z0-9_.]*)\((?P<args>.*)\)$", line)
    if call_match:
        dst = call_match.group("dst") or ""
        op = call_match.group("op").lower().replace(".", "_")
        return IROperation(dst=dst, op=op, args=split_args(call_match.group("args")), source=original)

    assign_match = re.match(r"^(?P<dst>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<expr>.+)$", line)
    if assign_match:
        dst = assign_match.group("dst")
        expr = assign_match.group("expr").strip()

        mem_match = re.match(r"^(?:\*?\(?\s*)?\[(?P<base>[A-Za-z_][A-Za-z0-9_]*)(?:\s*\+\s*(?P<offset>[-+]?\d+))?\]\s*\)?$", expr)
        if mem_match:
            args = [mem_match.group("base")]
            if mem_match.group("offset"):
                args.append(mem_match.group("offset"))
            return IROperation(dst=dst, op="load", args=tuple(args), source=original)

        infix_match = re.match(r"^(?P<a>[A-Za-z_][A-Za-z0-9_]*|-?\d+)\s*(?P<op>\+|-|\*|<<|>>|&|\||\^)\s*(?P<b>[A-Za-z_][A-Za-z0-9_]*|-?\d+)$", expr)
        if infix_match:
            op_map = {
                "+": "add",
                "-": "sub",
                "*": "mul",
                "<<": "shl",
                ">>": "shr",
                "&": "and",
                "|": "or",
                "^": "xor",
            }
            return IROperation(
                dst=dst,
                op=op_map[infix_match.group("op")],
                args=(infix_match.group("a"), infix_match.group("b")),
                source=original,
            )

        return IROperation(dst=dst, op="assign", args=(expr,), source=original)

    store_match = re.match(r"^(?:\[(?P<base>[A-Za-z_][A-Za-z0-9_]*)(?:\s*\+\s*(?P<offset>[-+]?\d+))?\]|\*\s*(?P<ptr>[A-Za-z_][A-Za-z0-9_]*))\s*=\s*(?P<src>[A-Za-z_][A-Za-z0-9_]*|-?\d+)$", line)
    if store_match:
        base = store_match.group("base") or store_match.group("ptr") or ""
        args = [base, store_match.group("src")]
        if store_match.group("offset"):
            args.append(store_match.group("offset"))
        return IROperation(dst="", op="store", args=tuple(args), source=original)

    asm_match = re.match(r"^(?P<op>[A-Za-z_.][A-Za-z0-9_.]*)\s+(?P<args>.+)$", line)
    if asm_match:
        op = asm_match.group("op").lower().replace(".", "_")
        args = split_args(asm_match.group("args"))
        if op in {"add", "sub", "mul", "and", "or", "xor", "shl", "shr", "mov", "ldr", "lw"} and args:
            normalized_op = {"ldr": "load", "lw": "load", "mov": "assign"}.get(op, op)
            normalized_args = normalize_memory_operand(args[1]) if normalized_op == "load" and len(args) > 1 else args[1:]
            return IROperation(dst=args[0], op=normalized_op, args=normalized_args, source=original)
        if op in {"str", "sw"}:
            return IROperation(dst="", op="store", args=args, source=original)
        return IROperation(dst="", op=op, args=args, source=original)

    return IROperation(dst="", op="raw", args=(line,), source=original)


def parse_ir(path: Path) -> list[IROperation]:
    ops: list[IROperation] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = strip_inline_comment(raw).strip()
        if not line:
            continue
        ops.append(parse_ir_line(line))
    return ops


def is_structural_ir_op(op: IROperation) -> bool:
    return op.op in {
        "label",
        "br",
        "b",
        "jmp",
        "jump",
        "br_if",
        "br_if_eq",
        "br_if_ne",
        "beq",
        "bne",
        "blt",
        "ble",
        "bgt",
        "bge",
        "cbz",
        "cbnz",
        "ret",
        "return",
        "raw",
    }


def mock_provider_can_handle(op: IROperation) -> bool:
    return op.op in {"add", "mul", "load", "loop", "mul_by_const_8"}


class RuleDatabase:
    def __init__(self, path: Path):
        self.path = path
        self.data = load_json(path)

    @property
    def rules(self) -> dict[str, Any]:
        return self.data.setdefault("rules", {})

    def find(self, op: IROperation) -> dict[str, Any] | None:
        return self.rules.get(op.op)

    def add_or_replace(self, rule: dict[str, Any], proof: dict[str, Any]) -> str:
        assert proof.get("ok"), "Cannot evaluate replacement for a failed candidate"
        op = rule["operation"]
        old = self.rules.get(op)
        rule["actual_cycles"] = proof["cycles"]
        decision = "added"
        if old and proof["cycles"] < old.get("actual_cycles", 10**9):
            decision = "replaced"
        elif old:
            decision = "kept_existing"
        if decision in {"added", "replaced"}:
            self.rules[op] = rule
        self.data.setdefault("history", []).append(
            {
                "ts": int(time.time()),
                "decision": decision,
                "operation": op,
                "rule_id": rule.get("rule_id"),
                "proof": proof,
            }
        )
        save_json(self.path, self.data)
        return decision


class StaticChecker:
    def __init__(self, hardware_facts: dict[str, Any]):
        scope = hardware_facts.get("compiler_scope_v0")
        if scope:
            allowed_ops = scope["allowed_ops"]
            allowed_regs = scope["allowed_integer_registers"]
        else:
            allowed_ops = hardware_facts.get("isa", {}).keys()
            allowed_regs = hardware_facts.get("registers", [])
        self.allowed_ops = {str(op).upper() for op in allowed_ops}
        self.allowed_regs = set(allowed_regs)

    def check(self, candidate: dict[str, Any]) -> dict[str, Any]:
        instructions = candidate.get("instructions")
        if not isinstance(instructions, list) or not instructions:
            return {"ok": False, "error": "candidate has no instructions"}
        for index, inst in enumerate(instructions):
            op = inst.get("op")
            if str(op).upper() not in self.allowed_ops:
                return {"ok": False, "error": f"instruction {index} uses unsupported op {op}"}
            for field in ("dst", "src1", "src2", "base"):
                reg = inst.get(field)
                if reg and isinstance(reg, str) and reg.startswith("r") and reg not in self.allowed_regs:
                    return {"ok": False, "error": f"instruction {index} uses invalid register {reg}"}
            if inst.get("dst") == "r0":
                return {"ok": False, "error": "candidate writes constant register r0"}
        return {"ok": True}


class LocalSemanticSimulator:
    def __init__(self, hardware_facts: dict[str, Any]):
        self.latencies = hardware_facts.get("isa", {})

    def _reg(self, regs: dict[str, int], name: str | None) -> int:
        if not name:
            return 0
        return int(regs.get(name, 0))

    def run(self, op: IROperation, candidate: dict[str, Any]) -> dict[str, Any]:
        initial_regs, initial_mem = get_initial_state()
        trace = []
        cycles = 0

        for inst in candidate.get("instructions", []):
            name = str(inst.get("op", "")).upper()
            latency = int(self.latencies.get(name, {}).get("latency", 1))
            cycles += latency
            if name not in {"ADD", "SUB", "MUL", "ADDI", "SUBI", "SLLI", "LW", "SW", "LABEL", "BNE", "BEQ"}:
                return {"ok": False, "error": f"unsupported local op {name}", "trace": trace}
            trace.append({"cycle": cycles, "inst": inst})

        isa = ReferenceISA()
        try:
            final_regs = isa.execute(candidate.get("instructions", []), initial_regs, initial_mem)
        except Exception as exc:
            return {"ok": False, "error": f"local execution failed: {exc}", "trace": trace}
        expected = compute_expected(op, initial_regs, initial_mem)
        actual = final_regs.get(op.dst, 0) if op.dst else 0
        return {
            "ok": actual == expected,
            "expected": expected,
            "actual": actual,
            "cycles": cycles,
            "trace": trace,
            "simulator": "local_semantic",
            "error": None if actual == expected else "candidate failed local verification",
        }


class VortexSimulator:
    def __init__(self, vortex_home: Path, emitter: "VortexArtifactEmitter", sim_target: str = "simx"):
        self.vortex_home = vortex_home
        self.emitter = emitter
        if sim_target not in ("simx", "rtlsim"):
            raise CompilerError(f"Unknown sim_target: {sim_target}. Choose 'simx' or 'rtlsim'.")
        self.sim_target = sim_target
        # rtlsim is slower; give it a bigger wall-clock budget
        self.timeout_inner = 10 if sim_target == "simx" else 120
        self.timeout_outer = 15 if sim_target == "simx" else 130

    def run(self, op: IROperation, candidate: dict[str, Any]) -> dict[str, Any]:
        try:
            # Emit the C++ artifact locally.
            project_dir = self.emitter.emit(op, candidate, {"ok": False, "cycles": 0})
            project_name = project_dir.name

            # Sync ONLY this specific project directory to WSL thread-safely
            win_project_path = project_dir.absolute().as_posix()
            wsl_src = win_project_path.replace("C:/", "/mnt/c/")
            
            # Extract the worker_root name, e.g., vortex_tests_12
            worker_root_name = project_dir.parent.name
            wsl_dest_parent = f"~/hackathon-project/artifacts/{worker_root_name}"
            wsl_dest = f"{wsl_dest_parent}/{project_name}"
            
            sync_cmd = f"mkdir -p {wsl_dest_parent} && cp -r '{wsl_src}' {wsl_dest_parent}/"
            subprocess.run(
                ["wsl.exe", "-e", "bash", "-c", sync_cmd],
                check=True,
                capture_output=True,
                text=True
            )

            # Run the chosen simulator via WSL.
            make_target = f"run-{self.sim_target}"
            cmd = (
                f"cd ~/hackathon-project && source .wsl_env && "
                f"timeout {self.timeout_inner} make -C artifacts/{worker_root_name}/{project_name} {make_target}"
            )
            result = subprocess.run(
                ["wsl.exe", "-e", "bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=self.timeout_outer
            )

            stdout = result.stdout
            if "Passed!" in stdout or "Failed!" in stdout:
                cycles_match = re.search(r"SIMX_CYCLES=(\d+)", stdout)
                cycles = int(cycles_match.group(1)) if cycles_match else 0
                result_match = re.search(r"SIMX_RESULT=(-?\d+)", stdout)
                actual_val = int(result_match.group(1)) if result_match else 0
                expected_match = re.search(r"SIMX_EXPECTED=(-?\d+)", stdout)
                expected_val = int(expected_match.group(1)) if expected_match else 0
                
                is_ok = ("Passed!" in stdout)
                return {
                    "ok": is_ok,
                    "expected": expected_val,
                    "actual": actual_val,
                    "cycles": cycles,
                    "trace": [{"cycle": cycles, "inst": {"op": "VORTEX_ASM"}, "dst_value": actual_val}],
                    "simulator": f"vortex_{self.sim_target}",
                    "error": "result mismatched" if not is_ok else None
                }
            else:
                return {
                    "ok": False,
                    "error": f"{self.sim_target} execution failed to print results. Output: " + stdout[-500:],
                    "trace": []
                }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": f"{self.sim_target} simulation timed out (likely infinite loop in candidate)",
                "trace": []
            }
        except subprocess.CalledProcessError as e:
            return {
                "ok": False,
                "error": f"Vortex compilation/execution crashed: {e.stderr}",
                "trace": []
            }


class VortexArtifactEmitter:
    def __init__(self, out_dir: Path, vortex_home: Path):
        self.out_dir = out_dir
        self.vortex_home = vortex_home

    def emit(self, op: IROperation, candidate: dict[str, Any], proof: dict[str, Any]) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_]+", "_", candidate["candidate_id"]).strip("_").lower()
        project = f"agent_{op.op}_{safe_id}"
        project_dir = self.out_dir / project
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
        (project_dir / "proof.local.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
        (project_dir / "main.cpp").write_text(self._main_cpp(op, candidate), encoding="utf-8")
        (project_dir / "Makefile").write_text(self._makefile(project), encoding="utf-8")
        return project_dir

    def _makefile(self, project: str) -> str:
        vortex_home = str(self.vortex_home).replace("\\", "/")
        return f"""# Generated by grow_compiler.py.
# This wrapper stages the test inside the Vortex build tree because
# Vortex's tests/kernel/common.mk expects ../../.. to contain config.mk.

VORTEX_HOME ?= {vortex_home}
VORTEX_BUILD_DIR ?= $(VORTEX_HOME)/build
PROJECT := {project}
STAGED_DIR := $(VORTEX_BUILD_DIR)/tests/kernel/$(PROJECT)

.PHONY: stage run-simx run-rtlsim clean-stage

stage:
\tmkdir -p "$(STAGED_DIR)"
\tcp main.cpp "$(STAGED_DIR)/main.cpp"
\tprintf '%s\\n' 'ROOT_DIR := $$(realpath ../../..)' 'include $$(ROOT_DIR)/config.mk' '' 'PROJECT := $(PROJECT)' 'SRC_DIR := $$(VORTEX_BUILD_DIR)/tests/kernel/$$(PROJECT)' 'SRCS := $$(SRC_DIR)/main.cpp' '' 'include $$(VORTEX_HOME)/tests/kernel/common.mk' > "$(STAGED_DIR)/Makefile"

run-simx: stage
\t$(MAKE) -C "$(STAGED_DIR)" run-simx

run-rtlsim: stage
\t$(MAKE) -C "$(STAGED_DIR)" run-rtlsim

clean-stage:
\trm -rf "$(STAGED_DIR)"
"""

    def _main_cpp(self, op: IROperation, candidate: dict[str, Any]) -> str:
        body = self._cpp_body(op, candidate)
        return f"""#include <stdint.h>
#include <vx_print.h>
#include <vx_intrinsics.h>

// Generated from candidate {candidate["candidate_id"]}.
// This uses inline assembly only as a mechanical instruction encoder.

int main() {{
{body}
}}
"""

    def _cpp_body(self, op: IROperation, candidate: dict[str, Any]) -> str:
        instructions = candidate.get("instructions", [])
        if not instructions:
            return self._unsupported_body(op, "candidate has no instructions")
        
        # All candidates use the dynamic multi-instruction emitter
        return self._multi_asm_body(op, candidate)

    def _multi_asm_body(self, op: IROperation, candidate: dict[str, Any]) -> str:
        """General emitter for multi-instruction candidates (loops, etc).
        
        Dynamically collects every register name referenced by the LLM's instruction
        list and declares them all in the GCC inline-asm constraint list.  This removes
        the need for any operation-specific hardcoding — the LLM can introduce whatever
        temp registers it likes and GCC will correctly allocate them.
        """
        instructions = candidate.get("instructions", [])
        
        # --- collect all unique register names used across all instructions ---
        reg_fields = ("dst", "src1", "src2", "base")
        all_regs: dict[str, str] = {}  # name -> initial C value
        
        initial_regs, initial_mem = get_initial_state()
        
        # Set up memory array (4KB)
        decls = "  volatile int32_t memory[1024] = {0};\n"
        # Seed it with the Phase 1 initial_mem values
        for addr, val in initial_mem.items():
            idx = addr // 4
            decls += f"  memory[{idx}] = {val};\n"
            
        for inst in instructions:
            for field in reg_fields:
                reg = inst.get(field)
                if reg and isinstance(reg, str) and reg.startswith("r"):
                    if reg not in all_regs:
                        # Detect if this register is used as a base address in any LW/SW instruction.
                        # If so, map it to (intptr_t)memory + byte_offset so that
                        # lw dst, 0(reg) loads from the correct slot in the C memory array.
                        # This is the general fix for the load-base address model mismatch bug.
                        is_mem_base = any(
                            instr.get("op", "").upper() in ("LW", "SW")
                            and (instr.get("base") == reg or instr.get("src1") == reg)
                            for instr in instructions
                        )
                        if is_mem_base:
                            byte_offset = initial_regs.get(reg, 0)
                            all_regs[reg] = f"(int32_t)((intptr_t)memory + {byte_offset})"
                        else:
                            all_regs[reg] = str(initial_regs.get(reg, 0))

        # --- build the asm instruction strings ---
        asm_lines = []
        for inst in instructions:
            name = inst.get("op", "").upper()
            if name == "LABEL":
                label = inst.get("label", "1")
                if "%=" not in label:
                    label = label.rstrip("bf") + "%="
                asm_lines.append(f"{label}:")
            elif name in ("ADD", "SUB", "MUL"):
                asm_lines.append(
                    f"{name.lower()} %[{inst['dst']}], %[{inst['src1']}], %[{inst['src2']}]"
                )
            elif name == "LW":
                offset = int(inst.get("offset", 0))
                base = inst.get("base") or inst.get("src1")
                asm_lines.append(f"lw %[{inst['dst']}], {offset}(%[{base}])")
            elif name == "SW":
                offset = int(inst.get("offset", 0))
                base = inst.get("base") or inst.get("src1")
                # src2 is the value to store
                asm_lines.append(f"sw %[{inst['src2']}], {offset}(%[{base}])")
            elif name in ("ADDI", "SUBI", "SLLI"):
                real_op = name.lower()
                imm = int(inst.get("imm", -1))
                if name == "SUBI":
                    real_op = "addi"
                    imm = -abs(imm)
                asm_lines.append(
                    f"{real_op} %[{inst['dst']}], %[{inst['src1']}], {imm}"
                )
            elif name in ("BNE", "BEQ"):
                label = inst.get("label", "1f")
                if "%=" not in label:
                    direction = label[-1] if label and label[-1] in "bf" else "f"
                    label = label.rstrip("bf") + "%=" + direction
                asm_lines.append(
                    f"{name.lower()} %[{inst['src1']}], %[{inst['src2']}], {label}"
                )
                asm_lines.append("1%=:")
            else:
                asm_lines.append(f"// Unsupported op: {name}")

        has_labels = any(inst.get("op", "").upper() in ("LABEL", "BNE", "BEQ") for inst in instructions)
        # Use unroll_factor = 1 to avoid loop boundary WAR hazards
        unroll_factor = 1

        asm_body_single = ' "\\n\\t" '.join(f'"{line}"' for line in asm_lines)
        for reg, init in all_regs.items():
            decls += f"  int32_t {reg} = {init};\n"
        
        # --- build GCC constraint lists ---
        # Registers that are written (dst) are output (+r), rest are input (r)
        written = set()
        for inst in instructions:
            if inst.get("dst"):
                written.add(inst["dst"])
        
        outputs = ", ".join(
            f'[{r}] "+r" ({r})' for r in all_regs if r in written
        )
        inputs = ", ".join(
            f'[{r}] "r" ({r})' for r in all_regs if r not in written
        )
        constraint_str = f"      : {outputs}\n"
        if inputs:
            constraint_str += f"      : {inputs}\n"
        else:
            constraint_str += f"      :\n"

        # Expected result oracle
        try:
            expected_val = str(compute_expected(op, initial_regs, initial_mem))
        except Exception:
            expected_val = "0"

        primary_reg = op.dst if op.dst in all_regs else list(all_regs.keys())[0]

        asm_blocks = ""
        reset_stmts = ""
        for reg, init in all_regs.items():
            if reg in written:
                reset_stmts += f"  {reg} = {init};\n"

        for i in range(unroll_factor):
            asm_blocks += f"""  __asm__ volatile (
      {asm_body_single}
{constraint_str}  );\n"""
            if i < unroll_factor - 1:
                asm_blocks += reset_stmts

        return f"""{decls}  int32_t expected = {expected_val};
  volatile int32_t sink = 0;
  uint64_t start = vx_rdcycle();
{asm_blocks}
  uint64_t end = vx_rdcycle();
  sink += {primary_reg};
  vx_printf("SIMX_RESULT=%d\\n", {primary_reg});
  vx_printf("SIMX_EXPECTED=%d\\n", expected);
  vx_printf("SIMX_CYCLES=%d\\n", (int)((end - start) / {unroll_factor}));
  if ({primary_reg} == expected) {{
      vx_printf("Passed! result matched expected\\n");
      return 0;
  }}
  vx_printf("Failed! result mismatched\\n");
  return 1;
"""

    def _unsupported_body(self, op: IROperation, reason: str) -> str:
        return f"""  vx_printf("Unsupported generated artifact for {op.source}: {reason}\\n");
  return 1;
"""


class LLMProvider:
    def generate_candidate(self, prompt: str, op: IROperation) -> dict[str, Any]:
        raise NotImplementedError

    def extract_rule(self, prompt: str, op: IROperation, candidate: dict[str, Any], proof: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def summarize_failure(self, prompt: str) -> dict[str, Any]:
        raise NotImplementedError


class MockProvider(LLMProvider):
    def generate_candidate(self, prompt: str, op: IROperation) -> dict[str, Any]:
        del prompt
        if op.op == "add":
            src1, src2 = op.args
            return {
                "candidate_id": "add_direct_v1",
                "operation": "add",
                "instructions": [{"op": "ADD", "dst": op.dst, "src1": src1, "src2": src2}],
                "expected_cycles": 1,
                "assumptions": ["integer add maps to base RISC-V ADD"],
                "why_valid": "ADD writes dst = src1 + src2."
            }
        if op.op == "mul":
            src1, src2 = op.args
            return {
                "candidate_id": "mul_direct_v1",
                "operation": "mul",
                "instructions": [{"op": "MUL", "dst": op.dst, "src1": src1, "src2": src2}],
                "expected_cycles": 3,
                "assumptions": ["M extension is enabled in Vortex RV32IMAF/RV64IMAFD"],
                "why_valid": "MUL writes dst = src1 * src2."
            }
        if op.op == "load":
            base = op.args[0]
            return {
                "candidate_id": "load_word_v1",
                "operation": "load",
                "instructions": [{"op": "LW", "dst": op.dst, "base": base, "offset": 0}],
                "expected_cycles": 4,
                "assumptions": ["load maps to a 32-bit word load with zero offset"],
                "why_valid": "LW writes dst = memory[base + offset]."
            }
        raise CompilerError(f"Mock provider cannot generate {op.op}")

    def extract_rule(self, prompt: str, op: IROperation, candidate: dict[str, Any], proof: dict[str, Any]) -> dict[str, Any]:
        del prompt
        bindings = {"dst": op.dst}
        for idx, arg in enumerate(op.args, start=1):
            bindings[f"src{idx}"] = arg
        pattern = []
        for inst in candidate["instructions"]:
            out = dict(inst)
            for key, value in list(out.items()):
                if value == op.dst:
                    out[key] = "$dst"
                for idx, arg in enumerate(op.args, start=1):
                    if value == arg:
                        out[key] = f"$src{idx}"
            pattern.append(out)
        return {
            "rule_id": f"{op.op.upper()}_INT_v1",
            "operation": op.op,
            "pattern": pattern,
            "constraints": [],
            "estimated_cycles": proof["cycles"],
            "replaces": None,
            "explanation": f"Reusable {op.op} rule extracted from verified candidate."
        }

    def summarize_failure(self, prompt: str) -> dict[str, Any]:
        del prompt
        return {
            "summary": "candidate failed local verification",
            "likely_cause": "instruction sequence did not produce the expected destination value",
            "retry_hint": "use the direct instruction for this operation if available"
        }


class FireworksProvider(LLMProvider):
    def __init__(self) -> None:
        self.api_key = os.environ.get("FIREWORKS_API_KEY")
        if not self.api_key:
            raise CompilerError("FIREWORKS_API_KEY is not set")
        self.candidate_model = os.environ.get(
            "CANDIDATE_MODEL",
            "accounts/fireworks/models/kimi-k2p6"
        )
        self.rule_model = os.environ.get(
            "RULE_MODEL",
            "accounts/fireworks/models/kimi-k2p6"
        )
        self.endpoint = os.environ.get(
            "FIREWORKS_CHAT_ENDPOINT",
            "https://api.fireworks.ai/inference/v1/chat/completions"
        )

    def _chat_json(self, model: str, prompt: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": "Return strict JSON only. No markdown."},
            {"role": "user", "content": prompt}
        ]
        
        for attempt in range(3):
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 4096,
                "response_format": {"type": "json_object"}
            }
            req = urllib.request.Request(
                os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai/inference/v1/chat/completions"),
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                method="POST"
            )
            
            content = ""
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                content = raw["choices"][0]["message"]["content"]
                if not content.strip():
                    raise CompilerError("Empty response from LLM")
                
                # Attempt to parse
                return parse_json_response(content)
                
            except CompilerError as comp_err:
                if "Failed to parse LLM JSON response" in str(comp_err):
                    # SELF-CORRECTING RETRY
                    print(f"[LLM] JSON parse failed on attempt {attempt+1}: {comp_err}")
                    if attempt < 2:
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": f"Your response was not valid JSON or contained markdown. Parse error: {comp_err}. Please output ONLY a single valid JSON object matching the requested schema, with absolutely no conversational text or markdown blocks."})
                    else:
                        raise
                else:
                    raise
                    
            except (urllib.error.HTTPError, TimeoutError, urllib.error.URLError) as exc:
                if attempt == 2:
                    if isinstance(exc, urllib.error.HTTPError):
                        body = exc.read().decode("utf-8", errors="replace")
                        raise CompilerError(f"HTTP {exc.code}: {body}") from exc
                    raise CompilerError(f"API Error: {exc}") from exc
                import time
                print(f"[llm] api error: {exc}, retrying ({attempt+1}/3)...")
                time.sleep(2 ** attempt)
        raise CompilerError("Failed after retries")

    def generate_candidate(self, prompt: str, op: IROperation) -> dict[str, Any]:
        del op
        return self._chat_json(self.candidate_model, prompt)

    def extract_rule(self, prompt: str, op: IROperation, candidate: dict[str, Any], proof: dict[str, Any]) -> dict[str, Any]:
        del op, candidate, proof
        return self._chat_json(self.rule_model, prompt)

    def summarize_failure(self, prompt: str) -> dict[str, Any]:
        return self._chat_json(self.rule_model, prompt)


def instantiate_rule(rule: dict[str, Any], op: IROperation) -> dict[str, Any]:
    values = {"$dst": op.dst}
    for idx, arg in enumerate(op.args, start=1):
        values[f"$src{idx}"] = arg
    instructions = []
    for inst in rule["pattern"]:
        out = {}
        for key, value in inst.items():
            out[key] = values.get(value, value)
        instructions.append(out)
    return {
        "candidate_id": rule["rule_id"] + "_instantiated",
        "operation": op.op,
        "instructions": instructions,
        "expected_cycles": rule.get("estimated_cycles"),
        "assumptions": ["instantiated from verified compiler rule"],
        "why_valid": rule.get("explanation", "")
    }


def build_prompt(name: str, values: dict[str, Any]) -> str:
    json_values = {
        key: json.dumps(value, indent=2)
        for key, value in values.items()
    }
    return render_template(ROOT / "prompts" / name, json_values)


def learn_operation(
    op: IROperation,
    provider: LLMProvider,
    rules: RuleDatabase,
    checker: StaticChecker,
    simulator: VortexSimulator,
    hardware_facts: dict[str, Any],
    max_retries: int,
    emitter: VortexArtifactEmitter | None = None,
    force_relearn: list[str] | None = None
) -> dict[str, Any]:
    force_relearn = force_relearn or []
    existing = rules.find(op)
    if existing and op.op not in force_relearn:
        candidate = instantiate_rule(existing, op)
        print(f"[rule] reused {existing['rule_id']} for {op.source}")
    else:
        feedback: dict[str, Any] = {}
        candidate = {}
        for attempt in range(1, max_retries + 1):
            prompt = build_prompt(
                "generate_candidate.txt",
                {
                    "operation_json": op.to_prompt_json(),
                    "hardware_facts_json": hardware_facts,
                    "rules_json": rules.data,
                    "feedback_json": feedback
                }
            )
            candidate = provider.generate_candidate(prompt, op)
            print(f"[llm] candidate {candidate.get('candidate_id')} for {op.source}")
            
            check = checker.check(candidate)
            if not check["ok"]:
                feedback = check
                print(f"[check] rejected: {check['error']}")
                # Log failed check
                log_entry = {
                    "ts": int(__import__("time").time()),
                    "op_type": op.op,
                    "simulator_target": "none",
                    "candidate_json": candidate,
                    "proof_cycles": -1,
                    "passed": False,
                    "error": check["error"]
                }
                with open(ROOT / "data" / "candidate_log.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
                continue
                
            proof = simulator.run(op, candidate)
            
            log_entry = {
                "ts": int(__import__("time").time()),
                "op_type": op.op,
                "simulator_target": proof.get("simulator", "unknown"),
                "candidate_json": candidate,
                "proof_cycles": proof.get("cycles", -1),
                "passed": proof["ok"],
                "error": proof.get("error")
            }
            with open(ROOT / "data" / "candidate_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            if proof["ok"]:
                print(f"[sim] passed cycles={proof['cycles']} expected={proof['expected']} actual={proof['actual']}")
                rule_prompt = build_prompt(
                    "extract_rule.txt",
                    {
                        "operation_json": op.to_prompt_json(),
                        "candidate_json": candidate,
                        "simulation_json": proof,
                        "rules_json": rules.data
                    }
                )
                rule = provider.extract_rule(rule_prompt, op, candidate, proof)
                decision = rules.add_or_replace(rule, proof)
                print(f"[gemma] {decision}: {rule['rule_id']}")
                artifact = str(emitter.emit(op, candidate, proof)) if emitter else None
                if artifact:
                    print(f"[emit] vortex test artifact: {artifact}")
                return {"op": op.source, "candidate": candidate, "proof": proof, "rule": rule, "decision": decision, "artifact": artifact}
            failure_prompt = build_prompt(
                "summarize_failure.txt",
                {
                    "operation_json": op.to_prompt_json(),
                    "candidate_json": candidate,
                    "failure_json": proof
                }
            )
            feedback = provider.summarize_failure(failure_prompt)
            print(f"[sim] failed: {feedback.get('summary')}")
        raise CompilerError(f"Could not learn rule for {op.source}")
    check = checker.check(candidate)
    if not check["ok"]:
        raise CompilerError(f"Stored rule failed static check: {check['error']}")
    proof = simulator.run(op, candidate)
    if not proof["ok"]:
        raise CompilerError(f"Stored rule failed simulation: expected={proof['expected']} actual={proof['actual']}")
    print(f"[sim] reused rule passed cycles={proof['cycles']}")
    artifact = str(emitter.emit(op, candidate, proof)) if emitter else None
    if artifact:
        print(f"[emit] vortex test artifact: {artifact}")
    return {"op": op.source, "candidate": candidate, "proof": proof, "rule": existing, "decision": "reused", "artifact": artifact}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Grow a tiny Vortex compiler backend.")
    parser.add_argument("--provider", choices=["mock", "fireworks"], default="mock")
    parser.add_argument("--program", type=Path, default=ROOT / "examples" / "demo_program.ir")
    parser.add_argument("--facts", type=Path, default=DEFAULT_FACTS)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--reset-rules", action="store_true")
    parser.add_argument("--simulator", choices=["local", "simx", "rtlsim"], default="local",
                        help="local=cheap semantic check; simx=fast Vortex sim; rtlsim=RTL/Verilator")
    parser.add_argument("--emit-vortex-tests", type=Path, default=ROOT / "artifacts" / "vortex_tests")
    parser.add_argument("--force-relearn", nargs="*", default=[], help="DEV-ONLY: Force regenerating candidates for these operations, bypassing the rule cache.")
    args = parser.parse_args(argv)

    load_dotenv(args.env_file)
    if args.reset_rules:
        save_json(args.rules, {"target": "vortex", "version": 1, "rules": {}, "history": []})

    hardware_facts = load_json(args.facts)
    rules = RuleDatabase(args.rules)
    checker = StaticChecker(hardware_facts)
    
    emitter = VortexArtifactEmitter(args.emit_vortex_tests, ROOT / "vendor" / "vortex")
    if args.simulator == "local":
        simulator = LocalSemanticSimulator(hardware_facts)
    else:
        simulator = VortexSimulator(ROOT / "vendor" / "vortex", emitter, sim_target=args.simulator)
    print(f"[sim] using simulator={args.simulator}")
        
    provider: LLMProvider = MockProvider() if args.provider == "mock" else FireworksProvider()

    ops = parse_ir(args.program)
    print(f"[start] provider={args.provider} program={args.program}")
    results = []
    for op in ops:
        if is_structural_ir_op(op):
            print(f"[boundary] {op.source}")
            results.append({"op": op.source, "decision": "boundary", "proof": {"cycles": 0}})
            continue
        if args.provider == "mock" and not mock_provider_can_handle(op):
            print(f"[unresolved] {op.source} => parsed as {op.op}{op.args}; needs Fireworks/lifter rule")
            results.append({"op": op.source, "decision": "unresolved", "proof": {"cycles": 0}})
            continue
        results.append(
            learn_operation(op, provider, rules, checker, simulator, hardware_facts, args.max_retries, emitter, args.force_relearn)
        )
    print("[done] learned/reused operations:")
    for result in results:
        print(f"  - {result['op']} => {result['decision']} ({result['proof']['cycles']} cycles)")
    return 0


def _test_add_or_replace() -> None:
    test_path = ROOT / "data" / "rules_test.json"
    save_json(test_path, {"rules": {}})
    db = RuleDatabase(test_path)
    db.rules.clear()
    
    # Test 1: Add new rule
    rule1 = {"operation": "add", "rule_id": "ADD_v1"}
    proof1 = {"ok": True, "cycles": 100}
    assert db.add_or_replace(rule1, proof1) == "added"
    assert db.rules["add"]["actual_cycles"] == 100
    
    # Test 2: Worse cycle count -> keep existing
    rule2 = {"operation": "add", "rule_id": "ADD_v2"}
    proof2 = {"ok": True, "cycles": 150}
    assert db.add_or_replace(rule2, proof2) == "kept_existing"
    assert db.rules["add"]["rule_id"] == "ADD_v1"
    
    # Test 3: Better cycle count -> replace
    rule3 = {"operation": "add", "rule_id": "ADD_v3"}
    proof3 = {"ok": True, "cycles": 50}
    assert db.add_or_replace(rule3, proof3) == "replaced"
    assert db.rules["add"]["rule_id"] == "ADD_v3"
    assert db.rules["add"]["actual_cycles"] == 50

    # Clean up test file
    import os
    if os.path.exists(db.path):
        os.remove(db.path)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-replace":
        _test_add_or_replace()
        print("Replacement unit tests passed!")
        sys.exit(0)
    try:
        raise SystemExit(main(sys.argv[1:]))
    except CompilerError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
