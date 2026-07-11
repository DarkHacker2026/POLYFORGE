#!/usr/bin/env python3
"""
oracle_standalone.py  —  Standalone Parallel Oracle Library (Item 4)

A race-detecting oracle that ingests a simple READ/WRITE/BARRIER IR JSON,
completely decoupled from any surface language or compiler.

Anyone can pip-install this and verify their own parallel compiler output.

IR Format (JSON):
{
  "num_threads": 4,
  "shared_memory_bytes": 256,
  "initial_memory": {"0": 1, "4": 2},   // byte_addr: int32_value
  "threads": [
    {
      "tid": 0,
      "instructions": [
        {"op": "READ",    "addr": 0,   "width": 4},
        {"op": "WRITE",   "addr": 128, "width": 4, "value": 42},
        {"op": "BARRIER"},
        {"op": "READ",    "addr": 128, "width": 4}
      ]
    }
  ]
}
"""
from __future__ import annotations
import json
import ctypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Public data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RaceReport:
    kind: str          # "RAW" | "WAR" | "WAW"
    tid: int           # thread that triggered the fault
    other_tid: int     # other thread involved
    byte_addr: int
    epoch: int
    message: str


@dataclass
class OracleResult:
    passed: bool
    races: list[RaceReport] = field(default_factory=list)
    epochs_executed: int = 0
    memory_final: bytes = field(default_factory=bytes)
    message: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# IR loader
# ─────────────────────────────────────────────────────────────────────────────

class OracleInput:
    """Parsed representation of a parallel kernel IR."""

    def __init__(self, num_threads: int, memory_size: int,
                 initial_memory: dict[int, int],
                 thread_instructions: list[list[dict]]):
        self.num_threads = num_threads
        self.memory_size = memory_size
        self.initial_memory = initial_memory          # {byte_addr: int32}
        self.thread_instructions = thread_instructions  # list[list[dict]]

    @classmethod
    def from_json(cls, path: str | Path) -> "OracleInput":
        """Load oracle IR from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "OracleInput":
        """Load oracle IR from a Python dict."""
        num_threads = int(data["num_threads"])
        memory_size = int(data.get("shared_memory_bytes", 65536))

        raw_mem = data.get("initial_memory", {})
        initial_memory = {int(k): int(v) for k, v in raw_mem.items()}

        threads_raw = data.get("threads", [])
        # Sort by tid to ensure correct ordering
        threads_raw.sort(key=lambda t: t.get("tid", 0))
        thread_instructions = [t["instructions"] for t in threads_raw]

        # If fewer threads than num_threads were declared, pad with empty
        while len(thread_instructions) < num_threads:
            thread_instructions.append([])

        return cls(num_threads, memory_size, initial_memory, thread_instructions)

    @classmethod
    def from_uniform_instructions(cls, instructions: list[dict],
                                   num_threads: int,
                                   memory_size: int = 65536,
                                   initial_memory: dict[int, int] | None = None
                                   ) -> "OracleInput":
        """All threads run the same instruction sequence."""
        return cls(
            num_threads=num_threads,
            memory_size=memory_size,
            initial_memory=initial_memory or {},
            thread_instructions=[list(instructions) for _ in range(num_threads)]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Oracle engine
# ─────────────────────────────────────────────────────────────────────────────

class StandaloneOracle:
    """
    Race-detecting parallel oracle.

    Detects:
      RAW  — Thread A reads a byte written by Thread B in the same sync epoch
      WAW  — Thread A writes a byte already written by Thread B in the same epoch
              (also catches partial overlaps byte-by-byte)
      WAR  — Thread A writes a byte already read by Thread B in the same epoch
    """

    def run(self, inp: OracleInput) -> OracleResult:
        mem = bytearray(inp.memory_size)
        # memory_writes[byte_addr] = (epoch, tid)
        memory_writes: dict[int, tuple[int, int]] = {}
        # memory_reads[byte_addr]  = (epoch, tid)
        memory_reads: dict[int, tuple[int, int]] = {}
        sync_epoch = 1
        races: list[RaceReport] = []

        # Load initial memory
        for addr, value in inp.initial_memory.items():
            b = ctypes.c_int32(value).value.to_bytes(4, byteorder="little", signed=True)
            mem[addr:addr+4] = b

        # Thread PCs
        pcs = [0] * inp.num_threads
        active = [True] * inp.num_threads

        def _read(tid: int, addr: int, width: int):
            for i in range(width):
                ba = addr + i
                if ba in memory_writes:
                    ep, w_tid = memory_writes[ba]
                    if ep == sync_epoch and w_tid != tid:
                        r = RaceReport("RAW", tid, w_tid, ba, sync_epoch,
                                       f"RAW: Thread {tid} read byte {ba} written by Thread {w_tid} in epoch {sync_epoch}")
                        races.append(r)
                        raise RuntimeError(r.message)
                memory_reads[ba] = (sync_epoch, tid)

        def _write(tid: int, addr: int, value: int, width: int):
            b = ctypes.c_int32(value).value.to_bytes(4, byteorder="little", signed=True)
            for i in range(width):
                ba = addr + i
                if ba in memory_writes:
                    ep, w_tid = memory_writes[ba]
                    if ep == sync_epoch and w_tid != tid:
                        r = RaceReport("WAW", tid, w_tid, ba, sync_epoch,
                                       f"WAW (Partial Overlap): Thread {tid} wrote byte {ba} written by Thread {w_tid} in epoch {sync_epoch}")
                        races.append(r)
                        raise RuntimeError(r.message)
                if ba in memory_reads:
                    ep, r_tid = memory_reads[ba]
                    if ep == sync_epoch and r_tid != tid:
                        r = RaceReport("WAR", tid, r_tid, ba, sync_epoch,
                                       f"WAR: Thread {tid} wrote byte {ba} read by Thread {r_tid} in epoch {sync_epoch}")
                        races.append(r)
                        raise RuntimeError(r.message)
                memory_writes[ba] = (sync_epoch, tid)
                mem[addr + i] = b[i]

        nonlocal_epoch = [sync_epoch]

        max_steps = 1_000_000
        step_counts = [0] * inp.num_threads

        while any(active[t] and pcs[t] < len(inp.thread_instructions[t])
                  for t in range(inp.num_threads)):
            for tid in range(inp.num_threads):
                instrs = inp.thread_instructions[tid]
                if not active[tid] or pcs[tid] >= len(instrs):
                    continue

                step_counts[tid] += 1
                if step_counts[tid] > max_steps:
                    return OracleResult(False, races, nonlocal_epoch[0],
                                        bytes(mem), f"Thread {tid}: execution limit exceeded")

                inst = instrs[pcs[tid]]
                op = inst.get("op", "").upper()

                try:
                    if op == "READ":
                        addr  = int(inst["addr"])
                        width = int(inst.get("width", 4))
                        _read(tid, addr, width)

                    elif op == "WRITE":
                        addr  = int(inst["addr"])
                        value = int(inst.get("value", 0))
                        width = int(inst.get("width", 4))
                        _write(tid, addr, value, width)

                    elif op == "BARRIER":
                        if tid == 0:
                            nonlocal_epoch[0] += 1
                            sync_epoch = nonlocal_epoch[0]  # type: ignore[assignment]
                            memory_reads.clear()

                    elif op in ("NOP", "LABEL"):
                        pass

                    else:
                        return OracleResult(False, races, nonlocal_epoch[0],
                                            bytes(mem), f"Unknown op '{op}' in thread {tid}")

                except RuntimeError as e:
                    return OracleResult(False, races, nonlocal_epoch[0], bytes(mem), str(e))

                pcs[tid] += 1

        # Patch the scope — Python closures make sync_epoch tricky
        final_epoch = nonlocal_epoch[0]
        return OracleResult(True, races, final_epoch, bytes(mem),
                            f"Passed — {final_epoch} sync epoch(s) executed")


# ─────────────────────────────────────────────────────────────────────────────
# CLI: python oracle_standalone.py <ir.json>
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python oracle_standalone.py <ir_file.json>")
        sys.exit(1)
    inp = OracleInput.from_json(sys.argv[1])
    oracle = StandaloneOracle()
    result = oracle.run(inp)
    if result.passed:
        print(f"PASS: {result.message}")
    else:
        print(f"RACE DETECTED: {result.message}")
        for r in result.races:
            print(f"  [{r.kind}] Thread {r.tid} vs Thread {r.other_tid} @ byte {r.byte_addr} epoch {r.epoch}")
    sys.exit(0 if result.passed else 1)
