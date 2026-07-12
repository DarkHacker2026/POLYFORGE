import ctypes
import copy
from typing import Dict, List, Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Single-threaded interpreter (unchanged — used by grow_compiler.py)
# ─────────────────────────────────────────────────────────────────────────────
class ReferenceISA:
    def __init__(self, memory_size=4096):
        self.regs = {f"r{i}": 0 for i in range(32)}
        self.memory = bytearray(memory_size)
        self.pc = 0
        self.labels = {}
        
    def _read_reg(self, r: str) -> int:
        if r == "r0":
            return 0
        return self.regs.get(r, 0)
        
    def _write_reg(self, r: str, val: int):
        if r != "r0":
            # 32-bit wrap
            self.regs[r] = ctypes.c_int32(val).value

    def _read_mem32(self, addr: int) -> int:
        if addr < 0 or addr + 4 > len(self.memory):
            raise MemoryError(f"Out of bounds read at {addr}")
        return int.from_bytes(self.memory[addr:addr+4], byteorder='little', signed=True)

    def _write_mem32(self, addr: int, val: int):
        if addr < 0 or addr + 4 > len(self.memory):
            raise MemoryError(f"Out of bounds write at {addr}")
        self.memory[addr:addr+4] = ctypes.c_int32(val).value.to_bytes(4, byteorder='little', signed=True)

    def execute(self, instructions: List[Dict[str, Any]], initial_regs: Dict[str, int] = None, initial_mem: Dict[int, int] = None) -> Dict[str, int]:
        """
        Executes a sequence of instructions. 
        initial_regs: dictionary of register names to initial 32-bit integer values.
        initial_mem: dictionary of byte addresses to initial 32-bit integer values.
        """
        if initial_regs:
            for r, v in initial_regs.items():
                self._write_reg(r, v)
                
        if initial_mem:
            for addr, v in initial_mem.items():
                self._write_mem32(addr, v)

        # First pass: map labels
        self.labels.clear()
        for i, inst in enumerate(instructions):
            op = inst.get("op", "").upper()
            if op == "LABEL":
                lbl = inst.get("label", "")
                if lbl.endswith("b") or lbl.endswith("f"):
                    lbl = lbl[:-1]
                self.labels[lbl] = i

        self.pc = 0
        inst_count = 0
        max_insts = 100000  # infinite loop protection

        while self.pc < len(instructions):
            if inst_count > max_insts:
                raise RuntimeError("Execution limit exceeded (infinite loop?)")
            
            inst = instructions[self.pc]
            op = inst.get("op", "").upper()
            inst_count += 1
            
            if op == "LABEL":
                self.pc += 1
                continue
                
            dst = inst.get("dst", "")
            src1 = inst.get("src1", "")
            src2 = inst.get("src2", "")
            base = inst.get("base", "")
            
            if op == "ADD":
                self._write_reg(dst, self._read_reg(src1) + self._read_reg(src2))
            elif op == "SUB":
                self._write_reg(dst, self._read_reg(src1) - self._read_reg(src2))
            elif op == "MUL":
                self._write_reg(dst, self._read_reg(src1) * self._read_reg(src2))
            elif op == "ADDI":
                imm = int(inst.get("imm", 0))
                self._write_reg(dst, self._read_reg(src1) + imm)
            elif op == "SLLI":
                imm = int(inst.get("imm", 0))
                self._write_reg(dst, self._read_reg(src1) << (imm & 31))
            elif op == "LW":
                offset = int(inst.get("offset", 0))
                addr = self._read_reg(base or src1) + offset
                self._write_reg(dst, self._read_mem32(addr))
            elif op == "SW":
                offset = int(inst.get("offset", 0))
                addr = self._read_reg(base or src1) + offset
                self._write_mem32(addr, self._read_reg(src2))
            elif op == "BNE":
                if self._read_reg(src1) != self._read_reg(src2):
                    lbl = inst.get("target", inst.get("label", ""))
                    if lbl.endswith("b") or lbl.endswith("f"):
                        lbl = lbl[:-1]
                    if lbl in self.labels:
                        self.pc = self.labels[lbl]
                        continue
                    else:
                        raise ValueError(f"Label not found: {lbl}")
            elif op == "BEQ":
                if self._read_reg(src1) == self._read_reg(src2):
                    lbl = inst.get("target", inst.get("label", ""))
                    if lbl.endswith("b") or lbl.endswith("f"):
                        lbl = lbl[:-1]
                    if lbl in self.labels:
                        self.pc = self.labels[lbl]
                        continue
                    else:
                        raise ValueError(f"Label not found: {lbl}")
            else:
                raise NotImplementedError(f"Unsupported op in ReferenceISA: {op}")
                
            self.pc += 1

        return self.regs

def get_initial_state():
    regs = {f"r{i}": i * 3 + 1 for i in range(32)}
    regs["r0"] = 0
    # Add explicit initialization for test ops
    regs["r1"] = 10
    regs["r2"] = 5
    regs["r6"] = 12  # base pointer to memory array (byte-aligned)
    regs["r4"] = 12  # FIX: was 13 (unaligned, caused false-positive), now 12 (aligned to mem[12]=101)
    # Memory array: emulate the C stack array {101, 202, 303, 404}
    # Stored at byte addresses 12, 16, 20, 24 (4-byte aligned word offsets).
    mem = {}
    mem[12] = 101
    mem[16] = 202
    mem[20] = 303
    mem[24] = 404
    return regs, mem

# ─────────────────────────────────────────────────────────────────────────────
# Parallel oracle: N hardware threads sharing one memory, per-thread registers.
#
# Surface-language pseudo-ops handled here:
#   THREAD_ID   : writes thread index to dst (vx_thread_id() equivalent)
#   NUM_THREADS : writes total thread count to dst (vx_num_threads() equivalent)
#   BARRIER     : no-op in the software oracle; threads are serialised so the
#                 invariant is trivially met (every thread has reached the barrier
#                 before any thread is allowed to proceed).
# ─────────────────────────────────────────────────────────────────────────────
class ParallelReferenceISA:
    """SIMT oracle: runs the same instruction sequence on N independent threads
    sharing one bytearray memory region.

    Usage::

        oracle = ParallelReferenceISA(num_threads=4, memory_size=4096)
        results = oracle.execute_parallel(
            instructions,           # list of dicts (same format as ReferenceISA)
            initial_regs_per_thread,# list[dict] — one per thread (or single dict
                                    # replicated to all)
            initial_mem             # dict {byte_addr: int32_value}
        )
        # results is a list[dict[str,int]] — final register state per thread
    """

    def __init__(self, num_threads: int = 4, memory_size: int = 65536):
        self.num_threads = num_threads
        self.memory_size = memory_size
        # Shared memory — all threads read/write the same bytes.
        self.memory = bytearray(memory_size)
        # Per-thread register files.
        self.thread_regs: list[dict[str, int]] = [
            {f"r{i}": 0 for i in range(32)} for _ in range(num_threads)
        ]
        self.labels: dict[str, int] = {}
        # Track memory writes: dict[byte_addr, (epoch, tid)]
        self.memory_writes: dict[int, tuple[int, int]] = {}
        # Track memory reads:  dict[byte_addr, (epoch, tid)]
        # Used to detect WAR (Write-After-Read) races
        self.memory_reads: dict[int, tuple[int, int]] = {}
        self.sync_epoch: int = 0

    # ── memory helpers (shared) ───────────────────────────────────────────────

    def _read_mem32(self, tid: int, addr: int) -> int:
        if addr < 0 or addr + 4 > self.memory_size:
            raise MemoryError(f"OOB read at {addr}")
        for i in range(4):
            byte_addr = addr + i
            # RAW: another thread wrote this byte in the same epoch
            if byte_addr in self.memory_writes:
                epoch, w_tid = self.memory_writes[byte_addr]
                if epoch == self.sync_epoch and w_tid != tid:
                    raise RuntimeError(
                        f"RAW Data Race: Thread {tid} read byte {byte_addr} "
                        f"written by Thread {w_tid} in epoch {self.sync_epoch}.")
            # Record the read for WAR detection
            self.memory_reads[byte_addr] = (self.sync_epoch, tid)
        return int.from_bytes(self.memory[addr:addr+4], byteorder='little', signed=True)

    def _write_mem32(self, tid: int, addr: int, val: int):
        if addr < 0 or addr + 4 > self.memory_size:
            raise MemoryError(f"OOB write at {addr}")
        for i in range(4):
            byte_addr = addr + i
            # WAW: another thread already wrote this byte in the same epoch
            if byte_addr in self.memory_writes:
                epoch, w_tid = self.memory_writes[byte_addr]
                if epoch == self.sync_epoch and w_tid != tid:
                    raise RuntimeError(
                        f"WAW Data Race (Partial Overlap): Thread {tid} wrote byte "
                        f"{byte_addr} already written by Thread {w_tid} in epoch {self.sync_epoch}.")
            # WAR: another thread already read this byte in the same epoch
            if byte_addr in self.memory_reads:
                epoch, r_tid = self.memory_reads[byte_addr]
                if epoch == self.sync_epoch and r_tid != tid:
                    raise RuntimeError(
                        f"WAR Data Race: Thread {tid} wrote byte {byte_addr} "
                        f"already read by Thread {r_tid} in epoch {self.sync_epoch}.")
            self.memory_writes[byte_addr] = (self.sync_epoch, tid)

        self.memory[addr:addr+4] = ctypes.c_int32(val).value.to_bytes(
            4, byteorder='little', signed=True)

    # ── register helpers (per-thread) ─────────────────────────────────────────

    def _read_reg(self, tid: int, r: str) -> int:
        if r == "r0":
            return 0
        return self.thread_regs[tid].get(r, 0)

    def _write_reg(self, tid: int, r: str, val: int):
        if r != "r0":
            self.thread_regs[tid][r] = ctypes.c_int32(val).value

    # ── execution ─────────────────────────────────────────────────────────────

    def execute_parallel(
        self,
        instructions: List[Dict[str, Any]],
        initial_regs: "List[Dict[str,int]] | Dict[str,int] | None" = None,
        initial_mem: "Dict[int,int] | None" = None,
    ) -> List[Dict[str, int]]:
        """Run *instructions* on all N threads and return per-thread final regs.

        *initial_regs* may be:
          - a single dict → replicated to every thread
          - a list of N dicts → one per thread (allows thread-specific init)
          - None → all regs start at 0
        """
        # Initialise memory
        self.memory = bytearray(self.memory_size)
        self.memory_writes = {}
        self.memory_reads = {}
        self.sync_epoch = 0
        if initial_mem:
            for addr, v in initial_mem.items():
                self._write_mem32(-1, addr, v)
        # Advance epoch so initial writes don't cause WAW/WAR races with Thread 0
        self.sync_epoch = 1
        # Clear read tracking — init writes should not block Thread 0 reads
        self.memory_reads = {}

        # Initialise per-thread regs
        if initial_regs is None:
            base = {f"r{i}": 0 for i in range(32)}
            regs_list = [dict(base) for _ in range(self.num_threads)]
        elif isinstance(initial_regs, dict):
            regs_list = [dict(initial_regs) for _ in range(self.num_threads)]
        else:
            regs_list = [dict(r) for r in initial_regs]
        for tid in range(self.num_threads):
            for reg, val in regs_list[tid].items():
                self._write_reg(tid, reg, ctypes.c_int32(val).value)

        # Label pass
        self.labels = {}
        for idx, inst in enumerate(instructions):
            if inst.get("op", "").upper() == "LABEL":
                lbl = inst.get("label", "")
                lbl = lbl.rstrip("bf")
                self.labels[lbl] = idx

        # Per-thread PC and active flag
        pcs = [0] * self.num_threads
        active = [True] * self.num_threads
        step_counts = [0] * self.num_threads
        max_steps = 1_000_000

        # Run until all threads have finished
        while any(active[tid] and pcs[tid] < len(instructions)
                  for tid in range(self.num_threads)):
            for tid in range(self.num_threads):
                if not active[tid] or pcs[tid] >= len(instructions):
                    continue

                step_counts[tid] += 1
                if step_counts[tid] > max_steps:
                    raise RuntimeError(f"Thread {tid}: execution limit exceeded")

                inst = instructions[pcs[tid]]
                op = inst.get("op", "").upper()
                dst   = inst.get("dst", "")
                src1  = inst.get("src1", "")
                src2  = inst.get("src2", "")
                base  = inst.get("base", "")

                if op == "LABEL":
                    pcs[tid] += 1
                    continue

                # ── surface pseudo-ops ────────────────────────────────────────
                if op == "THREAD_ID":
                    self._write_reg(tid, dst, tid)
                elif op == "NUM_THREADS":
                    self._write_reg(tid, dst, self.num_threads)
                elif op == "BARRIER":
                    # Advance sync epoch when tid==0 executes barrier.
                    # Clear BOTH read and write tracking: after a barrier,
                    # all previous writes are globally visible and committed.
                    # Keeping stale write records would cause false WAW/WAR
                    # positives on legitimate cross-epoch accesses.
                    if tid == 0:
                        self.sync_epoch += 1
                        self.memory_reads = {}
                        self.memory_writes = {}
                # ── standard ISA ─────────────────────────────────────────────
                elif op == "ADD":
                    self._write_reg(tid, dst,
                        self._read_reg(tid, src1) + self._read_reg(tid, src2))
                elif op == "SUB":
                    self._write_reg(tid, dst,
                        self._read_reg(tid, src1) - self._read_reg(tid, src2))
                elif op == "MUL":
                    self._write_reg(tid, dst,
                        self._read_reg(tid, src1) * self._read_reg(tid, src2))
                elif op == "ADDI":
                    imm = int(inst.get("imm", 0))
                    self._write_reg(tid, dst, self._read_reg(tid, src1) + imm)
                elif op == "SLLI":
                    imm = int(inst.get("imm", 0))
                    self._write_reg(tid, dst, self._read_reg(tid, src1) << (imm & 31))
                elif op == "LW":
                    offset = int(inst.get("offset", 0))
                    addr = self._read_reg(tid, base or src1) + offset
                    self._write_reg(tid, dst, self._read_mem32(tid, addr))
                elif op == "SW":
                    offset = int(inst.get("offset", 0))
                    addr = self._read_reg(tid, base or src1) + offset
                    self._write_mem32(tid, addr, self._read_reg(tid, src2))
                elif op == "BNE":
                    if self._read_reg(tid, src1) != self._read_reg(tid, src2):
                        lbl = inst.get("target", inst.get("label", "")).rstrip("bf")
                        if lbl not in self.labels:
                            raise ValueError(f"BNE: label not found: {lbl}")
                        pcs[tid] = self.labels[lbl]
                        continue
                elif op == "BEQ":
                    if self._read_reg(tid, src1) == self._read_reg(tid, src2):
                        lbl = inst.get("target", inst.get("label", "")).rstrip("bf")
                        if lbl not in self.labels:
                            raise ValueError(f"BEQ: label not found: {lbl}")
                        pcs[tid] = self.labels[lbl]
                        continue
                elif op == "TMC_ZERO":
                    # Deactivate this thread (vx_tmc_zero equivalent)
                    active[tid] = False
                    continue
                else:
                    raise NotImplementedError(f"ParallelISA: unsupported op {op}")

                pcs[tid] += 1

        return [dict(self.thread_regs[tid]) for tid in range(self.num_threads)]


def verify_parallel_kernel(
    instructions: List[Dict[str, Any]],
    num_threads: int,
    initial_regs_per_thread: "List[Dict[str,int]] | None" = None,
    initial_mem: "Dict[int,int] | None" = None,
    check_fn = None,
) -> dict:
    """Run a parallel kernel through the oracle and optionally verify correctness.

    *check_fn* receives (thread_final_regs: list[dict], memory: bytearray)
    and should return (ok: bool, message: str).
    Returns a dict: {ok, thread_results, message}.
    """
    oracle = ParallelReferenceISA(num_threads=num_threads)
    results = oracle.execute_parallel(instructions, initial_regs_per_thread, initial_mem)
    if check_fn is None:
        return {"ok": True, "thread_results": results, "memory": oracle.memory,
                "message": "no check function provided"}
    result = check_fn(results, oracle.memory)
    if len(result) == 3:
        ok, msg, skipped = result
    else:
        ok, msg = result
        skipped = False
    return {"ok": ok, "thread_results": results, "memory": oracle.memory, "message": msg, "skipped": skipped}


# ─────────────────────────────────────────────────────────────────────────────
# Legacy single-threaded helpers (unchanged API)
# ─────────────────────────────────────────────────────────────────────────────
def compute_expected(op, initial_regs, initial_mem) -> int:
    isa = ReferenceISA()
    insts = []
    dst = op.dst
    args = op.args

    if op.op == "add":
        insts = [{"op": "ADD", "dst": dst, "src1": args[0], "src2": args[1]}]
    elif op.op == "mul":
        insts = [{"op": "MUL", "dst": dst, "src1": args[0], "src2": args[1]}]
    elif op.op == "mul_by_const_8":
        insts = [{"op": "SLLI", "dst": dst, "src1": args[0], "imm": 3}]
    elif op.op == "load":
        insts = [{"op": "LW", "dst": dst, "src1": args[0], "offset": 0}]
    elif op.op == "loop":
        # dst = loop(dst)
        insts = [
            {"op": "LABEL", "label": "1"}, 
            {"op": "ADDI", "dst": dst, "src1": dst, "imm": -1},
            {"op": "BNE", "src1": dst, "src2": "r0", "target": "1"}
        ]
    else:
        raise ValueError(f"Unknown IR op {op.op}")

    final_regs = isa.execute(insts, initial_regs, initial_mem)
    return final_regs.get(dst, 0)

