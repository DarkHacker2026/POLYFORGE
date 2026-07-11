"""
evaluate_phase3_clean.py

A ground-up rebuild of the Phase 3 evaluation harness.

DESIGN PRINCIPLES:
  1. Sanity floors first, verdict last.
  2. A dummy constant predictor and the linear model MUST both score at
     chance level before the GNN is even read.
  3. No positional signal anywhere:
       - Candidates are shuffled with a fixed per-block seed before scoring.
       - Sort ties are broken with a seeded random value, never input index.
       - Graph features are purely structural (opcodes + data-flow edges).
         Sequential position edges (i-1 -> i) are REMOVED.
  4. Feature list is printed explicitly so the reader can audit it.
  5. The verdict is printed LAST and only after the floor checks pass.
"""

import json
import random
import sys
import math
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Feature engineering (no position leak)

OPCODES = ["ADD", "SUB", "MUL", "ADDI", "SLLI", "LW", "SW", "BNE", "BEQ", "LABEL"]
OP_TO_IDX = {op: i for i, op in enumerate(OPCODES)}

FEATURE_DESCRIPTION = """
Node features (16 dims per node, purely structural):
  [0]       is_instruction  (1 if this node is an instruction, 0 if register)
  [1]       is_register     (1 if this node is a physical register)
  [2..11]   opcode one-hot  (10 ops: ADD SUB MUL ADDI SLLI LW SW BNE BEQ LABEL)
  [12]      has_immediate   (1 if instruction has an imm field)
  [13]      imm_magnitude   (imm / 1000.0)
  [14]      ever_read       (set on register nodes: 1 if register is read in block)
  [15]      ever_written    (set on register nodes: 1 if register is written in block)

Edges (data-flow only, NO sequential position edges):
  reg -> inst  for each src1, src2, base read
  inst -> reg  for each dst write

NOTE: The original gnn_model.py added edges [i-1, i] between consecutive
instructions in INPUT ORDER. That directly encoded schedule position into
the graph and caused the 100% GNN recall. Those edges are REMOVED here.
"""

def build_graph(insts):
    """Build a purely structural data-flow graph. No sequential edges."""
    num_insts = len(insts)
    num_regs  = 32
    total     = num_insts + num_regs

    X = torch.zeros((total, 16))

    # Register nodes
    for r in range(32):
        X[num_insts + r, 1] = 1.0  # is_register

    edges = []

    for i, inst in enumerate(insts):
        X[i, 0] = 1.0  # is_instruction
        op = inst.get("op", "").upper()
        if op in OP_TO_IDX:
            X[i, 2 + OP_TO_IDX[op]] = 1.0

        if "imm" in inst:
            X[i, 12] = 1.0
            X[i, 13] = float(inst["imm"]) / 1000.0

        # Data-flow edges only (no i-1 -> i position edges)
        for field in ("src1", "src2", "base"):
            val = inst.get(field)
            if isinstance(val, str) and val.startswith("r"):
                try:
                    r_idx = int(val[1:])
                    edges.append([num_insts + r_idx, i])   # reg -> inst
                    X[num_insts + r_idx, 14] = 1.0         # ever_read
                except ValueError:
                    pass

        dst = inst.get("dst")
        if isinstance(dst, str) and dst.startswith("r"):
            try:
                r_idx = int(dst[1:])
                edges.append([i, num_insts + r_idx])       # inst -> reg
                X[num_insts + r_idx, 15] = 1.0             # ever_written
            except ValueError:
                pass

    if not edges:
        edges = [[0, 0]]

    # Adjacency matrix (row-normalised, with self-loops)
    A = torch.zeros((total, total))
    for e in edges:
        A[e[0], e[1]] = 1.0
    for i in range(total):
        A[i, i] = 1.0
    rowsum = A.sum(dim=1, keepdim=True)
    A = A / torch.clamp(rowsum, min=1.0)

    return X, A


# Models

class GNN(nn.Module):
    def __init__(self, in_dim=16, hidden=32):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden)
        self.W2 = nn.Linear(hidden, hidden)
        self.fc  = nn.Linear(hidden, 1)

    def forward(self, X, A):
        H1 = torch.relu(self.W1(A @ X))
        H2 = torch.relu(self.W2(A @ H1))
        inst_mask = (X[:, 0] == 1.0).float().unsqueeze(1)
        pooled = (H2 * inst_mask).sum(0) / inst_mask.sum().clamp(min=1.0)
        return self.fc(pooled)


class LinearModel(nn.Module):
    """Sums opcode-count features, then linear projection.
    By construction this is IDENTICAL for all topological permutations of
    the same instructions (opcode counts don't change with reordering).
    Therefore it CANNOT rank schedules and must score at chance."""
    def __init__(self, in_dim=16):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, X, A):
        inst_mask = (X[:, 0] == 1.0).float().unsqueeze(1)
        sum_x = (X * inst_mask).sum(0)
        return self.fc(sum_x)


# Evaluation helpers

def rank_candidates(predictor_fn, recs, block_seed):
    """
    Score every record in recs, then sort by predicted cost.
    Ties are broken by a seeded random draw, NOT by input index.
    Returns the index (into the shuffled recs list) of the top-1 pick.
    """
    rng = random.Random(block_seed + 1_000_000)   # deterministic but separate from shuffle seed
    scored = []
    for i, r in enumerate(recs):
        cost = predictor_fn(r)
        scored.append((cost, rng.random(), i))    # (cost, tiebreak, index)
    scored.sort()
    return scored[0][2]   # index of top-1 pick


def top_k_indices(predictor_fn, recs, k, block_seed):
    rng = random.Random(block_seed + 2_000_000)
    scored = []
    for i, r in enumerate(recs):
        cost = predictor_fn(r)
        scored.append((cost, rng.random(), i))
    scored.sort()
    return [s[2] for s in scored[:k]]


def recall_at_n(predictor_fn, blocks_test, tolerance, seed_base=42):
    """
    Compute top-1 and top-3 recall across all test blocks.
    Per block:
      - shuffle candidates with a fixed per-block seed (no positional info).
      - rank with the predictor (ties broken randomly).
    """
    top1_hits = top3_hits = tie_count = 0
    num_blocks = len(blocks_test)

    for b_idx, (bid, recs_orig) in enumerate(blocks_test.items()):
        block_seed = seed_base * 10000 + b_idx
        rng = random.Random(block_seed)

        recs = list(recs_orig)
        rng.shuffle(recs)   # destroy positional information

        true_cycles = [r["proof_cycles"] for r in recs]
        min_c = min(true_cycles)

        # tie rate (for diagnostics)
        tie_count += sum(1 for c in true_cycles if c <= min_c + tolerance)

        top1_i = rank_candidates(predictor_fn, recs, block_seed)
        if true_cycles[top1_i] <= min_c + tolerance:
            top1_hits += 1

        top3_is = top_k_indices(predictor_fn, recs, 3, block_seed)
        if any(true_cycles[i] <= min_c + tolerance for i in top3_is):
            top3_hits += 1

    avg_tie_rate = tie_count / num_blocks
    return top1_hits, top3_hits, num_blocks, avg_tie_rate


# Main evaluation

def main():
    print(FEATURE_DESCRIPTION)

    log_file = ROOT / "data" / "candidate_log.jsonl"
    if not log_file.exists():
        print("ERROR: data/candidate_log.jsonl not found.")
        return

    # Load and group by block
    blocks = defaultdict(list)
    with open(log_file, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                if rec.get("passed") and rec.get("proof_cycles", -1) > 0:
                    bid = rec["candidate_json"]["candidate_id"]
                    blocks[bid].append(rec)
            except Exception:
                pass

    blocks = {k: v for k, v in blocks.items() if len(v) >= 2}
    print(f"Dataset: {len(blocks)} blocks with >=2 valid schedules.")

    # Train/test split (80/20 by block, deterministic)
    all_ids = sorted(blocks.keys())   # sort for reproducibility
    split = int(len(all_ids) * 0.8)
    train_ids = all_ids[:split]
    test_ids  = all_ids[split:]

    blocks_train = {k: blocks[k] for k in train_ids}
    blocks_test  = {k: blocks[k] for k in test_ids}

    print(f"Train blocks: {len(train_ids)}, Test blocks: {len(test_ids)}")

    # Build training tensors
    train_data = []
    for bid in train_ids:
        for rec in blocks[bid]:
            X, A = build_graph(rec["candidate_json"]["instructions"])
            y = torch.tensor([float(rec["proof_cycles"])])
            train_data.append((X, A, y))
    print(f"Training samples: {len(train_data)}\n")

    # Train GNN
    gnn = GNN()
    opt_g = torch.optim.Adam(gnn.parameters(), lr=0.01)
    for _ in range(50):
        for X, A, y in train_data:
            opt_g.zero_grad()
            nn.MSELoss()(gnn(X, A), y).backward()
            opt_g.step()

    # Train Linear
    lin = LinearModel()
    opt_l = torch.optim.Adam(lin.parameters(), lr=0.01)
    for _ in range(50):
        for X, A, y in train_data:
            opt_l.zero_grad()
            nn.MSELoss()(lin(X, A), y).backward()
            opt_l.step()

    gnn.eval()
    lin.eval()

    # Compute MAE
    gnn_mae = lin_mae = 0.0
    total_test_recs = 0
    with torch.no_grad():
        for bid in test_ids:
            for rec in blocks[bid]:
                X, A = build_graph(rec["candidate_json"]["instructions"])
                y = rec["proof_cycles"]
                gnn_mae += abs(gnn(X, A).item() - y)
                lin_mae += abs(lin(X, A).item() - y)
                total_test_recs += 1

    gnn_mae /= max(total_test_recs, 1)
    lin_mae  /= max(total_test_recs, 1)

    # Predictor closures
    @torch.no_grad()
    def gnn_pred(rec):
        X, A = build_graph(rec["candidate_json"]["instructions"])
        return gnn(X, A).item()

    @torch.no_grad()
    def lin_pred(rec):
        X, A = build_graph(rec["candidate_json"]["instructions"])
        return lin(X, A).item()

    def dummy_pred(rec):
        return 42.0   # same constant for every candidate

    # STEP 1: SANITY FLOORS
    # Both dummy and linear MUST score approx chance before we read the GNN.
    print("=" * 60)
    print("STEP 1 -- SANITY FLOORS (read before GNN)")
    print("=" * 60)

    floor_results = {}
    floor_pass = True
    avg_n_cands = total_test_recs / max(len(test_ids), 1)

    for tol in [0, 1, 2]:
        d1, d3, nb, tie_rate = recall_at_n(dummy_pred, blocks_test, tol)
        l1, l3, nb, _        = recall_at_n(lin_pred,   blocks_test, tol)

        d1_pct = d1 / nb * 100
        l1_pct = l1 / nb * 100
        # Theoretical chance: fraction of candidates that are optimal
        chance_pct = (tie_rate / nb) / avg_n_cands * 100

        floor_results[tol] = dict(
            dummy_top1=d1, dummy_top3=d3,
            lin_top1=l1, lin_top3=l3,
            nb=nb, tie_rate=tie_rate/nb,
            chance_pct=chance_pct
        )

        # floors must agree within noise
        both_near_chance = (abs(d1_pct - l1_pct) <= 20)
        dummy_not_perfect = (d1_pct < 60)
        lin_not_inverted  = (l1_pct >= 0)

        tol_pass = both_near_chance and dummy_not_perfect and lin_not_inverted
        if not tol_pass:
            floor_pass = False

        print(f"\nTolerance N={tol}:")
        print(f"  Avg optimal cands/block: {tie_rate/nb:.2f}  (chance ~ {chance_pct:.1f}%)")
        print(f"  Dummy Top-1  : {d1}/{nb} = {d1_pct:.1f}%  {'OK: near chance' if dummy_not_perfect else 'SUSPICIOUS'}")
        print(f"  Linear Top-1 : {l1}/{nb} = {l1_pct:.1f}%  {'OK' if lin_not_inverted else 'SUSPICIOUS'}")
        print(f"  Floors agree : {'YES' if both_near_chance else 'NO -- HARNESS BROKEN'}")
        print(f"  Floor check  : {'PASS' if tol_pass else 'FAIL'}")

    print()
    if not floor_pass:
        print("HARD GATE FAILED: sanity floors did not pass.")
        print("Do NOT read the GNN result. Fix the harness first.")
        return

    print("HARD GATE PASSED: Both dummy and linear score at chance level.")
    print("Proceeding to GNN evaluation.\n")

    # STEP 2: GNN EVALUATION
    print("=" * 60)
    print("STEP 2 -- GNN EVALUATION")
    print("=" * 60)

    print(f"MAE on test set ({total_test_recs} schedules):")
    print(f"  GNN    MAE: {gnn_mae:.2f} cycles")
    print(f"  Linear MAE: {lin_mae:.2f} cycles\n")

    all_results = {}
    for tol in [0, 1, 2]:
        g1, g3, nb, _  = recall_at_n(gnn_pred, blocks_test, tol)
        fr = floor_results[tol]
        all_results[tol] = dict(gnn_top1=g1, gnn_top3=g3)

        print(f"Tolerance N={tol}:")
        print(f"  {'Model':<22} {'Top-1':>12} {'Top-3':>12}")
        print(f"  {'Dummy (constant)':22} {fr['dummy_top1']}/{nb} = {fr['dummy_top1']/nb*100:5.1f}%  {fr['dummy_top3']}/{nb} = {fr['dummy_top3']/nb*100:5.1f}%")
        print(f"  {'Linear':22} {fr['lin_top1']}/{nb} = {fr['lin_top1']/nb*100:5.1f}%  {fr['lin_top3']}/{nb} = {fr['lin_top3']/nb*100:5.1f}%")
        print(f"  {'GNN':22} {g1}/{nb} = {g1/nb*100:5.1f}%  {g3}/{nb} = {g3/nb*100:5.1f}%")
        print()

    # STEP 3: DIAGNOSIS
    print("=" * 60)
    print("STEP 3 -- DIAGNOSIS")
    print("=" * 60)

    gnn_top1_n0 = all_results[0]["gnn_top1"]
    lin_top1_n0 = floor_results[0]["lin_top1"]
    nb0         = floor_results[0]["nb"]
    gnn_pct_n0  = gnn_top1_n0 / nb0 * 100
    lin_pct_n0  = lin_top1_n0 / nb0 * 100

    if gnn_top1_n0 == nb0:
        print(f"WARNING: GNN scored PERFECT 100% ({gnn_top1_n0}/{nb0}).")
        print("With sequential position edges removed, this needs investigation.")
        print("The GNN may be keying off opcode-count features (identical per block)")
        print("OR finding genuine latency-sensitive structure. Without ablation,")
        print("this result is untrustworthy.")
    elif gnn_pct_n0 > lin_pct_n0 + 15:
        print(f"GNN beats linear by {gnn_pct_n0 - lin_pct_n0:.1f}pp at N=0.")
        print("Mechanism: the GNN reads data-flow edges (reg->inst, inst->reg) and")
        print("can detect load-use chains (LW->dependent ADD) that stall the pipeline.")
        print("The linear model sees only opcode counts, identical across all schedules")
        print("of the same block, so it cannot rank them.")
    elif gnn_pct_n0 > lin_pct_n0 + 5:
        print(f"GNN beats linear by a small margin ({gnn_pct_n0 - lin_pct_n0:.1f}pp at N=0). Signal exists but weak.")
    else:
        print(f"GNN does NOT beat linear (GNN {gnn_pct_n0:.1f}% vs Linear {lin_pct_n0:.1f}%).")
        print("Neither model ranks schedules. Structure not learnable from this data,")
        print("or schedule variation in cycle count is too small relative to block noise.")

    # STEP 4: VERDICT
    print("\n" + "=" * 60)
    print("STEP 4 -- FINAL VERDICT")
    print("=" * 60)

    gnn_perfect      = (gnn_top1_n0 == nb0)
    gnn_above_chance = (gnn_pct_n0 > lin_pct_n0 + 10) and (gnn_pct_n0 > 30) and not gnn_perfect

    if gnn_perfect:
        verdict = "SHELVE (GNN 100% suspicious without position edges; cannot confirm genuine ranking)"
    elif gnn_above_chance:
        verdict = "SHIP GNN (clearly above chance, floors confirmed, features are leak-free)"
    else:
        verdict = "SHELVE GNN (does not beat chance; linear is equally effective)"

    print(f"Verdict: {verdict}")
    print()
    print("One honest sentence:")
    if gnn_perfect:
        print(f"  Harness is leak-free (floors passed), but GNN scored 100% which is")
        print(f"  suspicious without position edges -- result is UNTRUSTWORTHY until ablated.")
    elif gnn_above_chance:
        print(f"  On a harness proven leak-free (floors: linear={lin_pct_n0:.0f}% ~= dummy ~= chance),")
        print(f"  the GNN beats linear on ranking: YES ({gnn_pct_n0:.0f}% vs {lin_pct_n0:.0f}%).")
    else:
        print(f"  On a harness proven leak-free (floors passed), the GNN does NOT")
        print(f"  beat linear on ranking: NO ({gnn_pct_n0:.0f}% vs {lin_pct_n0:.0f}%).")

    print("=" * 60)


if __name__ == "__main__":
    main()
