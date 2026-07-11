import json
import torch
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))
from gnn_model import SimpleGCN, LinearBaseline, process_candidate_to_graph

def evaluate(tolerance=2):
    log_file = ROOT / "data" / "candidate_log.jsonl"
    if not log_file.exists():
        print("No log file found!")
        return

    # Group records by block (candidate_id contains "real_b{block_idx}")
    # candidate_id format: "real_b{block_idx}"
    blocks = defaultdict(list)
    
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                if record.get("passed") and record.get("proof_cycles", -1) > 0:
                    cand_id = record["candidate_json"]["candidate_id"]
                    blocks[cand_id].append(record)
            except Exception:
                pass
                
    if len(blocks) < 2:
        print("Not enough blocks found for evaluation.")
        return
        
    print(f"Loaded {len(blocks)} blocks for evaluation.")
    
    # Filter blocks with fewer than 2 candidates to make top-1 meaningful
    blocks = {k: v for k, v in blocks.items() if len(v) >= 2}
    print(f"Filtered to {len(blocks)} blocks with >= 2 valid schedules.")
    
    # Train-test split by block
    block_ids = list(blocks.keys())
    split_idx = int(len(block_ids) * 0.8)
    train_ids = block_ids[:split_idx]
    test_ids = block_ids[split_idx:]
    
    # Prepare training data
    train_data = []
    for bid in train_ids:
        for rec in blocks[bid]:
            X, A = process_candidate_to_graph(rec["candidate_json"]["instructions"])
            y = torch.tensor([float(rec["proof_cycles"])])
            train_data.append((X, A, y))
            
    print(f"Training on {len(train_data)} records...")
    
    # Train GCN
    gcn = SimpleGCN()
    optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01)
    for epoch in range(50):
        for X, A, y in train_data:
            optimizer_gcn.zero_grad()
            pred = gcn(X, A)
            loss = torch.nn.MSELoss()(pred, y)
            loss.backward()
            optimizer_gcn.step()
            
    # Train Linear
    linear = LinearBaseline()
    optimizer_lin = torch.optim.Adam(linear.parameters(), lr=0.01)
    for epoch in range(50):
        for X, A, y in train_data:
            optimizer_lin.zero_grad()
            pred = linear(X, A)
            loss = torch.nn.MSELoss()(pred, y)
            loss.backward()
            optimizer_lin.step()
            
    # Evaluation
    gcn.eval()
    linear.eval()
    
    # Calculate Train MAE
    gcn_train_err = 0
    lin_train_err = 0
    with torch.no_grad():
        for X, A, y in train_data:
            gcn_train_err += abs(gcn(X, A).item() - y.item())
            lin_train_err += abs(linear(X, A).item() - y.item())
            
    print(f"Train MAE - GNN: {gcn_train_err / len(train_data):.2f}, Linear: {lin_train_err / len(train_data):.2f}")
    
    gcn_err = 0
    lin_err = 0
    total_test_records = 0
    
    import random
    random.seed(42)
    
    gcn_err = 0
    lin_err = 0
    total_test_records = 0
    
    num_test_blocks = len(test_ids)
    print("\n" + "="*50)
    print("PHASE 3 EVALUATION RESULTS")
    print("="*50)
    
    # Calculate MAE
    for bid in test_ids:
        recs = blocks[bid]
        total_test_records += len(recs)
        for r in recs:
            X, A = process_candidate_to_graph(r["candidate_json"]["instructions"])
            y_true = r["proof_cycles"]
            with torch.no_grad():
                gcn_err += abs(gcn(X, A).item() - y_true)
                lin_err += abs(linear(X, A).item() - y_true)
                
    print(f"Test Set: {num_test_blocks} blocks, {total_test_records} total candidate schedules")
    print(f"Average candidates per block: {total_test_records / num_test_blocks:.1f}")
    print(f"GNN MAE: {gcn_err / total_test_records:.2f}")
    print(f"Linear MAE: {lin_err / total_test_records:.2f}\n")
    
    # Check metrics for multiple tolerances
    for tolerance in [0, 1, 2]:
        gcn_top1_hits = 0
        gcn_top3_hits = 0
        lin_top1_hits = 0
        lin_top3_hits = 0
        dummy_top1_hits = 0
        ties_count = 0
        
        for bid in test_ids:
            recs = list(blocks[bid])
            # SHUFFLE to destroy the input-order leak (the baseline BFS order)
            random.shuffle(recs)
            
            true_cycles = [r["proof_cycles"] for r in recs]
            min_cycle = min(true_cycles)
            
            optimal_cands = sum(1 for c in true_cycles if c <= min_cycle + tolerance)
            ties_count += optimal_cands
            
            gcn_preds = []
            lin_preds = []
            dummy_preds = []
            with torch.no_grad():
                for i, r in enumerate(recs):
                    X, A = process_candidate_to_graph(r["candidate_json"]["instructions"])
                    gcn_preds.append((gcn(X, A).item(), i))
                    lin_preds.append((linear(X, A).item(), i))
                    dummy_preds.append((42.0, i)) # Constant output dummy
                    
            # Sort by predicted cycles. Tie-break randomly to eliminate Python stable sort leak.
            gcn_preds.sort(key=lambda x: (x[0], random.random()))
            lin_preds.sort(key=lambda x: (x[0], random.random()))
            dummy_preds.sort(key=lambda x: (x[0], random.random()))
            
            if true_cycles[gcn_preds[0][1]] <= min_cycle + tolerance:
                gcn_top1_hits += 1
            if true_cycles[lin_preds[0][1]] <= min_cycle + tolerance:
                lin_top1_hits += 1
            if true_cycles[dummy_preds[0][1]] <= min_cycle + tolerance:
                dummy_top1_hits += 1
                
            if any(true_cycles[idx] <= min_cycle + tolerance for _, idx in gcn_preds[:3]):
                gcn_top3_hits += 1
            if any(true_cycles[idx] <= min_cycle + tolerance for _, idx in lin_preds[:3]):
                lin_top3_hits += 1
                
        print(f"--- Metrics at Tolerance N={tolerance} ---")
        print(f"Average tie rate: {ties_count / num_test_blocks:.1f} per block")
        print(f"Top-1 Recall - GNN: {gcn_top1_hits}/{num_test_blocks} ({gcn_top1_hits/num_test_blocks*100:.1f}%), Linear: {lin_top1_hits}/{num_test_blocks} ({lin_top1_hits/num_test_blocks*100:.1f}%), Dummy Constant: {dummy_top1_hits}/{num_test_blocks} ({dummy_top1_hits/num_test_blocks*100:.1f}%)")
        print(f"Top-3 Recall - GNN: {gcn_top3_hits}/{num_test_blocks} ({gcn_top3_hits/num_test_blocks*100:.1f}%), Linear: {lin_top3_hits}/{num_test_blocks} ({lin_top3_hits/num_test_blocks*100:.1f}%)\n")
        
        if tolerance == 0:
            rec_gcn_top1 = gcn_top1_hits
            rec_lin_top1 = lin_top1_hits
            rec_gcn_top3 = gcn_top3_hits
            rec_lin_top3 = lin_top3_hits
            
    print("--- Recommendation ---")
    if rec_gcn_top1 > rec_lin_top1 or (rec_gcn_top1 == rec_lin_top1 and rec_gcn_top3 > rec_lin_top3):
        print("SHIP GNN: The graph model meaningfully outperforms the linear baseline on recall metrics at strict tolerance.")
    else:
        print("SHELVE GNN: The graph model fails to beat the simpler linear opcode-count baseline. Ship the linear cost model instead.")
    print("="*50)
    
if __name__ == "__main__":
    evaluate()
