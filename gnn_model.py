import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent

OPCODES = ["ADD", "SUB", "MUL", "ADDI", "SLLI", "LW", "SW", "BNE", "BEQ", "LABEL"]
OP_TO_IDX = {op: i for i, op in enumerate(OPCODES)}

def process_candidate_to_graph(candidate_insts):
    # Nodes: instructions + registers
    # We will assign indices: 0..N-1 for instructions, N..N+31 for physical registers r0-r31
    # Features per node: [is_inst, is_reg, opcode_onehot (10), has_imm, imm_mag, read_flag, write_flag] -> total 15 dims
    
    num_insts = len(candidate_insts)
    num_regs = 32
    total_nodes = num_insts + num_regs
    
    X = torch.zeros((total_nodes, 16))
    
    # Init register nodes
    for r in range(32):
        idx = num_insts + r
        X[idx, 1] = 1.0 # is_reg
    
    edges = []
    
    for i, inst in enumerate(candidate_insts):
        X[i, 0] = 1.0 # is_inst
        op = inst.get("op", "").upper()
        if op in OP_TO_IDX:
            X[i, 2 + OP_TO_IDX[op]] = 1.0
            
        if "imm" in inst:
            X[i, 12] = 1.0
            X[i, 13] = float(inst["imm"]) / 1000.0 # simple magnitude bucket
            
        # Edges
        # src -> inst (read)
        for field in ["src1", "src2", "base"]:
            val = inst.get(field)
            if isinstance(val, str) and val.startswith("r"):
                try:
                    r_idx = int(val[1:])
                    # edge: reg -> inst
                    edges.append([num_insts + r_idx, i])
                    X[num_insts + r_idx, 14] = 1.0 # read_flag
                except:
                    pass
                    
        # inst -> dst (write)
        dst = inst.get("dst")
        if isinstance(dst, str) and dst.startswith("r"):
            try:
                r_idx = int(dst[1:])
                # edge: inst -> reg
                edges.append([i, num_insts + r_idx])
                X[num_insts + r_idx, 15] = 1.0 # write_flag
            except:
                pass
                
        # Instruction ordering (sequential)
        if i > 0:
            edges.append([i-1, i])
            
    if len(edges) == 0:
        edges = [[0, 0]]
        
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Build Adjacency matrix for simple GCN
    A = torch.zeros((total_nodes, total_nodes))
    for e in edges:
        A[e[0], e[1]] = 1.0
        # self loops
    for i in range(total_nodes):
        A[i, i] = 1.0
        
    # Row normalize A
    rowsum = A.sum(dim=1, keepdim=True)
    A = A / torch.clamp(rowsum, min=1.0)
        
    return X, A

class SimpleGCN(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, X, A):
        # Layer 1
        H1 = torch.relu(self.W1(torch.matmul(A, X)))
        # Layer 2
        H2 = torch.relu(self.W2(torch.matmul(A, H1)))
        # Graph pooling (mean of instruction nodes only)
        # The first N nodes are instructions. But we can just mean pool all nodes with is_inst==1
        inst_mask = (X[:, 0] == 1.0).float().unsqueeze(1)
        sum_h = (H2 * inst_mask).sum(dim=0)
        num_inst = inst_mask.sum().clamp(min=1.0)
        pooled = sum_h / num_inst
        
        return self.fc(pooled)
        
class LinearBaseline(nn.Module):
    def __init__(self, in_dim=16):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
        
    def forward(self, X, A):
        # Use sum of features (opcode counts), not mean!
        # This aligns the features with the target (total block cycles).
        inst_mask = (X[:, 0] == 1.0).float().unsqueeze(1)
        sum_x = (X * inst_mask).sum(dim=0)
        return self.fc(sum_x)

def train():
    log_file = ROOT / "data" / "candidate_log.jsonl"
    if not log_file.exists():
        print("No log file found!")
        return
        
    data = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            # We only train on candidates that passed (otherwise cycles is meaningless for absolute regression)
            if record.get("passed") and record.get("proof_cycles", -1) > 0:
                X, A = process_candidate_to_graph(record["candidate_json"]["instructions"])
                y = torch.tensor([float(record["proof_cycles"])])
                data.append((X, A, y))
                
    if len(data) < 5:
        print(f"Not enough passing records to train ({len(data)}). Need volume!")
        return
        
    print(f"Loaded {len(data)} graph records for training.")
    
    # Train test split (80/20)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]
    
    # 1. Train GCN
    gcn = SimpleGCN()
    optimizer_gcn = optim.Adam(gcn.parameters(), lr=0.01)
    
    print("Training GNN...")
    for epoch in range(50):
        total_loss = 0
        for X, A, y in train_data:
            optimizer_gcn.zero_grad()
            pred = gcn(X, A)
            loss = nn.MSELoss()(pred, y)
            loss.backward()
            optimizer_gcn.step()
            total_loss += loss.item()
            
    # 2. Train Linear
    linear = LinearBaseline()
    optimizer_lin = optim.Adam(linear.parameters(), lr=0.01)
    
    print("Training Linear Baseline...")
    for epoch in range(50):
        total_loss = 0
        for X, A, y in train_data:
            optimizer_lin.zero_grad()
            pred = linear(X, A)
            loss = nn.MSELoss()(pred, y)
            loss.backward()
            optimizer_lin.step()
            total_loss += loss.item()
            
    # Evaluate
    gcn.eval()
    linear.eval()
    
    gcn_err = 0
    lin_err = 0
    with torch.no_grad():
        for X, A, y in test_data:
            pred_gcn = gcn(X, A)
            pred_lin = linear(X, A)
            gcn_err += abs(pred_gcn.item() - y.item())
            lin_err += abs(pred_lin.item() - y.item())
            
    print(f"GNN Mean Absolute Error on Test: {gcn_err / len(test_data):.2f} cycles")
    print(f"Linear Baseline MAE on Test: {lin_err / len(test_data):.2f} cycles")
    
    if gcn_err < lin_err:
        print("Conclusion: GNN successfully leverages topology to beat linear opcode counts!")
    else:
        print("Conclusion: Linear baseline is nearly as good; graph topology provides minimal gain on this dataset.")
        
    # Save the GNN model so it can be used for ranking
    torch.save(gcn.state_dict(), ROOT / "data" / "gnn_model.pt")
    print("GNN saved to data/gnn_model.pt")
    
if __name__ == "__main__":
    train()
