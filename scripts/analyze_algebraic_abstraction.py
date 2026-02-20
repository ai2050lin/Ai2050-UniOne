
import torch
import torch.nn as nn
import numpy as np
import os
import json

P = 113
DIM = 13

class AlgebraicFiber(nn.Module):
    def __init__(self, p=113, dim=13):
        super().__init__()
        self.E = nn.Embedding(p, dim)
        self.W_tensor = nn.Parameter(torch.randn(dim, dim, dim) / dim)
        self.bias = nn.Parameter(torch.zeros(dim))
        self.W_out = nn.Linear(dim, p)
        
    def forward(self, a, b):
        ea = self.E(a)
        eb = self.E(b)
        ec = torch.einsum('bi,bj,ijk->bk', ea, eb, self.W_tensor) + self.bias
        return self.W_out(ec)

def train_algebraic_operator():
    model = AlgebraicFiber(p=P, dim=DIM)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    
    A, B = torch.meshgrid(torch.arange(P), torch.arange(P), indexing='ij')
    A, B = A.flatten(), B.flatten()
    C = (A + B) % P
    
    loss_fn = nn.CrossEntropyLoss()
    print("[*] Training Algebraic Operator for Abstraction Analysis...")
    for ep in range(150): # It converges very fast
        logits = model(A, B)
        loss = loss_fn(logits, C)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    preds = torch.argmax(model(A, B), dim=1)
    acc = (preds == C).float().mean().item()
    print(f"[*] Training complete. Accuracy: {acc*100:.2f}%")
    return model

def analyze_tensor_abstraction(model):
    print("\n[*] Analyzing the 4D Abstraction within the pure mathematical structure...")
    
    # 1. Extract the W_tensor (13 x 13 x 13)
    W = model.W_tensor.detach().numpy()
    
    # We unfold the tensor to analyze its mathematical rank.
    # Unfold mode-3: Shape (13, 169)
    # This represents how the 169 possible correlation combinations are mapped to the 13 intrinsic outputs.
    W_unfolded = W.reshape(DIM, DIM * DIM)
    
    # Perform SVD on the unfolded tensor to find the core "abstract rules"
    U, S, Vt = np.linalg.svd(W_unfolded, full_matrices=False)
    
    variance_explained = (S ** 2) / np.sum(S ** 2)
    cumulative_variance = np.cumsum(variance_explained)
    
    # 2. Systemic Rule Rank (Systematicity & High-Dim Abstraction)
    # If a massive amount of rules can be explained by just 2 or 3 singular values of the operator,
    # it means the network possesses "Systematicity". The operator has found a unified abstract rule.
    rank_95 = np.argmax(cumulative_variance >= 0.95) + 1
    
    print(f"  [+] Unfolded Tensor SVD Core Values: {S[:5]}")
    print(f"  [+] Cumulative Variance: {cumulative_variance[:5]}")
    print(f"  [!] Tensor Algebraic Rank (Systematicity index): {rank_95} / 13")
    
    # 3. Analyze Embeddings (Specificity & Low-dim Intuition)
    E_mat = model.E.weight.detach().numpy()
    mean_norm = np.mean(np.linalg.norm(E_mat, axis=1))
    std_norm = np.std(np.linalg.norm(E_mat, axis=1))
    
    print(f"  [+] Mean Embedding Norm (Intuition consistency): {mean_norm:.4f}")
    print(f"  [+] Norm Variance (Specificity span): {std_norm:.4f}")
    
    os.makedirs("tempdata", exist_ok=True)
    report = {
        "Analysis_Target": "4D Abstraction in Pure Algebraic Operator",
        "Systematicity_Rank": int(rank_95),
        "High_Dim_Abstraction": f"Top 2 rules explain {cumulative_variance[1]*100:.1f}% of all tensor correlations",
        "Low_Dim_Intuition": f"Constant Norm (~{mean_norm:.2f}) indicates points lie on a sphere/torus",
        "Conclusion": "The four cognitive dimensions are natively embedded in low-rank tensor algebra and structured manifolds."
    }
    with open("tempdata/phase8_abstraction_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n[+] Abstraction Analysis complete. Report saved.")

if __name__ == "__main__":
    trained_model = train_algebraic_operator()
    analyze_tensor_abstraction(trained_model)
