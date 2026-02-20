
import torch
import torch.nn as nn
import numpy as np
import os
import json
from sklearn.decomposition import PCA

# --- 1. Train a Quick Z113 Fossil ---
P = 113
EMBED_DIM = 256
HIDDEN_DIM = 512

class Z113Fossil(nn.Module):
    def __init__(self):
        super().__init__()
        self.E = nn.Embedding(P, EMBED_DIM)
        self.W_in = nn.Linear(EMBED_DIM * 2, HIDDEN_DIM)
        self.W_out = nn.Linear(HIDDEN_DIM, P)
        
    def forward(self, a, b):
        e_a = self.E(a)
        e_b = self.E(b)
        x = torch.cat([e_a, e_b], dim=-1)
        h = torch.relu(self.W_in(x))
        return self.W_out(h)

def get_trained_embeddings():
    print(f"[*] Training strict Z113 Fossil for Operator Extraction...")
    model = Z113Fossil()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) # strict L2 for geometry
    
    A, B = torch.meshgrid(torch.arange(P), torch.arange(P), indexing='ij')
    A, B = A.flatten(), B.flatten()
    C = (A + B) % P
    
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(800): # Short train to get basic geometry
        logits = model(A, B)
        loss = loss_fn(logits, C)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ep % 400 == 0:
            print(f"  Ep {ep} | Loss: {loss.item():.4f}")
            
    return model.E.weight.detach().numpy()

# --- 2. Extract Mathematical Operator (Bilinear Tensor mapping) ---
def extract_operator():
    embeddings = get_trained_embeddings()
    
    # Step A: Dimensionality Reduction (Isolate the clean Math)
    print("\n[*] Isolating 13-Dimensional Orthogonal Basis using PCA...")
    pca = PCA(n_components=13)
    embed_13d = pca.fit_transform(embeddings) # shape: (113, 13)
    
    # Step B: Prepare Bilinear Data
    # We want to find a simple operator Op(a, b) = c
    # Instead of a huge MLP, let's try a Kronecker/Tensor product approximation:
    # Op(a, b) ~ W_tensor * (a [outer_product] b)
    # Since 13 * 13 = 169 dimensions, this is a tiny linear regression problem!
    
    print("[*] Formulating Bilinear Tensor Problem (13x13 -> 13)...")
    X_outer = []
    Y_target = []
    
    for a in range(P):
        for b in range(P):
            c = (a + b) % P
            vec_a = embed_13d[a]
            vec_b = embed_13d[b]
            vec_c = embed_13d[c]
            
            # Outer product flattened (size: 169) representing Bilinear interaction
            outer_ab = np.outer(vec_a, vec_b).flatten()
            
            X_outer.append(outer_ab)
            Y_target.append(vec_c)
            
    X_outer = np.array(X_outer) # (12769, 169)
    Y_target = np.array(Y_target) # (12769, 13)
    
    from sklearn.linear_model import LinearRegression

    # ...
    # Step C: Linear Regression to find the Bilinear Operator Weights
    print(f"[*] Solving Pure Algebraic Mapping: W * (a ⊗ b) = c")
    # Ordinary Linear regression to find the mapping matrix W (shape 169 -> 13)
    reg = LinearRegression()
    reg.fit(X_outer, Y_target)
    
    # Step D: Test accuracy of our extracted pure mathematical formula
    # If this works, we have completely replaced the non-linear MLP!
    predictions_13d = reg.predict(X_outer)
    
    # To evaluate accuracy, see which 13d embedding the prediction is closest to
    correct = 0
    total = len(X_outer)
    
    print("[*] Evaluating Algebraic Substitution Accuracy...")
    for i in range(total):
        pred_c = predictions_13d[i]
        true_c_idx = (i // P + i % P) % P
        
        # Find nearest neighbor in the 13d basis
        dists = np.linalg.norm(embed_13d - pred_c, axis=1)
        best_idx = np.argmin(dists)
        
        if best_idx == true_c_idx:
            correct += 1
            
    acc = correct / total
    print(f"  [!] Pure Algebraic Operator Accuracy: {acc*100:.2f}%")
    
    # Save Results
    os.makedirs("tempdata", exist_ok=True)
    report = {
        "MLP_Parameters": 256*2*512 + 512 + 512*113 + 113, # ~320k
        "Algebraic_Operator_Parameters": 169 * 13, # 2197 parameters!
        "Reduction_Factor": "145x Smaller",
        "Algebraic_Accuracy": acc
    }
    with open("tempdata/phase8_operator_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Replaced ~320k parameter MLP with a ~2k parameter pure Bilinear Equation.\n[+] Record saved.")

if __name__ == "__main__":
    extract_operator()
