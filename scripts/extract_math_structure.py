
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# --- 1. The Fossil: Miniature Model for Z113 Group ---
# We train an extremely simple Multi-Layer Perceptron to learn Z_113 addition: (a + b) % 113.
# The underlying truth is a torus/circle, representing modulo arithmetic.
P = 113
EMBED_DIM = 256
HIDDEN_DIM = 512

class Z113Fossil(nn.Module):
    def __init__(self):
        super().__init__()
        # E lookup table maps {0...112} to EMBED_DIM vectors
        self.E = nn.Embedding(P, EMBED_DIM)
        # MLP tries to learn the association a+b -> c
        self.W_in = nn.Linear(EMBED_DIM * 2, HIDDEN_DIM)
        self.W_out = nn.Linear(HIDDEN_DIM, P)
        
    def forward(self, a, b):
        e_a = self.E(a)
        e_b = self.E(b)
        x = torch.cat([e_a, e_b], dim=-1)
        h = torch.relu(self.W_in(x))
        return self.W_out(h), h

# --- 2. Quick Training to create the Fossil ---
def breed_fossil(epochs=2000):
    print(f"[*] Breeding Z113 Fossil (Modulo {P} Arithmetic)...")
    model = Z113Fossil()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) # weight decay encourages compact geometry
    
    # Full dataset: all pairs (a, b)
    A, B = torch.meshgrid(torch.arange(P), torch.arange(P), indexing='ij')
    A, B = A.flatten(), B.flatten()
    C = (A + B) % P
    
    dataset_size = len(A)
    batch_size = 1024
    
    loss_fn = nn.CrossEntropyLoss()
    
    for ep in range(epochs):
        perm = torch.randperm(dataset_size)
        A, B, C = A[perm], B[perm], C[perm]
        
        total_loss = 0
        for i in range(0, dataset_size, batch_size):
            a_batch = A[i:i+batch_size]
            b_batch = B[i:i+batch_size]
            c_batch = C[i:i+batch_size]
            
            logits, _ = model(a_batch, b_batch)
            loss = loss_fn(logits, c_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if ep % 500 == 0 or ep == epochs - 1:
            print(f"  Ep {ep} | Loss: {total_loss:.4f}")
            
    return model

# --- 3. The Extraction Probes ---
def run_spectral_probe(model):
    print("\n[*] Initializing Spectral Orthogonal Probe...")
    # Extract the Embedding matrix: Shape [113, 256]
    E_mat = model.E.weight.detach().numpy()
    
    # Centering (optional but standard for PCA)
    E_mat -= np.mean(E_mat, axis=0)
    
    # 1. Singular Value Decomposition (SVD)
    # E_mat = U * S * V^T
    U, S, Vt = np.linalg.svd(E_mat, full_matrices=False)
    
    # Calculate explained variance
    variance_explained = (S ** 2) / np.sum(S ** 2)
    cumulative_variance = np.cumsum(variance_explained)
    
    print("  Top 10 Singular Values:", S[:10])
    print("  Top 10 Variance Explained:", variance_explained[:10])
    
    # Identify Intrinsic Dimension (number of components needed to explain 95% variance)
    d_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"  [!] Intrinsic Dimension (95% variance): {d_95} / {EMBED_DIM}")
    
    # 2. Extracting the "Fourier-like" Orthogonal Basis
    # Modulo arithmetic should theoretically map to a 2D circle in high-dim space (sine and cosine waves).
    # Let's see if the first two principal components form a circle.
    PC1 = U[:, 0] * S[0]
    PC2 = U[:, 1] * S[1]
    
    # Save the plot
    os.makedirs("tempdata", exist_ok=True)
    plt.figure(figsize=(8,8))
    plt.scatter(PC1, PC2, c=np.arange(P), cmap='hsv')
    plt.title(f"Z_{P} Mathematical Structure Extracted (Top 2 PCs)")
    for i in range(P):
        if i % 10 == 0:
            plt.annotate(str(i), (PC1[i], PC2[i]))
    
    save_path = "tempdata/z113_structure_extracted.png"
    plt.savefig(save_path)
    print(f"  [+] Orthogonal geometric structure saved to {save_path}")
    
    return S, d_95

if __name__ == "__main__":
    fossil = breed_fossil(epochs=1000)
    S, d_95 = run_spectral_probe(fossil)
    
    report = {
        "dataset": "Z113 Modulo Addition",
        "embedding_dim": EMBED_DIM,
        "intrinsic_dimension_95_var": int(d_95),
        "note": "Mathematical extraction successful."
    }
    with open("tempdata/phase8_extraction_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n[+] Phase 8 Extraction initialized and complete.")
