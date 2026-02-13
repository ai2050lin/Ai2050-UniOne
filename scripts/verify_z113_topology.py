import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

# --- 1. Define Model & Dataset (Minimal Version) ---

class Z113Dataset(Dataset):
    def __init__(self, n=113, size=5000):
        self.n = n
        self.size = size
        self.data = torch.randint(0, n, (size, 2))
        self.targets = (self.data[:, 0] + self.data[:, 1]) % n

    def __len__(self): return self.size
    def __getitem__(self, idx): return self.data[idx], self.targets[idx]

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32): # Small d_model is enough for Z_113
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        # No pos encoding needed for a+b vs b+a, but let's keep it simple
        self.blocks = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        # We need a bilinear interaction to simulate addition in embedding space
        # Standard Transformer has Attention (Bilinear). 
        # Here we use a simplified Attention-like mechanism or just specific MLP
        # To really check if it learns modular arithmetic, let's use a standard 1-layer Transformer
        self.tf_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size, bias=False) # Tied weights usually help
        
    def forward(self, x):
        # x: [batch, 2]
        emb = self.embed(x) # [batch, 2, d]
        out = self.tf_layer(emb)
        # Mean pooling
        out = out.mean(dim=1)
        return self.head(out)

# --- 2. Train Function ---

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    n = 113
    d_model = 64
    model = SimpleTransformer(n, d_model).to(device)
    dataset = Z113Dataset(n=n, size=8000)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4) # Added weight decay for regularization
    criterion = nn.CrossEntropyLoss()
    
    print("Training SimpleTransformer on Z_113...")
    for epoch in range(40): # Train a bit longer to ensure structure converges
        total_loss = 0
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}, Acc {correct/total:.2%}")
            
    return model

# --- 3. Analysis: Topology Extraction ---

def analyze_topology(model):
    print("\n--- Analysing Topology ---")
    model.eval()
    
    # 1. Extract Embeddings
    # E: [113, d_model]
    W_E = model.embed.weight.detach().cpu().numpy()
    
    # 2. PCA
    pca = PCA(n_components=4) # Check first 4 components
    pca_res = pca.fit_transform(W_E)
    
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
    
    # Plot PCA 1 vs 2
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(pca_res[:, 0], pca_res[:, 1], c=np.arange(113), cmap='hsv', s=50)
    for i in [0, 20, 40, 60, 80, 100]:
        plt.text(pca_res[i, 0], pca_res[i, 1], str(i), fontsize=9)
    plt.title("Embedding PCA (Dim 1 vs 2)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Token ID (0-112)")
    
    # Plot PCA 3 vs 4
    plt.subplot(1, 2, 2)
    plt.scatter(pca_res[:, 2], pca_res[:, 3], c=np.arange(113), cmap='hsv', s=50)
    plt.title("Embedding PCA (Dim 3 vs 4)")
    plt.xlabel("PC3")
    plt.ylabel("PC4")
    
    save_path = "tempdata/z113_topology_pca.png"
    if not os.path.exists("tempdata"): os.makedirs("tempdata")
    plt.savefig(save_path)
    print(f"Saved PCA plot to {save_path}")

    # 3. Fourier Analysis
    # Check if dimensions correspond to sin/cos frequencies
    # For Z_n, expected representation is cos(2*pi*k*i/n) and sin(...)
    
    print("Performing Fourier Analysis on Embedding Dimensions...")
    n = 113
    x = np.arange(n)
    fft_spectrum = np.fft.fft(W_E, axis=0) # [n, d_model]
    power_spectrum = np.abs(fft_spectrum)
    
    # Average power across all embedding dimensions
    avg_power = np.mean(power_spectrum, axis=1) # [n]
    
    # Plot Power Spectrum
    plt.figure(figsize=(10, 4))
    # Only plot 0 to n//2 (Nyquist)
    plt.plot(np.arange(n)[:n//2+1], avg_power[:n//2+1])
    plt.title("Fourier Power Spectrum of Embeddings (Frequency Domain)")
    plt.xlabel("Frequency k")
    plt.ylabel("Avg Power")
    plt.grid(True)
    
    save_path_fft = "tempdata/z113_topology_fft.png"
    plt.savefig(save_path_fft)
    print(f"Saved Fourier plot to {save_path_fft}")
    
    # Check for dominant frequencies
    dominant_freqs = np.argsort(avg_power[:n//2+1])[::-1][:5]
    print(f"Top 5 Dominant Frequencies (k): {dominant_freqs}")

if __name__ == "__main__":
    model = train_model()
    analyze_topology(model)
