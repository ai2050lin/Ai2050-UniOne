import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# --- 1. Model Definition (Same as before) ---
class Z113Dataset(Dataset):
    def __init__(self, n=113, size=5000):
        self.n = n
        self.size = size
        self.data = torch.randint(0, n, (size, 2))
        self.targets = (self.data[:, 0] + self.data[:, 1]) % n
    def __len__(self): return self.size
    def __getitem__(self, idx): return self.data[idx], self.targets[idx]

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.tf_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, x):
        emb = self.embed(x)
        out = self.tf_layer(emb)
        out = out.mean(dim=1)
        return self.head(out)

# --- 2. Train Helper ---
def train_and_get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = 113
    d_model = 64
    model = SimpleTransformer(n, d_model).to(device)
    dataset = Z113Dataset(n=n, size=8000)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("Training (Scanning) Structure...")
    for epoch in range(30):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
    return model, device

# --- 3. Analysis Tools ---

def visualize_frequency_circles(model, device):
    print("Visualizing Frequency Circles...")
    W_E = model.embed.weight.detach().cpu().numpy() # [113, d]
    n = 113
    
    # FFT to find dominant freqs
    fft = np.fft.fft(W_E, axis=0)
    power = np.abs(fft)
    avg_power = np.mean(power, axis=1)
    
    # Get top 3 freqs (ignore DC k=0)
    idxs = np.argsort(avg_power[:n//2+1])[::-1]
    top_k = [k for k in idxs if k != 0][:3]
    print(f"Top 3 Frequencies: {top_k}")
    
    # Project onto these frequencies
    # For a frequency k, we want to find the 2D subspace in W_E that correlates most with cos(kx) and sin(kx)
    # Simple way: Just look at the columns of W_E in the Frequency Domain? 
    # Or PCA the W_E but specifically looking for these cyclic patterns?
    
    # Better approach: Cosine/Sine Similarity Projection
    # Generate reference waves
    x = np.arange(n)
    save_dir = "tempdata"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    plt.figure(figsize=(15, 5))
    
    for i, k in enumerate(top_k):
        # Create reference basis for freq k
        # We want to project W_E onto the plane spanned by Fourier modes k
        # But W_E columns might be mixtures.
        # Let's just pick the two dimensions of W_E that have the highest power at k.
        
        # Power per dimension at freq k
        power_k = power[k, :] # [d_model]
        # Top 2 dims for this freq
        dims = np.argsort(power_k)[::-1][:2]
        
        dim1, dim2 = dims[0], dims[1]
        
        plt.subplot(1, 3, i+1)
        plt.scatter(W_E[:, dim1], W_E[:, dim2], c=np.arange(n), cmap='hsv', s=50, alpha=0.8)
        plt.title(f"Freq k={k} (Dims {dim1}, {dim2})")
        plt.colorbar(label="Token 0-112")
        plt.axis('equal')
        
    plt.suptitle("Embedding Projected on Dominant Frequency Dimensions")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/z113_circles.png")
    print(f"Saved circles to {save_dir}/z113_circles.png")

def analyze_readout_duality(model):
    print("Analyzing Embed-Unembed Duality...")
    W_E = model.embed.weight.detach().cpu().numpy()
    W_U = model.head.weight.detach().cpu().numpy() # [vocab, d] -> usually [d, vocab] in TF Lens but here Linear is [out, in] so [113, d]
    
    # Correlate W_E and W_U
    # In a perfect world, W_U should be similar to W_E (transposed/inverse)
    
    # Compute Cosine Similarity between W_E[i] and W_U[i]
    # Ideally W_E[i] \cdot W_U[j] should relate to (i+j) logic? No, W_U is reading out the result.
    # If the output is 'c', W_U[c] should align with the hidden state.
    
    # Let's simply check if W_U also has the same Fourier structure
    fft_U = np.fft.fft(W_U, axis=0)
    power_U = np.abs(fft_U).mean(axis=1)
    
    plt.figure(figsize=(10, 4))
    n = 113
    plt.plot(np.arange(n)[:n//2+1], power_U[:n//2+1], label='Unembedding (Readout)', color='orange')
    
    # Overlay Embedding power for comparison
    fft_E = np.fft.fft(W_E, axis=0)
    power_E = np.abs(fft_E).mean(axis=1)
    plt.plot(np.arange(n)[:n//2+1], power_E[:n//2+1], label='Embedding (Input)', color='blue', alpha=0.5, linestyle='--')
    
    plt.title("Fourier Spectrum Comparison: Input vs Output Matrix")
    plt.legend()
    plt.grid(True)
    plt.savefig("tempdata/z113_duality.png")
    print("Saved duality plot.")

if __name__ == "__main__":
    model, device = train_and_get_model()
    visualize_frequency_circles(model, device)
    analyze_readout_duality(model)
