import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer, HookedTransformerConfig
import os

# Settings
P = 113
D_MODEL = 128
DEVICE = "cpu"

def analyze():
    print("Starting Post-Grokking Fourier Analysis...")
    
    # 1. Reconstruct config
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=D_MODEL,
        d_head=32,
        n_heads=4,
        d_mlp=512,
        d_vocab=P + 1,
        n_ctx=2,
        act_fn="relu",
        normalization_type=None,
        device=DEVICE
    )
    model = HookedTransformer(cfg)
    
    # 2. Find latest checkpoint
    import glob
    checkpoints = glob.glob("tests/gemini/data/model_grokking_*.pt")
    final_model = "tests/gemini/data/model_final.pt"
    if os.path.exists(final_model):
        latest_ckpt = final_model
    elif checkpoints:
        latest_ckpt = max(checkpoints, key=os.path.getmtime)
    else:
        print("No checkpoint found. Please ensure training finished with grokking detected.")
        return
    print(f"Loading checkpoint: {latest_ckpt}")
    model.load_state_dict(torch.load(latest_ckpt, map_location=DEVICE))
    model.eval()
    
    # 3. Analyze Embeddings (W_E)
    W_E = model.W_E[:P, :].detach().cpu().numpy() # [P, D_MODEL]
    
    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    W_E_pca = pca.fit_transform(W_E)
    
    print(f"PCA Variance Explained: {pca.explained_variance_ratio_}")
    
    # Fourier Transform on Embeddings
    # Modular addition models usually learn k*2pi/p frequencies.
    # W_E is [P, D_MODEL]. We take DFT along the vocabulary axis (P).
    W_E_fft = np.fft.rfft(W_E, axis=0) # [P//2+1, D_MODEL]
    W_E_fft_power = np.abs(W_E_fft)**2
    total_power_per_freq = W_E_fft_power.sum(axis=1)
    
    # Find top frequencies
    top_freqs = np.argsort(total_power_per_freq)[-6:][::-1]
    print(f"Top Frequencies (k): {top_freqs}")
    
    results = {
        "pca_variance": pca.explained_variance_ratio_.tolist(),
        "top_freqs": top_freqs.tolist(),
        "total_power": total_power_per_freq.tolist()
    }
    
    import json
    with open("tests/gemini/data/grokking_analysis_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("Analysis complete. Results saved to tests/gemini/data/grokking_analysis_results.json")

    # Plot PCA 1 vs 2 (expect circle)
    plt.figure(figsize=(8,8))
    plt.scatter(W_E_pca[:, 0], W_E_pca[:, 1], c=range(P), cmap='hsv')
    plt.title("W_E PCA 1 vs 2 (Modular Addition Circularity)")
    plt.colorbar(label='Number (0..P-1)')
    plt.savefig("tests/gemini/data/grokking_pca_circle.png")
    print("PCA plot saved to tests/gemini/data/grokking_pca_circle.png")

if __name__ == "__main__":
    analyze()
