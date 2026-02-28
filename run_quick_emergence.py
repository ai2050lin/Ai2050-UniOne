"""
Feature Emergence Tracking - Quick Demo

Runtime: ~1 minute
Goal: Demonstrate tracking method, get quick results
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import time


class MiniTransformer(nn.Module):
    """Minimal Transformer for quick demo"""
    
    def __init__(self, vocab_size=100, d_model=32, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(n_layers)
        ])
        self.unembed = nn.Linear(d_model, vocab_size)
        self.activations = {}
    
    def forward(self, x):
        x = self.embed(x)
        for i, layer in enumerate(self.layers):
            x = x + layer(x)
            self.activations[i] = x.detach()
        return self.unembed(x)


def quick_emergence_tracking():
    """Quick emergence tracking demo"""
    print("=" * 60)
    print("Feature Emergence Tracking - Quick Demo")
    print("=" * 60)
    print()
    
    # Config
    vocab_size, d_model, n_layers = 100, 32, 2
    steps = 500
    save_interval = 50
    
    print(f"Model: {n_layers} layers, {d_model} dims")
    print(f"Training: {steps} steps")
    print()
    
    # Create model
    model = MiniTransformer(vocab_size, d_model, n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Tracking data
    emergence_history = defaultdict(dict)
    
    print("Training and tracking feature emergence...")
    print("-" * 60)
    
    start = time.time()
    
    for step in range(steps + 1):
        # Synthetic data
        x = torch.randint(0, vocab_size, (16, 10))
        y = torch.cat([x[:, 1:], torch.zeros(16, 1).long()], dim=1)
        
        # Forward
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track
        if step % save_interval == 0:
            with torch.no_grad():
                # Analyze activations
                for layer_idx in range(n_layers):
                    act = model.activations[layer_idx]
                    
                    # Sparsity
                    threshold = act.abs().mean() * 0.1
                    sparsity = (act.abs() < threshold).float().mean().item()
                    
                    # Norm
                    norm = act.norm().item()
                    
                    # Entropy
                    flat = act.flatten()
                    hist = torch.histc(flat, bins=20)
                    hist = hist / hist.sum()
                    entropy = -(hist * torch.log2(hist + 1e-8)).sum().item()
                    
                    emergence_history[step][layer_idx] = {
                        "sparsity": sparsity,
                        "norm": norm,
                        "entropy": entropy
                    }
                
                print(f"Step {step:4d}: Loss={loss.item():.4f}, "
                      f"L0 sparsity={emergence_history[step][0]['sparsity']:.2%}, "
                      f"L1 sparsity={emergence_history[step][1]['sparsity']:.2%}")
    
    elapsed = time.time() - start
    print(f"\nDone! Time: {elapsed:.1f}s")
    print()
    
    # Analyze emergence pattern
    print("-" * 60)
    print("Emergence Pattern Analysis:")
    print()
    
    for layer_idx in range(n_layers):
        print(f"Layer {layer_idx}:")
        
        sparsities = []
        norms = []
        
        for step in sorted(emergence_history.keys()):
            data = emergence_history[step][layer_idx]
            sparsities.append(data["sparsity"])
            norms.append(data["norm"])
        
        # Detect emergence
        # Does sparsity increase?
        if sparsities[-1] > sparsities[0] * 1.5:
            print(f"  [+] Sparsity emergence: {sparsities[0]:.1%} -> {sparsities[-1]:.1%}")
        else:
            print(f"  Sparsity change: {sparsities[0]:.1%} -> {sparsities[-1]:.1%}")
        
        # Does norm increase?
        if norms[-1] > norms[0] * 2:
            print(f"  [+] Activation norm growth: {norms[0]:.1f} -> {norms[-1]:.1f}")
        else:
            print(f"  Activation norm: {norms[0]:.1f} -> {norms[-1]:.1f}")
        
        print()
    
    # Key findings
    print("-" * 60)
    print("Key Findings:")
    print()
    
    # Compare two layers
    l0_final = emergence_history[steps][0]["sparsity"]
    l1_final = emergence_history[steps][1]["sparsity"]
    
    if l1_final > l0_final:
        print("  - Deep layer (L1) is more sparse than shallow (L0)")
    else:
        print("  - Shallow layer (L0) is more sparse than deep (L1)")
    
    # Check emergence order
    for step in sorted(emergence_history.keys())[1:]:
        for layer_idx in range(n_layers):
            prev_step = step - save_interval
            if prev_step >= 0:
                prev = emergence_history[prev_step][layer_idx]["sparsity"]
                curr = emergence_history[step][layer_idx]["sparsity"]
                
                # First time above 30%
                if curr > 0.3 and prev <= 0.3:
                    print(f"  - Layer {layer_idx} reached 30% sparsity at Step {step}")
    
    print()
    print("=" * 60)
    print("Demo Complete!")
    print()
    print("Full version: analysis/mechanism_analysis/run_feature_emergence.py")
    print("Runtime: ~5 min (GPU) / ~30 min (CPU)")
    print("=" * 60)


if __name__ == "__main__":
    quick_emergence_tracking()
