import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from experiments.toy_experiment.group_theory_dataset import GroupTheoryDataset

# --- SIMPLIFIED MODELS FOR SMOKE TEST ---
# We discovered that using full Sequence Transformers for a simple (a, b) -> c task 
# introduced unnecessary complexity (padding, masking, pos encoding errors).
# For Z_n (Modular Addition), a simple geometric MLP should solve it instantly.
# If this works, we know the data is fine, and we can slowly add complexity back.

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        # Simple attention or just MLP? Let's use a tiny Transformer Block manually
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, a, b):
        # Input: [batch, 2]
        x = torch.stack([a, b], dim=1)
        x = self.embed(x) # [batch, 2, d]
        
        # Self Attention
        # No mask needed, they can see each other
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # MLP
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        # Pool: Mean of the two tokens
        x = x.mean(dim=1) # [batch, d]
        return self.head(x)

class SimpleFiberNet(nn.Module):
    """
    A minimal Geometric Network.
    Theory: a + b = c  =>  Manifold is a flat torus.
    We just need to learn the 'shift' operator.
    """
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # "Connection" Layer: Mixes a and b geometrically
        self.manifold_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        
        # The 'Logic' of addition
        self.logic = nn.Linear(d_model * 2, d_model)
        
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, a, b):
        # 1. Lift to Bundle
        fa = self.embed(a)
        fb = self.embed(b)
        
        # 2. Geometric Interaction (Concatenation represents product bundle)
        # In a real FiberNet, this would be Parallel Transport.
        # Here we approximate it as: Logic(fa, fb)
        combined = torch.cat([fa, fb], dim=1)
        
        # 3. Project back
        fiber_out = self.logic(combined)
        return self.head(fiber_out)

def test_generalization(group_type='Z_n', order=113):
    # Plan C (Revised): Smoke Test with Simplified Architectures
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    full_dataset = GroupTheoryDataset(group_type=group_type, order=order, num_samples=8000, seed=42)
    train_indices = list(range(0, 6000))
    test_indices = list(range(6000, 8000))
    train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=256, shuffle=True)
    test_loader = DataLoader(Subset(full_dataset, test_indices), batch_size=256, shuffle=False)
    
    vocab_size = order
    print(f"\n--- Plan C (Fix): Smoke Test (Group: {group_type}_{order}) ---")
    
    transformer = SimpleTransformer(vocab_size, d_model=128).to(device)
    fibernet = SimpleFiberNet(vocab_size, d_model=128).to(device)
    
    def train_and_eval(model, name):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001) # AdamW + lower LR
        
        model.train()
        epochs = 20
        print(f"Training {name}...")
        
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs[:, 0], inputs[:, 1])
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                 print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")

        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs[:, 0], inputs[:, 1])
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        acc = 100 * correct / total
        print(f"{name} Test Accuracy: {acc:.2f}%")

    train_and_eval(transformer, "Transformer (Simple)")
    train_and_eval(fibernet, "FiberNet (Simple)")

if __name__ == "__main__":
    test_generalization()
