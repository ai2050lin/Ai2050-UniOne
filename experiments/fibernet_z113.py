
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to sys.path to allow importing from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fibernet_v2 import DecoupledFiberNet, LieGroupEmbedding

# --- Configuration ---
GROUP_ORDER = 113
D_MODEL = 64 # Small model
N_LAYERS = 2
EPOCHS = 1000 # Give enough time for both
BATCH_SIZE = 64
LR_TRANSFORMER = 1e-3
LR_FIBERNET = 1e-3 

# --- Standard Transformer Baseline (Simplified) ---
class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Using standard attention
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        emb = self.embedding(x)
        out = self.transformer(emb)
        # Predict based on last token or mean? Let's use last token for simplicity in seq tasks
        # But for a+b=c, usually we train it to predict c given a, b.
        # Let's align task: Input [a, b, =], Output [., ., c]
        return self.head(out)

# --- Data Generation (Z_113) ---
def generate_zn_data(n, train_split=0.8):
    # Generate all pairs (a, b)
    X = []
    Y = []
    for a in range(n):
        for b in range(n):
            res = (a + b) % n
            X.append([a, b])
            Y.append(res)
            
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    
    # Shuffle
    idx = torch.randperm(len(X))
    X, Y = X[idx], Y[idx]
    
    split = int(len(X) * train_split)
    return X[:split], Y[:split], X[split:], Y[split:]

# --- Training Helper ---
def train_model(model, name, train_loader, val_X, val_Y, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) # AdamW is better
    
    history = {'loss': [], 'acc': []}
    model.to(device)
    val_X, val_Y = val_X.to(device), val_Y.to(device)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            # Predict only the result? 
            # Our models output [batch, seq, vocab]. We want the last token's prediction.
            logits = model(bx) 
            # We assume the model predicts the answer at the last position or based on the sequence
            # For [a, b], let's assume we want to predict c.
            # But wait, standard transformer usually predicts next token.
            # Here we simplify: use the last token projection to predict c.
            # Input is [a, b]. Output at pos 1 (corresponding to b) should be c?
            # Or we can just pool.
            
            # Let's use MEAN pooling for both models to be fair and robust for this simple task
            # (as used in previous simple scripts)
            if len(logits.shape) == 3:
                logits = logits.mean(dim=1)
            
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                vlogits = model(val_X)
                if len(vlogits.shape) == 3:
                     vlogits = vlogits.mean(dim=1)
                preds = vlogits.argmax(dim=-1)
                acc = (preds == val_Y).float().mean().item()
                history['acc'].append(acc)
                
            if epoch % 100 == 0:
                print(f"[{name}] Epoch {epoch}: Loss {avg_loss:.4f}, Acc {acc:.2%}")
                
            if acc > 0.999:
                print(f"[{name}] Converged at Epoch {epoch}!")
                break
                
    duration = time.time() - start_time
    print(f"[{name}] Finished in {duration:.2f}s. Final Acc: {history['acc'][-1] if history['acc'] else 0:.2%}")
    return history

# --- Main Experiment ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    # Data
    X_train, Y_train, X_val, Y_val = generate_zn_data(GROUP_ORDER)
    train_ds = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    
    # 1. Train Standard Transformer
    print("\n--- Training Standard Transformer ---")
    std_model = StandardTransformer(GROUP_ORDER, D_MODEL, N_LAYERS)
    print(f"Standard Params: {sum(p.numel() for p in std_model.parameters())}")
    std_hist = train_model(std_model, "Standard", train_loader, X_val, Y_val, EPOCHS, LR_TRANSFORMER, device)
    
    # 2. Train FiberNet (Circle)
    print("\n--- Training FiberNet (Circle Prior) ---")
    fiber_model = DecoupledFiberNet(GROUP_ORDER, D_MODEL, N_LAYERS, group_type='circle', max_len=10)
    print(f"FiberNet Params: {sum(p.numel() for p in fiber_model.parameters())}")
    fiber_hist = train_model(fiber_model, "FiberNet", train_loader, X_val, Y_val, EPOCHS, LR_FIBERNET, device)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(std_hist['loss'], label='Standard Transformer', alpha=0.7)
    plt.plot(fiber_hist['loss'], label='FiberNet (Circle)', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    # Acc is recorded every 10 epochs
    x_axis = range(0, len(std_hist['acc']) * 10, 10)
    plt.plot(x_axis, std_hist['acc'], label='Standard Transformer')
    # Use array slicing if lengths differ (due to early stopping)
    x_axis_fiber = range(0, len(fiber_hist['acc']) * 10, 10)
    plt.plot(x_axis_fiber, fiber_hist['acc'], label='FiberNet (Circle)', linewidth=2)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    os.makedirs('tempdata/fibernet', exist_ok=True)
    save_path = 'tempdata/fibernet/z113_comparison.png'
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    main()
