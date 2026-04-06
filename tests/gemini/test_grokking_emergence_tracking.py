import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time
from transformer_lens import HookedTransformer, HookedTransformerConfig

# Global Settings
P = 113  # Multiplier for modular addition
FRAC_TRAIN = 0.3  # Fraction of data used for training
NUM_EPOCHS = 10000
LR = 1e-3
WEIGHT_DECAY = 1.0
BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_dataset(p):
    """Create modular addition dataset x + y = z mod p."""
    x = torch.arange(p)
    y = torch.arange(p)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    x_flat = x_grid.reshape(-1)
    y_flat = y_grid.reshape(-1)
    z_flat = (x_flat + y_flat) % p
    
    # Input format: [x, y, p] -> x + y = p (modular addition)
    # Actually, simpler: [x, y, =] -> z
    # We use tokens: 0..p-1 are numbers, p is '+', p+1 is '='
    inputs = torch.stack([x_flat, y_flat], dim=1)
    labels = z_flat
    return inputs, labels

def train():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Grokking Emergence Tracking...")
    os.makedirs("tests/gemini/data", exist_ok=True)
    
    # 1. Dataset
    inputs, labels = create_dataset(P)
    dataset_size = len(inputs)
    indices = torch.randperm(dataset_size)
    
    train_size = int(dataset_size * FRAC_TRAIN)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_inputs, train_labels = inputs[train_indices].to(DEVICE), labels[train_indices].to(DEVICE)
    test_inputs, test_labels = inputs[test_indices].to(DEVICE), labels[test_indices].to(DEVICE)
    
    # 2. Model (1-layer Transformer)
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=128,
        d_head=32,
        n_heads=4,
        d_mlp=512,
        d_vocab=P + 1,  # Numbers 0..p-1, and we'll just use 2 input tokens
        n_ctx=2,        # [x, y]
        act_fn="relu",
        normalization_type=None,
        device=DEVICE
    )
    model = HookedTransformer(cfg).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    
    pbar = tqdm(range(NUM_EPOCHS), desc="Training")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        # Forward
        logits = model(train_inputs) # [batch, n_ctx, d_vocab]
        # We only care about the last token prediction for x+y
        # Wait, for [x, y], where do we predict? 
        # Usually we use [x, y, =] as input and predict z at '='.
        # Here we just use [x, y] and predict at index 1.
        logits = logits[:, -1, :P] # Prediction of z at the second position
        
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()
        
        # Eval
        if epoch % 100 == 0 or epoch == NUM_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                train_logits = model(train_inputs)[:, -1, :P]
                train_acc = (train_logits.argmax(dim=-1) == train_labels).float().mean().item()
                
                test_logits = model(test_inputs)[:, -1, :P]
                test_acc = (test_logits.argmax(dim=-1) == test_labels).float().mean().item()
                
                # Check Effective Rank of Residual Stream
                # We use a sample of test inputs to get residual stream
                _, cache = model.run_with_cache(test_inputs[:100])
                resid_post = cache["blocks.0.hook_resid_post"] # [100, 2, 128]
                resid_flat = resid_post[:, -1, :].cpu().numpy()
                u, s, v = np.linalg.svd(resid_flat)
                eff_rank = np.exp(-np.sum((s**2 / np.sum(s**2)) * np.log(s**2 / np.sum(s**2) + 1e-10)))
                
                history.append({
                    "epoch": epoch,
                    "loss": loss.item(),
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "eff_rank": eff_rank
                })
                
                pbar.set_postfix(T_Acc=f"{train_acc:.2f}", V_Acc=f"{test_acc:.2f}", Rank=f"{eff_rank:.2f}")
                
                # Detection of Grokking (Phase Transition)
                if test_acc > 0.5 and len(history) > 1 and history[-2]["test_acc"] < 0.1:
                    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Phase Transition Detected at Epoch {epoch}!")
                    # Save model and dynamics for "Deep Analysis"
                    torch.save(model.state_dict(), f"tests/gemini/data/model_grokking_{epoch}.pt")
    
    # Save statistics
    df = pd.DataFrame(history)
    df.to_csv("tests/gemini/data/grokking_history.csv", index=False)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training complete. Results saved to tests/gemini/data/")

if __name__ == "__main__":
    train()
