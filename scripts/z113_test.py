
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 配置
P = 113  # The Prime Number for Cyclic Group Z_p
LR = 1e-3
EPOCHS = 2000
TRAIN_RATIO = 0.5  # 50% for training
Hidden_Dim = 128
SEED = 42

# 设置随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)

class FiberNetZp(nn.Module):
    def __init__(self, p, hidden_dim):
        super().__init__()
        self.p = p
        # Embedding: Maps integer 0..p-1 to vector space
        self.embedding = nn.Embedding(self.p, hidden_dim)
        
        # Manifold Network: The "Logic"
        # We use a simple MLP to approximate the group operation
        # In a real FiberNet, this would be a Transformer or RNN
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.act = nn.ReLU()
        
        # Unembedding: Maps back to logits over p
        self.unembed = nn.Linear(hidden_dim, self.p, bias=False)
        
        # Tie weights for geometric consistency (optional but recommended for Zp)
        # self.unembed.weight = self.embedding.weight 

    def forward(self, a, b):
        # a, b: (batch_size)
        emb_a = self.embedding(a)  # (B, H)
        emb_b = self.embedding(b)  # (B, H)
        
        # Concatenate: "The combined state"
        x = torch.cat([emb_a, emb_b], dim=1) # (B, 2H)
        
        # Manifold processing
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        
        # Project to fiber (logits)
        logits = self.unembed(x)
        return logits

def train_z113():
    # 1. Prepare Data: All pairs (a, b) -> (a + b) % p
    pairs = []
    labels = []
    for i in range(P):
        for j in range(P):
            pairs.append([i, j])
            labels.append((i + j) % P)
            
    pairs = torch.tensor(pairs).long()
    labels = torch.tensor(labels).long()
    
    # Shuffle and Split
    dataset_size = len(pairs)
    indices = torch.randperm(dataset_size)
    train_size = int(dataset_size * TRAIN_RATIO)
    
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    train_pairs = pairs[train_idx]
    train_labels = labels[train_idx]
    
    test_pairs = pairs[test_idx]
    test_labels = labels[test_idx]
    
    # 2. Model & Optimizer
    model = FiberNetZp(P, Hidden_Dim).cuda() if torch.cuda.is_available() else FiberNetZp(P, Hidden_Dim)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1.0) # Weight decay prompts grokking
    loss_fn = nn.CrossEntropyLoss()
    
    # 3. Training Loop
    history = {"train_acc": [], "test_acc": [], "epoch": []}
    
    print(f"Starting Z{P} Training with {train_size} samples...")
    print(f"Goal: Watch for 'Grokking' - sudden jump in test accuracy.")

    os.makedirs("experiments/z113_visuals", exist_ok=True)
    
    pbar = tqdm(range(EPOCHS))
    for epoch in pbar:
        model.train()
        # Full batch training for grokking experiments
        if torch.cuda.is_available():
            batch_x = train_pairs.cuda()
            batch_y = train_labels.cuda()
            test_x = test_pairs.cuda()
            test_y = test_labels.cuda()
        else:
            batch_x = train_pairs
            batch_y = train_labels
            test_x = test_pairs
            test_y = test_labels

        optimizer.zero_grad()
        logits = model(batch_x[:,0], batch_x[:,1])
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            # Evaluate
            model.eval()
            with torch.no_grad():
                # Train Acc
                train_pred = logits.argmax(dim=1)
                train_acc = (train_pred == batch_y).float().mean().item()
                
                # Test Acc
                test_logits = model(test_x[:,0], test_x[:,1])
                test_pred = test_logits.argmax(dim=1)
                test_acc = (test_pred == test_y).float().mean().item()
                
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["epoch"].append(epoch)
            
            pbar.set_description(f"Epoch {epoch} | Train: {train_acc:.4f} | Test: {test_acc:.4f}")
            
            # Save periodic snapshot of embeddings for visualization
            if epoch % 100 == 0 or test_acc > 0.99:
                save_embedding_snapshot(model, epoch)
                if test_acc > 0.995:
                    print(f"\n\n🔥 GROKKED at Epoch {epoch}! Test Acc: {test_acc:.4f}")
                    break

    # 4. Save Results
    with open("experiments/z113_visuals/training_log.json", "w") as f:
        json.dump(history, f)
        
    print("Training Complete. Data saved to experiments/z113_visuals/")

def save_embedding_snapshot(model, epoch):
    """Save the current embedding weights (the 'Fiber') for visualization."""
    weights = model.embedding.weight.detach().cpu().numpy()
    # Perform PCA to 3D for visualization readiness
    from sklearn.decomposition import PCA
    if weights.shape[1] > 3:
        pca = PCA(n_components=3)
        weights_3d = pca.fit_transform(weights)
    else:
        weights_3d = weights
        
    data = {"epoch": epoch, "embeddings": weights_3d.tolist()}
    with open(f"experiments/z113_visuals/embeddings_epoch_{epoch}.json", "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    train_z113()
