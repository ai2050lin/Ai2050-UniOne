
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from experiments.toy_experiment.group_theory_dataset import GroupTheoryDataset
from experiments.toy_experiment.toy_models import ToyFiberNet, ToyTransformer
from scripts.ricci_optimizer import RicciFlowOptimizer

LOG_FILE = "d:\\develop\\TransformerLens-main\\experiments\\toy_experiment\\training_log.json"

def log_metrics(epoch, loss, accuracy, model_name):
    # Read existing logs or create new
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
        except:
            logs = {"Transformer": [], "FiberNet": []}
    else:
        logs = {"Transformer": [], "FiberNet": []}
    
    # Check if key exists (in case of fresh start)
    if model_name not in logs:
        logs[model_name] = []
        
    logs[model_name].append({
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy,
        "timestamp": time.time()
    })
    
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f)

def train_model(model, train_loader, epochs=500, lr=0.001, name="Model", opt_type="Adam", ricci_alpha=0.01, d_manifold=None):
    criterion = nn.CrossEntropyLoss()
    if opt_type == "Ricci":
        optimizer = RicciFlowOptimizer(model.parameters(), lr=lr, ricci_alpha=ricci_alpha)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n--- Training {name} ---")
    start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # inputs: [batch, 2] (a, b)
            # targets: [batch] (result)
            a_idx = inputs[:, 0]
            b_idx = inputs[:, 1]
            
            optimizer.zero_grad()
            outputs = model(a_idx, b_idx) # [batch, vocab]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Log to JSON
        log_metrics(epoch+1, avg_loss, accuracy, name)
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
            
        if accuracy > 99.5:
            print(f"converged at epoch {epoch+1}")
            break
            
    end_time = time.time()
    print(f"{name} Training Time: {end_time - start_time:.2f}s")
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, default="Z_n", choices=["Z_n", "S_3"])
    parser.add_argument("--order", type=int, default=113)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    # Configuration
    GROUP_TYPE = args.group
    GROUP_ORDER = args.order
    NUM_SAMPLES = 5000
    BATCH_SIZE = 64
    EPOCHS = args.epochs
    
    ricci_alpha = 0.001
    d_manifold = None # Default for RicciFlowOptimizer

    if GROUP_TYPE == "S_3":
        ricci_alpha = 0.0001 # Reduced for S3
        d_manifold = 100 # Increased for S3
    
    # Clear log file at start
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        
    # Dataset
    dataset = GroupTheoryDataset(group_type=GROUP_TYPE, order=GROUP_ORDER, num_samples=NUM_SAMPLES)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Dataset: {GROUP_TYPE} (Order {GROUP_ORDER}). {len(dataset)} samples.")
    
    # Models
    vocab_size = GROUP_ORDER if GROUP_TYPE == "Z_n" else 6
    
    # Transformer Baseline
    transformer = ToyTransformer(vocab_size=vocab_size, d_model=64, n_head=4, n_layer=2)
    train_model(transformer, train_loader, epochs=EPOCHS, name="Transformer")
    
    if args.compare:
        # FiberNet with softer Ricci Flow Optimization
        fibernet = ToyFiberNet(vocab_size=vocab_size, d_model=128)
        train_model(fibernet, train_loader, epochs=EPOCHS * 2, name="FiberNet", opt_type="Ricci",
                    ricci_alpha=ricci_alpha, d_manifold=d_manifold)

if __name__ == "__main__":
    main()
