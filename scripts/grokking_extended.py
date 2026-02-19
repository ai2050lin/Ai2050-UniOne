"""
Grokking 延长训练测试
====================

基于发现: p97_h128_wd1.0_frac40 在 10000 epochs 达到 42.2%
继续延长训练观察是否达到完全 Grokking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time

print("=" * 60)
print("Grokking 延长训练测试")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

def extended_grokking_test(
    p=97,
    train_frac=0.4,
    hidden_dim=128,
    n_epochs=50000,          # 极长训练
    weight_decay=1.0,
    lr=0.001,
    seed=42
):
    """延长 Grokking 测试"""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\nZ_{p}: hidden={hidden_dim}, wd={weight_decay}")
    print(f"训练数据: {train_frac*100:.0f}%, 轮次: {n_epochs}")
    
    # 生成数据
    all_a = torch.arange(p).unsqueeze(1).expand(p, p).reshape(-1)
    all_b = torch.arange(p).unsqueeze(0).expand(p, p).reshape(-1)
    all_labels = (all_a + all_b) % p
    
    n_total = p * p
    n_train = int(n_total * train_frac)
    
    indices = torch.randperm(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    def encode(a, b, p):
        n = len(a)
        inp = torch.zeros(n, p * 2)
        for i in range(n):
            inp[i, a[i]] = 1
            inp[i, p + b[i]] = 1
        return inp
    
    train_input = encode(all_a[train_idx], all_b[train_idx], p).to(device)
    train_labels = all_labels[train_idx].to(device)
    test_input = encode(all_a[test_idx], all_b[test_idx], p).to(device)
    test_labels = all_labels[test_idx].to(device)
    
    print(f"训练: {n_train}, 测试: {n_total - n_train}")
    
    model = nn.Sequential(
        nn.Linear(p * 2, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, p)
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = nn.CrossEntropyLoss()
    
    grokking_epoch = None
    best_test = 0
    history = []
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_input)
        loss = criterion(out, train_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 1000 == 0:
            model.eval()
            with torch.no_grad():
                train_pred = model(train_input).argmax(1)
                train_acc = (train_pred == train_labels).float().mean().item()
                test_pred = model(test_input).argmax(1)
                test_acc = (test_pred == test_labels).float().mean().item()
            
            best_test = max(best_test, test_acc)
            history.append((epoch+1, train_acc, test_acc))
            
            if grokking_epoch is None and test_acc > 0.95:
                grokking_epoch = epoch + 1
                print(f"  *** GROKKING at Epoch {grokking_epoch}! Test={test_acc:.1%} ***")
            
            print(f"  Epoch {epoch+1:5d}: Train={train_acc:.1%}, Test={test_acc:.1%}, Best={best_test:.1%}")
    
    return {
        'final_test': history[-1][2] if history else 0,
        'best_test': best_test,
        'grokking_epoch': grokking_epoch,
        'grokked': grokking_epoch is not None,
        'history': history
    }

# 测试
print("\n开始延长训练测试...")
result = extended_grokking_test(p=97, train_frac=0.4, hidden_dim=128, n_epochs=30000)

# 保存
import os
os.makedirs("tempdata", exist_ok=True)

def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native(v) for v in obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

with open("tempdata/grokking_extended_report.json", "w") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "result": convert_to_native(result)
    }, f, indent=2)

print(f"\n报告保存到: tempdata/grokking_extended_report.json")

if result['grokked']:
    print(f"\n结论: Grokking 成功! 发生在 Epoch {result['grokking_epoch']}")
else:
    print(f"\n结论: 最佳测试准确率 {result['best_test']:.1%}")
    if result['best_test'] > 0.5:
        print("观察到部分 Grokking 现象")
    else:
        print("需要进一步调整参数")
