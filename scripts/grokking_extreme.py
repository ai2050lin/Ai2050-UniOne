"""
Grokking 极端测试 - 复现 Power et al. 2022
==========================================

关键发现: Grokking 需要特定的超参数组合
- 极小模型 (甚至单层)
- 高 weight decay (1.0+)
- 足够长的训练 (>5000 epochs)
- 正确的数据比例
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import math

print("=" * 60)
print("Grokking 极端参数测试")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

results = {}

def extreme_grokking_test(
    p=97,
    train_frac=0.5,
    hidden_dim=32,          # 极小隐藏层
    n_layers=1,             # 单层!
    n_epochs=10000,         # 超长训练
    weight_decay=10.0,      # 极高 weight decay
    lr=0.001,
    seed=42
):
    """极端 Grokking 测试"""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Z_{p}: hidden={hidden_dim}, layers={n_layers}, wd={weight_decay}")
    print(f"训练数据: {train_frac*100:.0f}%, 轮次: {n_epochs}")
    print(f"{'='*60}")
    
    # 生成数据
    all_a = torch.arange(p).unsqueeze(1).expand(p, p).reshape(-1)
    all_b = torch.arange(p).unsqueeze(0).expand(p, p).reshape(-1)
    all_labels = (all_a + all_b) % p
    
    n_total = p * p
    n_train = int(n_total * train_frac)
    
    # 随机划分
    indices = torch.randperm(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    # One-hot 编码
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
    
    print(f"训练样本: {n_train}, 测试样本: {n_total - n_train}")
    
    # 极简模型
    model = nn.Sequential(
        nn.Linear(p * 2, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, p)
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}, 过参数化: {n_params/n_train:.1f}x")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = nn.CrossEntropyLoss()
    
    grokking_epoch = None
    best_test = 0
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_input)
        loss = criterion(out, train_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                train_pred = model(train_input).argmax(1)
                train_acc = (train_pred == train_labels).float().mean().item()
                test_pred = model(test_input).argmax(1)
                test_acc = (test_pred == test_labels).float().mean().item()
            
            best_test = max(best_test, test_acc)
            
            if grokking_epoch is None and test_acc > 0.95:
                grokking_epoch = epoch + 1
                print(f"  *** GROKKING at Epoch {grokking_epoch}! Test={test_acc:.1%} ***")
            
            if (epoch + 1) % 1000 == 0:
                print(f"  Epoch {epoch+1}: Train={train_acc:.1%}, Test={test_acc:.1%}")
    
    model.eval()
    with torch.no_grad():
        test_pred = model(test_input).argmax(1)
        final_test = (test_pred == test_labels).float().mean().item()
    
    return {
        'final_test': final_test,
        'best_test': best_test,
        'grokking_epoch': grokking_epoch,
        'grokked': grokking_epoch is not None
    }

# ============================================================================
# 测试不同配置
# ============================================================================

configs = [
    # (p, train_frac, hidden, layers, epochs, weight_decay)
    (97, 0.5, 32, 1, 10000, 10.0),    # 极端
    (97, 0.5, 64, 1, 10000, 5.0),     # 高 decay
    (97, 0.3, 64, 1, 15000, 10.0),    # 少数据
    (97, 0.4, 128, 1, 10000, 1.0),    # 标准
    (59, 0.5, 32, 1, 8000, 10.0),     # 小群
    (113, 0.5, 64, 1, 12000, 10.0),   # 大群
]

for i, (p, train_frac, hidden, layers, epochs, wd) in enumerate(configs):
    print(f"\n[{i+1}/{len(configs)}]")
    result = extreme_grokking_test(
        p=p, train_frac=train_frac, hidden_dim=hidden,
        n_layers=layers, n_epochs=epochs, weight_decay=wd
    )
    key = f"p{p}_h{hidden}_wd{wd}_frac{int(train_frac*100)}"
    results[key] = result

# ============================================================================
# 总结
# ============================================================================

print("\n" + "="*60)
print("Grokking 测试总结")
print("="*60)

print("\n| 配置 | 最终测试 | 最佳测试 | Grokking |")
print("|------|----------|----------|----------|")

grokked = 0
for name, res in results.items():
    status = "OK" if res['grokked'] else "-"
    grokked += res['grokked']
    print(f"| {name:25s} | {res['final_test']:.1%} | {res['best_test']:.1%} | {res['grokking_epoch'] or 'N/A':>8} {status} |")

print(f"\nGrokking 成功: {grokked}/{len(results)}")

# 保存
import os
os.makedirs("tempdata", exist_ok=True)

def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

with open("tempdata/grokking_extreme_report.json", "w") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_grokked": grokked,
        "n_total": len(results),
        "results": convert_to_native(results)
    }, f, indent=2)

print(f"\n报告保存到: tempdata/grokking_extreme_report.json")

if grokked > 0:
    print("\n结论: Grokking 现象成功复现!")
else:
    print("\n结论: Grokking 仍未观察到")
    print("可能原因:")
    print("1. 需要更长的训练时间 (>20000 epochs)")
    print("2. 需要特定的模型初始化")
    print("3. 需要不同的优化器设置")
