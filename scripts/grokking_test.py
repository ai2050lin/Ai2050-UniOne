"""
Grokking 现象复现测试
====================

基于 Power et al. 2022 "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"

关键设置:
- 小模型: hidden_dim=64-128, layers=1-2
- 训练数据比例: 30-50%
- 训练轮次: 2000+
- weight_decay: 1.0
- 素数群: p=97, 113
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import math

print("=" * 60)
print("Grokking 现象复现测试")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n设备: {device}")

results = {}

# ============================================================================
# 标准 Grokking 设置
# ============================================================================

def grokking_test_standard(
    p=97,                    # 素数群 Z_p
    train_frac=0.3,          # 训练数据比例
    hidden_dim=128,          # 隐藏层维度
    n_layers=2,              # 层数
    n_epochs=2000,           # 训练轮次
    weight_decay=1.0,        # 权重衰减
    lr=0.001,                # 学习率
    operation='addition',    # 运算类型
    seed=42
):
    """
    标准 Grokking 测试
    
    Grokking 发生的条件:
    1. 过参数化模型
    2. 小训练集
    3. 高 weight decay
    4. 足够长的训练时间
    """
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Grokking 测试: Z_{p} {operation}")
    print(f"训练数据: {train_frac*100:.0f}, 隐藏层: {hidden_dim}, 层数: {n_layers}")
    print(f"Weight decay: {weight_decay}, 学习率: {lr}")
    print(f"{'='*60}")
    
    # 生成所有可能的数据点
    all_a = torch.arange(p).unsqueeze(1).expand(p, p).reshape(-1)
    all_b = torch.arange(p).unsqueeze(0).expand(p, p).reshape(-1)
    
    if operation == 'addition':
        all_labels = (all_a + all_b) % p
    elif operation == 'subtraction':
        all_labels = (all_a - all_b) % p
    elif operation == 'multiplication':
        all_labels = (all_a * all_b) % p
    elif operation == 'division':
        # 只使用非零除数
        mask = all_b != 0
        all_a = all_a[mask]
        all_b = all_b[mask]
        all_labels = (all_a * torch.pow(all_b, p-2, dtype=torch.long)) % p  # 费马小定理
    
    n_total = len(all_labels)
    n_train = int(n_total * train_frac)
    n_test = n_total - n_train
    
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
    
    print(f"训练样本: {n_train}, 测试样本: {n_test}")
    
    # 构建模型 (简单 MLP)
    layers = [nn.Linear(p * 2, hidden_dim), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, p))
    
    model = nn.Sequential(*layers).to(device)
    
    # 计算参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")
    print(f"过参数化比: {n_params / n_train:.1f}x")
    
    # 优化器 (关键: AdamW + 高 weight decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    train_accs = []
    test_accs = []
    losses = []
    
    grokking_epoch = None
    best_test_acc = 0
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_input)
        loss = criterion(out, train_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        # 每 100 epoch 评估一次
        if (epoch + 1) % 100 == 0 or epoch < 100:
            model.eval()
            with torch.no_grad():
                train_pred = model(train_input).argmax(1)
                train_acc = (train_pred == train_labels).float().mean().item()
                test_pred = model(test_input).argmax(1)
                test_acc = (test_pred == test_labels).float().mean().item()
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            best_test_acc = max(best_test_acc, test_acc)
            
            # 检测 Grokking (测试准确率突然提升)
            if grokking_epoch is None and test_acc > 0.9 and len(test_accs) > 1:
                grokking_epoch = epoch + 1
                print(f"\n  *** GROKKING 发生在 Epoch {grokking_epoch}! ***")
                print(f"      测试准确率: {test_acc:.2%}")
            
            if (epoch + 1) % 200 == 0 or epoch < 50:
                print(f"  Epoch {epoch+1:4d}: Loss={loss.item():.4f}, Train={train_acc:.2%}, Test={test_acc:.2%}")
    
    return {
        'final_train_acc': train_accs[-1] if train_accs else 0,
        'final_test_acc': test_accs[-1] if test_accs else 0,
        'best_test_acc': best_test_acc,
        'grokking_epoch': grokking_epoch,
        'grokked': grokking_epoch is not None,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'n_params': n_params,
        'n_train': n_train
    }

# ============================================================================
# 测试 1: 标准设置 (Z_97, 30% 训练数据)
# ============================================================================

print("\n" + "="*60)
print("[1] 标准 Grokking 测试 (Z_97)")
print("="*60)

result1 = grokking_test_standard(
    p=97,
    train_frac=0.3,
    hidden_dim=128,
    n_layers=2,
    n_epochs=2000,
    weight_decay=1.0,
    operation='addition'
)
results['Z_97_standard'] = result1

# ============================================================================
# 测试 2: 更小模型 (更容易 Grokking)
# ============================================================================

print("\n" + "="*60)
print("[2] 小模型测试 (Z_97, hidden=64)")
print("="*60)

result2 = grokking_test_standard(
    p=97,
    train_frac=0.3,
    hidden_dim=64,
    n_layers=1,
    n_epochs=2000,
    weight_decay=1.0,
    operation='addition'
)
results['Z_97_small'] = result2

# ============================================================================
# 测试 3: 更少训练数据
# ============================================================================

print("\n" + "="*60)
print("[3] 极少训练数据测试 (Z_97, 20%)")
print("="*60)

result3 = grokking_test_standard(
    p=97,
    train_frac=0.2,
    hidden_dim=128,
    n_layers=2,
    n_epochs=3000,
    weight_decay=1.0,
    operation='addition'
)
results['Z_97_20percent'] = result3

# ============================================================================
# 测试 4: 乘法运算 (更难)
# ============================================================================

print("\n" + "="*60)
print("[4] 乘法运算测试 (Z_97)")
print("="*60)

result4 = grokking_test_standard(
    p=97,
    train_frac=0.3,
    hidden_dim=128,
    n_layers=2,
    n_epochs=2000,
    weight_decay=1.0,
    operation='multiplication'
)
results['Z_97_multiplication'] = result4

# ============================================================================
# 测试 5: 更大素数 (Z_113)
# ============================================================================

print("\n" + "="*60)
print("[5] 更大素数测试 (Z_113)")
print("="*60)

result5 = grokking_test_standard(
    p=113,
    train_frac=0.3,
    hidden_dim=128,
    n_layers=2,
    n_epochs=2000,
    weight_decay=1.0,
    operation='addition'
)
results['Z_113'] = result5

# ============================================================================
# 总结
# ============================================================================

print("\n" + "="*60)
print("Grokking 测试总结")
print("="*60)

print("\n| 测试 | 训练准确率 | 测试准确率 | 最佳测试 | Grokking Epoch |")
print("|------|------------|------------|----------|----------------|")

grokked_count = 0
for name, res in results.items():
    status = "OK" if res['grokked'] else "FAIL"
    grokked_count += res['grokked']
    print(f"| {name:20s} | {res['final_train_acc']:.1%} | {res['final_test_acc']:.1%} | {res['best_test_acc']:.1%} | {res['grokking_epoch'] or 'N/A':>14} | {status}")

print(f"\nGrokking 成功: {grokked_count}/{len(results)}")

# 保存结果
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

with open("tempdata/grokking_report.json", "w") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "n_grokked": grokked_count,
        "n_total": len(results),
        "results": convert_to_native(results)
    }, f, indent=2)

print(f"\n报告已保存到: tempdata/grokking_report.json")

if grokked_count > 0:
    print("\n结论: Grokking 现象成功复现!")
else:
    print("\n结论: Grokking 现象未观察到，需要调整超参数")
    print("建议: 减小模型、减少训练数据、增加训练轮次")
