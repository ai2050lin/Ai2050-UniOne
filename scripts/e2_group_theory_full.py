"""
E2: 大规模群论测试 - 完整版
============================

增强配置:
- 隐藏层: 256-512
- 网络深度: 4-6 层
- 训练数据: 5000-10000
- 训练轮次: 200-500
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import math

print("=" * 60)
print("E2: 大规模群论测试 (完整版)")
print("=" * 60)

# 检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n设备: {device}")

results = {}

# ============================================================================
# 测试 1: Z_n 循环群 - 大规模测试
# ============================================================================

print("\n" + "=" * 60)
print("[1] Z_n 循环群 - 大规模测试")
print("=" * 60)

def test_cyclic_group_full(n_elements, n_train=10000, n_test=1000, 
                           hidden_dim=256, n_layers=5, n_epochs=300):
    """完整版循环群测试"""
    
    print(f"\n  Z_{n_elements} (隐藏层={hidden_dim}, 层数={n_layers}, 数据={n_train})")
    
    # 生成数据
    train_a = torch.randint(0, n_elements, (n_train,))
    train_b = torch.randint(0, n_elements, (n_train,))
    train_labels = (train_a + train_b) % n_elements
    
    train_input = torch.zeros(n_train, n_elements * 2)
    for i in range(n_train):
        train_input[i, train_a[i]] = 1
        train_input[i, n_elements + train_b[i]] = 1
    
    test_a = torch.randint(0, n_elements, (n_test,))
    test_b = torch.randint(0, n_elements, (n_test,))
    test_labels = (test_a + test_b) % n_elements
    
    test_input = torch.zeros(n_test, n_elements * 2)
    for i in range(n_test):
        test_input[i, test_a[i]] = 1
        test_input[i, n_elements + test_b[i]] = 1
    
    # 移动到设备
    train_input = train_input.to(device)
    train_labels = train_labels.to(device)
    test_input = test_input.to(device)
    test_labels = test_labels.to(device)
    
    # 构建深度网络
    layers = [nn.Linear(n_elements * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
    for _ in range(n_layers - 2):
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ])
    layers.append(nn.Linear(hidden_dim, n_elements))
    
    model = nn.Sequential(*layers).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    best_acc = 0
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_input)
        loss = criterion(out, train_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 50 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                pred = model(test_input).argmax(1)
                acc = (pred == test_labels).float().mean().item()
            best_acc = max(best_acc, acc)
            print(f"    Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.2%}")
    
    return best_acc

# 测试不同规模
cyclic_configs = [
    (20, 5000, 500, 256, 4, 200),    # 小规模快速
    (50, 8000, 800, 384, 5, 300),    # 中规模
    (100, 10000, 1000, 512, 6, 400), # 大规模
    (200, 15000, 1500, 512, 6, 500), # 超大规模
]

cyclic_results = {}
for n, n_train, n_test, hidden, layers, epochs in cyclic_configs:
    acc = test_cyclic_group_full(n, n_train, n_test, hidden, layers, epochs)
    cyclic_results[n] = {
        "accuracy": acc,
        "hidden_dim": hidden,
        "layers": layers,
        "epochs": epochs,
        "n_train": n_train
    }

results["cyclic_groups"] = cyclic_results

# ============================================================================
# 测试 2: S_n 置换群 - 完整版
# ============================================================================

print("\n" + "=" * 60)
print("[2] S_n 置换群 - 完整版测试")
print("=" * 60)

def test_permutation_group_full(n, n_train=8000, n_epochs=300, hidden_dim=384, n_layers=5):
    """完整版置换群测试"""
    
    from itertools import permutations
    
    n_perms = math.factorial(n)
    print(f"\n  S_{n} ({n_perms} 元素, 隐藏层={hidden_dim}, 层数={n_layers})")
    
    # 生成所有置换
    perms = list(permutations(range(n)))
    
    def compose(p1, p2):
        return tuple(p1[p2[i]] for i in range(len(p1)))
    
    def perm_to_tensor(p):
        t = torch.zeros(n * n)
        for i, v in enumerate(p):
            t[i * n + v] = 1
        return t
    
    # 生成训练数据
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    train_data = []
    train_labels = []
    for _ in range(n_train):
        i, j = np.random.randint(0, n_perms, 2)
        p1, p2 = perms[i], perms[j]
        result = compose(p1, p2)
        
        inp = torch.cat([perm_to_tensor(p1), perm_to_tensor(p2)])
        train_data.append(inp)
        train_labels.append(perm_to_idx[result])
    
    train_input = torch.stack(train_data).to(device)
    train_labels = torch.tensor(train_labels).to(device)
    
    # 测试数据
    n_test = 1000
    test_data = []
    test_labels = []
    for _ in range(n_test):
        i, j = np.random.randint(0, n_perms, 2)
        p1, p2 = perms[i], perms[j]
        result = compose(p1, p2)
        
        inp = torch.cat([perm_to_tensor(p1), perm_to_tensor(p2)])
        test_data.append(inp)
        test_labels.append(perm_to_idx[result])
    
    test_input = torch.stack(test_data).to(device)
    test_labels = torch.tensor(test_labels).to(device)
    
    # 模型
    input_dim = n * n * 2
    layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
    for _ in range(n_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, n_perms))
    
    model = nn.Sequential(*layers).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_input)
        loss = criterion(out, train_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 50 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                pred = model(test_input).argmax(1)
                acc = (pred == test_labels).float().mean().item()
            best_acc = max(best_acc, acc)
            print(f"    Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.2%}")
    
    return best_acc

# 测试 S_3, S_4, S_5
perm_results = {}
for n in [3, 4, 5]:
    hidden = 256 if n <= 4 else 512
    layers = 4 if n <= 4 else 6
    epochs = 300 if n <= 4 else 400
    n_train = 10000 if n <= 4 else 20000
    
    acc = test_permutation_group_full(n, n_train, epochs, hidden, layers)
    perm_results[n] = {
        "accuracy": acc,
        "n_elements": math.factorial(n),
        "hidden_dim": hidden,
        "layers": layers
    }

results["permutation_groups"] = perm_results

# ============================================================================
# 测试 3: Grokking 现象 - 完整版
# ============================================================================

print("\n" + "=" * 60)
print("[3] Grokking 现象 - 完整版测试")
print("=" * 60)

def test_grokking_full(n_elements=113, n_train=1000, n_epochs=1000, hidden_dim=256, n_layers=3):
    """
    完整版 Grokking 测试
    基于 Power et al. 2022 的设置:
    - 使用素数群 Z_p
    - 小训练集
    - 长训练时间
    """
    
    print(f"\n  Z_{n_elements} Grokking 测试 (训练数据={n_train}, 轮次={n_epochs})")
    
    n_test = 500
    
    # 生成数据 (使用素数)
    train_a = torch.randint(0, n_elements, (n_train,))
    train_b = torch.randint(0, n_elements, (n_train,))
    train_labels = (train_a + train_b) % n_elements
    
    train_input = torch.zeros(n_train, n_elements * 2)
    for i in range(n_train):
        train_input[i, train_a[i]] = 1
        train_input[i, n_elements + train_b[i]] = 1
    
    test_a = torch.randint(0, n_elements, (n_test,))
    test_b = torch.randint(0, n_elements, (n_test,))
    test_labels = (test_a + test_b) % n_elements
    
    test_input = torch.zeros(n_test, n_elements * 2)
    for i in range(n_test):
        test_input[i, test_a[i]] = 1
        test_input[i, n_elements + test_b[i]] = 1
    
    train_input = train_input.to(device)
    train_labels = train_labels.to(device)
    test_input = test_input.to(device)
    test_labels = test_labels.to(device)
    
    # 使用标准 Grokking 设置
    layers_list = [nn.Linear(n_elements * 2, hidden_dim), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers_list.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers_list.append(nn.Linear(hidden_dim, n_elements))
    
    model = nn.Sequential(*layers_list).to(device)
    
    # AdamW + weight decay 是 Grokking 的关键
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()
    
    train_accs = []
    test_accs = []
    grokking_epoch = None
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_input)
        loss = criterion(out, train_labels)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                train_pred = model(train_input).argmax(1)
                train_acc = (train_pred == train_labels).float().mean().item()
                test_pred = model(test_input).argmax(1)
                test_acc = (test_pred == test_labels).float().mean().item()
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            # 检测 Grokking
            if grokking_epoch is None and test_acc > 0.9 and train_acc > 0.95:
                grokking_epoch = epoch + 1
                print(f"    ★ Grokking 发生在 Epoch {grokking_epoch}!")
            
            if (epoch + 1) % 100 == 0:
                print(f"    Epoch {epoch+1}: Train={train_acc:.2%}, Test={test_acc:.2%}")
    
    return test_accs[-1], grokking_epoch, train_accs[-1], test_accs, train_accs

grok_acc, grok_epoch, grok_train, test_accs, train_accs = test_grokking_full(
    n_elements=113, n_train=500, n_epochs=800, hidden_dim=256, n_layers=3
)

results["grokking"] = {
    "final_test_acc": grok_acc,
    "final_train_acc": grok_train,
    "grokking_epoch": grok_epoch,
    "grokked": grok_epoch is not None
}

# ============================================================================
# 测试 4: 群同态学习
# ============================================================================

print("\n" + "=" * 60)
print("[4] 群同态学习 - 完整版")
print("=" * 60)

def test_homomorphism_full(n_source=100, n_target=50, n_train=10000, n_epochs=300):
    """
    群同态学习: Z_n_source -> Z_n_target
    测试模型能否学习同态映射
    """
    
    print(f"\n  同态 Z_{n_source} -> Z_{n_target}")
    
    n_test = 1000
    
    # 同态: f(x) = x mod n_target
    train_a = torch.randint(0, n_source, (n_train,))
    train_b = torch.randint(0, n_source, (n_train,))
    train_labels = ((train_a + train_b) % n_target)
    
    train_input = torch.zeros(n_train, n_source * 2)
    for i in range(n_train):
        train_input[i, train_a[i]] = 1
        train_input[i, n_source + train_b[i]] = 1
    
    test_a = torch.randint(0, n_source, (n_test,))
    test_b = torch.randint(0, n_source, (n_test,))
    test_labels = ((test_a + test_b) % n_target)
    
    test_input = torch.zeros(n_test, n_source * 2)
    for i in range(n_test):
        test_input[i, test_a[i]] = 1
        test_input[i, n_source + test_b[i]] = 1
    
    train_input = train_input.to(device)
    train_labels = train_labels.to(device)
    test_input = test_input.to(device)
    test_labels = test_labels.to(device)
    
    model = nn.Sequential(
        nn.Linear(n_source * 2, 512),
        nn.LayerNorm(512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.LayerNorm(512),
        nn.ReLU(),
        nn.Linear(512, n_target)
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_input)
        loss = criterion(out, train_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(test_input).argmax(1)
                acc = (pred == test_labels).float().mean().item()
            best_acc = max(best_acc, acc)
            print(f"    Epoch {epoch+1}: Acc={acc:.2%}")
    
    return best_acc

homo_acc = test_homomorphism_full(100, 50, 10000, 300)
results["homomorphism"] = {"accuracy": homo_acc}

# ============================================================================
# 测试 5: 子群识别
# ============================================================================

print("\n" + "=" * 60)
print("[5] 子群识别测试")
print("=" * 60)

def test_subgroup_recognition(n_group=60, n_subgroup=12, n_train=8000, n_epochs=300):
    """
    测试模型能否识别子群
    Z_60 中识别 Z_12 子群 (12 是 60 的因数)
    """
    
    print(f"\n  Z_{n_group} 中识别 Z_{n_subgroup} 子群")
    
    n_test = 1000
    
    # 生成数据: 判断元素是否在子群中
    # 子群 Z_12 = {0, 5, 10, ..., 55} (步长 = 60/12 = 5)
    step = n_group // n_subgroup
    subgroup = set(range(0, n_group, step))
    
    train_elements = torch.randint(0, n_group, (n_train,))
    train_labels = torch.tensor([1 if x.item() in subgroup else 0 for x in train_elements])
    
    train_input = torch.zeros(n_train, n_group)
    for i, e in enumerate(train_elements):
        train_input[i, e] = 1
    
    test_elements = torch.randint(0, n_group, (n_test,))
    test_labels = torch.tensor([1 if x.item() in subgroup else 0 for x in test_elements])
    
    test_input = torch.zeros(n_test, n_group)
    for i, e in enumerate(test_elements):
        test_input[i, e] = 1
    
    train_input = train_input.to(device)
    train_labels = train_labels.to(device)
    test_input = test_input.to(device)
    test_labels = test_labels.to(device)
    
    model = nn.Sequential(
        nn.Linear(n_group, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 2)
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_input)
        loss = criterion(out, train_labels)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(test_input).argmax(1)
                acc = (pred == test_labels).float().mean().item()
            best_acc = max(best_acc, acc)
            print(f"    Epoch {epoch+1}: Acc={acc:.2%}")
    
    return best_acc

subgroup_acc = test_subgroup_recognition(60, 12, 8000, 300)
results["subgroup_recognition"] = {"accuracy": subgroup_acc}

# ============================================================================
# 总结
# ============================================================================

print("\n" + "=" * 60)
print("E2 完整测试总结")
print("=" * 60)

# 收集所有结果
all_results = []

print("\n循环群 Z_n:")
for n, res in cyclic_results.items():
    status = "OK" if res["accuracy"] > 0.5 else ("WARN" if res["accuracy"] > 0.2 else "FAIL")
    print(f"  [{status}] Z_{n}: {res['accuracy']:.2%} (隐藏层={res['hidden_dim']}, 层数={res['layers']})")
    all_results.append(res["accuracy"])

print("\n置换群 S_n:")
for n, res in perm_results.items():
    status = "OK" if res["accuracy"] > 0.3 else "FAIL"
    print(f"  [{status}] S_{n} ({res['n_elements']}元素): {res['accuracy']:.2%}")
    all_results.append(res["accuracy"])

print("\n高级测试:")
status = "OK" if grok_acc > 0.5 else "WARN"
print(f"  [{status}] Grokking (Z_113): {grok_acc:.2%}" + (f", 发生在 Epoch {grok_epoch}" if grok_epoch else ""))
all_results.append(grok_acc)

status = "OK" if homo_acc > 0.5 else "WARN"
print(f"  [{status}] 群同态学习: {homo_acc:.2%}")
all_results.append(homo_acc)

status = "OK" if subgroup_acc > 0.8 else "WARN"
print(f"  [{status}] 子群识别: {subgroup_acc:.2%}")
all_results.append(subgroup_acc)

avg_acc = np.mean(all_results)
n_passed_50 = sum(1 for a in all_results if a > 0.5)
n_passed_30 = sum(1 for a in all_results if a > 0.3)

print(f"\n平均准确率: {avg_acc:.2%}")
print(f"通过 (>50%): {n_passed_50}/{len(all_results)}")
print(f"通过 (>30%): {n_passed_30}/{len(all_results)}")

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

with open("tempdata/e2_full_report.json", "w") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "average_accuracy": avg_acc,
        "n_passed_50": n_passed_50,
        "n_passed_30": n_passed_30,
        "n_total": len(all_results),
        "results": convert_to_native(results)
    }, f, indent=2)

print(f"\n报告已保存到: tempdata/e2_full_report.json")

# 最终评估
if avg_acc > 0.6:
    print("\n结论: 群论学习能力良好，模型能有效学习代数结构")
elif avg_acc > 0.4:
    print("\n结论: 群论学习能力中等，建议继续优化模型架构")
else:
    print("\n结论: 群论学习能力不足，需要重新设计模型")
