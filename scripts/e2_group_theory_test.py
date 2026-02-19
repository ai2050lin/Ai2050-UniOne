"""
E2: 大规模群论测试 - 增强版
============================

测试目标: 验证模型学习群论结构的能力
改进:
- 增加训练轮次
- 使用更深的网络
- 测试不同群规模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import math

print("=" * 60)
print("E2: 大规模群论测试 (增强版)")
print("=" * 60)

results = {}

# ============================================================================
# 测试 1: Z_n 循环群 (加法模 n)
# ============================================================================

print("\n[1] Z_n 循环群测试...")

def test_cyclic_group(n_elements, n_epochs=100, hidden_dim=128, n_layers=3):
    """测试循环群 Z_n 的学习"""
    
    n_train, n_test = 2000, 500
    
    # 生成数据
    train_a = torch.randint(0, n_elements, (n_train,))
    train_b = torch.randint(0, n_elements, (n_train,))
    train_labels = (train_a + train_b) % n_elements
    
    # One-hot 编码
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
    
    # 构建更深网络
    layers = [nn.Linear(n_elements * 2, hidden_dim), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, n_elements))
    
    model = nn.Sequential(*layers)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = model(train_input)
        loss = criterion(out, train_labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                pred = model(test_input).argmax(1)
                acc = (pred == test_labels).float().mean().item()
            print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.2%}")
    
    # 最终测试
    with torch.no_grad():
        pred = model(test_input).argmax(1)
        acc = (pred == test_labels).float().mean().item()
    
    return acc, losses[-1]

# 测试不同规模
group_sizes = [20, 50, 100]
cyclic_results = {}

for n in group_sizes:
    print(f"\n  Z_{n} 群测试:")
    acc, final_loss = test_cyclic_group(n, n_epochs=100, hidden_dim=128, n_layers=3)
    cyclic_results[n] = {"accuracy": acc, "final_loss": final_loss}
    print(f"  最终准确率: {acc:.2%}")

results["cyclic_groups"] = cyclic_results

# ============================================================================
# 测试 2: S_n 置换群 (简化版)
# ============================================================================

print("\n[2] S_n 置换群测试...")

def test_permutation_group(n, n_epochs=150, hidden_dim=256, n_layers=4):
    """测试置换群 S_n (n! 元素)"""
    
    from math import factorial
    
    n_perms = min(factorial(n), 1000)  # 限制大小
    
    # 生成置换
    def generate_permutations(n, max_count=1000):
        perms = []
        import itertools
        for i, p in enumerate(itertools.permutations(range(n))):
            if i >= max_count:
                break
            perms.append(list(p))
        return perms
    
    perms = generate_permutations(n, n_perms)
    n_perms = len(perms)
    
    # 生成训练数据: 置换乘法
    n_train = min(n_perms * n_perms // 2, 5000)
    n_test = min(n_perms * n_perms // 4, 1000)
    
    def compose(p1, p2):
        """置换复合 p1 ∘ p2"""
        return [p1[p2[i]] for i in range(len(p1))]
    
    def perm_to_tensor(p):
        """置换转为 tensor"""
        t = torch.zeros(len(p) * len(p))
        for i, v in enumerate(p):
            t[i * len(p) + v] = 1
        return t
    
    # 训练数据
    train_data = []
    train_labels = []
    for _ in range(n_train):
        i, j = np.random.randint(0, n_perms, 2)
        p1, p2 = perms[i], perms[j]
        result = compose(p1, p2)
        
        inp = torch.cat([perm_to_tensor(p1), perm_to_tensor(p2)])
        train_data.append(inp)
        train_labels.append(perms.index(result) if result in perms else 0)
    
    train_input = torch.stack(train_data)
    train_labels = torch.tensor(train_labels)
    
    # 测试数据
    test_data = []
    test_labels = []
    for _ in range(n_test):
        i, j = np.random.randint(0, n_perms, 2)
        p1, p2 = perms[i], perms[j]
        result = compose(p1, p2)
        
        inp = torch.cat([perm_to_tensor(p1), perm_to_tensor(p2)])
        test_data.append(inp)
        test_labels.append(perms.index(result) if result in perms else 0)
    
    test_input = torch.stack(test_data)
    test_labels = torch.tensor(test_labels)
    
    # 模型
    input_dim = n * n * 2
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, n_perms))
    
    model = nn.Sequential(*layers)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = model(train_input)
        loss = criterion(out, train_labels)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 30 == 0:
            with torch.no_grad():
                pred = model(test_input).argmax(1)
                acc = (pred == test_labels).float().mean().item()
            print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.2%}")
    
    with torch.no_grad():
        pred = model(test_input).argmax(1)
        acc = (pred == test_labels).float().mean().item()
    
    return acc

# 测试 S_3, S_4
perm_results = {}
for n in [3, 4]:
    print(f"\n  S_{n} 置换群测试 ({math.factorial(n)} 元素):")
    acc = test_permutation_group(n, n_epochs=100, hidden_dim=128, n_layers=3)
    perm_results[n] = {"accuracy": acc}
    print(f"  最终准确率: {acc:.2%}")

results["permutation_groups"] = perm_results

# ============================================================================
# 测试 3: 群同态测试
# ============================================================================

print("\n[3] 群同态学习测试...")

def test_group_homomorphism(n_elements, n_epochs=100):
    """测试模型能否学习群同态"""
    
    n_train, n_test = 2000, 500
    
    # 定义同态: Z_{2n} -> Z_n (模运算)
    train_a = torch.randint(0, 2 * n_elements, (n_train,))
    train_b = torch.randint(0, 2 * n_elements, (n_train,))
    
    # 同态性质: f(a + b) = f(a) + f(b) mod n
    train_labels = ((train_a + train_b) % n_elements)
    
    # 输入: a, b 的 one-hot
    train_input = torch.zeros(n_train, 4 * n_elements)
    for i in range(n_train):
        train_input[i, train_a[i]] = 1
        train_input[i, 2 * n_elements + train_b[i]] = 1
    
    test_a = torch.randint(0, 2 * n_elements, (n_test,))
    test_b = torch.randint(0, 2 * n_elements, (n_test,))
    test_labels = ((test_a + test_b) % n_elements)
    
    test_input = torch.zeros(n_test, 4 * n_elements)
    for i in range(n_test):
        test_input[i, test_a[i]] = 1
        test_input[i, 2 * n_elements + test_b[i]] = 1
    
    # 模型
    model = nn.Sequential(
        nn.Linear(4 * n_elements, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, n_elements)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = model(train_input)
        loss = criterion(out, train_labels)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        pred = model(test_input).argmax(1)
        acc = (pred == test_labels).float().mean().item()
    
    return acc

homo_acc = test_group_homomorphism(20, n_epochs=100)
results["homomorphism"] = {"accuracy": homo_acc}
print(f"  同态学习准确率: {homo_acc:.2%}")

# ============================================================================
# 测试 4: Grokking 现象测试
# ============================================================================

print("\n[4] Grokking 现象测试...")

def test_grokking(n_elements, n_epochs=500):
    """测试 Grokking 现象 - 模型突然泛化"""
    
    n_train, n_test = 500, 200
    
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
    
    # 使用较小模型 (更容易观察 Grokking)
    model = nn.Sequential(
        nn.Linear(n_elements * 2, 64),
        nn.ReLU(),
        nn.Linear(64, n_elements)
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    
    train_accs = []
    test_accs = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = model(train_input)
        loss = criterion(out, train_labels)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                train_pred = model(train_input).argmax(1)
                train_acc = (train_pred == train_labels).float().mean().item()
                test_pred = model(test_input).argmax(1)
                test_acc = (test_pred == test_labels).float().mean().item()
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            print(f"  Epoch {epoch+1}: Train={train_acc:.2%}, Test={test_acc:.2%}")
    
    # 检测 Grokking: 测试准确率突然提升
    grokked = test_accs[-1] > 0.8 and test_accs[0] < 0.3
    
    return test_accs[-1], grokked, train_accs[-1]

final_acc, grokked, train_acc = test_grokking(30, n_epochs=300)
results["grokking"] = {
    "final_test_acc": final_acc,
    "final_train_acc": train_acc,
    "grokked": grokked
}
print(f"  最终测试准确率: {final_acc:.2%}")
print(f"  Grokking 现象: {'是' if grokked else '否'}")

# ============================================================================
# 总结
# ============================================================================

print("\n" + "=" * 60)
print("E2 测试总结")
print("=" * 60)

# 计算总体通过率
all_accs = []
for n, res in cyclic_results.items():
    all_accs.append(res["accuracy"])
for n, res in perm_results.items():
    all_accs.append(res["accuracy"])
all_accs.append(homo_acc)
all_accs.append(final_acc)

avg_acc = np.mean(all_accs)
n_passed = sum(1 for acc in all_accs if acc > 0.3)

print(f"\n平均准确率: {avg_acc:.2%}")
print(f"通过测试数: {n_passed}/{len(all_accs)} (阈值 30%)")

print("\n详细结果:")
for n, res in cyclic_results.items():
    status = "OK" if res["accuracy"] > 0.3 else "FAIL"
    print(f"  [{status}] Z_{n} 循环群: {res['accuracy']:.2%}")

for n, res in perm_results.items():
    status = "OK" if res["accuracy"] > 0.3 else "FAIL"
    print(f"  [{status}] S_{n} 置换群: {res['accuracy']:.2%}")

status = "OK" if homo_acc > 0.3 else "FAIL"
print(f"  [{status}] 群同态学习: {homo_acc:.2%}")

status = "OK" if final_acc > 0.3 else "FAIL"
print(f"  [{status}] Grokking 测试: {final_acc:.2%}")

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

with open("tempdata/e2_group_theory_report.json", "w") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "average_accuracy": avg_acc,
        "n_passed": n_passed,
        "n_total": len(all_accs),
        "results": convert_to_native(results)
    }, f, indent=2)

print(f"\n报告已保存到: tempdata/e2_group_theory_report.json")

# 最终评估
if avg_acc > 0.5:
    print("\n结论: 群论学习能力良好")
elif avg_acc > 0.3:
    print("\n结论: 群论学习能力一般，需要改进")
else:
    print("\n结论: 群论学习能力不足，需要重新设计")
