# -*- coding: utf-8 -*-
"""
================================================================================
SCRC 验证实验 V2：修正版稀疏竞争循环回路
================================================================================
V1 问题诊断：
  1. 纯 Hebbian 外积导致所有模板坍缩到数据均值方向（缺少竞争性分化）
  2. 没有输入归一化，内积尺度不可控
  3. 缺少"反 Hebbian"机制——未被选中的模板不会远离输入

V2 修正：
  1. 竞争性 Hebbian（Oja's Rule）：自动归一化，防止模板坍缩
  2. Winner-Take-All + 负反馈：赢家靠近输入，输家远离
  3. 输入 L2 归一化
  
核心公式不变：Z = top_k(W · X)
但学习规则升级为竞争性 Oja 规则：
  对赢家: Δw_j = η · (x - (w_j·x)·w_j)    [靠近输入，同时保持单位范数]
  对输家: 不更新（生物学中的沉默神经元不可塑）
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[设备] 使用: {device}")
if device.type == 'cuda':
    print(f"[GPU] {torch.cuda.get_device_name(0)}")

# 数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
train_dataset = datasets.MNIST('./tempdata', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./tempdata', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


class SCRC_v2(nn.Module):
    """
    稀疏竞争循环回路 V2
    
    修正版使用竞争性 Hebbian 学习（Oja's Rule 变体）：
    - 赢家（top-k 激活的单元）：模板向输入方向移动
    - 输家：保持不变（生物学中沉默神经元不改变突触）
    - 所有模板始终保持单位范数（自动归一化）
    
    这更接近大脑皮层的真实学习机制。
    """
    def __init__(self, input_dim, num_units, k, lr=0.01):
        super().__init__()
        self.k = k
        self.lr = lr
        self.num_units = num_units
        self.input_dim = input_dim
        
        # 初始化为单位范数随机方向
        W = torch.randn(num_units, input_dim, device=device)
        W = F.normalize(W, dim=1)
        self.W = nn.Parameter(W, requires_grad=False)
        
    def forward(self, x):
        # L2 归一化输入
        x_norm = F.normalize(x, dim=1)
        
        # 兴奋投射：余弦相似度
        scores = x_norm @ self.W.T  # [batch, num_units]
        
        # 竞争抑制：Top-K
        topk_vals, topk_idx = torch.topk(scores, self.k, dim=1)
        
        # 稀疏输出
        z = torch.zeros_like(scores)
        z.scatter_(1, topk_idx, topk_vals.clamp(min=0))
        
        return z, x_norm, topk_idx
    
    def learn(self, x_norm, topk_idx):
        """
        竞争性 Hebbian 学习（Oja's Rule 批量版）
        
        对每个样本，只有 top-k 赢家的模板会更新：
        Δw_j = η * (x - (w_j·x) * w_j)
        
        这保证了：
        1. 模板向输入方向移动（学习新特征）
        2. ||w_j|| 始终保持为 1（Oja 归一化）
        3. 只有赢家更新（竞争分化）
        """
        batch_size = x_norm.shape[0]
        
        # 展平 topk_idx
        for b in range(min(batch_size, 64)):  # 限制批量更新避免过慢
            x_b = x_norm[b]  # [D]
            winners = topk_idx[b]  # [k]
            
            for j in winners:
                w_j = self.W.data[j]  # [D]
                proj = (w_j @ x_b)  # 标量
                # Oja's rule
                delta = self.lr * (x_b - proj * w_j)
                self.W.data[j] += delta
        
        # 重新归一化
        self.W.data = F.normalize(self.W.data, dim=1)


class SCRC_Classifier_v2(nn.Module):
    def __init__(self):
        super().__init__()
        # 两级 SCRC
        self.scrc1 = SCRC_v2(784, 500, k=25, lr=0.05)
        self.scrc2 = SCRC_v2(500, 200, k=10, lr=0.05)
        # 线性读出
        self.readout = nn.Linear(200, 10).to(device)
        
    def forward(self, x, learn=False):
        z1, x1_norm, idx1 = self.scrc1(x)
        z2, z1_norm, idx2 = self.scrc2(z1)
        
        if learn:
            self.scrc1.learn(x1_norm, idx1)
            self.scrc2.learn(z1_norm, idx2)
        
        logits = self.readout(z2)
        return logits, z1, z2


class SimpleAttention(nn.Module):
    """Attention 对照组（与 V1 相同）"""
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        h = F.relu(self.embed(x))
        Q, K, V = self.W_Q(h), self.W_K(h), self.W_V(h)
        attn = F.softmax(Q * K / (self.hidden_dim ** 0.5), dim=-1)
        return self.head(attn * V)


class SimpleMLP(nn.Module):
    """朴素 MLP 对照组——最基础的基线"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 10)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train_bp_model(model, name, epochs=5):
    """用反向传播训练一个模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    t0 = time.time()
    for epoch in range(epochs):
        correct = total = 0
        loss_sum = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == target).sum().item()
            total += target.size(0)
            loss_sum += loss.item()
        print(f"  [{name}] Epoch {epoch+1}/{epochs} | 损失: {loss_sum/len(train_loader):.4f} | 准确率: {correct/total*100:.2f}%")
    
    train_time = time.time() - t0
    
    # 测试
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            correct += (logits.argmax(1) == target).sum().item()
            total += target.size(0)
    test_acc = correct / total * 100
    params = sum(p.numel() for p in model.parameters())
    return test_acc, train_time, params


def train_scrc(epochs=5):
    """训练 SCRC（Hebbian + 读出头 BP 混合）"""
    model = SCRC_Classifier_v2().to(device)
    optimizer = torch.optim.Adam(model.readout.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    params = sum(p.numel() for p in model.parameters())
    
    t0 = time.time()
    for epoch in range(epochs):
        correct = total = 0
        loss_sum = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            logits, _, _ = model(data, learn=True)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            correct += (logits.argmax(1) == target).sum().item()
            total += target.size(0)
            loss_sum += loss.item()
            
            if (batch_idx+1) % 50 == 0:
                print(f"  [SCRC] Epoch {epoch+1} | 批次 {batch_idx+1}/{len(train_loader)} | 准确率: {correct/total*100:.2f}%")
        
        print(f"  [SCRC] Epoch {epoch+1}/{epochs} | 损失: {loss_sum/len(train_loader):.4f} | 准确率: {correct/total*100:.2f}%")
    
    train_time = time.time() - t0
    
    # 测试
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits, z1, z2 = model(data, learn=False)
            correct += (logits.argmax(1) == target).sum().item()
            total += target.size(0)
    test_acc = correct / total * 100
    
    # 稀疏性分析
    with torch.no_grad():
        sample_data = next(iter(test_loader))[0].to(device)
        _, z1, z2 = model(sample_data, learn=False)
        sp1 = (z1 == 0).float().mean().item() * 100
        sp2 = (z2 == 0).float().mean().item() * 100
        print(f"\n  [SCRC 稀疏性] 第1层: {sp1:.1f}% 零激活 | 第2层: {sp2:.1f}% 零激活")
    
    # 模板多样性分析
    W1 = model.scrc1.W.data
    W1_norm = F.normalize(W1, dim=1)
    sim = (W1_norm @ W1_norm.T)
    mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=device)
    avg_sim = sim[mask].mean().item()
    print(f"  [SCRC 模板多样性] 第1层模板平均余弦相似度: {avg_sim:.4f}")
    
    return test_acc, train_time, params, model


if __name__ == '__main__':
    print("="*70)
    print("  SCRC V2 验证实验：竞争性 Hebbian 稀疏回路")
    print("="*70)
    
    # 1. SCRC V2
    print("\n" + "-"*60)
    print("  [A] SCRC V2（竞争性 Hebbian + 线性读出 BP）")
    print("-"*60)
    scrc_acc, scrc_time, scrc_params, scrc_model = train_scrc(epochs=5)
    print(f"\n  ✅ SCRC V2 测试准确率: {scrc_acc:.2f}%")
    print(f"  ⏱️  耗时: {scrc_time:.1f}s | 📊 参数: {scrc_params:,}")
    
    # 2. Attention 对照
    print("\n" + "-"*60)
    print("  [B] Attention 对照组（全程 BP）")
    print("-"*60)
    attn_model = SimpleAttention(784, 200, 10).to(device)
    attn_acc, attn_time, attn_params = train_bp_model(attn_model, "Attn", 5)
    print(f"\n  ✅ Attention 测试准确率: {attn_acc:.2f}%")
    print(f"  ⏱️  耗时: {attn_time:.1f}s | 📊 参数: {attn_params:,}")
    
    # 3. MLP 基线
    print("\n" + "-"*60)
    print("  [C] MLP 基线对照（全程 BP）")
    print("-"*60)
    mlp_model = SimpleMLP().to(device)
    mlp_acc, mlp_time, mlp_params = train_bp_model(mlp_model, "MLP", 5)
    print(f"\n  ✅ MLP 测试准确率: {mlp_acc:.2f}%")
    print(f"  ⏱️  耗时: {mlp_time:.1f}s | 📊 参数: {mlp_params:,}")
    
    # 总结
    print("\n" + "="*70)
    print("  🏆 实验总结")
    print("="*70)
    print(f"  {'模型':<22} {'准确率':>8} {'参数量':>12} {'时间':>8} {'学习方式'}")
    print(f"  {'-'*65}")
    print(f"  {'SCRC V2 (Hebbian)':<22} {scrc_acc:>7.2f}% {scrc_params:>11,} {scrc_time:>7.1f}s {'Hebbian+BP读出'}")
    print(f"  {'Attention (BP)':<22} {attn_acc:>7.2f}% {attn_params:>11,} {attn_time:>7.1f}s {'全程BP'}")
    print(f"  {'MLP (BP)':<22} {mlp_acc:>7.2f}% {mlp_params:>11,} {mlp_time:>7.1f}s {'全程BP'}")
    
    # 保存
    results = {
        'scrc_v2': {'acc': scrc_acc, 'time': scrc_time, 'params': scrc_params},
        'attention': {'acc': attn_acc, 'time': attn_time, 'params': attn_params},
        'mlp': {'acc': mlp_acc, 'time': mlp_time, 'params': mlp_params},
    }
    with open('tempdata/scrc_v2_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  📁 结果已保存到 tempdata/scrc_v2_results.json")
