# -*- coding: utf-8 -*-
"""
================================================================================
SCRC V3: 统一结构——学习与推理是同一回路的两种模式
================================================================================

核心洞察：
  学习和推理不是两套独立系统，而是同一个神经回路的两种运行模式。

统一回路：
  前向（推理模式）: z = top_k(W · x)          ← 稀疏特征匹配
  反向（学习模式）: x̂ = W^T · z              ← 用稀疏码重建输入
                    e = x - x̂                ← 预测误差（局部计算！）
                    ΔW = η · z · e^T          ← 误差驱动的局部学习

这就是"预测编码 + 稀疏编码"：
  - 推理时：W·x 然后 top-k（快速、稀疏、低能耗）
  - 学习时：同一个 W 的转置做重建，误差反传（局部、无需全局BP）

关键区别 vs 纯 Hebbian：
  Hebbian:  ΔW = η · z · x^T       ← 只看"输入是什么"
  预测编码: ΔW = η · z · (x-W^Tz)^T ← 看"我没学到的是什么"（误差驱动）

这就是同一个结构的两种模式：
  - W 做前向投射（推理）
  - W^T 做反向重建（学习时生成预测）
  - 误差 e 是局部可计算的，不需要全局梯度
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os

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


class UnifiedCircuit(nn.Module):
    """
    统一神经回路：学习和推理是同一结构的两种模式
    
    结构（只有一个矩阵 W）：
    
    推理模式（前向）：
      scores = W · x       ← 内积匹配
      z = top_k(scores)    ← 竞争稀疏
      
    学习模式（前向+反向）：
      scores = W · x       ← 同一个前向
      z = top_k(scores)    ← 同一个竞争
      x̂ = W^T · z         ← W 的转置做重建（不是新参数！）
      e = x - x̂           ← 预测误差（纯局部！）
      ΔW = η · z · e^T     ← 用误差而非原始输入更新
    
    关键：W 和 W^T 是同一个矩阵，不是两个独立参数。
    前向用 W，反向重建用 W^T。这就是"同一结构的两种模式"。
    """
    def __init__(self, input_dim, num_units, k, lr=0.01):
        super().__init__()
        self.k = k
        self.lr = lr
        self.num_units = num_units
        self.input_dim = input_dim
        
        # 唯一的参数：特征字典 W
        W = torch.randn(num_units, input_dim, device=device) * 0.01
        W = F.normalize(W, dim=1)
        self.W = nn.Parameter(W, requires_grad=False)
        
    def forward(self, x, mode='infer'):
        """
        mode='infer': 只做前向推理（快速、低能耗）
        mode='learn': 前向 + 反向重建 + 误差驱动更新
        """
        # ===== 前向（推理模式）=====
        x_norm = F.normalize(x, dim=1)
        scores = x_norm @ self.W.T  # [batch, num_units]
        
        # Top-K 竞争抑制
        topk_vals, topk_idx = torch.topk(scores, self.k, dim=1)
        z = torch.zeros_like(scores)
        z.scatter_(1, topk_idx, topk_vals.clamp(min=0))
        
        if mode == 'learn':
            # ===== 反向（学习模式）=====
            # 用同一个 W 的转置重建输入
            x_hat = z @ self.W  # [batch, input_dim]  ← W^T · z
            
            # 预测误差（局部计算，无需全局梯度！）
            error = x_norm - x_hat  # [batch, input_dim]
            
            # 误差驱动的学习
            # ΔW = η · z^T · error / batch_size
            batch_size = x.shape[0]
            delta_W = (z.T @ error) / batch_size  # [num_units, input_dim]
            self.W.data += self.lr * delta_W
            
            # 归一化保持稳定
            self.W.data = F.normalize(self.W.data, dim=1)
        
        return z


class UnifiedClassifier(nn.Module):
    """多级统一回路 + 线性读出"""
    def __init__(self):
        super().__init__()
        self.layer1 = UnifiedCircuit(784, 500, k=50, lr=0.1)
        self.layer2 = UnifiedCircuit(500, 200, k=20, lr=0.1)
        self.readout = nn.Linear(200, 10).to(device)
        
    def forward(self, x, mode='infer'):
        z1 = self.layer1(x, mode=mode)
        z2 = self.layer2(z1, mode=mode)
        logits = self.readout(z2)
        return logits, z1, z2


class SimpleMLP(nn.Module):
    """MLP 对照组"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 10)
    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


def train_unified(epochs=10):
    """训练统一回路"""
    model = UnifiedClassifier().to(device)
    # 只有读出头用梯度
    optimizer = torch.optim.Adam(model.readout.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    params = sum(p.numel() for p in model.parameters())
    
    print(f"  参数量: {params:,}")
    print(f"  结构: 784→500(k=50)→200(k=20)→10")
    print(f"  特征学习: 预测编码（局部误差驱动）")
    print(f"  读出学习: Adam（梯度下降）")
    
    t0 = time.time()
    for epoch in range(epochs):
        correct = total = 0
        loss_sum = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 学习模式：前向+反向重建+误差更新
            logits, z1, z2 = model(data, mode='learn')
            
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            correct += (logits.argmax(1) == target).sum().item()
            total += target.size(0)
            loss_sum += loss.item()
        
        acc = correct / total * 100
        avg_loss = loss_sum / len(train_loader)
        
        # 计算重建质量
        with torch.no_grad():
            sample = data[:100]
            sample_norm = F.normalize(sample, dim=1)
            z1_test = model.layer1(sample, mode='infer')
            recon = z1_test @ model.layer1.W
            recon_error = (sample_norm - recon).pow(2).mean().item()
        
        print(f"  Epoch {epoch+1:2d}/{epochs} | 损失: {avg_loss:.4f} | "
              f"准确率: {acc:.2f}% | 重建误差: {recon_error:.4f}")
    
    train_time = time.time() - t0
    
    # 测试（纯推理模式——不更新权重）
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits, _, _ = model(data, mode='infer')  # 推理模式！
            correct += (logits.argmax(1) == target).sum().item()
            total += target.size(0)
    test_acc = correct / total * 100
    
    # 稀疏性
    with torch.no_grad():
        sample_data = next(iter(test_loader))[0].to(device)
        _, z1, z2 = model(sample_data, mode='infer')
        sp1 = (z1 == 0).float().mean().item() * 100
        sp2 = (z2 == 0).float().mean().item() * 100
    
    # 模板多样性
    W1 = model.layer1.W.data
    W1n = F.normalize(W1, dim=1)
    sim = (W1n @ W1n.T)
    mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=device)
    avg_sim = sim[mask].mean().item()
    
    print(f"\n  ✅ 测试准确率: {test_acc:.2f}%")
    print(f"  ⏱️  训练时间: {train_time:.1f}s")
    print(f"  🔬 稀疏性: L1={sp1:.1f}%, L2={sp2:.1f}%")
    print(f"  📊 模板多样性: 平均余弦相似度={avg_sim:.4f}")
    
    return test_acc, train_time, params, model


def train_mlp(epochs=10):
    """MLP 对照组"""
    model = SimpleMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    params = sum(p.numel() for p in model.parameters())
    
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
        print(f"  Epoch {epoch+1:2d}/{epochs} | 损失: {loss_sum/len(train_loader):.4f} | 准确率: {correct/total*100:.2f}%")
    
    train_time = time.time() - t0
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            correct += (model(data).argmax(1) == target).sum().item()
            total += target.size(0)
    test_acc = correct / total * 100
    print(f"\n  ✅ 测试准确率: {test_acc:.2f}%")
    print(f"  ⏱️  训练时间: {train_time:.1f}s")
    return test_acc, train_time, params


if __name__ == '__main__':
    print("="*70)
    print("  SCRC V3: 统一回路——学习与推理是同一结构的两种模式")
    print("="*70)
    print()
    print("  前向 = 推理: z = top_k(W · x)")
    print("  反向 = 学习: e = x - W^T·z, ΔW = η·z·e^T")
    print("  关键: W 和 W^T 是同一个矩阵的两个方向")
    print()
    
    print("-"*60)
    print("  [A] 统一回路（预测编码 + Top-K 稀疏）")
    print("-"*60)
    unified_acc, unified_time, unified_params, unified_model = train_unified(10)
    
    print()
    print("-"*60)
    print("  [B] MLP 对照组（全程反向传播）")
    print("-"*60)
    mlp_acc, mlp_time, mlp_params = train_mlp(10)
    
    # 总结
    print()
    print("="*70)
    print("  🏆 实验总结：统一回路 vs 反向传播")
    print("="*70)
    print(f"  {'模型':<25} {'准确率':>8} {'参数量':>12} {'时间':>8} {'学习方式'}")
    print(f"  {'-'*70}")
    print(f"  {'统一回路(预测编码)':<23} {unified_acc:>7.2f}% {unified_params:>11,} {unified_time:>7.1f}s {'局部误差+BP读出'}")
    print(f"  {'MLP(全程BP)':<24} {mlp_acc:>7.2f}% {mlp_params:>11,} {mlp_time:>7.1f}s {'全局反向传播'}")
    print()
    
    improvement_vs_v2 = unified_acc - 21.06
    print(f"  📈 vs SCRC V2 (纯Hebbian 21.06%): {'+' if improvement_vs_v2 > 0 else ''}{improvement_vs_v2:.2f}%")
    print(f"  📈 vs MLP: {unified_acc - mlp_acc:+.2f}%")
    
    results = {
        'unified_circuit': {'acc': unified_acc, 'time': unified_time, 'params': unified_params},
        'mlp_bp': {'acc': mlp_acc, 'time': mlp_time, 'params': mlp_params},
        'scrc_v2_reference': 21.06,
        'scrc_v1_reference': 17.65,
    }
    os.makedirs('tempdata', exist_ok=True)
    with open('tempdata/scrc_v3_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  📁 结果已保存到 tempdata/scrc_v3_results.json")
