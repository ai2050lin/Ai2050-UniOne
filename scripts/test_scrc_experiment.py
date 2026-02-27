# -*- coding: utf-8 -*-
"""
================================================================================
SCRC 验证实验：稀疏竞争循环回路 vs Attention
================================================================================
实验目标：
  1. 实现 SCRC（Sparse Competitive Recurrent Circuit）核心结构
  2. 实验一：SCRC vs Attention 在 MNIST 上的特征拟合能力对比
  3. 实验二：SCRC 多级串联时是否自动涌现层次化特征

核心公式：
  Z = top_k(W · X)
  ΔW = η · Z · X^T    (Hebbian Learning)
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

# ============================================================
# 确保 GPU
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[设备] 使用: {device}")
if device.type == 'cuda':
    print(f"[GPU] {torch.cuda.get_device_name(0)}, 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# 数据加载
# ============================================================
print("\n[数据] 加载 MNIST ...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # 展平为 784 维
])
train_dataset = datasets.MNIST('./tempdata', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./tempdata', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
print(f"[数据] 训练集: {len(train_dataset)} 样本, 测试集: {len(test_dataset)} 样本")


# ============================================================
# 核心结构一：SCRC — 稀疏竞争循环回路
# ============================================================
class SCRC(nn.Module):
    """
    稀疏竞争循环回路 (Sparse Competitive Recurrent Circuit)
    
    这是我们提出的大脑核心计算原语：
      Z = top_k(W · X)           -- 兴奋投射 + 竞争抑制
      ΔW = η · Z · X^T           -- Hebbian 可塑性
    
    三个组分：
      E (兴奋投射)：W · X，每个神经元的突触权重与输入做内积
      I (竞争抑制)：top_k，只保留最强的 k 个激活
      P (Hebbian 可塑)：ΔW = η · z · x^T，同时激活则加强连接
    """
    def __init__(self, input_dim, num_units, sparsity_k, lr_hebbian=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.num_units = num_units
        self.k = sparsity_k
        self.lr = lr_hebbian
        
        # W：特征模板矩阵 [num_units, input_dim]
        # 每一行是一个"特征检测器"
        self.W = nn.Parameter(
            torch.randn(num_units, input_dim, device=device) * 0.01,
            requires_grad=False  # 不用梯度！用 Hebbian！
        )
        
    def forward(self, x):
        """
        x: [batch, input_dim]
        返回: [batch, num_units] 稀疏激活
        """
        # 兴奋投射：每个神经元与输入做内积
        scores = x @ self.W.T  # [batch, num_units]
        
        # 竞争抑制：只保留 Top-K 最强的
        topk_vals, topk_idx = torch.topk(scores, self.k, dim=1)
        
        # 构建稀疏输出
        z = torch.zeros_like(scores)
        z.scatter_(1, topk_idx, F.relu(topk_vals))  # ReLU 确保非负
        
        return z
    
    def hebbian_update(self, x, z):
        """
        Hebbian 学习：同时激活则连接加强
        ΔW = η · (z^T · x) / batch_size
        
        加入权重归一化防止爆炸
        """
        batch_size = x.shape[0]
        # 外积更新
        delta_W = (z.T @ x) / batch_size  # [num_units, input_dim]
        self.W.data += self.lr * delta_W
        
        # L2 归一化每一行（保持模板的方向，控制幅值）
        norms = self.W.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W.data = self.W.data / norms


# ============================================================
# 核心结构二：简单 Attention 对照组
# ============================================================
class SimpleAttention(nn.Module):
    """
    标准自注意力 + 分类头，用于公平对比。
    使用反向传播训练。
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads=4):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        h = F.relu(self.embed(x))  # [batch, hidden]
        Q = self.W_Q(h)
        K = self.W_K(h)
        V = self.W_V(h)
        # 自注意力（单令牌情况下等价于加权投射）
        attn = F.softmax(Q * K / (self.hidden_dim ** 0.5), dim=-1)
        out = attn * V
        return self.head(out)


# ============================================================
# SCRC 分类器（多级 SCRC + 简单线性读出）
# ============================================================
class SCRCClassifier(nn.Module):
    """
    多级 SCRC 串联 + 线性读出头
    
    验证：层次特征是否自动涌现
    """
    def __init__(self, dims, ks, lr_hebbian=0.01):
        """
        dims: [input_dim, layer1_units, layer2_units, ...]
        ks:   [k1, k2, ...]  每一级的稀疏度
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(SCRC(dims[i], dims[i+1], ks[i], lr_hebbian))
        
        # 线性读出头（这个用梯度训练，因为它只是一个标签映射）
        self.readout = nn.Linear(dims[-1], 10).to(device)
        
    def forward(self, x, learn=False):
        activations = [x]
        for layer in self.layers:
            x = layer(x)
            activations.append(x)
            
        logits = self.readout(x)
        
        if learn:
            # Hebbian 更新每一层
            for i, layer in enumerate(self.layers):
                layer.hebbian_update(activations[i], activations[i+1])
        
        return logits, activations


# ============================================================
# 实验一：SCRC vs Attention 特征拟合对比
# ============================================================
def experiment_1_comparison():
    print("\n" + "="*70)
    print("  实验一：SCRC vs Attention — MNIST 特征拟合能力对比")
    print("="*70)
    
    results = {}
    
    # ---------- SCRC 方案 ----------
    print("\n--- [A] SCRC 方案 ---")
    print("  结构: 784 → 500(k=25) → 200(k=10) → 线性读出(10)")
    print("  学习: SCRC 层用 Hebbian，读出头用梯度下降")
    
    scrc_model = SCRCClassifier(
        dims=[784, 500, 200],
        ks=[25, 10],
        lr_hebbian=0.005
    ).to(device)
    
    # 读出头的优化器
    readout_optimizer = torch.optim.Adam(scrc_model.readout.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    scrc_params = sum(p.numel() for p in scrc_model.parameters())
    print(f"  总参数量: {scrc_params:,}")
    
    # 训练
    scrc_train_start = time.time()
    for epoch in range(5):
        correct = 0
        total = 0
        epoch_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            logits, _ = scrc_model(data, learn=True)  # Hebbian 更新
            
            loss = criterion(logits, target)
            readout_optimizer.zero_grad()
            loss.backward()
            readout_optimizer.step()
            
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            epoch_loss += loss.item()
            
        acc = correct / total * 100
        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/5 | 损失: {avg_loss:.4f} | 训练准确率: {acc:.2f}%")
    
    scrc_train_time = time.time() - scrc_train_start
    
    # 测试
    scrc_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits, _ = scrc_model(data, learn=False)
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    scrc_test_acc = correct / total * 100
    print(f"\n  ✅ SCRC 测试准确率: {scrc_test_acc:.2f}%")
    print(f"  ⏱️  训练耗时: {scrc_train_time:.2f}s")
    print(f"  📊 参数量: {scrc_params:,}")
    
    # 检查稀疏性
    with torch.no_grad():
        sample = next(iter(test_loader))[0][:1].to(device)
        _, acts = scrc_model(sample, learn=False)
        for i, act in enumerate(acts[1:]):
            sparsity = (act == 0).float().mean().item() * 100
            print(f"  🔬 第{i+1}层稀疏率: {sparsity:.1f}%")
    
    results['scrc'] = {
        'test_acc': scrc_test_acc,
        'train_time': scrc_train_time,
        'params': scrc_params,
    }
    
    # ---------- Attention 方案 ----------
    print("\n--- [B] Attention 对照组 ---")
    print("  结构: 784 → 200(Attention) → 线性读出(10)")
    print("  学习: 全程反向传播")
    
    attn_model = SimpleAttention(784, 200, 10).to(device)
    attn_params = sum(p.numel() for p in attn_model.parameters())
    print(f"  总参数量: {attn_params:,}")
    
    attn_optimizer = torch.optim.Adam(attn_model.parameters(), lr=0.001)
    
    attn_train_start = time.time()
    for epoch in range(5):
        correct = 0
        total = 0
        epoch_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            logits = attn_model(data)
            loss = criterion(logits, target)
            
            attn_optimizer.zero_grad()
            loss.backward()
            attn_optimizer.step()
            
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            epoch_loss += loss.item()
            
        acc = correct / total * 100
        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/5 | 损失: {avg_loss:.4f} | 训练准确率: {acc:.2f}%")
    
    attn_train_time = time.time() - attn_train_start
    
    # 测试
    attn_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = attn_model(data)
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    attn_test_acc = correct / total * 100
    print(f"\n  ✅ Attention 测试准确率: {attn_test_acc:.2f}%")
    print(f"  ⏱️  训练耗时: {attn_train_time:.2f}s")
    print(f"  📊 参数量: {attn_params:,}")
    
    results['attention'] = {
        'test_acc': attn_test_acc,
        'train_time': attn_train_time,
        'params': attn_params,
    }
    
    # ---------- 总结对比 ----------
    print("\n" + "="*70)
    print("  实验一总结：SCRC vs Attention")
    print("="*70)
    print(f"  {'指标':<20} {'SCRC':<20} {'Attention':<20}")
    print(f"  {'-'*60}")
    print(f"  {'测试准确率':<18} {scrc_test_acc:.2f}%{'':<14} {attn_test_acc:.2f}%")
    print(f"  {'训练时间':<19} {scrc_train_time:.2f}s{'':<14} {attn_train_time:.2f}s")
    print(f"  {'参数量':<20} {scrc_params:<20,} {attn_params:<20,}")
    print(f"  {'学习方式':<19} {'Hebbian(局部)':<20} {'BP(全局梯度)'}")
    print(f"  {'稀疏性':<20} {'✅ Top-K 硬稀疏':<20} {'❌ 密集激活'}")
    
    return results, scrc_model


# ============================================================
# 实验二：SCRC 层次涌现分析
# ============================================================
def experiment_2_hierarchy(scrc_model):
    print("\n" + "="*70)
    print("  实验二：SCRC 层次涌现 — 每一级学到了什么？")
    print("="*70)
    
    for i, layer in enumerate(scrc_model.layers):
        W = layer.W.data.cpu()
        
        # 分析权重模板的统计特性
        print(f"\n--- 第 {i+1} 层 (输入:{layer.input_dim} → 单元:{layer.num_units}, k={layer.k}) ---")
        
        # 1. 权重的平均非零率（模板的"复杂度"）
        nonzero_rate = (W.abs() > 0.01).float().mean().item() * 100
        print(f"  活跃权重占比: {nonzero_rate:.1f}%")
        
        # 2. 模板间的平均余弦相似度（多样性）
        W_norm = F.normalize(W, dim=1)
        sim_matrix = W_norm @ W_norm.T
        # 去掉对角线
        mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool)
        avg_sim = sim_matrix[mask].mean().item()
        max_sim = sim_matrix[mask].max().item()
        print(f"  模板间平均余弦相似度: {avg_sim:.4f} (越低越多样)")
        print(f"  模板间最大余弦相似度: {max_sim:.4f}")
        
        # 3. 权重的有效维度（PCA 方差解释率）
        try:
            U, S, V = torch.linalg.svd(W, full_matrices=False)
            explained = (S ** 2).cumsum(0) / (S ** 2).sum()
            eff_dim_90 = (explained < 0.9).sum().item() + 1
            eff_dim_99 = (explained < 0.99).sum().item() + 1
            print(f"  有效维度 (90%方差): {eff_dim_90} / {min(W.shape)}")
            print(f"  有效维度 (99%方差): {eff_dim_99} / {min(W.shape)}")
        except Exception:
            print(f"  SVD 分析跳过")
        
        # 4. 第一层的可视化分析（如果是 784 维输入 = 28x28 图像）
        if layer.input_dim == 784:
            # 找出最活跃的 10 个模板
            activation_strength = W.norm(dim=1)
            top10_idx = activation_strength.topk(10).indices
            
            print(f"\n  📊 最强10个特征检测器的模式类型分析:")
            for rank, idx in enumerate(top10_idx):
                template = W[idx].reshape(28, 28)
                # 分析模板的空间频率
                high_freq = (template[:-1, :] - template[1:, :]).abs().mean() + \
                           (template[:, :-1] - template[:, 1:]).abs().mean()
                spatial_std = template.std()
                peak_loc = template.abs().argmax().item()
                peak_y, peak_x = peak_loc // 28, peak_loc % 28
                
                pattern_type = "边缘/纹理" if high_freq > spatial_std * 2 else "块状/区域"
                print(f"    #{rank+1} 单元{idx.item():3d}: "
                      f"类型={pattern_type}, "
                      f"峰值位置=({peak_y},{peak_x}), "
                      f"空间频率={high_freq:.4f}")


# ============================================================
# 实验三：SCRC 一次性学习（One-shot）vs Attention 多轮训练
# ============================================================
def experiment_3_oneshot():
    print("\n" + "="*70)
    print("  实验三：SCRC 一次性学习 — 只看一遍数据能学到多少？")
    print("="*70)
    
    # SCRC 只过一遍训练集
    model = SCRCClassifier(
        dims=[784, 500, 200],
        ks=[25, 10],
        lr_hebbian=0.01
    ).to(device)
    
    readout_opt = torch.optim.Adam(model.readout.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    
    print("\n  [只训练 1 个 epoch — 每个样本只看一次]")
    t0 = time.time()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        logits, _ = model(data, learn=True)
        
        loss = criterion(logits, target)
        readout_opt.zero_grad()
        loss.backward()
        readout_opt.step()
        
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
        
        if (batch_idx + 1) % 50 == 0:
            print(f"    进度: {batch_idx+1}/{len(train_loader)} | 当前准确率: {correct/total*100:.2f}%")
    
    oneshot_time = time.time() - t0
    
    # 测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits, _ = model(data, learn=False)
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    oneshot_acc = correct / total * 100
    print(f"\n  ✅ SCRC 一次性学习测试准确率: {oneshot_acc:.2f}%")
    print(f"  ⏱️  训练耗时: {oneshot_time:.2f}s (仅1个epoch)")
    
    return oneshot_acc


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print("="*70)
    print("  SCRC 验证实验：稀疏竞争循环回路 vs Attention")
    print("  Smart Competitive Recurrent Circuit Verification")
    print("="*70)
    print(f"  核心公式: Z = top_k(W · X), ΔW = η · Z · Xᵀ")
    print(f"  设备: {device}")
    print()
    
    # 实验一：对比
    results, scrc_model = experiment_1_comparison()
    
    # 实验二：层次涌现
    experiment_2_hierarchy(scrc_model)
    
    # 实验三：一次性学习
    oneshot_acc = experiment_3_oneshot()
    
    # 最终总结
    print("\n" + "="*70)
    print("  🏆 全部实验完成 — 最终总结")
    print("="*70)
    print(f"  SCRC (5 epochs)    → 准确率: {results['scrc']['test_acc']:.2f}%, "
          f"参数: {results['scrc']['params']:,}, "
          f"时间: {results['scrc']['train_time']:.1f}s")
    print(f"  Attention (5 epochs) → 准确率: {results['attention']['test_acc']:.2f}%, "
          f"参数: {results['attention']['params']:,}, "
          f"时间: {results['attention']['train_time']:.1f}s")
    print(f"  SCRC (1 epoch 一次性) → 准确率: {oneshot_acc:.2f}%")
    print()
    
    # 保存结果
    os.makedirs('tempdata', exist_ok=True)
    final_results = {
        'scrc_5epoch': results['scrc'],
        'attention_5epoch': results['attention'],
        'scrc_oneshot_acc': oneshot_acc,
    }
    with open('tempdata/scrc_experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print("  📁 结果已保存到 tempdata/scrc_experiment_results.json")
