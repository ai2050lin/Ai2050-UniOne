# -*- coding: utf-8 -*-
"""
GLM5路线 - Phase 1: 特征涌现追踪
此实验从随机初始化开始训练一个小型的 Transformer 模型（或MLP层），
每100步记录激活状态，追踪有效秩和稀疏度的变化，以揭示特征是如何从无到有、从分化到组合的。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import os

# 确保输出目录存在
output_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tempdata', 'glm5_emergence')
os.makedirs(output_dir, exist_ok=True)

class SimpleTransformerLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 简化的层结构：线性层模拟特征提取
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_model * 4, d_model)
        
    def forward(self, x):
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        return out + x, h  # 返回输出和隐藏层激活以供追踪

class SimpleTransformer(nn.Module):
    def __init__(self, num_layers=4, d_model=128):
        super().__init__()
        self.embedding = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([SimpleTransformerLayer(d_model) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        x = self.embedding(x)
        activations = []
        for layer in self.layers:
            x, act = layer(x)
            activations.append(act)
        out = self.head(x)
        return out, activations

def calculate_effective_rank(act):
    """计算特征矩阵的有效秩 (基于奇异值分解熵)"""
    # 压平 batch 和 sequence 维度
    flat_act = act.view(-1, act.size(-1))
    if flat_act.size(0) < flat_act.size(1):
        # 确保奇异值分解能够运行
        return 0.0
    
    # 随机采样以加快计算
    if flat_act.size(0) > 1000:
        indices = torch.randperm(flat_act.size(0))[:1000]
        flat_act = flat_act[indices]
        
    flat_act = flat_act - flat_act.mean(dim=0)
    try:
        _, S, _ = torch.svd(flat_act)
        # 计算归一化的奇异值频率
        P = S / S.sum()
        # 计算香农熵
        entropy = -torch.sum(P * torch.log(P + 1e-9))
        # 有效秩 = exp(entropy)
        effective_rank = torch.exp(entropy).item()
        return effective_rank
    except Exception:
        return 0.0

def calculate_sparsity(act):
    """计算稀疏度 (L0的近似)"""
    # 这里简单使用非零元素的比例，对于GELU来说，就是小于很小数的为零
    threshold = 1e-3
    zeros = (act.abs() < threshold).float().mean().item()
    return zeros * 100  # 转为百分比

def run_experiment():
    print("🚀 启动 GLM5 特征涌现追踪实验...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化模型
    d_model = 128
    num_layers = 4
    model = SimpleTransformer(num_layers=num_layers, d_model=d_model).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 训练循环
    total_steps = 3000
    tracking_interval = 100
    
    batch_size = 64
    seq_len = 32
    
    records = []
    
    start_time = time.time()
    
    print("\n开始训练与特征追踪...")
    print("Step | Loss | L0 Sparsity (Layer Avg) | Effective Rank (Layer 0->1->2->3)")
    print("-" * 80)
    
    for step in range(total_steps + 1):
        # 使用随机噪声模拟信号流
        # 在真实任务中这会被替换为确切的输入数据（如文本 token 或图像特征）
        x = torch.randn(batch_size, seq_len, d_model).to(device)
        target = torch.roll(x, shifts=-1, dims=1)  # 模拟预测下一步的自回归任务
        
        optimizer.zero_grad()
        out, activations = model(x)
        loss = criterion(out, target)
        
        loss.backward()
        optimizer.step()
        
        # 定期追踪
        if step % tracking_interval == 0:
            layer_ranks = []
            layer_sparsities = []
            
            with torch.no_grad():
                for act in activations:
                    rank = calculate_effective_rank(act)
                    sparsity = calculate_sparsity(act)
                    
                    layer_ranks.append(f"{rank:.1f}")
                    layer_sparsities.append(sparsity)
            
            avg_sparsity = sum(layer_sparsities) / len(layer_sparsities)
            
            print(f"{step:4d} | {loss.item():.4f} | {avg_sparsity:5.1f}% | {' -> '.join(layer_ranks)}")
            
            records.append({
                "step": step,
                "loss": loss.item(),
                "sparsities": layer_sparsities,
                "effective_ranks": [float(r) for r in layer_ranks]
            })

    end_time = time.time()
    runtime = end_time - start_time
    print(f"\n✅ 训练完成! 耗时: {runtime:.2f}秒")
    
    # 保存结果
    result_path = os.path.join(output_dir, 'emergence_tracking.json')
    result = {
        "metadata": {
            "model": "SimpleTransformer",
            "layers": num_layers,
            "d_model": d_model,
            "parameters": total_params,
            "total_steps": total_steps,
            "runtime_seconds": runtime,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "records": records
    }
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    print(f"数据已保存至: {result_path}")
    print("这一数据为前端 GLM5Tab.jsx 中 test-000b 测试记录提供了数学与物理基础。")

if __name__ == "__main__":
    run_experiment()
