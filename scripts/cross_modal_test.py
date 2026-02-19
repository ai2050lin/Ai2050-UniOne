"""
跨模态几何验证测试
==================

验证目标:
1. 视觉-语言嵌入的几何对齐
2. 跨模态测地线路径
3. 多模态 Ricci Flow 演化

核心假设: 不同模态共享底层几何结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time

print("=" * 60)
print("跨模态几何验证测试")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

results = {}

# ============================================================================
# 1. 模拟多模态嵌入
# ============================================================================

print("\n[1] 创建模拟多模态嵌入...")

# 假设共享嵌入维度
embed_dim = 512
n_concepts = 500

# 创建视觉和语言嵌入空间
# 理想情况: 相同概念在两个空间中应该对齐

# 基础概念嵌入 (共享语义空间)
base_embeddings = F.normalize(torch.randn(n_concepts, embed_dim, device=device), dim=1)

# 视觉嵌入 (添加模态特定噪声)
visual_noise = torch.randn_like(base_embeddings) * 0.3
visual_embeddings = F.normalize(base_embeddings + visual_noise, dim=1)

# 语言嵌入 (添加不同的模态特定噪声)
language_noise = torch.randn_like(base_embeddings) * 0.3
language_embeddings = F.normalize(base_embeddings + language_noise, dim=1)

print(f"  视觉嵌入: {visual_embeddings.shape}")
print(f"  语言嵌入: {language_embeddings.shape}")

# 计算对齐质量
alignment_scores = (visual_embeddings * language_embeddings).sum(dim=1)
print(f"\n  初始对齐质量: mean={alignment_scores.mean():.4f}, std={alignment_scores.std():.4f}")

results["initial_alignment"] = {
    "mean": float(alignment_scores.mean()),
    "std": float(alignment_scores.std())
}

# ============================================================================
# 2. 跨模态对齐学习
# ============================================================================

print("\n[2] 跨模态对齐学习...")

# 创建对齐映射
visual_proj = nn.Linear(embed_dim, embed_dim, bias=False).to(device)
language_proj = nn.Linear(embed_dim, embed_dim, bias=False).to(device)

# 对比学习目标
def contrastive_loss(v_emb, l_emb, temperature=0.1):
    """对比学习损失"""
    # 正样本: 相同索引
    # 负样本: 不同索引
    
    v_proj = visual_proj(v_emb)
    l_proj = language_proj(l_emb)
    
    # 归一化
    v_proj = F.normalize(v_proj, dim=1)
    l_proj = F.normalize(l_proj, dim=1)
    
    # 相似度矩阵
    sim = v_proj @ l_proj.T / temperature
    
    # InfoNCE 损失
    labels = torch.arange(len(v_emb), device=device)
    loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)
    
    return loss / 2

# 训练对齐
optimizer = torch.optim.Adam(
    list(visual_proj.parameters()) + list(language_proj.parameters()),
    lr=0.01
)

batch_size = 64
n_epochs = 100

for epoch in range(n_epochs):
    idx = torch.randperm(n_concepts)[:batch_size]
    v_batch = visual_embeddings[idx]
    l_batch = language_embeddings[idx]
    
    optimizer.zero_grad()
    loss = contrastive_loss(v_batch, l_batch)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 25 == 0:
        print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}")

# 评估对齐
with torch.no_grad():
    v_proj_emb = F.normalize(visual_proj(visual_embeddings), dim=1)
    l_proj_emb = F.normalize(language_proj(language_embeddings), dim=1)
    final_alignment = (v_proj_emb * l_proj_emb).sum(dim=1)

print(f"\n  训练后对齐质量: mean={final_alignment.mean():.4f}, std={final_alignment.std():.4f}")

alignment_improvement = final_alignment.mean() - alignment_scores.mean()
print(f"  对齐提升: {alignment_improvement:.4f}")

results["alignment_learning"] = {
    "final_mean": float(final_alignment.mean()),
    "final_std": float(final_alignment.std()),
    "improvement": float(alignment_improvement)
}

# ============================================================================
# 3. 跨模态测地线
# ============================================================================

print("\n[3] 跨模态测地线测试...")

def compute_geodesic_distance(x, y, n_steps=10):
    """计算测地线距离 (离散近似)"""
    path = torch.linspace(0, 1, n_steps).unsqueeze(1).to(x.device)
    
    # 球面插值
    cos_theta = (x * y).sum()
    if cos_theta > 0.9999:
        return (x - y).norm()
    
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    
    # 测地线路径
    points = []
    for t in path:
        p = (torch.sin((1 - t) * theta) / sin_theta) * x + \
            (torch.sin(t * theta) / sin_theta) * y
        points.append(F.normalize(p, dim=0))
    
    # 计算路径长度
    total_length = 0
    for i in range(len(points) - 1):
        total_length += (points[i+1] - points[i]).norm().item()
    
    return total_length

# 测试跨模态测地线
print("\n  计算跨模态测地线距离...")

# 视觉空间中的距离
v_distances = []
for i in range(20):
    for j in range(i+1, 20):
        d = compute_geodesic_distance(v_proj_emb[i], v_proj_emb[j])
        v_distances.append(d)

# 语言空间中的距离
l_distances = []
for i in range(20):
    for j in range(i+1, 20):
        d = compute_geodesic_distance(l_proj_emb[i], l_proj_emb[j])
        l_distances.append(d)

# 跨模态距离 (视觉概念 -> 对应语言概念)
cross_distances = []
for i in range(20):
    d = compute_geodesic_distance(v_proj_emb[i], l_proj_emb[i])
    cross_distances.append(d)

print(f"  视觉空间平均测地距离: {np.mean(v_distances):.4f}")
print(f"  语言空间平均测地距离: {np.mean(l_distances):.4f}")
print(f"  跨模态平均测地距离: {np.mean(cross_distances):.4f}")

# 测地线一致性: 相同概念对在不同模态中应该有相似的相对距离
v_dist_matrix = torch.zeros(20, 20)
l_dist_matrix = torch.zeros(20, 20)

for i in range(20):
    for j in range(20):
        if i != j:
            v_dist_matrix[i, j] = compute_geodesic_distance(v_proj_emb[i], v_proj_emb[j])
            l_dist_matrix[i, j] = compute_geodesic_distance(l_proj_emb[i], l_proj_emb[j])

# 相关性
v_flat = v_dist_matrix[v_dist_matrix > 0].cpu().numpy()
l_flat = l_dist_matrix[l_dist_matrix > 0].cpu().numpy()
correlation = np.corrcoef(v_flat, l_flat)[0, 1]
print(f"\n  测地线距离相关性: {correlation:.4f}")

results["geodesic_test"] = {
    "visual_mean_distance": float(np.mean(v_distances)),
    "language_mean_distance": float(np.mean(l_distances)),
    "cross_modal_distance": float(np.mean(cross_distances)),
    "distance_correlation": float(correlation)
}

# ============================================================================
# 4. 跨模态 Ricci Flow
# ============================================================================

print("\n[4] 跨模态 Ricci Flow 演化...")

def ricci_flow_step(embeddings, alpha=0.1, k_neighbors=10):
    """单步 Ricci Flow 演化"""
    n = len(embeddings)
    
    # 计算距离
    dist = torch.cdist(embeddings, embeddings)
    
    # 热核权重
    sigma = dist.mean()
    kernel = torch.exp(-dist ** 2 / (2 * sigma ** 2))
    
    # Ricci Flow 更新
    for i in range(n):
        # 找近邻
        _, idx = torch.topk(kernel[i], k_neighbors + 1, largest=True)
        neighbors = idx[1:]  # 排除自身
        
        # 曲率梯度
        weights = kernel[i, neighbors]
        weights = weights / weights.sum()
        
        # 平滑
        neighbor_center = (embeddings[neighbors] * weights.unsqueeze(1)).sum(dim=0)
        embeddings[i] = (1 - alpha) * embeddings[i] + alpha * neighbor_center
    
    return F.normalize(embeddings, dim=1)

# 联合 Ricci Flow
print("  应用联合 Ricci Flow...")
n_flow_steps = 10

v_flow = v_proj_emb.clone()
l_flow = l_proj_emb.clone()

for step in range(n_flow_steps):
    v_flow = ricci_flow_step(v_flow, alpha=0.1)
    l_flow = ricci_flow_step(l_flow, alpha=0.1)
    
    if (step + 1) % 3 == 0:
        alignment = (v_flow * l_flow).sum(dim=1).mean()
        print(f"    Step {step+1}: 对齐={alignment:.4f}")

# 计算曲率变化
def estimate_curvature(embeddings, k=5):
    """估计局部曲率"""
    dist = torch.cdist(embeddings, embeddings)
    curvatures = []
    
    for i in range(len(embeddings)):
        _, idx = torch.topk(dist[i], k + 1, largest=False)
        neighbors = embeddings[idx[1:]]
        curv = torch.var(neighbors).item()
        curvatures.append(curv)
    
    return np.mean(curvatures)

v_curv_before = estimate_curvature(v_proj_emb.cpu())
v_curv_after = estimate_curvature(v_flow.cpu())
l_curv_before = estimate_curvature(l_proj_emb.cpu())
l_curv_after = estimate_curvature(l_flow.cpu())

print(f"\n  视觉曲率: {v_curv_before:.4f} -> {v_curv_after:.4f} (变化 {(v_curv_after-v_curv_before)/v_curv_before*100:.1f}%)")
print(f"  语言曲率: {l_curv_before:.4f} -> {l_curv_after:.4f} (变化 {(l_curv_after-l_curv_before)/l_curv_before*100:.1f}%)")

results["ricci_flow"] = {
    "v_curv_before": v_curv_before,
    "v_curv_after": v_curv_after,
    "l_curv_before": l_curv_before,
    "l_curv_after": l_curv_after,
    "curvature_reduction_v": (v_curv_before - v_curv_after) / v_curv_before,
    "curvature_reduction_l": (l_curv_before - l_curv_after) / l_curv_before
}

# ============================================================================
# 5. 符号接地测试
# ============================================================================

print("\n[5] 符号接地测试...")

# 测试: 给定视觉概念，找到最相似的语言概念
grounding_correct = 0
grounding_total = 100

test_indices = torch.randperm(n_concepts)[:grounding_total]

for idx in test_indices:
    v_concept = v_proj_emb[idx]
    
    # 在语言空间中搜索最相似的
    similarities = (l_proj_emb @ v_concept)
    best_match = similarities.argmax().item()
    
    if best_match == idx.item():
        grounding_correct += 1

grounding_rate = grounding_correct / grounding_total
print(f"  符号接地准确率: {grounding_rate:.2%}")

# 计算接地分数的分布
all_similarities = (v_proj_emb @ l_proj_emb.T)
diagonal_sims = all_similarities.diag()
off_diagonal_max = all_similarities - torch.diag_embed(diagonal_sims)
off_diagonal_max = off_diagonal_max.max(dim=1).values

print(f"  正确接地相似度: mean={diagonal_sims.mean():.4f}")
print(f"  最佳错误接地相似度: mean={off_diagonal_max.mean():.4f}")
print(f"  分离度: {(diagonal_sims.mean() - off_diagonal_max.mean()):.4f}")

results["symbol_grounding"] = {
    "accuracy": grounding_rate,
    "correct_mean_sim": float(diagonal_sims.mean()),
    "incorrect_mean_sim": float(off_diagonal_max.mean()),
    "separation": float(diagonal_sims.mean() - off_diagonal_max.mean())
}

# ============================================================================
# 总结
# ============================================================================

print("\n" + "=" * 60)
print("跨模态几何验证总结")
print("=" * 60)

# 计算总体指标
alignment_score = min(final_alignment.mean().item(), 1.0)
geodesic_score = max(correlation, 0)
grounding_score = grounding_rate
ricci_score = (results["ricci_flow"]["curvature_reduction_v"] + 
               results["ricci_flow"]["curvature_reduction_l"]) / 2

print(f"\n对齐质量: {alignment_score:.2f}")
print(f"测地线一致性: {geodesic_score:.2f}")
print(f"符号接地准确率: {grounding_score:.2f}")
print(f"Ricci Flow 效果: {ricci_score:.2f}")

overall_score = (alignment_score + geodesic_score + grounding_score + ricci_score) / 4
print(f"\n总体跨模态分数: {overall_score:.2f}")

results["summary"] = {
    "alignment_score": alignment_score,
    "geodesic_score": geodesic_score,
    "grounding_score": grounding_score,
    "ricci_score": ricci_score,
    "overall_score": overall_score
}

# 保存
import os
os.makedirs("tempdata", exist_ok=True)

with open("tempdata/cross_modal_report.json", "w") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **results
    }, f, indent=2)

print(f"\n报告保存到: tempdata/cross_modal_report.json")

if overall_score > 0.5:
    print("\n结论: 跨模态几何结构基本对齐")
else:
    print("\n结论: 需要改进跨模态对齐机制")
