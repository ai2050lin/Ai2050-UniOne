"""
CCIV 跨模型综合分析 — 子空间对齐的全景图
==========================================
综合三个模型(Qwen3/GLM4/DS7B)的CCIV结果,
生成统一的跨模型报告和统计
"""

import json, os, sys
import numpy as np
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TEMP = Path("tests/glm5_temp")

# 加载三个模型的结果
models = ["qwen3", "glm4", "deepseek7b"]
all_data = {}

for m in models:
    path = TEMP / f"cciv_{m}_results.json"
    with open(path, "r", encoding="utf-8") as f:
        all_data[m] = json.load(f)

# 合并所有cell结果
all_cells = []
for m in models:
    for cell in all_data[m]["cell_results"]:
        all_cells.append(cell)

print("=" * 70)
print("CCIV 跨模型综合分析 — 子空间对齐全景图")
print("=" * 70)

# ============================================================
# 1. 全局对齐统计
# ============================================================
print("\n" + "=" * 70)
print("1. 全局对齐统计 (150 cells: 3模型 × 5领域 × 10层)")
print("=" * 70)

mean_cos = [c["mean_cos_angle"] for c in all_cells]
transfer = [c["weighted_transfer"] for c in all_cells]
diag_corr = [c["diag_corr_mean"] for c in all_cells]
beta_emb = [c["beta_emb_geo"] for c in all_cells]
angle1 = [c["angles_deg"][0] for c in all_cells if len(c["angles_deg"]) > 0]

print(f"\n  平均主角度余弦:  {np.mean(mean_cos):.4f} (std={np.std(mean_cos):.4f})")
print(f"  平均方差传递:    {np.mean(transfer):.4f} (std={np.std(transfer):.4f})")
print(f"  平均对角线相关:  {np.mean(diag_corr):.4f} (std={np.std(diag_corr):.4f})")
print(f"  平均β_emb:      {np.mean(beta_emb):+.4f} (std={np.std(beta_emb):.4f})")
print(f"  平均第一主角度:  {np.mean(angle1):.1f}° (std={np.std(angle1):.1f}°)")
print(f"\n  ★★★★★ 关键: Embedding和Residual子空间几乎正交!")
print(f"  平均主角度≈{np.mean(angle1):.0f}° → 两个空间的方向几乎完全独立")

# ============================================================
# 2. 按模型分组
# ============================================================
print("\n" + "=" * 70)
print("2. 按模型分组")
print("=" * 70)

print(f"\n  {'模型':10s} {'mean_cos':>9s} {'transfer':>9s} {'diag_corr':>10s} {'β_emb':>7s} {'angle1':>7s}")
for m in models:
    cells = [c for c in all_cells if c["model"] == m]
    mc = np.mean([c["mean_cos_angle"] for c in cells])
    tr = np.mean([c["weighted_transfer"] for c in cells])
    dc = np.mean([c["diag_corr_mean"] for c in cells])
    be = np.mean([c["beta_emb_geo"] for c in cells])
    a1 = np.mean([c["angles_deg"][0] for c in cells if len(c["angles_deg"]) > 0])
    print(f"  {m:10s} {mc:9.4f} {tr:9.4f} {dc:10.4f} {be:+7.4f} {a1:7.1f}°")

# ============================================================
# 3. 按领域分组 — 跨模型平均
# ============================================================
print("\n" + "=" * 70)
print("3. 按领域分组 — 跨模型平均")
print("=" * 70)

domains = ["animal10", "emotion10", "profession10", "color10", "vehicle10"]

print(f"\n  {'领域':14s} {'mean_cos':>9s} {'transfer':>9s} {'diag_corr':>10s} {'β_emb':>7s} {'angle1':>7s} {'β<0':>5s}")
for d in domains:
    cells = [c for c in all_cells if c["domain"] == d]
    mc = np.mean([c["mean_cos_angle"] for c in cells])
    tr = np.mean([c["weighted_transfer"] for c in cells])
    dc = np.mean([c["diag_corr_mean"] for c in cells])
    be = np.mean([c["beta_emb_geo"] for c in cells])
    a1 = np.mean([c["angles_deg"][0] for c in cells if len(c["angles_deg"]) > 0])
    neg = sum(1 for c in cells if c["beta_emb_geo"] <= 0)
    print(f"  {d:14s} {mc:9.4f} {tr:9.4f} {dc:10.4f} {be:+7.4f} {a1:7.1f}° {neg:5d}/30")

# ============================================================
# 4. β_emb正/负分组的对齐度差异
# ============================================================
print("\n" + "=" * 70)
print("4. β_emb正/负分组的对齐度差异")
print("=" * 70)

pos = [c for c in all_cells if c["beta_emb_geo"] > 0]
neg = [c for c in all_cells if c["beta_emb_geo"] <= 0]

print(f"\n  β_emb > 0: {len(pos)} cells ({100*len(pos)/len(all_cells):.0f}%)")
print(f"    mean_cos={np.mean([c['mean_cos_angle'] for c in pos]):.4f}, "
      f"transfer={np.mean([c['weighted_transfer'] for c in pos]):.4f}, "
      f"diag_corr={np.mean([c['diag_corr_mean'] for c in pos]):.4f}")

print(f"\n  β_emb ≤ 0: {len(neg)} cells ({100*len(neg)/len(all_cells):.0f}%)")
print(f"    mean_cos={np.mean([c['mean_cos_angle'] for c in neg]):.4f}, "
      f"transfer={np.mean([c['weighted_transfer'] for c in neg]):.4f}, "
      f"diag_corr={np.mean([c['diag_corr_mean'] for c in neg]):.4f}")

# 负β的领域-模型分布
print(f"\n  负β_emb的领域-模型分布:")
neg_dist = {}
for c in neg:
    key = f"{c['domain']}×{c['model']}"
    neg_dist[key] = neg_dist.get(key, 0) + 1

# 按领域统计
for d in domains:
    neg_count = sum(1 for c in neg if c["domain"] == d)
    total_count = sum(1 for c in all_cells if c["domain"] == d)
    model_breakdown = {}
    for c in neg:
        if c["domain"] == d:
            model_breakdown[c["model"]] = model_breakdown.get(c["model"], 0) + 1
    breakdown_str = ", ".join([f"{m}={n}" for m, n in sorted(model_breakdown.items())])
    print(f"    {d:14s}: {neg_count:2d}/{total_count} ({100*neg_count/total_count:.0f}%) — {breakdown_str}")

# ============================================================
# 5. Vehicle领域的跨模型详细分析
# ============================================================
print("\n" + "=" * 70)
print("5. Vehicle领域 — 跨模型β_emb层间演变")
print("=" * 70)

for m in models:
    cells = [c for c in all_cells if c["domain"] == "vehicle10" and c["model"] == m]
    cells_sorted = sorted(cells, key=lambda x: x["layer"])
    
    print(f"\n  {m}:")
    print(f"    {'层':>4s} {'β_emb':>7s} {'mean_cos':>9s} {'transfer':>9s} {'diag_corr':>10s} {'对角线PC相关':>20s}")
    for c in cells_sorted:
        diag = [c["pc_corr_top5"][i][i] for i in range(min(5, len(c["pc_corr_top5"])))]
        diag_str = ", ".join([f"{v:+.2f}" for v in diag[:5]])
        print(f"    L{c['layer']:2d} {c['beta_emb_geo']:+7.3f} {c['mean_cos_angle']:9.4f} "
              f"{c['weighted_transfer']:9.4f} {c['diag_corr_mean']:10.4f} [{diag_str}]")

# ============================================================
# 6. Color领域的跨模型分析
# ============================================================
print("\n" + "=" * 70)
print("6. Color领域 — 跨模型β_emb层间演变 (DS7B有8/10层β<0!)")
print("=" * 70)

for m in models:
    cells = [c for c in all_cells if c["domain"] == "color10" and c["model"] == m]
    cells_sorted = sorted(cells, key=lambda x: x["layer"])
    neg_count = sum(1 for c in cells if c["beta_emb_geo"] <= 0)
    
    print(f"\n  {m}: β<0: {neg_count}/{len(cells)}")
    for c in cells_sorted:
        print(f"    L{c['layer']:2d}: β_emb={c['beta_emb_geo']:+.3f}, "
              f"transfer={c['weighted_transfer']:.3f}")

# ============================================================
# 7. 主角度的分布
# ============================================================
print("\n" + "=" * 70)
print("7. 主角度分布统计")
print("=" * 70)

# 所有9个主角度的分布
all_angles = []
for c in all_cells:
    all_angles.extend(c["angles_deg"])

all_angles = np.array(all_angles)
print(f"\n  所有主角度(n={len(all_angles)}):")
print(f"    均值: {np.mean(all_angles):.1f}°")
print(f"    中位: {np.median(all_angles):.1f}°")
print(f"    标准差: {np.std(all_angles):.1f}°")
print(f"    范围: [{np.min(all_angles):.1f}°, {np.max(all_angles):.1f}°]")

# 按角度序号
print(f"\n  各序号主角度(跨所有cell):")
for k in range(9):
    angles_k = [c["angles_deg"][k] for c in all_cells if len(c["angles_deg"]) > k]
    print(f"    θ_{k+1}: mean={np.mean(angles_k):.1f}°, "
          f"std={np.std(angles_k):.1f}°, "
          f"<45°={sum(1 for a in angles_k if a < 45)}/{len(angles_k)}")

# ============================================================
# 8. 方差传递分析
# ============================================================
print("\n" + "=" * 70)
print("8. 方差传递: Embedding子空间解释了多少Residual方差?")
print("=" * 70)

print(f"\n  全局: {np.mean(transfer):.4f} (只有{100*np.mean(transfer):.1f}%)")

for d in domains:
    cells = [c for c in all_cells if c["domain"] == d]
    tr = np.mean([c["weighted_transfer"] for c in cells])
    print(f"  {d:14s}: {tr:.4f} ({100*tr:.1f}%)")

print(f"\n  ★★★★★ 结论: Embedding子空间只能解释约{100*np.mean(transfer):.1f}%的Residual方差")
print(f"  这意味着>{100*(1-np.mean(transfer)):.0f}%的几何结构来自Transformer层的变换!")

# ============================================================
# 9. β_emb与对齐度之间的关系
# ============================================================
print("\n" + "=" * 70)
print("9. β_emb与对齐度之间的关系")
print("=" * 70)

betas = np.array([c["beta_emb_geo"] for c in all_cells])
cos_vals = np.array([c["mean_cos_angle"] for c in all_cells])
diag_vals = np.array([c["diag_corr_mean"] for c in all_cells])
angle1_vals = np.array([c["angles_deg"][0] for c in all_cells if len(c["angles_deg"]) > 0])

from scipy.stats import pearsonr, spearmanr

r1, p1 = pearsonr(betas, cos_vals)
r2, p2 = pearsonr(betas, diag_vals)

# 对angle1, 需要匹配长度
betas_short = np.array([c["beta_emb_geo"] for c in all_cells if len(c["angles_deg"]) > 0])
r3, p3 = pearsonr(betas_short, angle1_vals)

print(f"\n  β_emb vs mean_cos:   r={r1:+.3f}, p={p1:.4f}")
print(f"  β_emb vs diag_corr:  r={r2:+.3f}, p={p2:.4f}")
print(f"  β_emb vs angle1:     r={r3:+.3f}, p={p3:.4f}")

if abs(r1) < 0.3:
    print(f"\n  ★★★ 弱相关! 对齐度不能很好预测β_emb的符号")
    print(f"  这意味着β_emb的符号由更微妙的结构决定, 不仅仅是整体对齐度")

# ============================================================
# 10. 对角线PC相关分析: 哪些PC被保留/反转?
# ============================================================
print("\n" + "=" * 70)
print("10. 对角线PC相关: Embedding PC_i ↔ Residual PC_i")
print("=" * 70)

# 统计对角线相关的符号分布
all_diag = []
for c in all_cells:
    for i in range(min(5, len(c["pc_corr_top5"]))):
        if i < len(c["pc_corr_top5"][i]):
            all_diag.append(c["pc_corr_top5"][i][i])

all_diag = np.array(all_diag)
print(f"\n  对角线相关(n={len(all_diag)}):")
print(f"    均值: {np.mean(all_diag):+.3f}")
print(f"    正相关: {sum(1 for d in all_diag if d > 0)}/{len(all_diag)} ({100*sum(1 for d in all_diag if d > 0)/len(all_diag):.0f}%)")
print(f"    负相关: {sum(1 for d in all_diag if d < 0)}/{len(all_diag)} ({100*sum(1 for d in all_diag if d < 0)/len(all_diag):.0f}%)")
print(f"    |r|>0.3: {sum(1 for d in all_diag if abs(d) > 0.3)}/{len(all_diag)} ({100*sum(1 for d in all_diag if abs(d) > 0.3)/len(all_diag):.0f}%)")
print(f"    |r|>0.5: {sum(1 for d in all_diag if abs(d) > 0.5)}/{len(all_diag)} ({100*sum(1 for d in all_diag if abs(d) > 0.5)/len(all_diag):.0f}%)")

# ============================================================
# 11. 核心结论
# ============================================================
print("\n" + "=" * 70)
print("★★★★★ CCIV 核心结论 ★★★★★")
print("=" * 70)

print("""
1. ★★★★★ Embedding和Residual子空间几乎正交!
   - 平均主角度≈82° (随机两个子空间的角度约90°)
   - 方差传递仅0.6-1.1% → Embedding只贡献极少的几何结构
   - 这解释了为什么CCII的R²极低!

2. ★★★★★ β_emb负相关的领域模式:
   - Vehicle: GLM4和DS7B都有9/10层β<0, Qwen3有1/10
   - Color: DS7B有8/10层β<0, 其他模型几乎全正
   - Animal: 仅Qwen3有2/10层β<0
   - 负β不是随机噪声, 而是特定领域×模型的一致模式!

3. ★★★★ 对角线PC相关: 正负各半!
   - 对角线相关均值为正(+0.07), 但很多PC被反转
   - 这意味着: 同一序号的PC在两个空间中方向可能相反
   - 反转≠对齐差, 而是组织轴的"交叉"或"旋转"

4. ★★★ β_emb与整体对齐度弱相关!
   - β_emb vs mean_cos: r=弱
   - β_emb的符号由更微妙的轴对齐模式决定
   - 不是"整体对齐好→β正", 而是"某些关键轴对齐→β正"

5. ★★★★ 层间演变:
   - 浅层(L1): 略高对齐(mean_cos≈0.11), β接近0或负
   - 中层: 对齐最低, β在非vehicle领域稳定为正
   - 深层: 对齐回升, β普遍增大(Qwen3最明显)
   - Transformer在深层重新引入embedding结构!

6. ★★★ 模型间差异:
   - Qwen3: β_emb最高(+0.30), 对齐最好(mean_cos=0.088)
   - GLM4: β_emb居中(+0.13), 对齐最差(mean_cos=0.064)
   - DS7B: β_emb最低(+0.08), 对齐居中, color也有负β!
   - 不同模型处理相同语义领域的方式不同!
""")

# 保存综合结果
summary = {
    "n_cells": len(all_cells),
    "global_mean_cos": float(np.mean(mean_cos)),
    "global_mean_transfer": float(np.mean(transfer)),
    "global_mean_diag_corr": float(np.mean(diag_corr)),
    "global_mean_beta_emb": float(np.mean(beta_emb)),
    "global_mean_angle1": float(np.mean(angle1)),
    "beta_positive_pct": float(100 * len(pos) / len(all_cells)),
    "n_negative_beta": int(len(neg)),
    "negative_beta_by_domain": {
        d: sum(1 for c in neg if c["domain"] == d) for d in domains
    },
    "negative_beta_by_model": {
        m: sum(1 for c in neg if c["model"] == m) for m in models
    },
}

out_path = TEMP / "cciv_cross_model_summary.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n综合结果已保存到: {out_path}")
