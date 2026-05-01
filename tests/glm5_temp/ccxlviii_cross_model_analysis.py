"""
CCXLVIII 跨模型综合分析
========================
汇总三个模型(Qwen3/GLM4/DS7B)在四个领域(emotion4/emotion6/animal4/color4)的结果
"""

import json, numpy as np
from pathlib import Path

TEMP = Path("tests/glm5_temp")

models = ["qwen3", "glm4", "deepseek7b"]
cat_sets = ["emotion4", "emotion6", "animal4", "color4"]

all_data = {}
for m in models:
    p = TEMP / f"ccxlviii_semantic_{m}.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            all_data[m] = json.load(f)

print("=" * 80)
print("CCXLVIII 跨模型综合分析")
print("=" * 80)

# ============================================================
# 1. Pairwise距离 - 跨模型一致性
# ============================================================
print("\n### 1. Pairwise距离 — 跨模型一致性")

for cat in cat_sets:
    print(f"\n--- {cat} ---")
    # 收集所有模型的距离排序
    all_rankings = {}
    for m in models:
        if cat in all_data[m].get("results", {}):
            res = all_data[m]["results"][cat]
            if res.get("geo") and res["geo"].get("sorted_pairs"):
                sorted_pairs = res["geo"]["sorted_pairs"]
                for rank, (pair, dist, norm_dist) in enumerate(sorted_pairs):
                    key = f"{pair[0]}-{pair[1]}"
                    if key not in all_rankings:
                        all_rankings[key] = {}
                    all_rankings[key][m] = {"rank": rank, "norm_dist": norm_dist}
    
    if not all_rankings:
        continue
    
    # 计算跨模型排名一致性
    print(f"  {'Pair':<20} {'Qwen3':>8} {'GLM4':>8} {'DS7B':>8} {'一致性':>8}")
    for pair_key in sorted(all_rankings.keys()):
        vals = all_rankings[pair_key]
        ranks = [vals.get(m, {}).get("rank", -1) for m in models]
        dists = [vals.get(m, {}).get("norm_dist", 0) for m in models]
        
        # 一致性: 排名的标准差
        valid_ranks = [r for r in ranks if r >= 0]
        consistency = f"σ={np.std(valid_ranks):.1f}" if len(valid_ranks) >= 2 else "N/A"
        
        q3_str = f"{dists[0]:.3f}×" if dists[0] > 0 else "N/A"
        g4_str = f"{dists[1]:.3f}×" if dists[1] > 0 else "N/A"
        d7_str = f"{dists[2]:.3f}×" if dists[2] > 0 else "N/A"
        
        print(f"  {pair_key:<20} {q3_str:>8} {g4_str:>8} {d7_str:>8} {consistency:>8}")

# ============================================================
# 2. 几何-语义相关性 — 领域依赖性
# ============================================================
print(f"\n### 2. 几何-语义相关性 — 领域依赖性")
print(f"  {'Domain':<12} {'Model':<12} {'Pearson':>8} {'Spearman':>10} {'方向':>6}")

for cat in cat_sets:
    for m in models:
        if cat in all_data[m].get("results", {}):
            res = all_data[m]["results"][cat]
            corr = res.get("corr")
            if corr:
                pr = corr.get("pearson_r", 0)
                sr = corr.get("spearman_r", 0)
                direction = "+" if pr > 0 else "-"
                print(f"  {cat:<12} {m:<12} {pr:>8.3f} {sr:>10.3f} {direction:>6}")

# ============================================================
# 3. Valence聚类
# ============================================================
print(f"\n### 3. Valence聚类测试")
print(f"  {'Domain':<12} {'Model':<12} {'Neg内部':>8} {'Pos-Neg':>8} {'比值':>6} {'结论':>12}")

for cat in ["emotion4", "emotion6"]:
    for m in models:
        if cat in all_data[m].get("results", {}):
            res = all_data[m]["results"][cat]
            val = res.get("valence")
            if val:
                print(f"  {cat:<12} {m:<12} {val['intra_negative_mean']:>8.3f} "
                      f"{val['positive_negative_mean']:>8.3f} "
                      f"{val['ratio']:>6.3f} {val['prediction']:>12}")

# ============================================================
# 4. 变形分析 — sad是最变形的顶点
# ============================================================
print(f"\n### 4. 变形分析 — 哪个类别最变形?")
print(f"  {'Domain':<12} {'Model':<12} {'变形比':>6} {'最大变形类别':>12} {'变形量':>8}")

for cat in cat_sets:
    for m in models:
        if cat in all_data[m].get("results", {}):
            res = all_data[m]["results"][cat]
            deform = res.get("deform")
            if deform and deform.get("vertex_deform_norms"):
                norms = deform["vertex_deform_norms"]
                # 获取类别名
                cat_def = {"emotion4": ["happy", "sad", "angry", "scared"],
                          "emotion6": ["happy", "sad", "angry", "scared", "surprise", "disgust"],
                          "animal4": ["mammal", "bird", "fish", "insect"],
                          "color4": ["red", "blue", "green", "yellow"]}
                order = cat_def.get(cat, [])
                if len(order) == len(norms):
                    max_idx = np.argmax(norms)
                    print(f"  {cat:<12} {m:<12} {deform['deformation_ratio']:>6.3f} "
                          f"{order[max_idx]:>12} {norms[max_idx]:>8.3f}")

# ============================================================
# 5. 核心发现总结
# ============================================================
print(f"\n{'='*80}")
print("核心发现总结")
print("=" * 80)

print("""
1. ★★★★★ 几何-语义相关性是领域依赖的:
   - 情感领域: 负相关(r=-0.07到-0.61) → 语义相反的类别几何更近!
   - 动物领域: 正相关(r=0.03到0.59) → 语义相似的类别几何更近
   - 颜色领域: 弱正相关(r=0.31到0.63)

2. ★★★★★ 情感领域: happy-sad是几何最近的(2/3模型)
   - 但happy-sad的语义距离最大(2.83)! 
   - 这是"对比组织"模式: 反义词在单纯形中相邻
   - angry-scared语义最近(1.00)但几何最远(2/3模型) → 需要更多分离来区分

3. ★★★★ Valence聚类: 负面情绪内部更近(比值<1, 2/3模型)
   - 但这不意味着"负面更近"是普遍规律
   - DS7B的emotion6比值=1.085(负面更远!)

4. ★★★★ sad是所有模型中变形最大的顶点
   - 变形量: sad > angry > scared/happy
   - sad在语义空间中是"最低唤醒度"的情感
   - 可能是因为sad在情感空间中是"基线"或"锚点"

5. ★★★ 变形比(deformation_ratio)领域差异大:
   - 颜色: 0.10-0.32 (最大变形)
   - 动物: 0.21 (中等)
   - 情感: 0.12-0.26 (中等)
   → 颜色空间的变形最大 → 颜色感知有更强的物理约束

6. ★★ "对比组织"假说:
   - 情感空间中, 语义相反的类别更近 → 二元对立结构
   - 动物空间中, 语义相似的类别更近 → 层级分类结构
   → 不同领域使用不同的几何组织原则!
""")
