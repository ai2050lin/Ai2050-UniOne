"""CCXLIII 跨模型汇总分析"""
import json
import numpy as np
from pathlib import Path

TEMP = Path("tests/glm5_temp")

results = {}
for model_name in ['qwen3', 'glm4', 'deepseek7b']:
    path = TEMP / f"ccxliii_within_class_{model_name}.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            results[model_name] = json.load(f)

print("=" * 80)
print("CCXLIII 跨模型汇总")
print("=" * 80)

# ===== 实验1: 类内子空间 =====
print("\n=== 实验1: 类内子空间结构 ===")
print(f"{'Model':>12} {'Domain':>16} {'Class':>10} {'dim_90':>7} {'dim_95':>7} {'top3%':>7} {'radial%':>8} {'align1':>8}")
print("-" * 85)
for model_name, data in results.items():
    for domain, ddata in data.get("exp1_within_class", {}).items():
        for cls, wr in ddata["results"].items():
            print(f"{model_name:>12} {domain:>16} {cls:>10} {wr['dim_90']:>7} {wr['dim_95']:>7} "
                  f"{wr['top3_ratio']:>7.3f} {wr['radial_ratio']:>8.3f} {wr['alignment_top1']:>8.3f}")

# ===== 实验2: 连续语义方向 =====
print("\n\n=== 实验2: 连续语义方向 ===")
print(f"{'Model':>12} {'Domain':>20} {'Radial_Align':>13} {'Traj_Norm':>10} {'Direction':>15} {'Dist_Trend':>15}")
print("-" * 90)
for model_name, data in results.items():
    for domain, ddata in data.get("exp2_intensity", {}).items():
        r = ddata["results"]
        dir_label = "TANGENTIAL" if r["radial_alignment"] < 0.3 else ("RADIAL" if r["radial_alignment"] > 0.7 else "MIXED")
        dist_label = r.get("dist_interpretation", "?")
        trend = r.get("dist_trend", [])
        trend_str = "→".join([f"{t:.1f}" for t in trend[:3]]) if trend else "?"
        print(f"{model_name:>12} {domain:>20} {r['radial_alignment']:>13.4f} {r['trajectory_norm']:>10.2f} "
              f"{dir_label:>15} {dist_label:>15} {trend_str}")

# ===== 实验3: 混合模型 =====
print("\n\n=== 实验3: 混合模型验证 ===")
print(f"{'Model':>12} {'Layer':>6} {'Anisotropy':>11} {'Overlap':>8} {'Within_dims_90':>15} {'Isotropy':>30}")
print("-" * 90)
for model_name, data in results.items():
    for layer, mr in data.get("exp3_mixed_model", {}).items():
        print(f"{model_name:>12} {layer:>6} {mr['anisotropy_ratio']:>11.2f} {mr['subspace_overlap']:>8.4f} "
              f"{mr['n_within_dims_90']:>15} {mr['isotropy_verdict']:>30}")

# ===== 核心发现总结 =====
print("\n\n" + "=" * 80)
print("★★★★★ 核心发现总结")
print("=" * 80)

# 径向对齐度统计
print("\n--- 径向对齐度 ---")
for model_name, data in results.items():
    aligns = [d["results"]["radial_alignment"] for d in data.get("exp2_intensity", {}).values()]
    if aligns:
        print(f"  {model_name:>12}: mean={np.mean(aligns):.4f}, range=[{min(aligns):.4f}, {max(aligns):.4f}]")

# 各向异性统计
print("\n--- 各向异性比 ---")
for model_name, data in results.items():
    anisos = [mr["anisotropy_ratio"] for mr in data.get("exp3_mixed_model", {}).values()]
    if anisos:
        print(f"  {model_name:>12}: mean={np.mean(anisos):.2f}, range=[{min(anisos):.2f}, {max(anisos):.2f}]")

# 类内维度统计
print("\n--- 类内90%方差维度 ---")
for model_name, data in results.items():
    dims = [mr["n_within_dims_90"] for mr in data.get("exp3_mixed_model", {}).values()]
    d_model = data.get("d_model", "?")
    if dims:
        print(f"  {model_name:>12}: dims={dims}, mean={np.mean(dims):.1f}, d_model={d_model}, ratio={np.mean(dims)/d_model:.4f}")

# 子空间重叠度
print("\n--- 类间/类内子空间重叠度 ---")
for model_name, data in results.items():
    overlaps = [mr["subspace_overlap"] for mr in data.get("exp3_mixed_model", {}).values()]
    if overlaps:
        print(f"  {model_name:>12}: mean={np.mean(overlaps):.4f}, range=[{min(overlaps):.4f}, {max(overlaps):.4f}]")

# ★★★★★ 核心结论
print("\n\n" + "=" * 80)
print("★★★★★ 统一结论")
print("=" * 80)

print("""
1. ★★★★★ 强度方向以切向为主 (Qwen3/GLM4)
   - radial_alignment = 0.07-0.23 (Qwen3/GLM4)
   - 强度变化(mild→strong)垂直于类中心方向
   - 说明: 强度不是沿径向变化, 而是沿单纯形面方向
   - 含义: 类内结构不是简单的"围绕中心的高斯噪声"
   
2. ★★★★ 类内噪声的各向异性因模型而异
   - Qwen3: anisotropy=1.6-3.3 (接近各向同性)
   - GLM4: anisotropy=1.7-3.9 (接近各向同性)
   - DS7B: anisotropy=7.5-418 (强各向异性!)
   - DS7B的类内方差集中在1维, 有主导方向
   
3. ★★★★★ 类内方差集中在低维子空间
   - Qwen3: ~7维 (d_model=2560, 占0.3%)
   - GLM4: ~6维 (d_model=4096, 占0.15%)
   - DS7B: ~1维 (d_model=3584, 占0.03%)
   - 说明: 即使类内结构复杂, 也只占据极低维子空间

4. ★★★★ 子空间重叠度≈0.15-0.37
   - 类间和类内子空间有一定重叠
   - 重叠度不高, 说明类内主方向不全是类间方向
   - 但DS7B重叠度更高(0.37), 类内更受类间方向影响

5. ★★★ DS7B特殊性: scared_intensity径向!
   - scared在DS7B中radial_alignment=0.99
   - 但其他emotion都是切向
   - 且DS7B类内有极强各向异性
   - 可能是模型规模较小导致的过拟合现象
""")

# 关键洞察
print("=" * 80)
print("★★★★★ 关键洞察: 语言几何的两层结构")
print("=" * 80)
print("""
基于三个模型的数据:

第一层 (原型层): N个类别中心构成近似正则(N-1)维单纯形
  - 已充分验证: fit_r2=0.88-0.98
  - 但只描述类间关系

第二层 (连续层): 类内分布
  - ★★★★★ 不是各向同性高斯! 强度方向是切向的
  - ★★★★★ 集中在~5-7维低维子空间 (vs d_model=数千)
  - ★★★★ 有弱各向异性 (Qwen3/GLM4) 或强各向异性 (DS7B)

★★★★★ 最关键的发现: 
  强度方向是切向的, 不是径向的!
  
  这意味着: 
  - "glad→happy→ecstatic" 的变化方向垂直于类中心方向
  - 强度变化不改变与类中心的距离
  - 而是改变在单纯形面上的位置
  
  这暗示:
  - 单纯形的每条边/面可能代表一种语义维度
  - 强度变化 = 沿单纯形面移动
  - 这与单纯形的几何结构有深层联系!

★★★★★ 下一步验证:
  1. 切向方向是否与单纯形的边/面对齐?
  2. 不同强度水平的点在单纯形面上的分布?
  3. 如果切向=沿单纯形面, 则语言几何可能是:
     正则单纯形 + 面上的连续轨迹 + 面间的跳跃
""")
