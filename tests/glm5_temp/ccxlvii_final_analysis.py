"""
CCXLVII 综合分析 — fit_r2问题 + 面内方向 + 统一结论
===================================================
"""

import json, numpy as np
from pathlib import Path

TEMP = Path("tests/glm5_temp")

# 收集所有模型的面内方向结果
print("=" * 70)
print("CCXLVII 综合分析")
print("=" * 70)

# ============================================================
# 1. fit_r2对比(不归一化 vs L2归一化)
# ============================================================
print("\n1. fit_r2对比:")
print("-" * 70)
print(f"{'模型':>12s} {'领域':>10s} {'不归一化':>10s} {'L2归一化':>10s} {'随机基线':>10s} {'vs随机':>8s}")
print("-" * 70)

for model_name in ["qwen3", "glm4", "deepseek7b"]:
    jp = TEMP / f"ccxlvii_unified_{model_name}.json"
    if not jp.exists():
        continue
    with open(jp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for domain_name in ["emotion4", "emotion6"]:
        if domain_name not in data:
            continue
        r = data[domain_name]
        raw = r["best_raw"]["fit_r2_raw"]
        norm = r["best_norm"]["fit_r2_norm"]
        N = int(domain_name[-1])
        bkey = f"baseline_N{N}"
        if bkey in data:
            b_mean = data[bkey]["mean"]
            b_std = data[bkey]["std"]
            z = (raw - b_mean) / (b_std + 1e-10)
            vs_random = f"z={z:.1f}σ"
        else:
            b_mean = 0
            vs_random = "N/A"
        
        print(f"{model_name:>12s} {domain_name:>10s} {raw:10.4f} {norm:10.4f} {b_mean:10.4f} {vs_random:>8s}")

print("\n★★★★★ fit_r2结论:")
print("  - 不归一化fit_r2=0.96-0.99, 但随机基线=0.999+")
print("  - 实际值低于随机基线(z=-14 to -145σ)!")
print("  - → fit_r2不能证明接近正则单纯形(高维中平凡)")

# ============================================================
# 2. edge_cv对比(正确的统计量)
# ============================================================
print("\n\n2. edge_cv(边长变异系数)对比:")
print("-" * 70)
print(f"{'模型':>12s} {'领域':>10s} {'实际edge_cv':>12s} {'随机均值':>12s} {'vs随机':>8s}")
print("-" * 70)

# 随机基线(从之前的分析)
random_edge_cv = {4: 0.0107, 6: 0.0123}

for model_name in ["qwen3", "glm4", "deepseek7b"]:
    jp = TEMP / f"ccxlvii_unified_{model_name}.json"
    if not jp.exists():
        continue
    with open(jp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for domain_name in ["emotion4", "emotion6"]:
        if domain_name not in data:
            continue
        N = int(domain_name[-1])
        r = data[domain_name]
        best = r["best_raw"]
        actual_cv = best["edge_cv_raw"]
        r_mean = random_edge_cv[N]
        z = (actual_cv - r_mean) / 0.003  # 近似std
        
        print(f"{model_name:>12s} {domain_name:>10s} {actual_cv:12.4f} {r_mean:12.4f} z={z:+.1f}σ")

print("\n★★★★★ edge_cv结论:")
print("  - 实际edge_cv=0.07-0.11, 远高于随机0.011")
print("  - → 模型表示比随机聚类更不均匀!")
print("  - → 某些类别对更近, 某些更远 → 有语义结构!")

# ============================================================
# 3. 面内流形方向(3模型对比)
# ============================================================
print("\n\n3. 面内流形方向(强度轨迹在面上朝哪里走?):")
print("-" * 70)

for model_name in ["qwen3", "glm4", "deepseek7b"]:
    jp = TEMP / f"ccxlvii_unified_{model_name}.json"
    if not jp.exists():
        continue
    with open(jp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if "tangential_analysis" not in data:
        continue
    
    ta = data["tangential_analysis"]
    print(f"\n  {model_name} (L{ta['best_layer']}):")
    print(f"  {'类别':>8s} {'径向对齐':>8s} {'切向占比':>8s} {'→最接近':>8s} {'对齐度':>8s}")
    for cls, t in ta["trajectories"].items():
        print(f"  {cls:>8s} {t['radial_alignment']:8.3f} {t['tangential_fraction']:8.3f} "
              f"{t['closest_other_class']:>8s} {t['closest_align']:+8.3f}")

print("\n★★★★★ 面内方向结论:")
print("  - happy: 3/3模型径向对齐最高(0.56-0.95) → 强度增加主要推离中心")
print("  - scared: 3/3模型切向占比最高(0.55-0.99) → 强度增加主要在面上移动")
print("  - sad: 2/3模型切向占比高(0.79-0.90) → 朝angry方向移动")
print("  - angry: 2/3模型径向较高(0.71-0.81) → 朝scared方向移动")
print("  - 共性: 强度轨迹的方向是类别特异的, 不是随机的!")

# ============================================================
# 4. 层间演化
# ============================================================
print("\n\n4. 层间演化(切向占比随层变化):")
print("-" * 70)

for model_name in ["qwen3", "glm4", "deepseek7b"]:
    jp = TEMP / f"ccxlvii_unified_{model_name}.json"
    if not jp.exists():
        continue
    with open(jp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if "layer_evolution" not in data:
        continue
    
    le = data["layer_evolution"]
    print(f"\n  {model_name}:")
    print(f"  {'层':>4s} {'fit_r2':>8s} {'径向':>8s} {'切向':>8s}")
    for l in le:
        print(f"  L{l['layer']:3d} {l['fit_r2']:8.4f} {l['avg_radial_align']:8.3f} {l['avg_tang_fraction']:8.3f}")

print("\n★★★★★ 层间演化结论:")
print("  - 前层(L3-8): 切向占比波动大(0.03-0.82)")
print("  - 中后层(L14+): 切向占比稳定(0.47-0.87)")
print("  - DS7B: L14-16有fit_r2崩溃, 但切向性仍保持")

# ============================================================
# 5. 综合结论
# ============================================================
print("\n\n" + "=" * 70)
print("★★★★★ CCXLVII 综合结论 ★★★★★")
print("=" * 70)

print("""
★★★★★ 1. fit_r2是错误的统计量!
  - 在高维空间(d=2560-4096)中, N=4-6个点的fit_r2天然≈1
  - 随机基线fit_r2=0.999+, 实际0.96-0.99 < 随机!
  - fit_r2不能区分"正则单纯形"和"随机聚类"
  - ★★★★★ 之前CCXXXIX-CCXLIV的fit_r2=0.88-0.99无统计意义!

★★★★★ 2. 模型表示比随机更不均匀(edge_cv更高)!
  - 随机edge_cv=0.011, 实际=0.07-0.11 (z=16-37σ更高!)
  - → 类别中心之间的距离不均匀 → 有语义结构
  - → 但不是正则单纯形(正则单纯形edge_cv=0)

★★★★★ 3. 单纯形的"等角性"是低维投影的必然结果!
  - N个去均值点在R^{N-1}中, cos角度均值=-1/(N-1)
  - 这是数学定理, 不是模型的性质!
  - → "等角性"是投影到N-1维的平凡性质

★★★★★ 4. 面内方向是真实且可解释的!
  - happy强度增加 → 径向(推离中心)
  - scared强度增加 → 切向(朝angry方向)
  - sad强度增加 → 切向(朝angry方向)
  - → 强度轨迹的方向有语义规律

★★★★★ 5. 修正的语言几何模型:
  原始模型: 正则单纯形(fit_r2=0.98) → 边轨迹 → 面混合
  CCXLV修正: 面上流形(切向性) → 边对齐是伪影
  CCXLVII最终修正:
    ★ 骨架: 近似单纯形但不均匀(edge_cv=0.07-0.11)
    ★ fit_r2无统计意义(高维平凡)
    ★ "等角性"是低维投影的必然(不是模型性质)
    ★ 流形: 强度轨迹方向有语义规律(类别特异)
    ★ 变换: 层间变换切向, 但方向有语义含义

★★★★★ 6. 严格审视:
  a) 之前所有基于fit_r2的结论都需要重新审视!
  b) 正则单纯形的证据只剩下:
     - angle_dev=1-5° (角度接近理想)
     - 但随机基线的angle_dev是多少? 需要计算!
  c) 真正有信息量的发现:
     - edge_cv高于随机 → 语义结构
     - 面内方向有语义规律 → 可解释的流形
     - 层间变换切向 → 涌现性质

★★★★★ 7. 下一步:
  1. ★★★★★ 用angle_dev和edge_cv与随机基线严格对比
  2. ★★★★ 分析不均匀性的语义含义(为什么某些类别更近?)
  3. ★★★ 面内方向的语义可解释性系统化
  4. ★★ 跨领域验证(occupation, habitat等)
""")
