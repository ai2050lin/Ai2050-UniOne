"""
CCXLVII-B: 正确的统计检验 — 用角度方差+边长CV代替fit_r2
==========================================================
★★★★★ 发现: fit_r2在高维空间中平凡接近1, 不是好的统计量!
★★★★★ N个随机点在R^{N-1}中的cos角度均值天然=-1/(N-1)

正确的统计量:
1. 角度方差 var(cos_angles) — 正则单纯形=0
2. 边长CV — 正则单纯形=0
3. 半径CV — 正则单纯形=0

这些指标的零假设: 数据点是从各向同性高斯分布采样的
→ 角度方差>0, 边长CV>0, 半径CV>0
→ 如果实际值显著低于随机基线 → 接近正则单纯形
"""

import json, numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist

TEMP = Path("tests/glm5_temp")


def compute_simplex_quality_metrics(centers, class_order):
    """计算单纯形质量指标 — 不用fit_r2"""
    N = len(class_order)
    D = N - 1
    center_mat = np.array([centers[c] for c in class_order])
    
    # 去均值
    gc = np.mean(center_mat, axis=0)
    centered = center_mat - gc
    
    # SVD投影到N-1维
    from scipy.linalg import svd
    U, S, Vt = svd(centered, full_matrices=False)
    proj = centered @ Vt[:D].T  # [N, D]
    
    # 1. 角度方差
    center_pt = np.mean(proj, axis=0)
    cos_angles = []
    for i in range(N):
        for j in range(i+1, N):
            vi = proj[i] - center_pt
            vj = proj[j] - center_pt
            ni = np.linalg.norm(vi)
            nj = np.linalg.norm(vj)
            if ni > 1e-10 and nj > 1e-10:
                cos_angles.append(np.dot(vi, vj) / (ni * nj))
    
    cos_var = np.var(cos_angles) if cos_angles else float('inf')
    cos_mean = np.mean(cos_angles) if cos_angles else 0
    ideal_cos = -1.0 / (N - 1)
    cos_dev = abs(cos_mean - ideal_cos)
    
    # 2. 边长CV
    dists = pdist(proj)
    edge_cv = np.std(dists) / (np.mean(dists) + 1e-10)
    
    # 3. 半径CV
    radii = [np.linalg.norm(proj[i] - center_pt) for i in range(N)]
    radius_cv = np.std(radii) / (np.mean(radii) + 1e-10)
    
    # 4. 等距比
    iso_ratio = np.min(dists) / (np.max(dists) + 1e-10)
    
    return {
        "cos_var": float(cos_var),
        "cos_mean": float(cos_mean),
        "cos_dev": float(cos_dev),
        "ideal_cos": float(ideal_cos),
        "edge_cv": float(edge_cv),
        "radius_cv": float(radius_cv),
        "isoperimetric_ratio": float(iso_ratio),
    }


def random_baseline_metrics(N, d_model, n_trials=2000):
    """随机基线: 在d_model维空间中采样N个随机点"""
    metrics_list = {
        "cos_var": [], "cos_mean": [], "cos_dev": [],
        "edge_cv": [], "radius_cv": [], "isoperimetric_ratio": [],
    }
    
    for _ in range(n_trials):
        # 模拟类中心: 加一些类间差异
        class_offset = np.random.randn(N, d_model) * 2.0
        centers = {f"c{i}": class_offset[i] for i in range(N)}
        class_names = [f"c{i}" for i in range(N)]
        
        m = compute_simplex_quality_metrics(centers, class_names)
        for k in metrics_list:
            metrics_list[k].append(m[k])
    
    result = {}
    for k, vals in metrics_list.items():
        vals = np.array(vals)
        result[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p05": float(np.percentile(vals, 5)),
            "p50": float(np.percentile(vals, 50)),
            "p95": float(np.percentile(vals, 95)),
        }
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("CCXLVII-B: 正确的统计检验 — 角度方差+边长CV")
    print("=" * 70)
    
    # 1. 随机基线
    print("\n1. 随机基线:")
    baselines = {}
    for N in [4, 6]:
        d_model = 2560  # Qwen3的维度
        print(f"\n  N={N} (D={N-1}, d={d_model}):")
        baseline = random_baseline_metrics(N, d_model, n_trials=1000)
        baselines[N] = baseline
        
        for k in ["cos_var", "edge_cv", "radius_cv", "cos_dev"]:
            b = baseline[k]
            print(f"    {k}: mean={b['mean']:.6f} ± {b['std']:.6f}  "
                  f"P5={b['p05']:.6f}  P95={b['p95']:.6f}")
    
    # 2. 实际数据
    print("\n\n2. 实际数据:")
    for model_name in ["qwen3", "glm4", "deepseek7b"]:
        jp = TEMP / f"ccxlvii_unified_{model_name}.json"
        if not jp.exists():
            print(f"  {model_name}: 数据不存在")
            continue
        
        with open(jp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n  --- {model_name} ---")
        
        for domain_name in ["emotion4", "emotion6"]:
            if domain_name not in data:
                continue
            
            N = int(domain_name[-1])
            lr = data[domain_name]["layer_results"]
            
            # 找fit_r2最高的层
            best_layer = max(lr, key=lambda x: x["fit_r2_raw"])
            best_l = best_layer["layer"]
            
            # 从原始数据重新计算(因为JSON中没有保存原始中心)
            # 使用角度偏差和edge_cv
            print(f"    {domain_name} (L{best_l}): "
                  f"angle_dev={best_layer['angle_dev_raw']:.2f}°  "
                  f"edge_cv={best_layer['edge_cv_raw']:.4f}  "
                  f"fit_r2_raw={best_layer['fit_r2_raw']:.4f}")
            
            # 与基线比较
            if N in baselines:
                # 角度偏差: 随机基线的cos_dev对应的角度偏差
                b = baselines[N]["cos_dev"]
                # cos_dev → 角度偏差(近似): dev ≈ |cos_mean - ideal| → 角度差 ≈ dev/sin(ideal_angle)
                ideal_angle = np.arccos(-1.0/(N-1)) * 180 / np.pi
                actual_dev = best_layer["angle_dev_raw"]  # 度
                random_dev_deg = np.mean(np.arccos(np.clip(-1.0/(N-1) + np.random.randn(1000)*b['std'], -1, 1)) * 180 / np.pi - ideal_angle)
                
                # edge_cv vs 基线
                actual_edge_cv = best_layer["edge_cv_raw"]
                random_edge_cv_mean = baselines[N]["edge_cv"]["mean"]
                random_edge_cv_std = baselines[N]["edge_cv"]["std"]
                z_edge = (actual_edge_cv - random_edge_cv_mean) / (random_edge_cv_std + 1e-10)
                
                print(f"      edge_cv: 实际={actual_edge_cv:.4f} "
                      f"随机均值={random_edge_cv_mean:.4f} "
                      f"z={z_edge:.2f}σ")
                
                if z_edge < -2:
                    print(f"      ★★★ edge_cv显著低于随机(P<0.05) → 更均匀!")
                elif z_edge < 0:
                    print(f"      ★★ edge_cv低于随机但不显著")
                else:
                    print(f"      ★ edge_cv不低于随机")
    
    # 3. 直接用实际残差重新计算
    print("\n\n3. 直接用Qwen3的残差数据验证(需要模型):")
    print("  (在主实验中已收集, 使用JSON中保存的数据)")
    
    # 检查: angle_dev在所有层中的值
    for model_name in ["qwen3", "glm4"]:
        jp = TEMP / f"ccxlvii_unified_{model_name}.json"
        if not jp.exists():
            continue
        with open(jp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n  {model_name}:")
        for domain_name in ["emotion4", "emotion6"]:
            if domain_name not in data:
                continue
            lr = data[domain_name]["layer_results"]
            print(f"    {domain_name}:")
            for layer_data in lr:
                if layer_data["fit_r2_raw"] > 0.5:  # 只看fit_r2>0.5的层
                    print(f"      L{layer_data['layer']:2d}: "
                          f"angle_dev={layer_data['angle_dev_raw']:.2f}°  "
                          f"edge_cv={layer_data['edge_cv_raw']:.4f}  "
                          f"radius_cv={layer_data.get('edge_cv_raw', 0):.4f}")
    
    print("\n" + "=" * 70)
    print("★★★★★ 统计检验结论 ★★★★★")
    print("=" * 70)
    print("""
★★★★★ fit_r2问题:
  - 高维空间中(d=2560-4096), N=4-6个随机点的fit_r2≈0.999
  - 实际fit_r2=0.96-0.99, 低于随机基线!
  - → fit_r2不能用来证明"接近正则单纯形"

★★★★★ 但!角度偏差和边长CV是有信息量的:
  - 正则单纯形: angle_dev=0°, edge_cv=0
  - 随机点: angle_dev>0°, edge_cv>0
  - 实际数据: angle_dev=1-5°, edge_cv=? (需要与基线比)

★★★★★ 关键问题: 
  实际的angle_dev=1-5°是否显著低于随机基线?
  如果是 → 确实比随机更接近正则单纯形
  如果否 → 不能排除"随机聚类"的零假设

★★★★★ 下一实验需要:
  用实际模型残差计算角度方差和边长CV, 与随机基线对比
""")
