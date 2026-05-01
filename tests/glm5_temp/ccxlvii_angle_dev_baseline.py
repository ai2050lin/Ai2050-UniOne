"""
CCXLVII-C: angle_dev随机基线 — 最终统计验证
=============================================
关键问题: 实际的angle_dev=1-5°, 随机基线是多少?
  - 如果随机angle_dev也≈1-5° → "等角性"不显著
  - 如果随机angle_dev≫5° → "等角性"是显著的
"""

import numpy as np
from scipy.linalg import svd

def compute_angle_dev(points):
    """计算角度偏差(与正则单纯形的理想角度之差)"""
    N = len(points)
    D = N - 1
    
    # 去均值
    center = np.mean(points, axis=0)
    centered = points - center
    
    # SVD投影到N-1维
    U, S, Vt = svd(centered, full_matrices=False)
    proj = centered @ Vt[:D].T
    
    # 计算角度
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
    
    if not cos_angles:
        return float('inf'), float('inf')
    
    ideal_cos = -1.0 / (N - 1)
    ideal_angle = np.arccos(ideal_cos) * 180 / np.pi
    actual_angle = np.arccos(np.clip(np.mean(cos_angles), -1, 1)) * 180 / np.pi
    angle_dev = abs(actual_angle - ideal_angle)
    
    return angle_dev, np.std(cos_angles)


if __name__ == "__main__":
    print("=" * 70)
    print("CCXLVII-C: angle_dev随机基线")
    print("=" * 70)
    
    # 1. 高维空间中的随机基线(d=2560)
    print("\n1. 高维空间(d=2560)中随机点的angle_dev:")
    for N in [4, 6]:
        dev_list = []
        for _ in range(2000):
            points = np.random.randn(N, 2560) * 2.0
            dev, _ = compute_angle_dev(points)
            dev_list.append(dev)
        
        dev_list = np.array(dev_list)
        print(f"  N={N}: mean={np.mean(dev_list):.2f}°  std={np.std(dev_list):.2f}°  "
              f"P5={np.percentile(dev_list, 5):.2f}°  P95={np.percentile(dev_list, 95):.2f}°")
    
    # 2. 低维空间中的随机基线(D=N-1)
    print("\n2. 低维空间(D=N-1)中随机点的angle_dev:")
    for N in [4, 6]:
        D = N - 1
        dev_list = []
        for _ in range(5000):
            points = np.random.randn(N, D)
            dev, _ = compute_angle_dev(points)
            dev_list.append(dev)
        
        dev_list = np.array(dev_list)
        print(f"  N={N} (D={D}): mean={np.mean(dev_list):.2f}°  std={np.std(dev_list):.2f}°  "
              f"P5={np.percentile(dev_list, 5):.2f}°  P95={np.percentile(dev_list, 95):.2f}°")
    
    # 3. 模拟更真实的随机聚类(类内方差)
    print("\n3. 模拟随机聚类(类间方差×2 + 类内方差×1, d=2560):")
    for N in [4, 6]:
        dev_list = []
        for _ in range(2000):
            # 类中心
            class_centers = np.random.randn(N, 2560) * 2.0
            # 每类12个词, 加类内噪声
            all_points = []
            for i in range(N):
                words = class_centers[i] + np.random.randn(12, 2560) * 1.0
                all_points.append(words)
            # 取每类均值
            centers = np.array([np.mean(p, axis=0) for p in all_points])
            dev, _ = compute_angle_dev(centers)
            dev_list.append(dev)
        
        dev_list = np.array(dev_list)
        print(f"  N={N}: mean={np.mean(dev_list):.2f}°  std={np.std(dev_list):.2f}°  "
              f"P5={np.percentile(dev_list, 5):.2f}°  P95={np.percentile(dev_list, 95):.2f}°")
    
    # 4. 对比实际数据
    print("\n4. 对比:")
    print(f"  N=4 实际angle_dev: 4.7-5.0° (跨3模型)")
    print(f"  N=6 实际angle_dev: 1.0-2.1° (跨3模型)")
    
    # 5. 另一个关键统计量: cos角度的方差
    print("\n5. cos角度方差(衡量角度均匀性):")
    for N in [4, 6]:
        print(f"\n  N={N} (理想cos={-1/(N-1):.4f}):")
        
        # 低维随机
        var_list_low = []
        for _ in range(5000):
            points = np.random.randn(N, N-1)
            _, var = compute_angle_dev(points)
            var_list_low.append(var)
        
        # 高维随机
        var_list_high = []
        for _ in range(2000):
            points = np.random.randn(N, 2560) * 2.0
            _, var = compute_angle_dev(points)
            var_list_high.append(var)
        
        print(f"    低维(D={N-1}): var(cos) mean={np.mean(var_list_low):.6f}  "
              f"P5={np.percentile(var_list_low, 5):.6f}  P95={np.percentile(var_list_low, 95):.6f}")
        print(f"    高维(d=2560): var(cos) mean={np.mean(var_list_high):.6f}  "
              f"P5={np.percentile(var_list_high, 5):.6f}  P95={np.percentile(var_list_high, 95):.6f}")
        print(f"    正则单纯形: var(cos) = 0.000000")
    
    print("\n" + "=" * 70)
    print("★★★★★ 最终统计结论 ★★★★★")
    print("=" * 70)
    print("""
如果高维随机angle_dev≈0° (与实际1-5°接近):
  → "等角性"不显著, 是高维空间的平凡性质

如果高维随机angle_dev≫5°:
  → "等角性"显著, 模型确实在构造等角结构

如果cos角度方差在随机中也很低:
  → 角度均匀性也是平凡的
  → 真正有信息量的只有edge_cv和面内方向
""")
