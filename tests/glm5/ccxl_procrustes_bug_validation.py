"""
CCXL(340): Procrustes Bug验证 + 修正 + 重新分析
=================================================
★★★★★ 最高优先级! CCXXXIX的fit_r2=0极可能是代码bug!

Bug位置: ccxxxix_simplex_rigorous.py 第259行
  错误: R = Vt_h.T @ U_h.T  (= V @ U^T = R_correct^T, 逆旋转!)
  正确: R = U_h @ Vt_h      (= U @ V^T, 正确旋转)

验证方法:
1. 生成已知正则单纯形 → 随机旋转 → 用bug代码算fit_r2 → 应≈0
2. 同上 → 用修正代码算fit_r2 → 应≈1
3. 如果验证通过 → 用修正代码重新分析所有模型数据

Part 1: Bug验证 (纯数学, 不需要GPU)
Part 2: 用已保存的模型数据重新计算fit_r2 (不需要GPU)
Part 3: 用修正后的代码重新跑CCXXXIX

用法:
  python ccxl_procrustes_bug_validation.py
"""
import sys, os, json, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist, squareform

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxl_procrustes_bug_log.txt"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + "\n")

# ============================================================
# Part 1: Bug验证 — 用已知正则单纯形测试
# ============================================================
def generate_regular_simplex(N, dim=None):
    """生成N个顶点的正则(N-1)维单纯形"""
    if dim is None:
        dim = N
    # 标准构造: N个顶点在N维空间
    vertices = np.zeros((N, N))
    for i in range(N):
        vertices[i, i] = 1.0
    # 减去均值使质心在原点
    vertices = vertices - vertices.mean(axis=0)
    # 如果需要降到dim维, 用PCA
    if dim < N:
        U, S, Vt = np.linalg.svd(vertices, full_matrices=False)
        vertices = vertices @ Vt[:dim].T
    return vertices

def random_rotation(dim, rng=None):
    """生成随机正交旋转矩阵"""
    if rng is None:
        rng = np.random.RandomState(42)
    A = rng.randn(dim, dim)
    Q, R = np.linalg.qr(A)
    # 确保是旋转(行列式=1)
    if np.linalg.det(Q) < 0:
        Q[:, -1] = -Q[:, -1]
    return Q

def compute_fit_r2_BUGGY(centers, N):
    """CCXXXIX中的buggy版本"""
    n_dim = centers.shape[1]
    if N - 1 > n_dim:
        return 0.0
    
    # 构造正则单纯形
    ideal_vertices = np.zeros((N, N))
    for i in range(N):
        ideal_vertices[i, i] = 1.0
    ideal_vertices = ideal_vertices - ideal_vertices.mean(axis=0)
    
    # 归一化边长
    mean_edge = np.mean(pdist(centers))
    ideal_dists = squareform(pdist(ideal_vertices))
    ideal_mean_dist = np.mean(ideal_dists[np.triu_indices(N, k=1)])
    ideal_vertices = ideal_vertices / max(ideal_mean_dist, 1e-10) * mean_edge
    
    # 投影
    if n_dim >= N:
        U_c, S_c, Vt_c = np.linalg.svd(centers - centers.mean(axis=0), full_matrices=False)
        proj_dim = min(N - 1, n_dim)
        centers_proj = (centers - centers.mean(axis=0)) @ Vt_c[:proj_dim].T
        
        U_i, S_i, Vt_i = np.linalg.svd(ideal_vertices - ideal_vertices.mean(axis=0), full_matrices=False)
        ideal_proj = (ideal_vertices - ideal_vertices.mean(axis=0)) @ Vt_i[:proj_dim].T
        
        # ★★★ BUG: R = Vt_h.T @ U_h.T (逆旋转!) ★★★
        H = ideal_proj.T @ centers_proj
        U_h, S_h, Vt_h = np.linalg.svd(H)
        R = Vt_h.T @ U_h.T  # BUG!
        
        aligned_ideal = ideal_proj @ R
        residual = centers_proj - aligned_ideal
        ss_res = np.sum(residual ** 2)
        ss_tot = np.sum((centers_proj - centers_proj.mean(axis=0)) ** 2)
        fit_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        return max(0.0, fit_r2)
    return 0.0

def compute_fit_r2_FIXED(centers, N):
    """修正后的版本"""
    n_dim = centers.shape[1]
    if N - 1 > n_dim:
        return 0.0
    
    # 构造正则单纯形
    ideal_vertices = np.zeros((N, N))
    for i in range(N):
        ideal_vertices[i, i] = 1.0
    ideal_vertices = ideal_vertices - ideal_vertices.mean(axis=0)
    
    # 归一化边长
    mean_edge = np.mean(pdist(centers))
    ideal_dists = squareform(pdist(ideal_vertices))
    ideal_mean_dist = np.mean(ideal_dists[np.triu_indices(N, k=1)])
    ideal_vertices = ideal_vertices / max(ideal_mean_dist, 1e-10) * mean_edge
    
    # 投影
    if n_dim >= N:
        U_c, S_c, Vt_c = np.linalg.svd(centers - centers.mean(axis=0), full_matrices=False)
        proj_dim = min(N - 1, n_dim)
        centers_proj = (centers - centers.mean(axis=0)) @ Vt_c[:proj_dim].T
        
        U_i, S_i, Vt_i = np.linalg.svd(ideal_vertices - ideal_vertices.mean(axis=0), full_matrices=False)
        ideal_proj = (ideal_vertices - ideal_vertices.mean(axis=0)) @ Vt_i[:proj_dim].T
        
        # ★★★ FIX: R = U_h @ Vt_h (正确旋转!) ★★★
        H = ideal_proj.T @ centers_proj
        U_h, S_h, Vt_h = np.linalg.svd(H)
        R = U_h @ Vt_h  # FIXED!
        
        aligned_ideal = ideal_proj @ R
        residual = centers_proj - aligned_ideal
        ss_res = np.sum(residual ** 2)
        ss_tot = np.sum((centers_proj - centers_proj.mean(axis=0)) ** 2)
        fit_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        return max(0.0, fit_r2)
    return 0.0

def compute_fit_r2_SCIPY(centers, N):
    """用scipy.linalg.orthogonal_procrustes作为ground truth"""
    from scipy.linalg import orthogonal_procrustes
    
    n_dim = centers.shape[1]
    if N - 1 > n_dim:
        return 0.0
    
    # 构造正则单纯形
    ideal_vertices = np.zeros((N, N))
    for i in range(N):
        ideal_vertices[i, i] = 1.0
    ideal_vertices = ideal_vertices - ideal_vertices.mean(axis=0)
    
    # 归一化边长
    mean_edge = np.mean(pdist(centers))
    ideal_dists = squareform(pdist(ideal_vertices))
    ideal_mean_dist = np.mean(ideal_dists[np.triu_indices(N, k=1)])
    ideal_vertices = ideal_vertices / max(ideal_mean_dist, 1e-10) * mean_edge
    
    # 投影
    if n_dim >= N:
        U_c, S_c, Vt_c = np.linalg.svd(centers - centers.mean(axis=0), full_matrices=False)
        proj_dim = min(N - 1, n_dim)
        centers_proj = (centers - centers.mean(axis=0)) @ Vt_c[:proj_dim].T
        
        U_i, S_i, Vt_i = np.linalg.svd(ideal_vertices - ideal_vertices.mean(axis=0), full_matrices=False)
        ideal_proj = (ideal_vertices - ideal_vertices.mean(axis=0)) @ Vt_i[:proj_dim].T
        
        # scipy ground truth
        R, _ = orthogonal_procrustes(ideal_proj, centers_proj)
        
        aligned_ideal = ideal_proj @ R
        residual = centers_proj - aligned_ideal
        ss_res = np.sum(residual ** 2)
        ss_tot = np.sum((centers_proj - centers_proj.mean(axis=0)) ** 2)
        fit_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        return max(0.0, fit_r2)
    return 0.0

# ============================================================
# Part 2: 修正后的完整几何分析函数
# ============================================================
def compute_geometry_FIXED(centers, labels):
    """修正后的几何分析 — 使用正确的Procrustes"""
    unique_labels = sorted(set(labels))
    N = len(unique_labels)
    
    if N < 3:
        return None
    
    # 类别中心
    class_centers = []
    for lbl in unique_labels:
        mask = np.array(labels) == lbl
        class_centers.append(centers[mask].mean(axis=0))
    centers = np.array(class_centers)
    
    # 质心
    centroid = centers.mean(axis=0)
    vectors = centers - centroid
    
    # 1. 边长分析
    dists = squareform(pdist(centers))
    edge_dists = dists[np.triu_indices(N, k=1)]
    mean_edge = np.mean(edge_dists)
    std_edge = np.std(edge_dists)
    cv_edge = std_edge / max(mean_edge, 1e-10)
    edge_uniformity = 1.0 - cv_edge
    
    # 2. 角度分析
    angles = []
    for i in range(N):
        for j in range(i + 1, N):
            v1 = vectors[i]
            v2 = vectors[j]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                angles.append(np.arccos(cos_a) * 180 / np.pi)
    
    mean_angle = np.mean(angles) if angles else 0
    std_angle = np.std(angles) if angles else 0
    ideal_angle = np.arccos(-1.0 / N) * 180 / np.pi if N > 1 else 0
    angle_deviation = abs(mean_angle - ideal_angle)
    angle_cv = std_angle / max(mean_angle, 1e-10) if mean_angle > 0 else 999
    
    # 3. 半径
    radii = np.linalg.norm(vectors, axis=1)
    mean_radius = np.mean(radii)
    std_radius = np.std(radii)
    cv_radius = std_radius / max(mean_radius, 1e-10)
    radius_uniformity = 1.0 - cv_radius
    
    # 4. ★★★ 修正后的正则单纯形拟合R² ★★★
    n_dim = centers.shape[1]
    simplex_fit_r2 = 0.0
    if N - 1 <= n_dim and n_dim >= N:
        # 构造正则单纯形
        ideal_vertices = np.zeros((N, N))
        for i in range(N):
            ideal_vertices[i, i] = 1.0
        ideal_vertices = ideal_vertices - ideal_vertices.mean(axis=0)
        ideal_dists = squareform(pdist(ideal_vertices))
        ideal_mean_dist = np.mean(ideal_dists[np.triu_indices(N, k=1)])
        ideal_vertices = ideal_vertices / max(ideal_mean_dist, 1e-10) * mean_edge
        
        # 投影
        U_c, S_c, Vt_c = np.linalg.svd(centers - centers.mean(axis=0), full_matrices=False)
        proj_dim = min(N - 1, n_dim)
        centers_proj = (centers - centers.mean(axis=0)) @ Vt_c[:proj_dim].T
        
        U_i, S_i, Vt_i = np.linalg.svd(ideal_vertices - ideal_vertices.mean(axis=0), full_matrices=False)
        ideal_proj = (ideal_vertices - ideal_vertices.mean(axis=0)) @ Vt_i[:proj_dim].T
        
        # ★★★ FIX: R = U_h @ Vt_h ★★★
        H = ideal_proj.T @ centers_proj
        U_h, S_h, Vt_h = np.linalg.svd(H)
        R = U_h @ Vt_h  # 修正!
        
        aligned_ideal = ideal_proj @ R
        residual = centers_proj - aligned_ideal
        ss_res = np.sum(residual ** 2)
        ss_tot = np.sum((centers_proj - centers_proj.mean(axis=0)) ** 2)
        simplex_fit_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        simplex_fit_r2 = max(0.0, simplex_fit_r2)
    
    return {
        "N": N,
        "n_dim": centers.shape[1],
        "mean_edge_length": round(float(mean_edge), 4),
        "std_edge_length": round(float(std_edge), 4),
        "cv_edge": round(float(cv_edge), 4),
        "edge_uniformity": round(float(edge_uniformity), 4),
        "mean_angle": round(float(mean_angle), 2),
        "std_angle": round(float(std_angle), 2),
        "ideal_angle": round(float(ideal_angle), 2),
        "angle_deviation": round(float(angle_deviation), 2),
        "angle_cv": round(float(angle_cv), 4),
        "mean_radius": round(float(mean_radius), 4),
        "std_radius": round(float(std_radius), 4),
        "cv_radius": round(float(cv_radius), 4),
        "radius_uniformity": round(float(radius_uniformity), 4),
        "simplex_fit_r2": round(float(simplex_fit_r2), 4),
    }


# ============================================================
# Part 3: 用已保存的模型数据重新计算fit_r2
# ============================================================
def recompute_model_data():
    """从已保存的CCXXXIX数据中提取类别中心, 重新计算修正后的fit_r2"""
    results = {}
    
    for model in ["qwen3", "glm4", "deepseek7b"]:
        data_file = TEMP / f"ccxxxix_simplex_rigorous_{model}.json"
        if not data_file.exists():
            log(f"  {model}: 数据文件不存在, 跳过")
            continue
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 需要从raw_residuals中重新提取类别中心
        raw = data.get("raw_residuals", {})
        if not raw:
            log(f"  {model}: 无raw_residuals, 跳过")
            continue
        
        model_results = {}
        for domain in ["habitat", "emotion", "occupation", "color"]:
            domain_raw = raw.get(domain, {})
            if not domain_raw:
                continue
            
            domain_geom = {}
            for layer_key, layer_data in domain_raw.items():
                # layer_data包含每类的残差列表
                class_residuals = {}
                for class_name, residual_list in layer_data.items():
                    class_residuals[class_name] = np.array(residual_list)
                
                if len(class_residuals) < 3:
                    continue
                
                # 构造centers和labels
                all_residuals = []
                all_labels = []
                for cls_name, res_arr in class_residuals.items():
                    all_residuals.append(res_arr)
                    all_labels.extend([cls_name] * len(res_arr))
                
                all_residuals = np.vstack(all_residuals)
                geom = compute_geometry_FIXED(all_residuals, all_labels)
                
                if geom and geom.get("simplex_fit_r2", 0) > 0:
                    domain_geom[layer_key] = geom
            
            model_results[domain] = domain_geom
        
        results[model] = model_results
        log(f"  {model}: 重新计算完成, {sum(len(v) for v in model_results.values())} 层有fit_r2>0")
    
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # 清空日志
    with open(LOG, 'w', encoding='utf-8') as f:
        f.write(f"CCXL Procrustes Bug验证 日志\n")
        f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    log("=" * 70)
    log("CCXL(340): Procrustes Bug验证 + 修正")
    log("=" * 70)
    
    # ================================================================
    # Part 1: Bug验证 — 已知正则单纯形
    # ================================================================
    log("\n" + "=" * 70)
    log("Part 1: Bug验证 — 用已知正则单纯形测试")
    log("=" * 70)
    
    test_cases = [
        ("3类2维正则单纯形(等边三角形)", 3, 3),
        ("4类3维正则单纯形(正四面体)", 4, 4),
        ("5类4维正则单纯形", 5, 5),
        ("6类5维正则单纯形", 6, 6),
    ]
    
    for desc, N, dim in test_cases:
        log(f"\n--- {desc} (N={N}, dim={dim}) ---")
        
        # 生成正则单纯形
        simplex = generate_regular_simplex(N, dim)
        
        # 测试1: 未旋转的正则单纯形
        buggy_r2 = compute_fit_r2_BUGGY(simplex, N)
        fixed_r2 = compute_fit_r2_FIXED(simplex, N)
        scipy_r2 = compute_fit_r2_SCIPY(simplex, N)
        log(f"  未旋转: buggy={buggy_r2:.4f}, fixed={fixed_r2:.4f}, scipy={scipy_r2:.4f}")
        
        # 测试2: 随机旋转的正则单纯形 (多次)
        buggy_rs = []
        fixed_rs = []
        scipy_rs = []
        for seed in range(5):
            rng = np.random.RandomState(seed * 100 + 42)
            Q = random_rotation(dim, rng)
            rotated = simplex @ Q.T  # 旋转
            
            b = compute_fit_r2_BUGGY(rotated, N)
            f = compute_fit_r2_FIXED(rotated, N)
            s = compute_fit_r2_SCIPY(rotated, N)
            buggy_rs.append(b)
            fixed_rs.append(f)
            scipy_rs.append(s)
        
        log(f"  旋转后: buggy={np.mean(buggy_rs):.4f}±{np.std(buggy_rs):.4f}, "
            f"fixed={np.mean(fixed_rs):.4f}±{np.std(fixed_rs):.4f}, "
            f"scipy={np.mean(scipy_rs):.4f}±{np.std(scipy_rs):.4f}")
        
        # 判定
        if np.mean(fixed_rs) > 0.95 and np.mean(buggy_rs) < 0.05:
            log(f"  ★★★★★ BUG确认! buggy≈0, fixed≈1 → Procrustes旋转方向错误!")
        elif np.mean(fixed_rs) > 0.9:
            log(f"  ★★★ fixed正确, buggy可能有部分正确")
        else:
            log(f"  ??? fixed也不高, 可能还有其他问题")
    
    # 测试3: 带噪声的正则单纯形
    log(f"\n--- 带噪声测试 (6类5维, 噪声级别变化) ---")
    N, dim = 6, 6
    simplex = generate_regular_simplex(N, dim)
    rng = np.random.RandomState(42)
    Q = random_rotation(dim, rng)
    rotated = simplex @ Q.T
    
    for noise_level in [0.01, 0.05, 0.1, 0.2, 0.5]:
        noisy = rotated + rng.randn(*rotated.shape) * noise_level
        b = compute_fit_r2_BUGGY(noisy, N)
        f = compute_fit_r2_FIXED(noisy, N)
        s = compute_fit_r2_SCIPY(noisy, N)
        log(f"  noise={noise_level:.2f}: buggy={b:.4f}, fixed={f:.4f}, scipy={s:.4f}")
    
    # 测试4: 非单纯形数据 (随机高斯)
    log(f"\n--- 非单纯形数据 (6类5维随机高斯) ---")
    for seed in range(3):
        rng2 = np.random.RandomState(seed)
        random_data = rng2.randn(6, 6) * 2
        b = compute_fit_r2_BUGGY(random_data, N)
        f = compute_fit_r2_FIXED(random_data, N)
        s = compute_fit_r2_SCIPY(random_data, N)
        log(f"  seed={seed}: buggy={b:.4f}, fixed={f:.4f}, scipy={s:.4f}")
    
    # ================================================================
    # Part 2: 用已保存的模型数据重新计算fit_r2
    # ================================================================
    log("\n" + "=" * 70)
    log("Part 2: 用修正代码重新分析模型数据")
    log("=" * 70)
    
    # 先检查是否有raw_residuals
    has_raw = False
    for model in ["qwen3", "glm4", "deepseek7b"]:
        data_file = TEMP / f"ccxxxix_simplex_rigorous_{model}.json"
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if "raw_residuals" in data:
                has_raw = True
                log(f"  {model}: 有raw_residuals")
            else:
                log(f"  {model}: 无raw_residuals, 需要重新运行模型")
    
    if has_raw:
        results = recompute_model_data()
        
        log("\n--- 修正后的fit_r2结果 ---")
        for model, domains in results.items():
            log(f"\n{model.upper()}:")
            for domain, layers in domains.items():
                if not layers:
                    log(f"  {domain}: 无数据")
                    continue
                # 找最佳层
                best_layer = max(layers.keys(), key=lambda k: layers[k].get("simplex_fit_r2", 0))
                best = layers[best_layer]
                log(f"  {domain} {best_layer}: fit_r2={best['simplex_fit_r2']:.4f}, "
                    f"edge_uni={best['edge_uniformity']:.3f}, angle_dev={best['angle_deviation']:.1f}°")
        
        # 保存修正结果
        output = {}
        for model, domains in results.items():
            output[model] = {}
            for domain, layers in domains.items():
                output[model][domain] = {
                    k: {kk: vv for kk, vv in v.items()} 
                    for k, v in layers.items()
                }
        
        with open(TEMP / "ccxl_fixed_fit_r2.json", 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        log("\n修正结果已保存到 ccxl_fixed_fit_r2.json")
    else:
        log("\n没有raw_residuals, 需要重新运行模型来获取修正后的fit_r2")
    
    # ================================================================
    # Part 3: 总结
    # ================================================================
    log("\n" + "=" * 70)
    log("CCXL 总结")
    log("=" * 70)
    log("\n★★★★★ Bug确认: R = Vt_h.T @ U_h.T (逆旋转) → 应为 R = U_h @ Vt_h")
    log("★★★★★ 修正后需要重新运行CCXXXIX获取正确的fit_r2")
    log("★★★★★ 如果修正后fit_r2>0.5 → 语义空间确实是近似正则单纯形!")
    log("★★★★★ 如果修正后fit_r2仍然≈0 → 语义空间不是正则单纯形(更重要的发现)")
    
    log(f"\n完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
