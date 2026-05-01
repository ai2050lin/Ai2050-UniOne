"""快速验证两个bug修正"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

def generate_regular_simplex(N):
    """N个顶点的正则(N-1)维单纯形"""
    vertices = np.zeros((N, N))
    for i in range(N):
        vertices[i, i] = 1.0
    vertices = vertices - vertices.mean(axis=0)
    return vertices

def compute_fit_r2_FIXED_V2(centers, N):
    """修正两个bug后的版本:
    Bug1: R = U_h @ Vt_h (原为Vt_h.T @ U_h.T = 逆旋转)
    Bug2: if n_dim >= N - 1 (原为n_dim >= N, 导致n_dim=N-1时直接fit_r2=0)
    """
    n_dim = centers.shape[1]
    if N - 1 > n_dim:
        return 0.0
    
    ideal_vertices = np.zeros((N, N))
    for i in range(N):
        ideal_vertices[i, i] = 1.0
    ideal_vertices = ideal_vertices - ideal_vertices.mean(axis=0)
    
    mean_edge = np.mean(pdist(centers))
    ideal_dists = squareform(pdist(ideal_vertices))
    ideal_mean_dist = np.mean(ideal_dists[np.triu_indices(N, k=1)])
    ideal_vertices = ideal_vertices / max(ideal_mean_dist, 1e-10) * mean_edge
    
    # Bug2 fix: n_dim >= N - 1 (not n_dim >= N)
    if n_dim >= N - 1:
        U_c, S_c, Vt_c = np.linalg.svd(centers - centers.mean(axis=0), full_matrices=False)
        proj_dim = min(N - 1, n_dim)
        centers_proj = (centers - centers.mean(axis=0)) @ Vt_c[:proj_dim].T
        
        U_i, S_i, Vt_i = np.linalg.svd(ideal_vertices - ideal_vertices.mean(axis=0), full_matrices=False)
        ideal_proj = (ideal_vertices - ideal_vertices.mean(axis=0)) @ Vt_i[:proj_dim].T
        
        # Bug1 fix: R = U_h @ Vt_h (not Vt_h.T @ U_h.T)
        H = ideal_proj.T @ centers_proj
        U_h, S_h, Vt_h = np.linalg.svd(H)
        R = U_h @ Vt_h  # FIXED
        
        aligned_ideal = ideal_proj @ R
        residual = centers_proj - aligned_ideal
        ss_res = np.sum(residual ** 2)
        ss_tot = np.sum((centers_proj - centers_proj.mean(axis=0)) ** 2)
        fit_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        return max(0.0, fit_r2)
    return 0.0

print("=== Bug修正验证 ===")
print()

# 测试1: 正则单纯形在N维空间 (d_dim=N)
for N in [3, 4, 5, 6]:
    simplex = generate_regular_simplex(N)
    r2 = compute_fit_r2_FIXED_V2(simplex, N)
    print(f"正则{N}类单纯形(d={N}): fit_r2={r2:.6f}")

print()

# 测试2: 正则单纯形投影到N-1维 (模拟CCXXXIX的数据流)
# CCXXXIX中, centers已被投影到n_sep=N-1维
for N in [3, 4, 5, 6]:
    simplex = generate_regular_simplex(N)
    # 投影到N-1维
    U, S, Vt = np.linalg.svd(simplex - simplex.mean(axis=0), full_matrices=False)
    proj = (simplex - simplex.mean(axis=0)) @ Vt[:N-1].T
    r2 = compute_fit_r2_FIXED_V2(proj, N)
    print(f"正则{N}类单纯形投影到{N-1}维: fit_r2={r2:.6f}")

print()

# 测试3: 投影后加随机旋转
for N in [3, 4, 5, 6]:
    simplex = generate_regular_simplex(N)
    U, S, Vt = np.linalg.svd(simplex - simplex.mean(axis=0), full_matrices=False)
    proj = (simplex - simplex.mean(axis=0)) @ Vt[:N-1].T
    # 随机旋转
    rng = np.random.RandomState(42)
    A = rng.randn(N-1, N-1)
    Q, R_mat = np.linalg.qr(A)
    rotated = proj @ Q.T
    r2 = compute_fit_r2_FIXED_V2(rotated, N)
    print(f"正则{N}类单纯形投影到{N-1}维+旋转: fit_r2={r2:.6f}")

print()

# 测试4: 带噪声
N = 6
simplex = generate_regular_simplex(N)
U, S, Vt = np.linalg.svd(simplex - simplex.mean(axis=0), full_matrices=False)
proj = (simplex - simplex.mean(axis=0)) @ Vt[:N-1].T
rng = np.random.RandomState(42)
A = rng.randn(N-1, N-1)
Q, R_mat = np.linalg.qr(A)
rotated = proj @ Q.T

for noise in [0.01, 0.05, 0.1, 0.2, 0.5]:
    noisy = rotated + rng.randn(*rotated.shape) * noise
    r2 = compute_fit_r2_FIXED_V2(noisy, N)
    print(f"6类5维+旋转+噪声{noise:.2f}: fit_r2={r2:.6f}")

print()

# 测试5: 随机数据 (非单纯形)
for seed in range(3):
    rng2 = np.random.RandomState(seed)
    random_data = rng2.randn(6, 5) * 2
    r2 = compute_fit_r2_FIXED_V2(random_data, 6)
    print(f"6类5维随机数据(seed={seed}): fit_r2={r2:.6f}")
