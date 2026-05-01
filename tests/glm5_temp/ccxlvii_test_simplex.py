import numpy as np
import sys
sys.path.insert(0, '.')
from tests.glm5.ccxlvii_unified_simplex_tangential import construct_regular_simplex

for N in [2, 3, 4, 5, 6]:
    v = construct_regular_simplex(N)
    print(f"N={N}: shape={v.shape}, center_norm={np.linalg.norm(np.mean(v,axis=0)):.6f}")
    dists = [np.linalg.norm(v[i]-v[j]) for i in range(N) for j in range(i+1, N)]
    print(f"  dists: mean={np.mean(dists):.4f} std={np.std(dists):.6f}")
    
    c = np.mean(v, axis=0)
    angles = []
    for i in range(N):
        for j in range(i+1, N):
            vi = v[i] - c
            vj = v[j] - c
            ni = np.linalg.norm(vi)
            nj = np.linalg.norm(vj)
            if ni > 1e-10 and nj > 1e-10:
                angles.append(np.dot(vi, vj) / (ni * nj))
    if angles:
        print(f"  cos_angles: mean={np.mean(angles):.4f} ideal={-1/N:.4f}")
