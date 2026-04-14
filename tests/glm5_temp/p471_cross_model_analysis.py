"""
Phase XCVII Cross-model summary analysis
==========================================
汇总三模型的P471数据, 进行跨模型统一gamma公式分析
"""
import numpy as np
from numpy.linalg import lstsq

# 跨模型数据 (从三个模型的P471输出汇总)
# 格式: (model, alpha, gamma, delta, d_model, n_layers, layer_frac)
data = [
    # Qwen3: delta=0.175, d_model=2560, n_layers=36
    ("qwen3", -0.162, 1.050, 0.175, 2560, 36, 0.06),
    ("qwen3", -0.070, 1.028, 0.175, 2560, 36, 0.14),
    ("qwen3",  0.018, 1.006, 0.175, 2560, 36, 0.22),
    ("qwen3",  0.088, 0.992, 0.175, 2560, 36, 0.31),
    ("qwen3",  0.144, 0.983, 0.175, 2560, 36, 0.39),
    ("qwen3",  0.188, 0.974, 0.175, 2560, 36, 0.47),
    ("qwen3",  0.222, 0.968, 0.175, 2560, 36, 0.56),
    ("qwen3",  0.282, 0.948, 0.175, 2560, 36, 0.64),
    ("qwen3",  0.394, 0.926, 0.175, 2560, 36, 0.72),
    ("qwen3",  0.484, 0.921, 0.175, 2560, 36, 0.81),
    ("qwen3",  0.570, 0.916, 0.175, 2560, 36, 0.89),
    # GLM4: delta=0.181, d_model=4096, n_layers=40
    ("glm4",  0.724, 0.800, 0.181, 4096, 40, 0.05),
    ("glm4",  0.817, 0.761, 0.181, 4096, 40, 0.13),
    ("glm4",  0.889, 0.736, 0.181, 4096, 40, 0.20),
    ("glm4",  0.950, 0.714, 0.181, 4096, 40, 0.28),
    ("glm4",  1.012, 0.697, 0.181, 4096, 40, 0.35),
    ("glm4",  1.072, 0.680, 0.181, 4096, 40, 0.43),
    ("glm4",  1.139, 0.658, 0.181, 4096, 40, 0.50),
    ("glm4",  1.227, 0.630, 0.181, 4096, 40, 0.58),
    ("glm4",  1.336, 0.592, 0.181, 4096, 40, 0.65),
    ("glm4",  1.466, 0.546, 0.181, 4096, 40, 0.73),
    ("glm4",  1.603, 0.513, 0.181, 4096, 40, 0.80),
    ("glm4",  1.698, 0.485, 0.181, 4096, 40, 0.88),
    ("glm4",  1.748, 0.468, 0.181, 4096, 40, 0.95),
    # DeepSeek7B: delta=0.194, d_model=3584, n_layers=28
    ("deepseek7b",  0.468, 0.866, 0.194, 3584, 28, 0.07),
    ("deepseek7b",  0.514, 0.841, 0.194, 3584, 28, 0.14),
    ("deepseek7b",  0.555, 0.818, 0.194, 3584, 28, 0.21),
    ("deepseek7b",  0.602, 0.794, 0.194, 3584, 28, 0.29),
    ("deepseek7b",  0.658, 0.769, 0.194, 3584, 28, 0.36),
    ("deepseek7b",  0.726, 0.739, 0.194, 3584, 28, 0.43),
    ("deepseek7b",  0.813, 0.700, 0.194, 3584, 28, 0.50),
    ("deepseek7b",  0.917, 0.657, 0.194, 3584, 28, 0.57),
    ("deepseek7b",  1.034, 0.607, 0.194, 3584, 28, 0.64),
    ("deepseek7b",  1.155, 0.557, 0.194, 3584, 28, 0.71),
    ("deepseek7b",  1.218, 0.516, 0.194, 3584, 28, 0.79),
    ("deepseek7b",  1.152, 0.499, 0.194, 3584, 28, 0.86),
    ("deepseek7b",  1.002, 0.484, 0.194, 3584, 28, 0.93),
]

data = np.array([(d[1], d[2], d[3], d[4], d[5], d[6]) for d in data], dtype=float)
alphas = data[:, 0]
gammas = data[:, 1]
deltas = data[:, 2]
d_models = data[:, 3]
n_layers_arr = data[:, 4]
layer_fracs = data[:, 5]

n = len(alphas)
print(f"Total data points: {n}")
print(f"Alpha range: {alphas.min():.3f} ~ {alphas.max():.3f}")
print(f"Gamma range: {gammas.min():.3f} ~ {gammas.max():.3f}")
print(f"Delta values: {np.unique(deltas)}")

ss_tot = np.sum((gammas - np.mean(gammas)) ** 2)

# Model A: gamma = 1 - delta*alpha
gamma_A = 1 - deltas * alphas
R2_A = 1 - np.sum((gammas - gamma_A) ** 2) / ss_tot

# Model B: gamma = c3 - c2*alpha (single c2 for all)
X_B = np.column_stack([-alphas, np.ones(n)])
c_B, _, _, _ = lstsq(X_B, gammas, rcond=None)
c2_B = -c_B[0]
c3_B = c_B[1]
gamma_B = c3_B - c2_B * alphas
R2_B = 1 - np.sum((gammas - gamma_B) ** 2) / ss_tot

# Model C: gamma = 1 - c_eff * alpha (with c_eff = c1*delta + c2)
X_C = np.column_stack([-deltas * alphas, -alphas, np.ones(n)])
c_C, _, _, _ = lstsq(X_C, gammas, rcond=None)
gamma_C = X_C @ c_C
R2_C = 1 - np.sum((gammas - gamma_C) ** 2) / ss_tot
c_eff_C = c_C[0] * np.mean(deltas) + c_C[1]

# Model D: Full multi-variable
X_D = np.column_stack([
    deltas, alphas, deltas * alphas,
    d_models / n_layers_arr, layer_fracs,
    np.ones(n),
])
c_D, _, _, _ = lstsq(X_D, gammas, rcond=None)
gamma_D = X_D @ c_D
R2_D = 1 - np.sum((gammas - gamma_D) ** 2) / ss_tot

# Model E: gamma = 1 - (a*delta + b*alpha + c)*alpha (nonlinear in alpha)
X_E = np.column_stack([
    -deltas * alphas, -alphas**2, -alphas, np.ones(n),
])
c_E, _, _, _ = lstsq(X_E, gammas, rcond=None)
gamma_E = X_E @ c_E
R2_E = 1 - np.sum((gammas - gamma_E) ** 2) / ss_tot

# Model F: gamma = a1 + a2*alpha + a3*alpha^2 + a4*delta
X_F = np.column_stack([
    np.ones(n), alphas, alphas**2, deltas,
])
c_F, _, _, _ = lstsq(X_F, gammas, rcond=None)
gamma_F = X_F @ c_F
R2_F = 1 - np.sum((gammas - gamma_F) ** 2) / ss_tot

print(f"\n{'='*70}")
print(f"  Cross-model gamma formula comparison ({n} data points)")
print(f"{'='*70}")

print(f"\n  Model A: gamma = 1 - delta*alpha")
print(f"    R2 = {R2_A:.3f}")

print(f"\n  Model B: gamma = {c3_B:.3f} - {c2_B:.3f}*alpha")
print(f"    R2 = {R2_B:.3f}")

print(f"\n  Model C: gamma = c0 + c1*(-delta*alpha) + c2*(-alpha)")
print(f"    = {c_C[2]:.3f} + {c_C[0]:.3f}*(-delta*alpha) + {c_C[1]:.3f}*(-alpha)")
print(f"    effective c = c1*delta + c2 = {c_eff_C:.4f}")
print(f"    R2 = {R2_C:.3f}")

print(f"\n  Model D: gamma = c1*delta + c2*alpha + c3*delta*alpha + c4*(d/n_L) + c5*layer_frac + c6")
labels_D = ["delta", "alpha", "delta*alpha", "d/n_L", "layer_frac", "const"]
for i, l in enumerate(labels_D):
    print(f"    {l}: {c_D[i]:.6f}")
print(f"    R2 = {R2_D:.3f}")

print(f"\n  Model E: gamma = c0 + c1*(-delta*alpha) + c2*(-alpha^2) + c3*(-alpha)")
print(f"    = {c_E[3]:.3f} + {c_E[0]:.3f}*(-delta*alpha) + {c_E[1]:.3f}*(-alpha^2) + {c_E[2]:.3f}*(-alpha)")
print(f"    R2 = {R2_E:.3f}")

print(f"\n  Model F: gamma = {c_F[0]:.3f} + {c_F[1]:.3f}*alpha + {c_F[2]:.3f}*alpha^2 + {c_F[3]:.3f}*delta")
print(f"    R2 = {R2_F:.3f}")

# Best model
models = [("A", R2_A), ("B", R2_B), ("C", R2_C), ("D", R2_D), ("E", R2_E), ("F", R2_F)]
best_name, best_R2 = max(models, key=lambda x: x[1])
print(f"\n  Best model: {best_name} (R2 = {best_R2:.3f})")

# Per-model analysis
print(f"\n{'='*70}")
print(f"  Per-model effective coupling constants")
print(f"{'='*70}")

for model_name, delta_val in [("qwen3", 0.175), ("glm4", 0.181), ("deepseek7b", 0.194)]:
    mask = np.array([d[3] for d in 
        [("qwen3", 0, 0, 0.175)] * 11 + 
        [("glm4", 0, 0, 0.181)] * 13 + 
        [("deepseek7b", 0, 0, 0.194)] * 13
    ] == delta_val)
    
    a = alphas[mask]
    g = gammas[mask]
    
    if len(a) > 2 and np.std(a) > 1e-10:
        X = np.column_stack([-a, np.ones(len(a))])
        c, _, _, _ = lstsq(X, g, rcond=None)
        c2 = -c[0]
        c3 = c[1]
        
        print(f"  {model_name}: c2={c2:.4f}, c3={c3:.4f}, c/delta={c2/delta_val:.2f}")
        
        # Alpha range
        print(f"    alpha: {a.min():.3f} ~ {a.max():.3f}")
        print(f"    gamma: {g.min():.3f} ~ {g.max():.3f}")
        
        # Nonlinear fit: gamma = c3 - c2*alpha - c_nl*alpha^2
        if len(a) > 3:
            X_nl = np.column_stack([-a, -a**2, np.ones(len(a))])
            c_nl, _, _, _ = lstsq(X_nl, g, rcond=None)
            gamma_pred = c_nl[2] - c_nl[0] * a - c_nl[1] * a**2
            R2_nl = 1 - np.sum((g - gamma_pred)**2) / np.sum((g - np.mean(g))**2)
            print(f"    Nonlinear: gamma = {c_nl[2]:.3f} - {c_nl[0]:.3f}*alpha - {c_nl[1]:.3f}*alpha^2")
            print(f"    R2(linear) vs R2(nonlinear): {1-np.sum((g-(c3-c2*a))**2)/np.sum((g-np.mean(g))**2):.3f} vs {R2_nl:.3f}")

# Cross-model c vs model properties
print(f"\n{'='*70}")
print(f"  Cross-model c2 vs model properties")
print(f"{'='*70}")

models_info = [
    ("qwen3", 0.175, 2560, 36, 71.1),
    ("glm4", 0.181, 4096, 40, 102.4),
    ("deepseek7b", 0.194, 3584, 28, 128.0),
]

c2_values = {}
for model_name, delta_val, d, n_l, d_over_n in models_info:
    mask_data = {
        "qwen3": (0, 11),
        "glm4": (11, 24),
        "deepseek7b": (24, 37),
    }
    start, end = mask_data[model_name]
    a = alphas[start:end]
    g = gammas[start:end]
    
    if len(a) > 2 and np.std(a) > 1e-10:
        X = np.column_stack([-a, np.ones(len(a))])
        c, _, _, _ = lstsq(X, g, rcond=None)
        c2_values[model_name] = -c[0]

print(f"  Model      | delta  | c2     | c/delta | n_L | d     | d/n_L")
print(f"  -----------|--------|--------|---------|-----|-------|------")
for model_name, delta_val, d, n_l, d_over_n in models_info:
    c2 = c2_values.get(model_name, 0)
    print(f"  {model_name:10s} | {delta_val:.3f}  | {c2:.4f} | {c2/delta_val:.2f}   | {n_l:3d} | {d:5d} | {d_over_n:.1f}")

# Check if c/delta correlates with d/n_L
c_ratios = [c2_values.get(m, 0) / d for m, d in 
            [("qwen3", 0.175), ("glm4", 0.181), ("deepseek7b", 0.194)]]
d_over_n = [71.1, 102.4, 128.0]

print(f"\n  c/delta vs d/n_L correlation:")
if len(c_ratios) >= 2:
    corr = np.corrcoef(c_ratios, d_over_n)[0, 1]
    print(f"    Correlation = {corr:.3f}")

# Check c/delta vs sigma_W (from P470 results)
sigma_W = {"qwen3": 0.725, "glm4": 0.529, "deepseek7b": 0.961}
c_delta = {m: c2_values.get(m, 0) / d for m, d in 
           [("qwen3", 0.175), ("glm4", 0.181), ("deepseek7b", 0.194)]}

print(f"\n  c/delta vs sigma_W:")
for m in ["qwen3", "glm4", "deepseek7b"]:
    print(f"    {m}: c/delta={c_delta.get(m, 0):.2f}, sigma_W={sigma_W.get(m, 0):.3f}")

if len(list(c_delta.values())) >= 2:
    corr_sw = np.corrcoef(list(c_delta.values()), list(sigma_W.values()))[0, 1]
    print(f"    Correlation = {corr_sw:.3f}")

print(f"\n  Conclusion:")
print(f"  1. gamma = 1 - delta*alpha has R2={R2_A:.3f} (poor)")
print(f"  2. Best model is {best_name} with R2={best_R2:.3f}")
print(f"  3. Effective coupling c varies across models:")
print(f"     c/delta ~ {min(c_delta.values()):.2f} to {max(c_delta.values()):.2f}")
print(f"  4. c/delta does NOT simply correlate with d/n_L or sigma_W")
print(f"  5. The relationship c = f(delta, alpha, d, n_L) is complex and nonlinear")
