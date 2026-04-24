"""Post-hoc analysis: compute H_true = ||[A,B]||_fix / A_eff_fix
This removes BOTH local perturbation bias AND forward propagation bias.
If this still varies with L1, it's truly intrinsic non-commutativity."""

import numpy as np

# DS7B data (from console.log)
ds7b = {
    'l1': [4, 9, 14, 18],
    'curv_fix': [158.94, 122.72, 136.23, 50.24],
    'A_eff_fix': [43.82, 43.38, 33.72, 41.34],
    'B_eff': [70.57, 70.57, 70.57, 70.57],
    'local_norm_fix': [58.615, 58.615, 58.615, 58.615],
    'C_norm_fix': [0.051401, 0.040089, 0.057245, 0.017222],
    'H_intr_fix': [0.038427, 0.029668, 0.032936, 0.012146],
    'J_fwd_fix': [0.748, 0.740, 0.575, 0.705],
}

# Qwen3 data
qwen3 = {
    'l1': [5, 12, 18, 24],
    'curv_fix': [66.82, 88.47, 37.68, 23.82],
    'A_eff_fix': [19.68, 12.51, 10.35, 11.46],
    'B_eff': [37.91, 37.91, 37.91, 37.91],
    'local_norm_fix': [15.741, 15.741, 15.741, 15.741],
    'C_norm_fix': [0.089570, 0.186616, 0.096020, 0.054813],
    'H_intr_fix': [0.111974, 0.148271, 0.063148, 0.039923],
    'J_fwd_fix': [1.250, 0.795, 0.658, 0.728],
}

# GLM4 data
glm4 = {
    'l1': [5, 13, 20, 26],
    'curv_fix': [256.12, 255.98, 147.79, 53.95],
    'A_eff_fix': [62.14, 36.14, 28.14, 20.92],
    'B_eff': [79.73, 79.73, 79.73, 79.73],
    'local_norm_fix': [3.101, 3.101, 3.101, 3.101],
    'C_norm_fix': [0.051694, 0.088849, 0.065885, 0.032342],
    'H_intr_fix': [1.217594, 1.035440, 0.597831, 0.218227],
    'J_fwd_fix': [23.554, 11.654, 9.074, 6.748],
}

print("=" * 80)
print("H_TRUE ANALYSIS: ||[A,B]||_fix / A_eff_fix")
print("This removes BOTH local perturbation AND forward propagation bias")
print("=" * 80)

for name, data in [("DS7B", ds7b), ("Qwen3", qwen3), ("GLM4", glm4)]:
    l1 = np.array(data['l1'])
    curv = np.array(data['curv_fix'])
    A_eff = np.array(data['A_eff_fix'])
    B_eff = np.array(data['B_eff'])
    local_norm = np.array(data['local_norm_fix'])
    C_norm = np.array(data['C_norm_fix'])
    H_intr = np.array(data['H_intr_fix'])
    J_fwd = np.array(data['J_fwd_fix'])

    # H_true: fully normalized - removes ALL propagation gains
    # H_true = ||[A,B]|| / (||delta_L1|| * A_eff * B_eff / (||delta_L1|| * ||delta_L2||))
    # Simplifies to: ||[A,B]||_fix / A_eff_fix (since ||delta_L1|| and ||delta_L2|| are constant)
    H_true = curv / A_eff

    print(f"\n--- {name} ---")
    print(f"  L1   ||[A,B]||_fix  A_eff_fix  H_true=||[A,B]||/A_eff  C_norm_fix  H_intr_fix  J_fwd")
    for i in range(len(l1)):
        print(f"  L{l1[i]:>2}  {curv[i]:>10.2f}  {A_eff[i]:>9.2f}  {H_true[i]:>10.4f}       {C_norm[i]:>10.6f}  {H_intr[i]:>10.6f}  {J_fwd[i]:>7.3f}")

    # Correlations
    corr_H_true = np.corrcoef(l1, H_true)[0, 1]
    corr_C_norm = np.corrcoef(l1, C_norm)[0, 1]
    corr_H_intr = np.corrcoef(l1, H_intr)[0, 1]
    corr_J_fwd = np.corrcoef(l1, J_fwd)[0, 1]
    corr_curv = np.corrcoef(l1, curv)[0, 1]

    print(f"\n  Corr(L1, H_true)        = {corr_H_true:>7.3f}  [fully normalized]")
    print(f"  Corr(L1, ||[A,B]||_fix) = {corr_curv:>7.3f}  [fixed local pert only]")
    print(f"  Corr(L1, H_intr_fix)    = {corr_H_intr:>7.3f}  [local pert + B_eff normalized]")
    print(f"  Corr(L1, C_norm_fix)    = {corr_C_norm:>7.3f}  [A_eff x B_eff normalized]")
    print(f"  Corr(L1, J_fwd_fix)     = {corr_J_fwd:>7.3f}  [forward gain]")

    # Interpretation
    if abs(corr_H_true) < 0.3:
        verdict = "PROPAGATION ARTIFACT - after removing ALL gains, no L1 dependence"
    elif corr_H_true < -0.3:
        verdict = "GENUINE - intrinsic non-commutativity stronger at shallow layers"
    else:
        verdict = "UNEXPECTED - deep layers have stronger intrinsic non-commutativity"
    print(f"\n  VERDICT: {verdict}")

print("\n" + "=" * 80)
print("CROSS-MODEL COMPARISON")
print("=" * 80)
print(f"\n  {'Metric':<25} {'DS7B':>8} {'Qwen3':>8} {'GLM4':>8} {'Consensus':>12}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")

metrics = {
    'Corr(L1, H_true)': [],
    'Corr(L1, ||[A,B]||_fix)': [],
    'Corr(L1, H_intr_fix)': [],
    'Corr(L1, C_norm_fix)': [],
    'Corr(L1, J_forward)': [],
}

for name, data in [("DS7B", ds7b), ("Qwen3", qwen3), ("GLM4", glm4)]:
    l1 = np.array(data['l1'])
    curv = np.array(data['curv_fix'])
    A_eff = np.array(data['A_eff_fix'])
    C_norm = np.array(data['C_norm_fix'])
    H_intr = np.array(data['H_intr_fix'])
    J_fwd = np.array(data['J_fwd_fix'])
    H_true = curv / A_eff

    metrics['Corr(L1, H_true)'].append(np.corrcoef(l1, H_true)[0, 1])
    metrics['Corr(L1, ||[A,B]||_fix)'].append(np.corrcoef(l1, curv)[0, 1])
    metrics['Corr(L1, H_intr_fix)'].append(np.corrcoef(l1, H_intr)[0, 1])
    metrics['Corr(L1, C_norm_fix)'].append(np.corrcoef(l1, C_norm)[0, 1])
    metrics['Corr(L1, J_forward)'].append(np.corrcoef(l1, J_fwd)[0, 1])

for metric, vals in metrics.items():
    consensus = "Yes" if all(v < -0.3 for v in vals) else ("No" if any(v > -0.3 for v in vals) else "Mixed")
    print(f"  {metric:<25} {vals[0]:>8.3f} {vals[1]:>8.3f} {vals[2]:>8.3f} {consensus:>12}")

print(f"\n  KEY INSIGHT:")
print(f"  - H_true (fully normalized) shows WEAKER but still negative correlation")
print(f"  - This means part of the 'shallow seeding' was propagation artifact,")
print(f"    but a genuine component remains")
print(f"  - The remaining L1 dependence in H_true could be due to:")
print(f"    (a) Genuinely stronger interaction tensor H at shallow layers")
print(f"    (b) J_{{L1->L2}} variation not fully captured by A_eff")
print(f"    (c) Nonlinear effects not captured by linear Jacobian decomposition")
