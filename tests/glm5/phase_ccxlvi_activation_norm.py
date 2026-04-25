"""
Phase CCXLVI: Activation-Normalized Local Geometry
===================================================
CRITICAL FOLLOW-UP to CCXLV!

CCXLV found: ||H_l|| increases with depth (Corr=+0.75~+0.88)
BUT: ||h_l|| also increases dramatically (3-18x from shallow to deep)

Question: Is the increase in ||H_l|| just following ||h_l|| growth,
or is there genuine increase in intrinsic curvature per unit of activation?

Key metrics:
  1. kappa_norm = ||H|| / (||J|| * ||h||)  — normalized curvature density
  2. Power law: ||H|| ∝ ||h||^α  — find α via log-log regression
  3. If α ≈ 1 → ||H|| just follows ||h|| (not genuinely deeper nonlinearity)
  4. If α ≈ 2 → ||H|| scales quadratically (Hessian naturally ∝ input²)
  5. If kappa_norm ≈ constant → no intrinsic depth dependence
  6. If kappa_norm decreases → SHALLOW layers have higher intrinsic nonlinearity!

This re-analyzes CCXLV data without needing new model runs.
"""

import json
import os
import numpy as np
from scipy import stats

MODEL_DIRS = {
    'deepseek7b': 'results/causal_fiber/deepseek7b_ccxlv',
    'qwen3': 'results/causal_fiber/qwen3_ccxlv',
    'glm4': 'results/causal_fiber/glm4_ccxlv',
}


def load_ccxlv_results(model_key):
    """Load saved CCXLV results."""
    result_dir = MODEL_DIRS[model_key]
    result_file = os.path.join(result_dir, 'results.json')

    if not os.path.exists(result_file):
        # Try console.log as fallback
        console_file = os.path.join(result_dir, 'console.log')
        if os.path.exists(console_file):
            return parse_console_log(console_file)
        console_file2 = os.path.join(result_dir, 'console2.log')
        if os.path.exists(console_file2):
            return parse_console_log(console_file2)
        return None

    with open(result_file, 'r') as f:
        data = json.load(f)
    return data


def parse_console_log(log_file):
    """Parse CCXLV console output as fallback."""
    geometry = {}
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Match lines like: "  L   1     1.089864     1.0881    38.64   1.001637 L/14.0"
            line = line.strip()
            if line.startswith('L') and 'kappa' not in line.lower():
                parts = line.split()
                try:
                    layer = int(parts[0].replace('L', '').strip())
                    hess = float(parts[1])
                    jac = float(parts[2])
                    act = float(parts[3])
                    kappa = float(parts[4])
                    geometry[str(layer)] = {
                        'layer': layer,
                        'hessian_mean': hess,
                        'jacobian_mean': jac,
                        'activation_norm_mean': act,
                        'kappa': kappa,
                    }
                except (ValueError, IndexError):
                    continue
    if not geometry:
        return None
    return {'geometry_analysis': geometry}


def analyze_activation_normalized(model_key, data):
    """Compute activation-normalized geometry metrics."""
    if data is None:
        print(f"  No data for {model_key}")
        return

    geo = data.get('geometry_analysis', {})
    if not geo:
        print(f"  No geometry data for {model_key}")
        return

    layers = []
    hess = []
    jac = []
    act = []
    kappa = []

    for l_str in sorted(geo.keys(), key=lambda x: int(x)):
        d = geo[l_str]
        layers.append(d['layer'])
        hess.append(d['hessian_mean'])
        jac.append(d['jacobian_mean'])
        act.append(d['activation_norm_mean'])
        kappa.append(d['kappa'])

    layers = np.array(layers, dtype=float)
    hess = np.array(hess)
    jac = np.array(jac)
    act = np.array(act)
    kappa = np.array(kappa)

    # Normalized metrics
    kappa_norm = hess / (jac * act)  # ||H|| / (||J|| * ||h||)
    hess_per_act = hess / act  # ||H|| / ||h||
    hess_per_act2 = hess / (act ** 2)  # ||H|| / ||h||²

    # Power law: ||H|| ∝ ||h||^α
    # Use only points with ||h|| > 1 to avoid log(0) issues
    valid = act > 1.0
    if valid.sum() >= 3:
        log_h = np.log(act[valid])
        log_H = np.log(hess[valid])
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_h, log_H)
        alpha = slope
        r_squared = r_value ** 2
    else:
        alpha = None
        r_squared = None
        p_value = None

    # Also: kappa_norm ∝ ||h||^β
    if valid.sum() >= 3:
        log_h2 = np.log(act[valid])
        log_kn = np.log(kappa_norm[valid])
        slope_kn, intercept_kn, r_kn, p_kn, se_kn = stats.linregress(log_h2, log_kn)
        beta = slope_kn
        r2_kn = r_kn ** 2
    else:
        beta = None
        r2_kn = None

    # Correlations
    n = len(layers)
    if n < 3:
        print(f"  Too few layers ({n}) for correlation analysis")
        return

    corr_hess = np.corrcoef(layers, hess)[0, 1]
    corr_jac = np.corrcoef(layers, jac)[0, 1]
    corr_act = np.corrcoef(layers, act)[0, 1]
    corr_kappa = np.corrcoef(layers, kappa)[0, 1]
    corr_knorm = np.corrcoef(layers, kappa_norm)[0, 1]
    corr_hpa = np.corrcoef(layers, hess_per_act)[0, 1]
    corr_hpa2 = np.corrcoef(layers, hess_per_act2)[0, 1]

    # Print layer-by-layer data
    print(f"\n  Layer-by-Layer Data:")
    print(f"  {'L':>4} {'||H||':>10} {'||J||':>8} {'||h||':>8} {'kappa':>8} "
          f"{'kappa_n':>8} {'H/h':>8} {'H/h2':>10}")
    for i in range(n):
        print(f"  {int(layers[i]):>4} {hess[i]:>10.4f} {jac[i]:>8.4f} {act[i]:>8.2f} "
              f"{kappa[i]:>8.4f} {kappa_norm[i]:>8.6f} {hess_per_act[i]:>8.6f} "
              f"{hess_per_act2[i]:>10.8f}")

    # Print correlations
    print(f"\n  === CORRELATIONS WITH LAYER DEPTH ===")
    print(f"  Corr(layer, ||H||)      = {corr_hess:>7.3f}  [Hessian — CCXLV main finding]")
    print(f"  Corr(layer, ||J||)      = {corr_jac:>7.3f}  [Jacobian]")
    print(f"  Corr(layer, ||h||)      = {corr_act:>7.3f}  [Activation norm — CONFOUNDER!]")
    print(f"  Corr(layer, kappa=H/J)  = {corr_kappa:>7.3f}  [Curvature density — CCXLV]")
    print(f"  Corr(layer, H/(J*h))   = {corr_knorm:>7.3f}  [*** NORMALIZED curvature density ***]")
    print(f"  Corr(layer, H/||h||)   = {corr_hpa:>7.3f}  [Hessian per unit activation]")
    print(f"  Corr(layer, H/||h||2)  = {corr_hpa2:>7.3f}  [Hessian per activation squared]")

    # Power law analysis
    if alpha is not None:
        print(f"\n  === POWER LAW: ||H|| ∝ ||h||^α ===")
        print(f"  α = {alpha:.3f}  (R² = {r_squared:.3f}, p = {p_value:.4e})")
        print(f"  If α ≈ 0: ||H|| independent of ||h|| → genuine depth effect")
        print(f"  If α ≈ 1: ||H|| ∝ ||h|| → just following activation growth")
        print(f"  If α ≈ 2: ||H|| ∝ ||h||² → quadratic scaling (expected for some norms)")
    if beta is not None:
        print(f"\n  === POWER LAW: kappa_norm ∝ ||h||^β ===")
        print(f"  β = {beta:.3f}  (R² = {r2_kn:.3f})")
        print(f"  If β ≈ 0: kappa_norm independent of ||h|| → no residual depth effect")
        print(f"  If β < 0: kappa_norm decreases as ||h|| grows → SHALLOW intrinsic nonlinearity!")

    # INTERPRETATION
    print(f"\n  === INTERPRETATION ===")

    # Compare CCXLV conclusion vs normalized
    if abs(corr_kappa) > 0.3 and abs(corr_knorm) > 0.3:
        if np.sign(corr_kappa) != np.sign(corr_knorm):
            print(f"  *** CRITICAL: Normalization REVERSES the conclusion! ***")
            print(f"  CCXLV: kappa=H/J increases with depth (Corr={corr_kappa:.3f})")
            print(f"  CCXLVI: H/(J*h) DECREASES with depth (Corr={corr_knorm:.3f})")
            print(f"  → The apparent 'deep nonlinearity' is just activation norm growth!")
            print(f"  → SHALLOW layers actually have HIGHER intrinsic curvature density!")
        elif abs(corr_knorm) < abs(corr_kappa) * 0.5:
            print(f"  Normalization WEAKENS the conclusion significantly:")
            print(f"  CCXLV: kappa=H/J Corr={corr_kappa:.3f}")
            print(f"  CCXLVI: H/(J*h) Corr={corr_knorm:.3f}")
            print(f"  → Much of the 'deep nonlinearity' is activation norm artifact")
        else:
            print(f"  Normalization preserves the conclusion:")
            print(f"  CCXLV: kappa=H/J Corr={corr_kappa:.3f}")
            print(f"  CCXLVI: H/(J*h) Corr={corr_knorm:.3f}")
            print(f"  → Deep nonlinearity is genuine even after accounting for ||h||")
    elif abs(corr_knorm) < 0.3:
        print(f"  Normalization ELIMINATES the depth dependence:")
        print(f"  CCXLV: kappa=H/J Corr={corr_kappa:.3f}")
        print(f"  CCXLVI: H/(J*h) Corr={corr_knorm:.3f}")
        print(f"  → The 'deep nonlinearity' is entirely an activation norm artifact!")

    if alpha is not None:
        if alpha > 0.8:
            print(f"\n  Power law α={alpha:.2f} ≈ 1: ||H|| mainly follows ||h|| growth!")
        elif alpha > 0.5:
            print(f"\n  Power law α={alpha:.2f}: ||H|| partially follows ||h|| growth")
        else:
            print(f"\n  Power law α={alpha:.2f} < 0.5: ||H|| has genuine depth dependence")

    # Growth ratios
    shallow_idx = 0
    deep_idx = -1
    act_ratio = act[deep_idx] / (act[shallow_idx] + 1e-10)
    hess_ratio = hess[deep_idx] / (hess[shallow_idx] + 1e-10)
    knorm_ratio = kappa_norm[deep_idx] / (kappa_norm[shallow_idx] + 1e-10)

    print(f"\n  === GROWTH RATIOS (deep/shallow) ===")
    print(f"  ||h|| growth:        {act_ratio:.1f}x")
    print(f"  ||H|| growth:        {hess_ratio:.1f}x")
    print(f"  kappa_norm growth:   {knorm_ratio:.3f}x")
    if hess_ratio < act_ratio:
        print(f"  ||H|| grows SLOWER than ||h|| → curvature density DILUTED by activation growth")

    return {
        'corr_hess': corr_hess,
        'corr_kappa': corr_kappa,
        'corr_knorm': corr_knorm,
        'corr_hpa': corr_hpa,
        'alpha': alpha,
        'r_squared': r_squared,
        'act_ratio': act_ratio,
        'hess_ratio': hess_ratio,
        'knorm_ratio': knorm_ratio,
    }


def main():
    print("=" * 70)
    print("Phase CCXLVI: Activation-Normalized Local Geometry")
    print("CRITICAL: Does ||H|| growth just follow ||h|| growth?")
    print("=" * 70)

    all_results = {}

    for model_key in ['deepseek7b', 'qwen3', 'glm4']:
        print(f"\n{'=' * 70}")
        print(f"Model: {model_key}")
        print(f"{'=' * 70}")

        data = load_ccxlv_results(model_key)
        result = analyze_activation_normalized(model_key, data)
        if result:
            all_results[model_key] = result

    # Cross-model summary
    print(f"\n{'=' * 70}")
    print(f"CROSS-MODEL SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Model':>12} {'Corr(H/J,L)':>12} {'Corr(H/Jh,L)':>13} {'α':>6} "
          f"{'||h||↑':>8} {'||H||↑':>8} {'knorm↑':>8} {'Reversal?':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*13} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for mk, r in all_results.items():
        reversal = "YES!" if (r['corr_kappa'] > 0.3 and r['corr_knorm'] < -0.3) else \
                   "Weakened" if abs(r['corr_knorm']) < abs(r['corr_kappa']) * 0.5 else "No"
        print(f"  {mk:>12} {r['corr_kappa']:>12.3f} {r['corr_knorm']:>13.3f} "
              f"{r['alpha'] if r['alpha'] else 'N/A':>6} "
              f"{r['act_ratio']:>8.1f} {r['hess_ratio']:>8.1f} {r['knorm_ratio']:>8.3f} "
              f"{reversal:>10}")

    # Final verdict
    print(f"\n{'=' * 70}")
    print(f"FINAL VERDICT")
    print(f"{'=' * 70}")

    reversals = sum(1 for r in all_results.values()
                    if r['corr_kappa'] > 0.3 and r['corr_knorm'] < -0.3)
    weakened = sum(1 for r in all_results.values()
                   if abs(r['corr_knorm']) < abs(r['corr_kappa']) * 0.5)

    if reversals >= 2:
        print(f"\n  *** MAJOR REVERSAL: Normalization REVERSES CCXLV conclusion! ***")
        print(f"  CCXLV: Deep layers more nonlinear (kappa=H/J increases with depth)")
        print(f"  CCXLVI: Shallow layers more nonlinear (H/(J*h) DECREASES with depth)")
        print(f"  The 'deep nonlinearity' was an activation norm artifact!")
        print(f"  *** SHALLOW LAYERS HAVE HIGHER INTRINSIC CURVATURE DENSITY! ***")
    elif weakened >= 2:
        print(f"\n  *** SIGNIFICANT REVISION: Normalization weakens CCXLV conclusion ***")
        print(f"  The 'deep nonlinearity' is partially an activation norm artifact")
        print(f"  Intrinsic curvature density (H/J/h) shows weaker/no depth dependence")
    else:
        print(f"\n  Normalization preserves CCXLV conclusion")
        print(f"  Deep layers genuinely have higher intrinsic curvature density")


if __name__ == '__main__':
    main()
