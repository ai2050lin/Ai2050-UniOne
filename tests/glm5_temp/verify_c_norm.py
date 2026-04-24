"""
CCXLIII-B: 关键验证 — 更严格的归一化
======================================
C_norm = ||[A,B]|| / (A_eff × B_eff)

如果"浅层播种"是真实的 → C_norm仍在浅层更高
如果"浅层播种"是A_eff小导致的假象 → C_norm在浅层反而更低(因为分母含A_eff)
"""

import json
import numpy as np

models = ['deepseek7b', 'qwen3', 'glm4']
model_names = ['DS7B', 'Qwen3', 'GLM4']

print("=" * 70)
print("CRITICAL VERIFICATION: C_norm = ||[A,B]|| / (A_eff × B_eff)")
print("=" * 70)

for model_key, model_name in zip(models, model_names):
    result_path = f"results/causal_fiber/{model_key}_ccxliii/results.json"
    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"\n{model_name}: No results found")
        continue
    
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    
    for pair_name, pair_data in data['pairs'].items():
        eed = pair_data.get('equal_effect_data', {})
        if not eed:
            continue
        
        target = pair_data.get('target_A_effect', 0)
        print(f"\n  Pair: {pair_name}")
        print(f"  Target A_eff = {target:.2f}")
        print(f"  {'L1':>4} {'α_A':>6} {'A_eff':>8} {'B_eff':>8} {'||[A,B]||':>10} {'Rel.Curv':>10} {'C_norm':>10}")
        
        l1_vals = []
        old_rel_curvs = []
        c_norms = []
        abs_curvs = []
        a_effs = []
        b_effs = []
        
        for key in sorted(eed.keys(), key=lambda x: eed[x]['l1']):
            d = eed[key]
            a_eff = d['mean_A_effect']
            b_eff = d['mean_B_effect']
            abs_curv = d['curvature_norm_mean']
            old_rel = d['relative_curvature']
            
            # New normalization: ||[A,B]|| / (A_eff * B_eff)
            c_norm = abs_curv / (a_eff * b_eff + 1e-10)
            
            l1_vals.append(d['l1'])
            old_rel_curvs.append(old_rel)
            c_norms.append(c_norm)
            abs_curvs.append(abs_curv)
            a_effs.append(a_eff)
            b_effs.append(b_eff)
            
            print(f"  L{d['l1']:>2} {d['alpha_A']:>6.3f} {a_eff:>8.2f} {b_eff:>8.2f} {abs_curv:>10.2f} {old_rel:>10.4f} {c_norm:>10.6f}")
        
        # Correlations
        if len(l1_vals) > 2:
            corr_l1_old = np.corrcoef(l1_vals, old_rel_curvs)[0, 1]
            corr_l1_cnorm = np.corrcoef(l1_vals, c_norms)[0, 1]
            corr_l1_abs = np.corrcoef(l1_vals, abs_curvs)[0, 1]
            corr_l1_aeff = np.corrcoef(l1_vals, a_effs)[0, 1]
            
            print(f"\n  Old: Corr(L1, Rel.Curv) = {corr_l1_old:.3f}")
            print(f"  New: Corr(L1, C_norm)   = {corr_l1_cnorm:.3f}")
            print(f"       Corr(L1, ||[A,B]||) = {corr_l1_abs:.3f}")
            print(f"       Corr(L1, A_eff)     = {corr_l1_aeff:.3f}")
            
            print(f"\n  C_norm values: {[f'{c:.6f}' for c in c_norms]}")
            print(f"  C_norm ratio (shallow/deep): {c_norms[0]/c_norms[-1]:.3f}")
            
            if corr_l1_cnorm < -0.3:
                print(f"  *** C_norm STILL higher at shallow L1 → 'Shallow seeding' is REAL ***")
            elif corr_l1_cnorm > 0.3:
                print(f"  *** C_norm HIGHER at deep L1 → OPPOSITE of shallow seeding! ***")
            else:
                print(f"  *** C_norm independent of L1 → 'Shallow seeding' was NORMALIZATION ARTIFACT ***")

# Also check CCXLII data with the new normalization
print(f"\n{'='*70}")
print("CCXLII PANORAMA DATA WITH C_norm")
print(f"{'='*70}")

for model_key, model_name in zip(models, model_names):
    result_path = f"results/causal_fiber/{model_key}_ccxlii/results.json"
    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"\n{model_name}: No results found")
        continue
    
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    
    for pair_name, pair_data in data['pairs'].items():
        pd = pair_data.get('panorama_data', {})
        if not pd:
            continue
        
        print(f"\n  Pair: {pair_name}")
        print(f"  {'L1':>4} {'L2':>4} {'Dist':>4} {'A_eff':>8} {'B_eff':>8} {'||[A,B]||':>10} {'Rel.Curv':>10} {'C_norm':>10}")
        
        l1_vals = []
        dist_vals = []
        old_rel_curvs = []
        c_norms = []
        
        for key in sorted(pd.keys(), key=lambda k: (pd[k]['l1'], pd[k]['l2'])):
            d = pd[key]
            a_eff = d['mean_A_effect']
            b_eff = d['mean_B_effect']
            abs_curv = d['curvature_norm_mean']
            old_rel = d['relative_curvature']
            c_norm = abs_curv / (a_eff * b_eff + 1e-10)
            
            l1_vals.append(d['l1'])
            dist_vals.append(d['distance'])
            old_rel_curvs.append(old_rel)
            c_norms.append(c_norm)
            
            print(f"  L{d['l1']:>2} L{d['l2']:>2} {d['distance']:>4} {a_eff:>8.2f} {b_eff:>8.2f} {abs_curv:>10.2f} {old_rel:>10.4f} {c_norm:>10.6f}")
        
        if len(l1_vals) > 3:
            corr_l1_old = np.corrcoef(l1_vals, old_rel_curvs)[0, 1]
            corr_l1_cnorm = np.corrcoef(l1_vals, c_norms)[0, 1]
            corr_dist_old = np.corrcoef(dist_vals, old_rel_curvs)[0, 1]
            corr_dist_cnorm = np.corrcoef(dist_vals, c_norms)[0, 1]
            
            print(f"\n  Old: Corr(L1, Rel.Curv) = {corr_l1_old:.3f}, Corr(dist, Rel.Curv) = {corr_dist_old:.3f}")
            print(f"  New: Corr(L1, C_norm)   = {corr_l1_cnorm:.3f}, Corr(dist, C_norm)   = {corr_dist_cnorm:.3f}")
            
            # Group by L1
            l1_groups = {}
            for l1, cn in zip(l1_vals, c_norms):
                l1_groups.setdefault(l1, []).append(cn)
            
            print(f"\n  C_norm by L1:")
            for l1 in sorted(l1_groups.keys()):
                mean_cn = np.mean(l1_groups[l1])
                print(f"    L1={l1}: mean C_norm = {mean_cn:.6f}")
