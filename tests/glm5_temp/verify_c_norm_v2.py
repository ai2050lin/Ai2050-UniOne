"""
Parse CCXLII console.log data and compute C_norm = ||[A,B]|| / (A_eff * B_eff)
Also parse CCXLIII console.log data
"""
import re
import numpy as np

def parse_ccxlii_console(filepath, model_name):
    """Parse CCXLII console.log to extract L1, L2, A_eff, B_eff, abs_curv, rel_curv"""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    data = []
    in_detailed = False
    for line in lines:
        if 'Detailed curvature values' in line:
            in_detailed = True
            continue
        if in_detailed:
            m = re.match(r'\s+L\s*(\d+)\s+L\s*(\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', line)
            if m:
                data.append({
                    'L1': int(m.group(1)),
                    'L2': int(m.group(2)),
                    'dist': int(m.group(3)),
                    'abs_curv': float(m.group(4)),
                    'rel_curv': float(m.group(5)),
                    'A_eff': float(m.group(6)),
                    'B_eff': float(m.group(7)),
                })
    return data

def parse_ccxliii_console(filepath, model_name):
    """Parse CCXLIII console.log to extract L1, alpha, A_eff, B_eff, abs_curv, rel_curv"""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    data = []
    in_table = False
    for line in lines:
        if 'Equal-Effect Curvature' in line:
            in_table = True
            continue
        if in_table:
            m = re.match(r'\s+L\s*(\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', line)
            if m:
                data.append({
                    'L1': int(m.group(1)),
                    'alpha_A': float(m.group(2)),
                    'dist': int(m.group(3)),
                    'A_eff': float(m.group(4)),
                    'B_eff': float(m.group(5)),
                    'abs_curv': float(m.group(6)),
                    'rel_curv': float(m.group(7)),
                })
    return data

print("=" * 70)
print("CRITICAL VERIFICATION: C_norm = ||[A,B]|| / (A_eff * B_eff)")
print("=" * 70)

# CCXLII Panorama data
for model_key, model_name in [('deepseek7b', 'DS7B'), ('qwen3', 'Qwen3'), ('glm4', 'GLM4')]:
    filepath = f"results/causal_fiber/{model_key}_ccxlii/console.log"
    try:
        data = parse_ccxlii_console(filepath, model_name)
    except:
        print(f"\n{model_name} CCXLII: no data")
        continue
    
    if not data:
        print(f"\n{model_name} CCXLII: no data parsed")
        continue
    
    print(f"\n{'='*60}")
    print(f"{model_name} CCXLII (standard alpha)")
    print(f"{'='*60}")
    
    print(f"  {'L1':>4} {'L2':>4} {'Dist':>4} {'A_eff':>8} {'B_eff':>8} {'||[A,B]||':>10} {'Rel.Curv':>10} {'C_norm':>10}")
    
    l1_vals = []
    dist_vals = []
    old_rel = []
    c_norms = []
    
    for d in data:
        cn = d['abs_curv'] / (d['A_eff'] * d['B_eff'] + 1e-10)
        l1_vals.append(d['L1'])
        dist_vals.append(d['dist'])
        old_rel.append(d['rel_curv'])
        c_norms.append(cn)
        print(f"  L{d['L1']:>2} L{d['L2']:>2} {d['dist']:>4} {d['A_eff']:>8.2f} {d['B_eff']:>8.2f} {d['abs_curv']:>10.2f} {d['rel_curv']:>10.4f} {cn:>10.6f}")
    
    if len(l1_vals) > 3:
        corr_l1_old = np.corrcoef(l1_vals, old_rel)[0, 1]
        corr_l1_cn = np.corrcoef(l1_vals, c_norms)[0, 1]
        corr_dist_old = np.corrcoef(dist_vals, old_rel)[0, 1]
        corr_dist_cn = np.corrcoef(dist_vals, c_norms)[0, 1]
        
        print(f"\n  Old: Corr(L1, Rel.Curv) = {corr_l1_old:.3f}, Corr(dist, Rel.Curv) = {corr_dist_old:.3f}")
        print(f"  New: Corr(L1, C_norm)   = {corr_l1_cn:.3f}, Corr(dist, C_norm)   = {corr_dist_cn:.3f}")
        
        l1_groups = {}
        for l1, cn in zip(l1_vals, c_norms):
            l1_groups.setdefault(l1, []).append(cn)
        print(f"\n  C_norm by L1:")
        for l1 in sorted(l1_groups.keys()):
            print(f"    L1={l1}: mean C_norm = {np.mean(l1_groups[l1]):.6f}")

# CCXLIII Equal-effect data
print(f"\n{'='*70}")
print("CCXLIII EQUAL-EFFECT DATA WITH C_norm")
print(f"{'='*70}")

for model_key, model_name in [('deepseek7b', 'DS7B'), ('qwen3', 'Qwen3'), ('glm4', 'GLM4')]:
    filepath = f"results/causal_fiber/{model_key}_ccxliii/console.log"
    try:
        data = parse_ccxliii_console(filepath, model_name)
    except:
        print(f"\n{model_name} CCXLIII: no data")
        continue
    
    if not data:
        print(f"\n{model_name} CCXLIII: no data parsed")
        continue
    
    print(f"\n{'='*60}")
    print(f"{model_name} CCXLIII (equal-effect alpha)")
    print(f"{'='*60}")
    
    print(f"  {'L1':>4} {'alpha':>6} {'A_eff':>8} {'B_eff':>8} {'||[A,B]||':>10} {'Rel.Curv':>10} {'C_norm':>10}")
    
    l1_vals = []
    old_rel = []
    c_norms = []
    
    for d in data:
        cn = d['abs_curv'] / (d['A_eff'] * d['B_eff'] + 1e-10)
        l1_vals.append(d['L1'])
        old_rel.append(d['rel_curv'])
        c_norms.append(cn)
        print(f"  L{d['L1']:>2} {d['alpha_A']:>6.3f} {d['A_eff']:>8.2f} {d['B_eff']:>8.2f} {d['abs_curv']:>10.2f} {d['rel_curv']:>10.4f} {cn:>10.6f}")
    
    if len(l1_vals) > 2:
        corr_l1_old = np.corrcoef(l1_vals, old_rel)[0, 1]
        corr_l1_cn = np.corrcoef(l1_vals, c_norms)[0, 1]
        
        print(f"\n  Old: Corr(L1, Rel.Curv) = {corr_l1_old:.3f}")
        print(f"  New: Corr(L1, C_norm)   = {corr_l1_cn:.3f}")
        
        if corr_l1_cn < -0.3:
            print(f"  *** C_norm STILL higher at shallow L1 -> Shallow seeding is REAL ***")
        elif corr_l1_cn > 0.3:
            print(f"  *** C_norm HIGHER at deep L1 -> OPPOSITE of shallow seeding! ***")
        else:
            print(f"  *** C_norm independent of L1 -> Shallow seeding was ARTIFACT ***")
