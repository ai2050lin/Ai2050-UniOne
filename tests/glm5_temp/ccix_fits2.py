"""CCIX algebraic fits - ASCII only output"""
import json, numpy as np
from pathlib import Path

def fit_algebraic_model(layers, l2_values):
    x = np.array(layers, dtype=float)
    y = np.array(l2_values, dtype=float)
    results = {}
    try:
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        r2 = 1 - np.sum((y-y_pred)**2) / np.sum((y-np.mean(y))**2)
        results['linear'] = {'a': float(coeffs[0]), 'b': float(coeffs[1]), 'r2': float(r2)}
    except: results['linear'] = {'r2': 0}
    try:
        y_pos = np.maximum(y, 1e-6)
        coeffs = np.polyfit(x, np.log(y_pos), 1)
        y_pred = np.exp(coeffs[1]) * np.exp(coeffs[0] * x)
        r2 = 1 - np.sum((y-y_pred)**2) / np.sum((y-np.mean(y))**2)
        results['exponential'] = {'a': float(np.exp(coeffs[1])), 'b': float(coeffs[0]), 'r2': float(r2)}
    except: results['exponential'] = {'r2': 0}
    try:
        x_pos = np.maximum(x, 1.0); y_pos = np.maximum(y, 1e-6)
        coeffs = np.polyfit(np.log(x_pos), np.log(y_pos), 1)
        y_pred = np.exp(coeffs[1]) * x_pos ** coeffs[0]
        r2 = 1 - np.sum((y-y_pred)**2) / np.sum((y-np.mean(y))**2)
        results['power_law'] = {'a': float(np.exp(coeffs[1])), 'b': float(coeffs[0]), 'r2': float(r2)}
    except: results['power_law'] = {'r2': 0}
    try:
        log_x = np.log(x + 1)
        coeffs = np.polyfit(log_x, y, 1)
        y_pred = coeffs[0] * log_x + coeffs[1]
        r2 = 1 - np.sum((y-y_pred)**2) / np.sum((y-np.mean(y))**2)
        results['logarithmic'] = {'a': float(coeffs[0]), 'b': float(coeffs[1]), 'r2': float(r2)}
    except: results['logarithmic'] = {'r2': 0}
    return results

models = ['deepseek7b', 'qwen3', 'glm4']
all_fits = {}

for model in models:
    path = Path(f'results/causal_fiber/{model}_ccix/full_results.json')
    if not path.exists(): 
        print(f'MISSING: {model}')
        continue
    with open(path) as f: 
        data = json.load(f)
    resid = data.get('resid', {})
    
    print(f'\n{"="*60}')
    print(f'  {model.upper()}')
    print(f'{"="*60}')
    
    # Residual median
    print(f'\n--- Residual median_l2 ---')
    for ln in sorted(resid.keys()):
        parts = []
        for feat in ['tense', 'polarity', 'number', 'sentiment', 'semantic_topic']:
            if feat in resid[ln]:
                parts.append(f"{feat}={resid[ln][feat]['median_l2']:.0f}")
        print(f'  {ln}: {", ".join(parts)}')
    
    # Fits
    model_fits = {}
    print(f'\n--- Algebraic Fits (median_l2) ---')
    for feat in ['tense', 'polarity', 'number', 'sentiment', 'semantic_topic']:
        layers_x, l2_medians = [], []
        for ln in sorted(resid.keys()):
            if feat in resid[ln]:
                lidx = int(ln.replace('L', ''))
                layers_x.append(float(lidx))
                l2_medians.append(resid[ln][feat]['median_l2'])
        if len(layers_x) >= 3:
            fits = fit_algebraic_model(layers_x, l2_medians)
            model_fits[feat] = fits
            best = max(fits.keys(), key=lambda k: fits[k].get('r2', 0))
            r2_parts = [f"{k}={v['r2']:.4f}" for k, v in fits.items()]
            print(f'  {feat}: {", ".join(r2_parts)}')
            print(f'    BEST: {best} R2={fits[best]["r2"]:.4f}')
    
    all_fits[model] = model_fits
    
    # Contribution
    contrib = data.get('contribution', {})
    if contrib:
        print(f'\n--- Attn vs MLP (median) ---')
        for ln in sorted(contrib.keys()):
            parts = []
            for feat in ['tense', 'polarity', 'number']:
                if feat in contrib[ln]:
                    d = contrib[ln][feat]
                    parts.append(f"{feat}:{d['attn_pct']:.0f}/{d['mlp_pct']:.0f}")
            print(f'  {ln}: {", ".join(parts)} (attn/mlp%)')
    
    # Semantic
    sem = data.get('semantic', {})
    if sem:
        print(f'\n--- Semantic median_l2 ---')
        for ln in sorted(sem.keys()):
            parts = []
            for feat in ['sentiment', 'semantic_topic']:
                if feat in sem[ln]:
                    parts.append(f"{feat}={sem[ln][feat]['median_l2']:.0f}")
            print(f'  {ln}: {", ".join(parts)}')

# Cross-model
print(f'\n\n{"="*60}')
print(f'CROSS-MODEL BEST FIT')
print(f'{"="*60}')
for feat in ['tense', 'polarity', 'number', 'sentiment', 'semantic_topic']:
    print(f'\n  {feat}:')
    for model in models:
        if model in all_fits and feat in all_fits[model]:
            fits = all_fits[model][feat]
            best = max(fits.keys(), key=lambda k: fits[k].get('r2', 0))
            print(f'    {model}: {best} R2={fits[best]["r2"]:.4f}')

# Save
with open('results/causal_fiber/ccix_fits_summary.json', 'w') as f:
    json.dump(all_fits, f, indent=2, default=str)
print(f'\nSaved to results/causal_fiber/ccix_fits_summary.json')
