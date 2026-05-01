import json

with open('tests/glm5_temp/ccxli_three_challenges_qwen3.json','r',encoding='utf-8') as f:
    data = json.load(f)
mr = data['model_results']

# Dirty data comparison
print('=== Hard Issue 2: Dirty vs Clean Data ===')
for k in ['emotion_6','emotion_dirty','emotion_behaviors']:
    d = mr.get(k,{})
    if d:
        fr = d['best_fit_r2']
        eu = d['best_edge_uni']
        ad = d['best_angle_dev']
        print(f'  {k:>20}: fit_r2={fr:.4f}, edge_uni={eu:.4f}, angle_dev={ad:.2f}')

# N=6 vs N=8
print()
print('=== N=6 vs N=8 ===')
for base in ['habitat','emotion','occupation']:
    k6 = f'{base}_6'
    k8 = f'{base}_8'
    r6 = mr.get(k6,{})
    r8 = mr.get(k8,{})
    if r6 and r8:
        delta = r8['best_fit_r2'] - r6['best_fit_r2']
        print(f'  {base:>12}: N=6 fit_r2={r6["best_fit_r2"]:.4f}, N=8 fit_r2={r8["best_fit_r2"]:.4f}, delta={delta:+.4f}')

# Hard Issue 3: F-ratio
print()
print('=== Hard Issue 3: Between vs Within F-ratio ===')
for domain_name, dr in mr.items():
    wi = dr.get('within_class', {})
    summary = wi.get('_summary', {})
    if summary:
        fr = summary['f_ratio']
        bv = summary['between_var']
        wv = summary['avg_within_var']
        print(f'  {domain_name:>20}: F_ratio={fr:.2f} (between={bv:.6f}, within={wv:.6f})')
