import json, numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

models = ['qwen3', 'glm4']
names = {'qwen3': 'Qwen3-4B', 'glm4': 'GLM4-9B(8bit)'}

for m in models:
    try:
        data = json.load(open(f'tests/glm5_temp/ccx_{m}_results.json', encoding='utf-8'))
    except:
        print(f'{m}: no results'); continue
    
    print(f'=== {names[m]} ===')
    
    # Part 1
    print('  Part 1: Random vs Trained Jacobian')
    for r in data['random_jacobian']:
        li = r['layer']
        tp = r['trained_pres_pca']
        rp = r['random_pres_pca']
        diff = r['pres_diff_trained_vs_random']
        print(f'  L{li:3d}  trained={tp:+.4f}  random={rp:+.4f}  diff={diff:+.4f}')
    
    # Part 2
    if 'swiglu_eigenvalue' in data:
        print('  Part 2: SwiGLU Eigenvalue')
        for r in data['swiglu_eigenvalue']:
            li = r['layer']
            pm = r['pres_mean']
            rq = r['rq_mean']
            d2p = r['D2_pos_frac']
            ss = r['same_sign_frac']
            print(f'  L{li:3d}  pres={pm:+.4f}  RQ={rq:+.4f}  D2+={d2p:.3f}  same={ss:.3f}')
    
    # Part 3
    if 'partial_random' in data:
        print('  Part 3: Partial Random')
        for r in data['partial_random']:
            li = r['layer']
            tp = r['trained_pres']
            rg = r['rand_gate_pres']
            ru = r['rand_up_pres']
            rd = r['rand_down_pres']
            dg = r['delta_gate']
            print(f'  L{li:3d}  train={tp:+.4f}  rg={rg:+.4f}  ru={ru:+.4f}  rd={rd:+.4f}  dg={dg:+.4f}')
    
    # Key summary
    mid = [r for r in data['random_jacobian'] if 3 <= r['layer'] <= data['n_layers']*0.7]
    t_avg = np.mean([r['trained_pres_pca'] for r in mid])
    r_avg = np.mean([r['random_pres_pca'] for r in mid])
    print(f'\n  KEY: trained_pres={t_avg:+.4f}, random_pres={r_avg:+.4f}, diff={t_avg-r_avg:+.4f}')
    
    if 'partial_random' in data:
        gd = np.mean([abs(r['delta_gate']) for r in data['partial_random']])
        ud = np.mean([abs(r['delta_up']) for r in data['partial_random']])
        dd = np.mean([abs(r['delta_down']) for r in data['partial_random']])
        print(f'  Partial: gate_d={gd:.4f}, up_d={ud:.4f}, down_d={dd:.4f}')
    print()
