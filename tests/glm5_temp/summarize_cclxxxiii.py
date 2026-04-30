import json
import numpy as np

for model in ['qwen3', 'glm4', 'deepseek7b']:
    print(f'\n===== {model.upper()} =====')
    
    # Exp1: kNN graph
    with open(f'results/causal_fiber/{model}_cclxxxiii/exp1_knn_graph.json') as f:
        d = json.load(f)
    print('\n--- Exp1: kNN Graph (k=5) ---')
    for r in d['results']:
        if r['k_nn'] == 5:
            print(f"  L{r['layer']}: within={r['within_ratio']:.3f} (random={r['random_within_ratio_mean']:.3f}), "
                  f"n_comp={r['n_components']}, purity={r['comp_purity_mean']:.3f}, "
                  f"within/random={r['within_vs_random_ratio']:.2f}")
    
    # Exp2: Distance decomposition - key stats
    with open(f'results/causal_fiber/{model}_cclxxxiii/exp2_distance_decomp.json') as f:
        d = json.load(f)
    print('\n--- Exp2: Distance Decomposition ---')
    for r in d['results']:
        if r['type'] == 'distance_stats':
            print(f"  L{r['layer']}: w/b={r['wb_ratio']:.3f}, cent/within={r['centroid_to_within_ratio']:.3f}")
    
    # Exp2: PC alignment - pick a middle layer
    pc_data = [r for r in d['results'] if r['type'] == 'pc_alignment']
    if pc_data:
        layers_seen = sorted(set(r['layer'] for r in pc_data))
        mid_layer = layers_seen[len(layers_seen)//2]
        print(f'  --- PC Alignment L{mid_layer} ---')
        for r in pc_data:
            if r['layer'] == mid_layer:
                print(f"    d={r['d_proj']}: top_d_acc={r['knn_acc_top_d']:.3f}, resid_acc={r['knn_acc_residual']:.3f}, var={r['var_in_top_d']:.3f}")
    
    # Exp3: Residual vs WDU
    with open(f'results/causal_fiber/{model}_cclxxxiii/exp3_resid_comparison.json') as f:
        d = json.load(f)
    print('\n--- Exp3: WDU vs Residual ---')
    wdu_data = [r for r in d['results'] if r['space'] == 'wdu']
    resid_data = [r for r in d['results'] if r['space'] == 'resid']
    for wr, rr in zip(wdu_data, resid_data):
        if wr['layer'] % 8 == 0:
            print(f"  L{wr['layer']}: wdu_knn={wr['knn_k1_acc']:.3f} resid_knn={rr['knn_k1_acc']:.3f}, "
                  f"wdu_ratio={wr['cat_vs_random_ratio']:.3f} resid_ratio={rr['cat_vs_random_ratio']:.3f}")
    
    # Exp4: Neighborhood purity
    with open(f'results/causal_fiber/{model}_cclxxxiii/exp4_neighborhood_purity.json') as f:
        d = json.load(f)
    print('\n--- Exp4: Neighborhood Purity (k=5) ---')
    for r in d['results']:
        if r['k_nn'] == 5:
            cat_str = ', '.join(f"{k}={v:.3f}" for k, v in sorted(r['cat_purity_means'].items()))
            print(f"  L{r['layer']}: purity={r['purity_mean']:.3f}+/-{r['purity_std']:.3f} ({cat_str})")
    
    # Cross-category edges (k=10)
    with open(f'results/causal_fiber/{model}_cclxxxiii/exp1_knn_graph.json') as f:
        d = json.load(f)
    print('\n--- Cross-category bridge edges (k=10, best layer) ---')
    best = [r for r in d['results'] if r['k_nn'] == 10]
    if best:
        best_r = max(best, key=lambda x: x['within_ratio'])
        print(f"  L{best_r['layer']}: {best_r['cross_cat_edges_top5']}")
