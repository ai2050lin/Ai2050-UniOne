import json

for model in ['qwen3', 'glm4', 'deepseek7b']:
    path = f'tests/glm5_temp/ccxxxvii_simplex_additivity_{model}.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f'\n===== {model.upper()} =====')
    
    # Part 1: Additivity
    add = data.get('additivity', {})
    best_layer = None
    best_matches = -1
    for layer_key, layer_data in add.items():
        matches = sum(1 for n_key, n_data in layer_data.items() if n_data.get('match', False))
        if matches > best_matches:
            best_matches = matches
            best_layer = layer_key
    
    if best_layer:
        print(f'\n--- Best Additivity Layer: {best_layer} ({best_matches} matches) ---')
        for n_key in sorted(add[best_layer].keys()):
            nd = add[best_layer][n_key]
            nc = nd["n_classes"]
            ns = nd["n_separating_PCs"]
            ex = nd["expected_N_minus_1"]
            dn = nd["delta_n_sep"]
            mt = "Y" if nd["match"] else "N"
            reg = ""
            if nd.get('geometry') and nd['geometry']:
                reg = f", reg={nd['geometry']['regularity_score']:.3f}"
            print(f'  {n_key}: N={nc}, n_sep={ns}, exp={ex}, d={dn}, match={mt}{reg}')
    
    # All layers additivity summary
    print(f'\n--- All Layers Additivity ---')
    for layer_key in sorted(add.keys()):
        layer_data = add[layer_key]
        matches = sum(1 for n_key, n_data in layer_data.items() if n_data.get('match', False))
        deltas = [layer_data[k]["delta_n_sep"] for k in sorted(layer_data.keys())]
        n_matches_up_to_6 = sum(1 for k in sorted(layer_data.keys()) 
                                if layer_data[k]["n_classes"] <= 6 and layer_data[k]["match"])
        print(f'  {layer_key}: matches={matches}/8, up_to_N6={n_matches_up_to_6}/4, deltas={deltas}')
    
    # Part 2: Attention
    attn = data.get('attention_propagation', {})
    head_stats = {}
    for pair_key, pair_data in attn.items():
        for layer_key, layer_data in pair_data.items():
            for hc in layer_data.get('head_contributions', [])[:5]:
                h_key = f'{layer_key}_H{hc["head"]}'
                if h_key not in head_stats:
                    head_stats[h_key] = {'last': [], 'comp': []}
                head_stats[h_key]['last'].append(hc['last_contribution'])
                head_stats[h_key]['comp'].append(hc['comp_contribution'])
    
    print(f'\n--- Top Attention Heads ---')
    sorted_heads = sorted(head_stats.items(), key=lambda x: max(x[1]['last']), reverse=True)
    for h_key, stats in sorted_heads[:8]:
        avg_last = sum(stats['last'])/len(stats['last'])
        avg_comp = sum(stats['comp'])/len(stats['comp'])
        max_last = max(stats['last'])
        print(f'  {h_key}: max_last={max_last:.4f}, avg_last={avg_last:.4f}, avg_comp={avg_comp:.4f}')
    
    # Part 3: Cross-domain
    cross = data.get('cross_domain', {})
    print(f'\n--- Cross-Domain ---')
    for key, val in cross.items():
        domain = val["domain"]
        nc = val["n_classes"]
        ns = val["n_separating_PCs"]
        ex = val["expected_N_minus_1"]
        mt = "Y" if val["match"] else "N"
        geo = val.get("geometry")
        if geo:
            print(f'  {key}: N={nc}, n_sep={ns}, exp={ex}, match={mt}, '
                  f'reg={geo["regularity_score"]:.3f}, angle_dev={geo["angle_deviation"]:.1f}')
        else:
            print(f'  {key}: N={nc}, n_sep={ns}, exp={ex}, match={mt}')
