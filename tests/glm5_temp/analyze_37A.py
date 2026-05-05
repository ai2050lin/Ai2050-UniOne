import json, numpy as np

for model in ['qwen3', 'glm4', 'deepseek7b']:
    with open(f'ccml_phase37_expA_{model}_results.json') as f:
        data = json.load(f)
    
    d_model = data['d_model']
    k_wu = data['k_wu']
    k_d = k_wu / d_model
    n_layers = data['n_layers']
    
    print(f'\n===== {model} (k/d={k_d:.4f}, n_layers={n_layers}) =====')
    print(f'  Late-layer injection results:')
    
    for concept in ['apple', 'dog', 'hammer']:
        for src_l_str in data['aligned_trajectories'].get(concept, {}):
            src_l = int(src_l_str)
            aligned = data['aligned_trajectories'][concept][src_l_str]
            orth = data['orthogonal_trajectories'][concept][src_l_str]
            
            final_l = str(n_layers - 1)
            if final_l not in aligned or final_l not in orth:
                continue
            
            a_final = aligned[final_l]
            o_final = orth[final_l]
            
            a_mult = a_final['wu_proj_ratio'] / k_d
            o_mult = o_final['wu_proj_ratio'] / k_d
            
            a_wu = a_final['wu_proj_ratio']
            o_wu = o_final['wu_proj_ratio']
            a_nr = a_final['norm_ratio']
            o_nr = o_final['norm_ratio']
            
            print(f'  {concept} L{src_l}->L{n_layers-1}: '
                  f'aligned WU={a_wu:.4f} ({a_mult:.1f}x), '
                  f'orth WU={o_wu:.4f} ({o_mult:.1f}x), '
                  f'norm: a={a_nr:.1f} o={o_nr:.1f}')
    
    # 也看中间层轨迹 (只看一个代表性的: apple L18 or L9)
    concept = 'apple'
    for src_l_str in data['aligned_trajectories'].get(concept, {}):
        src_l = int(src_l_str)
        if src_l not in [9, 10, 18, 20]:  # 只看早期注入
            continue
        aligned = data['aligned_trajectories'][concept][src_l_str]
        orth = data['orthogonal_trajectories'][concept][src_l_str]
        
        print(f'\n  apple L{src_l} trajectory (sample layers):')
        sample_ls = sorted([int(x) for x in aligned.keys()])
        # 只打印几个关键层
        step = max(1, len(sample_ls) // 6)
        for li in sample_ls[::step] + [sample_ls[-1]]:
            li_str = str(li)
            a = aligned[li_str]
            o = orth[li_str]
            a_m = a['wu_proj_ratio'] / k_d
            o_m = o['wu_proj_ratio'] / k_d
            print(f'    L{li}: aligned WU={a["wu_proj_ratio"]:.4f} ({a_m:.1f}x), '
                  f'orth WU={o["wu_proj_ratio"]:.4f} ({o_m:.1f}x), '
                  f'norm: a={a["norm_ratio"]:.2f} o={o["norm_ratio"]:.2f}')
