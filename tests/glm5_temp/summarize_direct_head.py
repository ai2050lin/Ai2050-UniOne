"""总结Phase CCV的直接head输出结果"""
import json

for model in ['deepseek7b', 'qwen3', 'glm4']:
    print(f'\n{"="*70}')
    print(f'{model.upper()} S1 Direct Head Alignment')
    print(f'{"="*70}')
    with open(f'results/causal_fiber/{model}_direct_head/s1_direct_head.json') as f:
        data = json.load(f)
    for layer_key in sorted(data.keys()):
        layer_data = data[layer_key]
        if 'error' in layer_data:
            print(f'  {layer_key}: {layer_data["error"]}')
            continue
        top_heads = layer_data.get('top_heads', [])
        if top_heads:
            h_name, alignment, norm = top_heads[0]
            all_heads = layer_data.get('all_heads', {})
            h_detail = all_heads.get(h_name, {})
            feat_str = ''
            if 'feature_cos' in h_detail:
                items = sorted(h_detail['feature_cos'].items(), key=lambda x: -x[1]['mean'])[:3]
                feat_str = ', '.join(f"{k}={v['mean']:.3f}" for k, v in items)
            print(f'  {layer_key}: top={h_name} align={alignment:.4f} norm={norm:.1f} | {feat_str}')
            # Top 3 heads
            for h_name2, alignment2, norm2 in top_heads[:3]:
                h_detail2 = all_heads.get(h_name2, {})
                feat_str2 = ''
                if 'feature_cos' in h_detail2:
                    items2 = sorted(h_detail2['feature_cos'].items(), key=lambda x: -x[1]['mean'])[:2]
                    feat_str2 = ', '.join(f"{k}={v['mean']:.3f}" for k, v in items2)
                print(f'         {h_name2}: align={alignment2:.4f} | {feat_str2}')

for model in ['deepseek7b', 'qwen3', 'glm4']:
    print(f'\n{"="*70}')
    print(f'{model.upper()} S3 Causal Atoms')
    print(f'{"="*70}')
    with open(f'results/causal_fiber/{model}_direct_head/s3_causal_atoms.json') as f:
        data = json.load(f)
    for layer_key in sorted(data.keys()):
        layer_data = data[layer_key]
        if 'error' in layer_data:
            print(f'  {layer_key}: {layer_data["error"]}')
            continue
        head_dom = layer_data.get('head_dominant', {})
        feat_dom = layer_data.get('feat_dominant', {})
        sorted_heads = sorted(head_dom.items(), key=lambda x: sum(x[1]['norm_profile'].values()), reverse=True)[:5]
        print(f'  {layer_key}:')
        for h_name, hd in sorted_heads:
            dom_feat = hd['dominant_feature']
            dom_score = hd['dominant_score']
            total_norm = sum(hd['norm_profile'].values())
            feat_norms = ', '.join(f"{k}={v:.0f}" for k, v in sorted(hd['norm_profile'].items(), key=lambda x: -x[1])[:3])
            print(f'    {h_name}: norm={total_norm:.0f}, dominant={dom_feat}({dom_score:.3f}) | {feat_norms}')
        for fname, fd in feat_dom.items():
            print(f'    {fname} -> {fd["dominant_head"]} (score={fd["dominant_score"]:.3f})')

for model in ['deepseek7b', 'qwen3', 'glm4']:
    print(f'\n{"="*70}')
    print(f'{model.upper()} S2 W_o SVD')
    print(f'{"="*70}')
    with open(f'results/causal_fiber/{model}_direct_head/s2_wo_svd.json') as f:
        data = json.load(f)
    for layer_key in sorted(data.keys()):
        ld = data[layer_key]
        if 'error' in ld:
            print(f'  {layer_key}: {ld["error"]}')
            continue
        print(f'  {layer_key}: top1={ld["top1_ratio"]*100:.1f}%, top5={ld["top5_ratio"]*100:.1f}%, eff_rank90={ld["eff_rank_90"]}, cond={ld["condition_number"]:.1f}, head_corr={ld["mean_head_corr"]:.4f}')
