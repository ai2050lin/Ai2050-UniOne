"""检查每个模型每层的差分向量样本数"""
import json

for model in ['deepseek7b', 'qwen3', 'glm4']:
    print(f'\n===== {model.upper()} sample counts =====')
    with open(f'results/causal_fiber/{model}_direct_head/s1_direct_head.json') as f:
        data = json.load(f)
    for layer_key in sorted(data.keys()):
        layer_data = data[layer_key]
        if 'error' in layer_data:
            print(f'  {layer_key}: {layer_data["error"]}')
            continue
        heads = layer_data['all_heads']
        total_diffs = sum(h['n_diffs'] for h in heads.values())
        n_heads = len(heads)
        print(f'  {layer_key}: {n_heads} heads, {total_diffs} total diffs, avg {total_diffs/max(n_heads,1):.0f}/head')
        # Per-head sample count
        sample_counts = [(h, heads[h]['n_diffs']) for h in heads]
        sample_counts.sort(key=lambda x: -x[1])
        print(f'    top: {sample_counts[:3]}')
