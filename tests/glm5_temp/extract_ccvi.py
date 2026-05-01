import sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

for model in ['qwen3', 'glm4', 'deepseek7b']:
    path = f'tests/glm5_temp/ccvi_semantic_to_logit_{model}.json'
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    
    print(f'\n===== {model} =====')
    r = data['wu_effective_rank']
    print(f'  W_U rank: r90={r["r90"]}, r95={r["r95"]}')
    
    print('  Logit shift SEM/RND:')
    for li_str in sorted(data['logit_analysis'].keys(), key=int):
        li_data = data['logit_analysis'][li_str]
        if '_summary' in li_data:
            s = li_data['_summary']
            print(f'    L{li_str}: SEM={s["avg_semantic_logit_shift"]:.4f}, RND={s["avg_random_logit_shift"]:.4f}, ratio={s["semantic_random_ratio"]:.2f}')
    
    print('  W_U projection SEM/RND:')
    for li_str in sorted(data['wu_projection'].keys(), key=int):
        li_data = data['wu_projection'][li_str]
        if '_summary' in li_data:
            s = li_data['_summary']
            print(f'    L{li_str}: SEM={s["avg_semantic_ratio"]:.4f}, RND={s["avg_random_ratio"]:.4f}, ratio={s["semantic_random_ratio"]:.2f}')
