import json

for model in ['qwen3', 'glm4', 'deepseek7b']:
    with open(f'results/causal_fiber/{model}_cclxxxii/exp3_knn_errors.json') as f:
        data = json.load(f)
    print(f'=== {model} kNN k=1 ===')
    for r in data['results']:
        if r['k'] == 1:
            print(f"  L{r['layer']}: acc={r['knn_acc']:.3f} errors={r['n_errors']} by_true={r['error_by_true_cat']}")
    print(f'=== {model} best k per layer ===')
    layers = sorted(set(r['layer'] for r in data['results']))
    for li in layers:
        entries = [r for r in data['results'] if r['layer'] == li]
        best = max(entries, key=lambda x: x['knn_acc'])
        print(f"  L{li}: best_k={best['k']} acc={best['knn_acc']:.3f}")
    print()
