import json, sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('results/causal_fiber/qwen3_cclxv/exp2_basin_boundary.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

pd = data['per_pair_data']
keys = list(pd.keys())

for k in keys[:2]:
    print(f"\nPair: {k}")
    for layer in ['0', '4', '16', '35']:
        if layer in pd[k]:
            curve = pd[k][layer]['alpha_curve']
            print(f"  L{layer}:")
            for i in range(0, len(curve), 5):
                pt = curve[i]
                print(f"    a={pt['alpha']:.2f} pA={pt['prob_A']:.6f} pB={pt['prob_B']:.6f} diff={pt['prob_diff']:.6f}")
