"""Debug probe: why is nearest-centroid accuracy 0?"""
import json, numpy as np

with open('results/causal_fiber/qwen3_cclxxvi/exp4_semantic_consistency.json') as f:
    d = json.load(f)

for r in d['layer_results'][:3]:
    li = r['layer']
    intra = r['cat_consistency']
    inter = r['inter_cat_mean_cos']
    print(f"L{li}: inter_cos={inter:.3f}")
    for cat, v in intra.items():
        print(f"  {cat}: intra_cos={v['mean_intra_cos']:.3f} n={v['n_words']}")

# The issue is clear: intra_cos and inter_cos are BOTH very high (0.65-0.98)
# This means ALL words have very similar W_down@u vectors regardless of category
# So nearest-centroid can't distinguish them because centroids are nearly identical

print("\n=== Diagnosis ===")
print("intra_cos > 0.6 and inter_cos > 0.6")
print("All W_down@u vectors are very similar (cos > 0.6)")
print("Category information is NOT in the absolute direction of W_down@u")
print("It must be in the RELATIVE differences (which is what Δg⊙ū and Δu capture)")
