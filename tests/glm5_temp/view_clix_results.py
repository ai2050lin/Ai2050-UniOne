import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('results/clix_adversarial_balance/clix_qwen3_results.json', encoding='utf-8') as f:
    data = json.load(f)
r = data['results']
for word, d in r.items():
    print(f"{word}: final={d['final_logit']:.2f}, cumG={d['cum_G']:.1f}, cumA={d['cum_A']:.1f}, G/A={d['G_A_ratio']:.3f}, global_bal={d['global_balance']:.4f}, global_cos={d['global_cos_GA']:.4f}, same={d['same_sign_count']}, opp={d['opposite_sign_count']}")
    iv = d['interventions']
    print(f"  normal_prob={iv['normal']['prob']:.6f}")
    print(f"  only_A: prob={iv['only_A']['prob']:.6f}(d={iv['only_A']['prob_delta']:+.6f})")
    print(f"  only_G: prob={iv['only_G']['prob']:.8f}(d={iv['only_G']['prob_delta']:+.8f})")
    print(f"  flip_G: prob={iv['flip_G']['prob']:.8f}(d={iv['flip_G']['prob_delta']:+.8f})")
    print(f"  amplify_G: prob={iv['amplify_G']['prob']:.8f}(d={iv['amplify_G']['prob_delta']:+.8f})")
    print(f"  last3_only: prob={iv['last3_only']['prob']:.2e}(d={iv['last3_only']['prob_delta']:+.2e})")
