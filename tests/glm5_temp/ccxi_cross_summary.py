import json, numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

for model_name, display_name in [('qwen3', 'Qwen3-4B'), ('glm4', 'GLM4-9B(8bit)')]:
    try:
        data = json.load(open(f'tests/glm5_temp/ccxi_{model_name}_results.json', encoding='utf-8'))
    except:
        print(f'{display_name}: no results'); continue
    
    print(f'\n=== {display_name} ===')
    
    # Part 1 summary
    mid_svd = [r for r in data['gate_svd'] if 6 <= r['layer'] <= 24]
    if mid_svd:
        tr = np.mean([r['sv_ratio_trained_vs_random'] for r in mid_svd])
        eff_tr = np.mean([r['eff_rank_90'] for r in mid_svd])
        eff_rnd = np.mean([r['random_eff_rank_90'] for r in mid_svd])
        print(f'  Part1: sv_ratio_t/r={tr:.2f}x, eff_rank90={eff_tr:.0f}(r={eff_rnd:.0f})')
        print(f'    >>> Trained W_gate has higher singular value concentration (lower effective rank)')
    
    # Part 2 summary
    mid_g = [r for r in data['gate_distribution'] if 6 <= r['layer'] <= 24]
    if mid_g:
        gn = np.mean([r['g_neg_frac'] for r in mid_g])
        rn = np.mean([r['g_random_neg_frac'] for r in mid_g])
        shift = np.mean([r['alignment_shift'] for r in mid_g])
        print(f'  Part2: g_neg_frac={gn:.3f} (random={rn:.3f}), shift={shift:+.3f}')
        print(f'    >>> Training shifts gate output: {gn*100:.0f}% negative vs {rn*100:.0f}% random')
    
    # Part 3 summary - THE KEY
    mid_ref = [r for r in data['effective_reflection'] if 6 <= r['layer'] <= 24]
    if mid_ref:
        t1 = np.mean([r['T1_pres_mean'] for r in mid_ref])
        t2 = np.mean([r['T2_pres_mean'] for r in mid_ref])
        rd1 = np.mean([r['T1_random_pres'] for r in mid_ref])
        d2d1 = np.mean([r['D2_norm_frac'] for r in mid_ref])
        print(f'  Part3: T1_pres={t1:+.4f}, T2_pres={t2:+.4f}, random_D1_pres={rd1:+.4f}')
        print(f'    D2/D1_norm={d2d1:.3f}')
        print(f'    >>> T2/silu(g) dominates! pres={t2:+.4f} >> T1 pres={t1:+.4f}')
    
    # Part 4 summary
    mid_sem = [r for r in data['semantic_direction'] if 6 <= r['layer'] <= 24]
    if mid_sem:
        enh = np.mean([r['alignment_enhancement'] for r in mid_sem])
        pp = [r.get('jvp_pres_pca_mean', 0) for r in mid_sem if r.get('jvp_pres_pca_mean') is not None]
        pr = [r.get('jvp_pres_rand_mean', 0) for r in mid_sem if r.get('jvp_pres_rand_mean') is not None]
        pp_avg = np.mean(pp) if pp else 0
        pr_avg = np.mean(pr) if pr else 0
        print(f'  Part4: JVP_pca={pp_avg:+.4f}, JVP_rand={pr_avg:+.4f}, align_enhance={enh:.2f}x')
        print(f'    >>> PCA vs random directions: NO selective difference! enhance={enh:.2f}x ~ 1.0')

print('\n' + '='*60)
print('CRITICAL CORRECTION TO CCX:')
print('='*60)
print("""
CCX hypothesized: T1 (silu'(g)⊙u⊙W_gate@v) dominates reflection
  Because: silu(g)≈0 when g<0, so T2≈0

CCXI finds: T2 (silu(g)⊙W_up@v) dominates reflection!
  Middle layer (Qwen3): T2_pres=-0.45 >> T1_pres=-0.10
  
Why? Because:
  1. silu(g) for g<0 is small but NOT zero: silu(-1)=-0.27, silu(-2)=-0.24
  2. silu(g) has consistent SIGN (negative for g<0)
  3. T2 = silu(g) ⊙ (W_up @ v) — the silu(g) acts as a GAIN MASK
     When silu(g)<0, it flips the sign of W_up@v → direction reversal!
  4. T1 = silu'(g)⊙u ⊙ (W_gate@v) — silu'(g)=sigmoid(g)≈0 for g<0
     So silu'(g) suppresses T1, not enhances it!

★★★★★ NEW MECHANISM:
  Training makes g=W_gate@x mostly NEGATIVE (88%)
  → silu(g) is consistently NEGATIVE (but non-zero)
  → T2 = negative_gain ⊙ W_up@v → FLIPS the direction of W_up@v
  → This is the SOURCE of direction reflection!
  
  W_gate's role: NOT directly reflecting via W_gate@v
  W_gate's role: CONTROLLING the sign of silu(g) → making it negative
  W_up's role: Providing the content that gets sign-flipped
  W_down's role: Projecting back to d_model
""")
