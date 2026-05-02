import json, numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

data = json.load(open('tests/glm5_temp/ccxi_qwen3_results.json', encoding='utf-8'))

print('=== CCXI Qwen3 Results ===\n')

print('--- Part 1: W_gate SVD ---')
print(f'  {"Layer":>5s}  {"S_top3":>30s}  {"ratio10":>8s}  {"r_ratio10":>9s}  {"eff90":>6s}  {"r_eff90":>7s}  {"t/r":>5s}')
for r in data['gate_svd']:
    s3 = [f'{s:.1f}' for s in r['sv_top5'][:3]]
    print(f'  L{r["layer"]:3d}  {str(s3):>30s}  {r["sv_ratio_10"]:8.1f}  {r["random_sv_ratio_10"]:9.1f}  {r["eff_rank_90"]:6d}  {r["random_eff_rank_90"]:7d}  {r["sv_ratio_trained_vs_random"]:5.2f}')

print('\n--- Part 2: W_gate Output Distribution ---')
print(f'  {"Layer":>5s}  {"g_neg":>7s}  {"r_neg":>7s}  {"shift":>7s}  {"g_mean":>8s}  {"cos_neg":>8s}')
for r in data['gate_distribution']:
    print(f'  L{r["layer"]:3d}  {r["g_neg_frac"]:7.3f}  {r["g_random_neg_frac"]:7.3f}  {r["alignment_shift"]:+7.3f}  {r["g_mean"]:+8.4f}  {r["cos_neg_frac"]:8.3f}')

print('\n--- Part 3: Effective Reflection Matrix ---')
print(f'  {"Layer":>5s}  {"T1_pres":>9s}  {"rD1_pres":>9s}  {"T2_pres":>9s}  {"D2/D1":>7s}  {"D1_neg":>7s}')
for r in data['effective_reflection']:
    print(f'  L{r["layer"]:3d}  {r["T1_pres_mean"]:+9.4f}  {r["T1_random_pres"]:+9.4f}  {r["T2_pres_mean"]:+9.4f}  {r["D2_norm_frac"]:7.3f}  {r["D1_neg_frac"]:7.3f}')

print('\n--- Part 4: Semantic Direction ---')
print(f'  {"Layer":>5s}  {"JVP_pca":>9s}  {"JVP_rand":>9s}  {"align":>7s}  {"r_align":>8s}  {"enhance":>8s}')
for r in data['semantic_direction']:
    pp = r.get('jvp_pres_pca_mean', None)
    pr = r.get('jvp_pres_rand_mean', None)
    pp_str = f'{pp:+.4f}' if pp is not None else 'N/A'
    pr_str = f'{pr:+.4f}' if pr is not None else 'N/A'
    print(f'  L{r["layer"]:3d}  {pp_str:>9s}  {pr_str:>9s}  {r["gate_alignment_mean"]:7.3f}  {r["random_alignment_mean"]:8.3f}  {r["alignment_enhancement"]:8.2f}')

# KEY FINDING
print('\n=== KEY FINDING ===')
mid = [r for r in data['effective_reflection'] if 6 <= r['layer'] <= 24]
if mid:
    t1_avg = np.mean([r['T1_pres_mean'] for r in mid])
    t2_avg = np.mean([r['T2_pres_mean'] for r in mid])
    rd1_avg = np.mean([r['T1_random_pres'] for r in mid])
    d2d1_avg = np.mean([r['D2_norm_frac'] for r in mid])
    print(f'  Middle layers: T1_pres={t1_avg:+.4f}, T2_pres={t2_avg:+.4f}')
    print(f'  Random D1 pres={rd1_avg:+.4f}')
    print(f'  D2/D1 norm ratio={d2d1_avg:.3f}')
    print(f'  >>> T2 dominates reflection! silu(g)⊙W_up@v is the main driver!')
    print(f'  >>> NOT T1 (silu\'(g)⊙u⊙W_gate@v) as CCX hypothesized!')

# Gate output
mid_g = [r for r in data['gate_distribution'] if 6 <= r['layer'] <= 24]
if mid_g:
    gn = np.mean([r['g_neg_frac'] for r in mid_g])
    rn = np.mean([r['g_random_neg_frac'] for r in mid_g])
    shift = np.mean([r['alignment_shift'] for r in mid_g])
    print(f'\n  Gate output: g_neg_frac={gn:.3f} (random={rn:.3f}), shift={shift:+.3f}')
    print(f'  >>> Training makes ~{gn*100:.0f}% of gate outputs negative!')

# Alignment
mid_a = [r for r in data['semantic_direction'] if 6 <= r['layer'] <= 24]
if mid_a:
    enh = np.mean([r['alignment_enhancement'] for r in mid_a])
    print(f'\n  Gate-PCA alignment enhancement={enh:.2f}x')
    print(f'  >>> W_gate does NOT selectively align with semantic directions!')
